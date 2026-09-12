from __future__ import annotations

import glob
import logging
import os
import struct
import subprocess
import tempfile

import httpx
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

log = logging.getLogger("whisper-bazarr-adapter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Port 8000 ist korrekt, da wir 8003 in der docker-compose darauf gemappt haben
UPSTREAM = os.getenv("FWSERVER", "http://faster-whisper:8000")
REQUEST_TIMEOUT = float(os.getenv("FWSERVER_TIMEOUT", "1200"))
FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")

OUTPUT_TO_RESPONSE_FORMAT = {
    "srt": "srt",
    "vtt": "vtt",
    "txt": "text",
    "text": "text",
}
RESPONSE_FORMAT_MEDIA = {"srt": "text/plain", "vtt": "text/vtt", "text": "text/plain"}

app = FastAPI(title="whisper-bazarr-adapter")
_upstream: httpx.Client | None = None


def _client() -> httpx.Client:
    global _upstream
    if _upstream is None:
        _upstream = httpx.Client(
            base_url=UPSTREAM,
            timeout=REQUEST_TIMEOUT,
        )
    return _upstream


def _health_sync() -> dict:
    r = _client().get("/health")
    r.raise_for_status()
    return r.json()


def _post_sync(endpoint: str, data: dict, files: dict) -> httpx.Response:
    return _client().post(endpoint, data=data, files=files)


def _pcm_to_wav(pcm_data: bytes) -> bytes:
    data_size = len(pcm_data)
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,
        1,
        16000,
        32000,
        2,
        16,
        b'data',
        data_size
    )
    return header + pcm_data


def _normalize_audio(raw: bytes, hint_filename: str | None = None) -> bytes:
    if not raw:
        raise ValueError("empty audio payload")
    ext = os.path.splitext(hint_filename or "")[1] or ".bin"
    tmpdir = tempfile.mkdtemp(prefix="wba_")
    inp = os.path.join(tmpdir, "in" + ext)
    try:
        with open(inp, "wb") as f:
            f.write(raw)
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", inp,
            "-f", "wav", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", "replace").strip()
            raise ValueError(f"ffmpeg normalisation failed: {err[-500:]}")
        if not proc.stdout:
            raise ValueError("ffmpeg produced no audio (input had no decodable audio track?)")
        return proc.stdout
    finally:
        for x in glob.glob(os.path.join(tmpdir, "*")):
            try:
                os.remove(x)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


@app.get("/status")
async def status():
    upstream_ok = True
    try:
        await run_in_threadpool(_health_sync)
    except Exception as exc: 
        upstream_ok = False
        log.warning("upstream unreachable: %s", exc)
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok" if upstream_ok else "upstream_unreachable",
            "engine": "whisperx",
            "upstream": "up" if upstream_ok else "down",
            "upstream_url": UPSTREAM,
            "models": ["whisperx-blackwell"],
        },
    )


@app.post("/asr")
async def asr(
    audio_file: UploadFile = File(...),
    task: str = Query("transcribe", pattern="^(transcribe|translate)$"),
    language: str | None = Query(None),
    output: str = Query("srt"),
    encode: bool = Query(True), 
    initial_prompt: str | None = Query(None),
):
    try:
        out_key = output.lower()
    except Exception:
        out_key = ""
    if out_key not in OUTPUT_TO_RESPONSE_FORMAT:
        return _error(400, f"unsupported output '{output}' (want srt|vtt|txt)")
    response_format = OUTPUT_TO_RESPONSE_FORMAT[out_key]

    raw = await audio_file.read()
    try:
        if not encode:
            wav = _pcm_to_wav(raw)
        else:
            wav = await run_in_threadpool(_normalize_audio, raw, audio_file.filename)
    except (ValueError, subprocess.TimeoutExpired) as exc:
        return _error(400, str(exc))

    endpoint = "/transcribe"
    files = {"file": (f"{audio_file.filename or 'audio'}.wav", wav, "audio/wav")}
    
    # Wir übergeben vorsichtshalber beide gängigen Parameter-Namen für das Ausgabeformat
    data = {
        "response_format": response_format,
        "output_format": response_format
    }
    if language:
        data["language"] = language

    try:
        r = await run_in_threadpool(_post_sync, endpoint, data, files)
    except httpx.HTTPError as exc:
        log.exception("upstream request failed")
        return _error(502, f"failed to reach upstream WhisperX: {exc}")

    if r.status_code >= 400:
        return _error(r.status_code, _friendly_error(r))

    media = RESPONSE_FORMAT_MEDIA.get(response_format, "text/plain")
    disposition = _content_disposition(audio_file.filename, out_key)
    headers = {"Content-Disposition": disposition} if disposition else {}
    
    # Optionaler Check: Falls WhisperX JSON statt SRT liefert
    if b'"segments":' in r.content[:100]:
        log.warning("WhisperX returned JSON instead of raw SRT! Converter might be needed.")

    log.info("transcribed %s bytes -> %s (%s)", len(raw), response_format, r.status_code)
    return Response(content=r.content, media_type=media, status_code=200, headers=headers)


@app.post("/detect-language")
async def detect_language(audio_file: UploadFile = File(...), encode: bool = Query(True)):
    raw = await audio_file.read()
    try:
        if not encode:
            wav = _pcm_to_wav(raw)
        else:
            wav = await run_in_threadpool(_normalize_audio, raw, audio_file.filename)
    except (ValueError, subprocess.TimeoutExpired) as exc:
        return _error(400, str(exc))

    files = {"file": (f"{audio_file.filename or 'audio'}.wav", wav, "audio/wav")}
    data = {"response_format": "json"}
    try:
        r = await run_in_threadpool(_post_sync, "/transcribe", data, files)
    except httpx.HTTPError as exc:
        return _error(502, f"failed to reach upstream: {exc}")
    if r.status_code >= 400:
        return _error(r.status_code, _friendly_error(r))

    try:
        body = r.json()
        lang_code = (body.get("language") or "").strip().lower()
    except Exception:
        lang_code = "und"
        
    return JSONResponse(content=_detect_language_payload(lang_code, {}))


def _error(code: int, msg: str):
    log.warning("returning %s: %s", code, msg)
    return JSONResponse(status_code=code, content={"detail": msg, "status": "error"})


def _friendly_error(r: httpx.Response) -> str:
    try:
        j = r.json()
        if isinstance(j, dict) and "detail" in j:
            return str(j["detail"])
    except Exception:
        pass
    text = (r.text or "").strip()
    return text[:500] or f"upstream returned HTTP {r.status_code}"


def _content_disposition(filename: str | None, out_key: str) -> str | None:
    if not filename:
        return None
    base = filename.rsplit(".", 1)[0] if "." in filename else filename
    return f'attachment; filename="{base}.{out_key}"'


def _detect_language_payload(lang_code: str, body: dict) -> dict:
    name_map = {"en": "english", "de": "german", "fr": "french", "es": "spanish",
                "it": "italian", "nl": "dutch", "pl": "polish", "pt": "portuguese",
                "ru": "russian", "zh": "chinese", "ja": "japanese", "ko": "korean",
                "tr": "turkish", "uk": "ukrainian", "sv": "swedish", "fi": "finnish",
                "da": "danish", "no": "norwegian", "cs": "czech", "el": "greek",
                "he": "hebrew", "ar": "arabic"}
    return {
        "detected_language": name_map.get(lang_code, lang_code or "und"),
        "language_code": lang_code or "und",
        "confidence": 1.0,
    }
