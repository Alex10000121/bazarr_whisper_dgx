from __future__ import annotations

import glob
import logging
import math
import os
import struct
import subprocess
import tempfile
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

log = logging.getLogger("whisper-bazarr-adapter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class _SuppressStatusAccessLog(logging.Filter):
    """Drop uvicorn access-log lines for the noisy /status healthcheck polling,
    while leaving /asr, /detect-language, and error logs untouched."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/status" not in record.getMessage()


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").addFilter(_SuppressStatusAccessLog())

UPSTREAM = os.getenv("FWSERVER", "http://whisperx:8003")
REQUEST_TIMEOUT = float(os.getenv("FWSERVER_TIMEOUT", "1200"))
FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")
FFMPEG_TIMEOUT = float(os.getenv("FFMPEG_TIMEOUT", "300"))

# Contract for encode=false: Bazarr sends raw PCM (its own ffmpeg output),
# no container. If quality looks wrong, verify these still match Bazarr's
# actual output via a manual /asr?encode=false test.
PCM_SAMPLE_RATE = int(os.getenv("PCM_SAMPLE_RATE", "16000"))
PCM_CHANNELS = int(os.getenv("PCM_CHANNELS", "1"))
PCM_BITS_PER_SAMPLE = int(os.getenv("PCM_BITS_PER_SAMPLE", "16"))

OUTPUT_FORMATS = {"srt", "vtt", "txt", "text"}
RESPONSE_FORMAT_MEDIA = {"srt": "text/plain", "vtt": "text/vtt", "txt": "text/plain", "text": "text/plain"}

_upstream: httpx.Client | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if _upstream is not None:
        _upstream.close()


app = FastAPI(title="whisper-bazarr-adapter", lifespan=lifespan)


def _client() -> httpx.Client:
    global _upstream
    if _upstream is None:
        _upstream = httpx.Client(base_url=UPSTREAM, timeout=REQUEST_TIMEOUT)
    return _upstream


def _health_sync() -> dict:
    r = _client().get("/health", timeout=3.0)
    r.raise_for_status()
    return r.json()


def _post_sync(endpoint: str, data: dict, files: dict) -> httpx.Response:
    return _client().post(endpoint, data=data, files=files)


def _pcm_to_wav(pcm_data: bytes) -> bytes:
    if not pcm_data:
        raise ValueError("empty audio payload")
    block_align = PCM_CHANNELS * (PCM_BITS_PER_SAMPLE // 8)
    byte_rate = PCM_SAMPLE_RATE * block_align
    data_size = len(pcm_data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE", b"fmt ",
        16, 1, PCM_CHANNELS, PCM_SAMPLE_RATE, byte_rate,
        block_align, PCM_BITS_PER_SAMPLE, b"data", data_size,
    )
    return header + pcm_data


def _normalize_audio(raw: bytes, hint_filename: str | None = None) -> bytes:
    """Decode an arbitrary container/codec to 16kHz mono WAV via ffmpeg.

    Only for encode=true. encode=false has no container to demux, so it
    uses _pcm_to_wav instead.
    """
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
            "-f", "wav", "-acodec", "pcm_s16le", "-ar", str(PCM_SAMPLE_RATE), "-ac", str(PCM_CHANNELS),
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT)
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


def _to_wav(raw: bytes, encode: bool, hint_filename: str | None) -> bytes:
    if not raw:
        raise ValueError("empty audio payload")
    if encode:
        return _normalize_audio(raw, hint_filename)
    return _pcm_to_wav(raw)


def _format_time(seconds: float, vtt: bool = False) -> str:
    frac, whole = math.modf(seconds)
    msecs = int(round(frac * 1000))
    if msecs == 1000:
        msecs = 0
        whole += 1
    whole = int(whole)
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    sep = "." if vtt else ","
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{msecs:03d}"


def _json_to_srt(data: dict) -> str:
    lines = []
    for i, seg in enumerate(data.get("segments", []), start=1):
        start = _format_time(seg.get("start", 0.0))
        end = _format_time(seg.get("end", 0.0))
        text = seg.get("text", "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _json_to_vtt(data: dict) -> str:
    lines = ["WEBVTT\n"]
    for i, seg in enumerate(data.get("segments", []), start=1):
        start = _format_time(seg.get("start", 0.0), vtt=True)
        end = _format_time(seg.get("end", 0.0), vtt=True)
        text = seg.get("text", "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _json_to_txt(data: dict) -> str:
    return "\n".join(seg.get("text", "").strip() for seg in data.get("segments", []))


_CONVERTERS = {"srt": _json_to_srt, "vtt": _json_to_vtt, "txt": _json_to_txt, "text": _json_to_txt}


_last_upstream_ok: bool | None = None


@app.get("/status")
async def status():
    global _last_upstream_ok
    upstream_ok = True
    try:
        await run_in_threadpool(_health_sync)
    except Exception as exc:
        upstream_ok = False
        if _last_upstream_ok is not False:
            log.warning("upstream unreachable: %s", exc)
    else:
        if _last_upstream_ok is False:
            log.info("upstream reachable again")
    _last_upstream_ok = upstream_ok
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
    out_key = output.lower()
    if out_key not in OUTPUT_FORMATS:
        return _error(400, f"unsupported output '{output}' (want srt|vtt|txt)")

    raw = await audio_file.read()
    try:
        wav = await run_in_threadpool(_to_wav, raw, encode, audio_file.filename)
    except (ValueError, struct.error, subprocess.TimeoutExpired) as exc:
        return _error(400, str(exc))

    files = {"file": (f"{audio_file.filename or 'audio'}.wav", wav, "audio/wav")}
    # Always request JSON so SRT/VTT/TXT formatting stays under our control.
    data = {"response_format": "json", "task": task}
    if language:
        data["language"] = language
    if initial_prompt:
        data["initial_prompt"] = initial_prompt

    try:
        r = await run_in_threadpool(_post_sync, "/transcribe", data, files)
    except httpx.HTTPError as exc:
        log.exception("upstream request failed")
        return _error(502, f"failed to reach upstream whisperx: {exc}")

    if r.status_code >= 400:
        return _error(r.status_code, _friendly_error(r))

    try:
        jdata = r.json()
    except ValueError:
        return _error(502, "upstream whisperx did not return valid JSON")

    if "segments" not in jdata:
        return _error(502, "upstream whisperx response missing 'segments'")

    content = _CONVERTERS[out_key](jdata).encode("utf-8")
    media = RESPONSE_FORMAT_MEDIA[out_key]
    headers = {}
    disposition = _content_disposition(audio_file.filename, out_key)
    if disposition:
        headers["Content-Disposition"] = disposition

    log.info("transcribed %s bytes -> %s (200)", len(raw), out_key)
    return Response(content=content, media_type=media, status_code=200, headers=headers)


@app.post("/detect-language")
async def detect_language(audio_file: UploadFile = File(...), encode: bool = Query(True)):
    raw = await audio_file.read()
    try:
        wav = await run_in_threadpool(_to_wav, raw, encode, audio_file.filename)
    except (ValueError, struct.error, subprocess.TimeoutExpired) as exc:
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
        lang_code = (r.json().get("language") or "").strip().lower()
    except ValueError:
        lang_code = "und"

    return JSONResponse(content=_detect_language_payload(lang_code))


def _error(code: int, msg: str):
    log.warning("returning %s: %s", code, msg)
    return JSONResponse(status_code=code, content={"detail": msg, "status": "error"})


def _friendly_error(r: httpx.Response) -> str:
    try:
        j = r.json()
        if isinstance(j, dict) and "detail" in j:
            return str(j["detail"])
    except ValueError:
        pass
    text = (r.text or "").strip()
    return text[:500] or f"upstream returned HTTP {r.status_code}"


def _content_disposition(filename: str | None, out_key: str) -> str | None:
    if not filename:
        return None
    base = filename.rsplit(".", 1)[0] if "." in filename else filename
    return f'attachment; filename="{base}.{out_key}"'


def _detect_language_payload(lang_code: str) -> dict:
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