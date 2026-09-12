from __future__ import annotations
 
import glob
import logging
import os
import subprocess
import tempfile
 
import httpx
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
 
log = logging.getLogger("whisper-bazarr-adapter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 
UPSTREAM = os.getenv("FWSERVER", "http://faster-whisper:8000")
MODEL = os.getenv("FWSERVER_MODEL", "")          # empty => auto-discover via /v1/models
API_KEY = os.getenv("FWSERVER_API_KEY", "")      # optional Bearer auth for upstream
REQUEST_TIMEOUT = float(os.getenv("FWSERVER_TIMEOUT", "1200"))  # GPU transcribes can be long
FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")
 
# Bazarr `output` (ahmetoner)  ->  faster-whisper `response_format` (OpenAI)
OUTPUT_TO_RESPONSE_FORMAT = {
    "srt": "srt",
    "vtt": "vtt",
    "txt": "text",
    "text": "text",
}
# faster-whisper content-type for each response_format (mirror of the server).
RESPONSE_FORMAT_MEDIA = {"srt": "text/plain", "vtt": "text/vtt", "text": "text/plain"}
 
app = FastAPI(title="whisper-bazarr-adapter")
# Keep one httpx client for the whole process; faster-whisper serialises work.
_upstream: httpx.Client | None = None
 
 
def _client() -> httpx.Client:
    global _upstream
    if _upstream is None:
        _upstream = httpx.Client(
            base_url=UPSTREAM,
            timeout=REQUEST_TIMEOUT,
            headers={"Authorization": f"Bearer {API_KEY}"} if API_KEY else {},
        )
    return _upstream
 
def _normalize_audio(raw: bytes, hint_filename: str | None = None) -> bytes:
    if not raw:
        raise ValueError("empty audio payload")
    log.info("received %d bytes, filename=%r, first 16 bytes: %s",
              len(raw), hint_filename, raw[:16].hex()) 
 
def _get_models_sync() -> list[dict]:
    """Blocking call -- always invoke via run_in_threadpool."""
    r = _client().get("/v1/models")
    r.raise_for_status()
    return r.json().get("data", [])
 
 
def _post_sync(endpoint: str, data: dict, files: dict) -> httpx.Response:
    """Blocking call -- always invoke via run_in_threadpool."""
    return _client().post(endpoint, data=data, files=files)
 
 
async def _resolve_model() -> str:
    """Return a usable model id.
 
    Honour an explicit FWSERVER_MODEL; otherwise discover the first model the
    server reports (never hard-code a placeholder like 'large-v3').
    """
    if MODEL:
        return MODEL
    data = await run_in_threadpool(_get_models_sync)
    if not data:
        raise RuntimeError(f"upstream {UPSTREAM} reported no models")
    return data[0]["id"]
 
 
def _normalize_audio(raw: bytes, hint_filename: str | None = None) -> bytes:
    """Decode any audio to 16 kHz mono WAV (PCM s16le) via ffmpeg.
 
    The input is written to a seekable temp file (MP4/MKV need seekable input;
    a pipe breaks for them), while the normalized WAV is produced on stdout.
    Raises ValueError with ffmpeg's stderr on failure.
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
    """Health endpoint (ahmetoner-compatible).
 
    Reports the upstream's advertised models when reachable; if the upstream is
    down we still return 200 with `upstream: "down"` so liveness can be told
    apart from reachability (Bazarr treats any 2xx as "connected").
    """
    upstream_ok, models = True, []
    try:
        data = await run_in_threadpool(_get_models_sync)
        models = [m.get("id") for m in data]
    except Exception as exc:  # upstream down / no network
        upstream_ok = False
        log.warning("upstream unreachable: %s", exc)
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok" if upstream_ok else "upstream_unreachable",
            "engine": "faster-whisper",
            "upstream": "up" if upstream_ok else "down",
            "upstream_url": UPSTREAM,
            "models": models,
        },
    )
 
 
@app.post("/asr")
async def asr(
    audio_file: UploadFile = File(...),
    task: str = Query("transcribe", pattern="^(transcribe|translate)$"),
    language: str | None = Query(None),
    output: str = Query("srt"),
    encode: bool = Query(True),  # accepted for ahmetoner compat; we always normalise
    initial_prompt: str | None = Query(None),
):
    """Transcribe/translate and return the subtitle body as plain text.
 
    Bazarr reads the response body verbatim (`subtitle.content = r.content`),
    so we return the upstream's SRT/VTT/text directly -- never JSON.
    """
    try:
        out_key = output.lower()
    except Exception:
        out_key = ""
    if out_key not in OUTPUT_TO_RESPONSE_FORMAT:
        return _error(400, f"unsupported output '{output}' (want srt|vtt|txt)")
    response_format = OUTPUT_TO_RESPONSE_FORMAT[out_key]
 
    raw = await audio_file.read()
    try:
        wav = await run_in_threadpool(_normalize_audio, raw, audio_file.filename)
    except (ValueError, subprocess.TimeoutExpired) as exc:
        return _error(400, str(exc))
 
    try:
        model = await _resolve_model()
    except (httpx.HTTPError, RuntimeError) as exc:
        return _error(502, f"could not resolve model from upstream: {exc}")
 
    endpoint = "/v1/audio/transcriptions" if task == "transcribe" else "/v1/audio/translations"
 
    files = {"file": (f"{audio_file.filename or 'audio'}.wav", wav, "audio/wav")}
    data = {"model": model, "response_format": response_format}
    if language:
        data["language"] = language
    if initial_prompt:
        data["prompt"] = initial_prompt
 
    try:
        r = await run_in_threadpool(_post_sync, endpoint, data, files)
    except httpx.HTTPError as exc:
        log.exception("upstream request failed")
        return _error(502, f"failed to reach upstream faster-whisper: {exc}")
 
    if r.status_code >= 400:
        return _error(r.status_code, _friendly_error(r))
 
    media = RESPONSE_FORMAT_MEDIA.get(response_format, "text/plain")
    disposition = _content_disposition(audio_file.filename, out_key)
    headers = {"Content-Disposition": disposition} if disposition else {}
    # Echo the detected language when the server provides it (ahmetoner does).
    if "x-detected-language" in r.headers:
        headers["X-Detected-Language"] = r.headers["x-detected-language"]
    log.info("transcribed %s bytes -> %s (%s)", len(raw), response_format, r.status_code)
    return Response(content=r.content, media_type=media, status_code=200, headers=headers)
 
 
@app.post("/detect-language")
async def detect_language(audio_file: UploadFile = File(...), encode: bool = Query(True)):
    """Return the detected language, in the ahmetoner JSON shape.
 
    Uses the server's own detection: a minimal transcription request, reading
    the `language` field from a verbose_json response (or an upstream-provided
    header). Bazarr uses `language_code` (alpha2) and maps `"und"` to failure.
    """
    raw = await audio_file.read()
    try:
        wav = await run_in_threadpool(_normalize_audio, raw, audio_file.filename)
    except (ValueError, subprocess.TimeoutExpired) as exc:
        return _error(400, str(exc))
 
    try:
        model = await _resolve_model()
    except (httpx.HTTPError, RuntimeError) as exc:
        return _error(502, f"could not resolve model from upstream: {exc}")
 
    files = {"file": (f"{audio_file.filename or 'audio'}.wav", wav, "audio/wav")}
    data = {"model": model, "response_format": "verbose_json"}
    try:
        r = await run_in_threadpool(_post_sync, "/v1/audio/transcriptions", data, files)
    except httpx.HTTPError as exc:
        return _error(502, f"failed to reach upstream: {exc}")
    if r.status_code >= 400:
        return _error(r.status_code, _friendly_error(r))
 
    body = r.json()
    lang_code = (body.get("language") or "").strip().lower()
    return JSONResponse(content=_detect_language_payload(lang_code, body))
 
 
# --- helpers -----------------------------------------------------------------
 
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
    # Strip an existing extension and append the requested one, ahmetoner-style.
    base = filename.rsplit(".", 1)[0] if "." in filename else filename
    return f'attachment; filename="{base}.{out_key}"'
 
 
def _detect_language_payload(lang_code: str, body: dict) -> dict:
    name_map = {"en": "english", "de": "german", "fr": "french", "es": "spanish",
                "it": "italian", "nl": "dutch", "pl": "polish", "pt": "portuguese",
                "ru": "russian", "zh": "chinese", "ja": "japanese", "ko": "korean",
                "tr": "turkish", "uk": "ukrainian", "sv": "swedish", "fi": "finnish",
                "da": "danish", "no": "norwegian", "cs": "czech", "el": "greek",
                "he": "hebrew", "ar": "arabic"}
    # confidence is not reliably available; report 1.0 when we have a code.
    return {
        "detected_language": name_map.get(lang_code, lang_code or "und"),
        "language_code": lang_code or "und",
        "confidence": 1.0,
    }
 
