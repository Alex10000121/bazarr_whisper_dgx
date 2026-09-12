"""Faithful stand-in for the fedirz/faster-whisper-server (speaches) ASR API.

It implements the exact subset of the OpenAI-compatible contract that the
adapter relies on, using the SAME faster-whisper / CTranslate2 library and the
SAME SRT/VTT formatters that the real server uses (copied verbatim from
speaches.text_utils). This lets the adapter be exercised end-to-end in an
environment that cannot run the GPU container.

Endpoints:
  GET  /v1/models                       -> {"data":[{"id":...}],"object":"list"}
  POST /v1/audio/transcriptions          -> form: file, model, response_format,
                                            language, prompt, ...
  POST /v1/audio/translations            -> same shape
  Response media: srt=text/plain, vtt=text/vtt (WEBVTT header), text=text/plain.
"""
import logging
import os
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from starlette.responses import Response

log = logging.getLogger("mock-fws")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

MODEL_ID = os.getenv("FWS_MODEL", "Systran/faster-whisper-small")
DEVICE = os.getenv("FWS_DEVICE", "cpu")
COMPUTE = os.getenv("FWS_COMPUTE", "int8")

app = FastAPI()
_model = WhisperModel(MODEL_ID, device=DEVICE, compute_type=COMPUTE)
log.info("backend ready: model=%s device=%s compute=%s", MODEL_ID, DEVICE, COMPUTE)


# ---- verbatim from speaches.text_utils ----
def srt_format_timestamp(ts: float) -> str:
    hours = ts // 3600
    minutes = (ts % 3600) // 60
    seconds = ts % 60
    milliseconds = (ts * 1000) % 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


def vtt_format_timestamp(ts: float) -> str:
    hours = ts // 3600
    minutes = (ts % 3600) // 60
    seconds = ts % 60
    milliseconds = (ts * 1000) % 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"


def format_as_srt(text, start, end, i) -> str:
    return f"{i + 1}\n{srt_format_timestamp(start)} --> {srt_format_timestamp(end)}\n{text}\n\n"


def format_as_vtt(text, start, end, i) -> str:
    start = start if i > 0 else 0.0
    result = f"{vtt_format_timestamp(start)} --> {vtt_format_timestamp(end)}\n{text}\n\n"
    if i == 0:
        return f"WEBVTT\n\n{result}"
    return result


def _run(task: str, language: str | None, prompt: str | None, response_format: str, audio: np.ndarray):
    t0 = time.time()
    kwargs = dict(task=task, vad_filter=True, beam_size=5)
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["initial_prompt"] = prompt
    segments, info = _model.transcribe(audio, **kwargs)
    segs = [(s.start, s.end, s.text.strip()) for s in segments]
    log.info("transcribed %s task=%s lang=%s prob=%.2f segs=%d in %.1fs",
             round(len(audio) / 16000, 1), task, info.language,
             info.language_probability, len(segs), time.time() - t0)
    if response_format == "srt":
        return "".join(format_as_srt(t, a, b, i) for i, (a, b, t) in enumerate(segs)), "text/plain"
    if response_format == "vtt":
        return "".join(format_as_vtt(t, a, b, i) for i, (a, b, t) in enumerate(segs)), "text/vtt"
    if response_format in ("json", "verbose_json"):
        import json
        return json.dumps({
            "language": info.language,
            "duration": round(len(audio) / 16000.0, 3),
            "text": " ".join(t for _, _, t in segs),
            "segments": [
                {"id": i, "seek": 0, "start": a, "end": b, "text": t,
                 "tokens": [], "temperature": 0.0, "avg_logprob": 0.0,
                 "compression_ratio": 0.0, "no_speech_prob": 0.0}
                for i, (a, b, t) in enumerate(segs)
            ],
        }), "application/json"
    # text (default)
    return " ".join(t for _, _, t in segs), "text/plain"


def _decode(file: UploadFile) -> np.ndarray:
    # mirror speaches: content_type audio/pcm|raw -> int16; else decode_audio
    if (file.content_type or "") in ("audio/pcm", "audio/raw"):
        arr = np.frombuffer(file.file.read(), dtype=np.int16).astype(np.float32) / 32768.0
        return arr
    return decode_audio(file.file, sampling_rate=16000)


@app.get("/v1/models")
def models():
    return {"data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}], "object": "list"}


@app.post("/v1/audio/transcriptions")
def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    response_format: str = Form("json"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
):
    log.info("POST /v1/audio/transcriptions model=%s rf=%s lang=%s file=%s ct=%s",
             model, response_format, language, file.filename, file.content_type)
    if model != MODEL_ID:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
    try:
        audio = _decode(file)
    except Exception as e:
        raise HTTPException(status_code=415, detail=f"Failed to decode audio: {e}") from e
    body, media = _run("transcribe", language, prompt, response_format, audio)
    return Response(content=body, media_type=media)


@app.post("/v1/audio/translations")
def translations(
    file: UploadFile = File(...),
    model: str = Form(...),
    response_format: str = Form("json"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
):
    log.info("POST /v1/audio/translations model=%s rf=%s file=%s", model, response_format, file.filename)
    if model != MODEL_ID:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
    try:
        audio = _decode(file)
    except Exception as e:
        raise HTTPException(status_code=415, detail=f"Failed to decode audio: {e}") from e
    body, media = _run("translate", None, prompt, response_format, audio)
    return Response(content=body, media_type=media)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("FWS_PORT", "8000")), log_level="info")
