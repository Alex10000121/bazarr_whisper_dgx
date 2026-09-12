# Repository: /workspace/project — Bazarr <-> faster-whisper adapter for DGX Spark (ARM64 + Blackwell)

## Verified API contracts (from source, 2026-09-12)

### fedirz/faster-whisper-server (a.k.a. speaches) — the EXISTING GPU service (DO NOT MODIFY)
- `GET /v1/models` -> `{"data":[{"id":"<MODEL_ID>","object":"model",...}],"object":"list"}` (model IDs are
  HuggingFace repo ids like `Systran/faster-whisper-large-v3`; NOT openai "large-v3"). ALWAYS discover via this call.
- `POST /v1/audio/transcriptions` (TRANSCRIBE) and `POST /v1/audio/translations` (TRANSLATE) — SEPARATE endpoints.
  - Multipart form fields (NOT query params): `file` (the audio UploadFile), `model` (REQUIRED), `language` (opt),
    `prompt` (opt), `response_format` in {text, json, verbose_json, srt, vtt}, `temperature`, `stream`,
    `timestamp_granularities[]`, `hotwords`, `without_timestamps`.
  - Response for srt = `text/plain`; vtt = `text/vtt` (with `WEBVTT\n\n` header); text = `text/plain`.
  - Audio decode: content_type audio/pcm|raw -> raw int16; else `faster_whisper.audio.decode_audio` (av/ffmpeg).
  - Optional Bearer auth if `api_key` set; NO language response header on plain srt/vtt/text responses.
  - verbose_json response is JSON and includes `language` (alpha2) — use this for language detection.

### ahmetoner/whisper-asr-webservice (the API Bazarr speaks — what the adapter must EMULATE)
- `POST /asr`: multipart field **`audio_file`**; query: `task`(transcribe|translate, default transcribe),
  `language` (opt, alpha2), `output`(txt|vtt|srt|tsv|json, default txt), `encode`(bool, default True),
  `initial_prompt`(opt). Response = `text/plain` + `Content-Disposition: attachment; filename="<name>.<output>"`.
- `POST /detect-language`: multipart field `audio_file`; returns JSON `{"detected_language","language_code","confidence"}`.
- No `/status` route in ahmetoner. Default port 9000.

### Bazarr (morpheus65535/bazarr) whisperai provider (custom_libs/subliminal_patch/providers/whisperai.py)
- `whisperai.endpoint` default = `http://127.0.0.1:9000` (port 9000 confirmed).
- Reads the /asr response body RAW (`subtitle.content = r.content`) -> MUST return plain SRT text, never JSON.
- /detect-language: uses `language_code`; `"und"` or missing => "detection failed" (sub.task=error).
- Connection-test button (Settings) = `providerUrlTest` -> proxies `GET <endpoint>/api/system/status` (Radarr/Sonarr
  style). whisper-asr has no such route => 404 => UI shows "Connected but no version found (possibly whisper-asr?)"
  which is the RECOGNIZED-working state for this provider. A plain FastAPI adapter naturally returns 404 here. Bazarr
  reads NO fields from /status.
- Bazarr pre-encodes audio to 16kHz mono PCM s16le WAV before posting (encode_audio_stream), so adapter ffmpeg
  normalization is an idempotent safety net for other clients.

## Environment reality (this sandbox)
- aarch64, NO GPU (no /dev/nvidia*, no nvidia-smi). User non-root, CapBnd lacks CAP_SYS_ADMIN + CAP_NET_ADMIN.
- `unshare --mount` FAILED, bind mount FAILED, dockerd iptables FAILED -> **containers CANNOT run here** (proven).
- `localhost:8000` is the OpenHands agent server, NOT faster-whisper. `faster-whisper` hostname does not resolve.
- So `docker compose up` cannot be executed in this sandbox; test the adapter natively against a faithful faster-whisper
  (CPU) backend that implements the verified fedirz contract. ffmpeg 7.1.5 + faster-whisper 1.2.1 + CTranslate2 4.8.2
  are installed and work on CPU (small model, int8). TTS via espeak-ng works for non-trivial test audio.
