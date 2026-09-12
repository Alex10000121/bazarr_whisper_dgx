# Slim adapter between Bazarr (ahmetoner whisper-asr API) and the
# faster-whisper GPU server. No ML libraries are installed here -- the heavy
# lifting happens in the existing `faster-whisper` container; we only need a
# FastAPI web stack plus ffmpeg for audio normalisation.
#
# ARM64 friendly: python:3.12-slim publishes aarch64 builds and the pip wheels
# we pull (fastapi/uvicorn/httpx/python-multipart) are pure-python or ship
# aarch64 wheels, so this builds cleanly on a DGX Spark.
FROM python:3.12-slim

# ffmpeg for the mandatory 16kHz/mono/PCM normalisation step.
# (libfontconfig1 etc. kept minimal; ffmpeg from the distro repo.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Pin web-stack dependencies for reproducibility.
RUN pip install --no-cache-dir \
    "fastapi==0.115.6" \
    "uvicorn[standard]==0.34.0" \
    "httpx==0.28.1" \
    "python-multipart==0.0.20"

COPY main.py /app/main.py

EXPOSE 9000

# liveness (adapter answers this even when the GPU server is down).
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9000/status', timeout=4).status==200 else 1)"

# Run as a non-root user.
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
