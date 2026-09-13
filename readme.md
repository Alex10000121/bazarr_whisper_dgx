# Bazarr Whisper Adapter (DGX Spark / ARM64 + Blackwell)

Automatische Untertitel-Generierung für Bazarr auf einer NVIDIA DGX Spark
(ARM64-Prozessor + Blackwell-GPU), basierend auf `whisperx` als
GPU-Transkriptions-Engine.

## Warum dieses Projekt existiert

Bazarr kann für die Whisper-Anbindung nur eine **veraltete, feste API**
sprechen (`POST /asr`, `POST /detect-language` – ursprünglich aus dem
`ahmetoner/whisper-asr-webservice`-Projekt). Moderne, GPU-beschleunigte
Whisper-Server sprechen dagegen OpenAI-kompatible oder eigene APIs.

Zusätzlich gibt es aktuell **kein offizielles, vorgefertigtes Docker-Image**,
das gleichzeitig:
- auf ARM64 läuft (DGX Spark ist kein x86_64-System),
- die Blackwell-GPU per CUDA nutzt, und
- die alte Bazarr-API bedient.

Dieses Projekt löst das mit zwei Containern:

```
Bazarr  --(alte /asr-API)-->  whisper-bazarr-adapter  --(intern)-->  whisperx (GPU)
```

- **`whisperx`**: Der eigentliche GPU-Transkriptions-Server
  (`mekopa/whisperx-blackwell`), läuft nativ auf ARM64 + Blackwell.
- **`whisper-bazarr-adapter`**: Ein schlanker, selbst gebauter
  FastAPI-Service, der Bazarrs alte API-Aufrufe entgegennimmt, die
  Audiodaten ins passende Format bringt und an `whisperx` weiterreicht.

## Voraussetzungen

- Docker + Docker Compose auf einem ARM64-System mit NVIDIA-GPU
  (NVIDIA Container Toolkit installiert und konfiguriert)
- Ein externes Docker-Netzwerk namens `whisper`:
  ```bash
  docker network create whisper
  ```
- Ein Hugging Face Account + Access Token

## Hugging Face Token & Modell-Zugriff (wichtig!)

`whisperx` nutzt intern Modelle von Hugging Face (u. a. für
Sprecher-Diarisierung/Alignment), die **nicht anonym** heruntergeladen
werden können. Ohne die folgenden Schritte startet `whisperx` zwar,
schlägt aber beim ersten echten Transkriptionsversuch fehl:

1. Auf https://huggingface.co/settings/tokens einen **Access Token**
   erstellen (Read-Rechte reichen).
2. Für jedes benötigte Modell **einmalig die Nutzungsbedingungen auf der
   jeweiligen Hugging-Face-Modellseite akzeptieren** ("Agree and access
   repository" / "You need to share your contact information..."). Das
   betrifft typischerweise Modelle wie `pyannote/speaker-diarization`
   und `pyannote/segmentation`. **Ohne diese manuelle Bestätigung im
   Browser bleibt der Download blockiert, selbst mit gültigem Token.**
3. Den Token in einer `.env`-Datei im Projekt-Root hinterlegen
   (diese Datei **nicht** committen):
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

## Deployment

### Produktiv (Adapter-Image wird von GHCR gezogen)

```bash
docker compose -f docker-compose.yml up -d
```

Das Adapter-Image wird automatisch per GitHub Actions gebaut, sobald sich
etwas im Ordner `whisper-bazarr-adapter/` ändert (Branch `main`), und nach
`ghcr.io/alex10000121/bazarr_whisper_dgx/whisper-bazarr-adapter:latest`
gepusht. Falls das GHCR-Package privat ist, einmalig einloggen:
```bash
docker login ghcr.io -u alex10000121
```

### Lokal (Adapter wird selbst gebaut, z. B. für Entwicklung/Debugging)

```bash
docker compose up -d --build
```
Compose lädt `docker-compose.override.yml` automatisch mit und baut den
Adapter aus dem lokalen `whisper-bazarr-adapter/`-Ordner statt ihn zu
pullen.

### Erster Start dauert länger als erwartet

Beim allerersten Start lädt `whisperx` seine Modelle herunter – das kann
mehrere Minuten dauern. Das ist normal (der Healthcheck ist entsprechend
grosszügig konfiguriert). Bei jedem weiteren Start sind die Modelle im
Volume `whisperx-cache` zwischengespeichert und der Start ist deutlich
schneller.

## Konfiguration

| Variable            | Service  | Bedeutung                                              |
|---------------------|----------|---------------------------------------------------------|
| `HF_TOKEN`           | whisperx | Hugging-Face-Zugriffstoken (siehe oben)                 |
| `HF_HOME`            | whisperx | Cache-Verzeichnis für HF-Modelle                        |
| `TORCH_HOME`         | whisperx | Cache-Verzeichnis für Torch-Modelle                     |
| `FWSERVER`           | Adapter  | Basis-URL des whisperx-Servers                          |
| `FWSERVER_TIMEOUT`   | Adapter  | Timeout (Sekunden) für Anfragen an whisperx              |
| `PCM_SAMPLE_RATE`    | Adapter  | Erwartete Sample-Rate bei `encode=false` (Default 16000) |
| `PCM_CHANNELS`       | Adapter  | Erwartete Kanalzahl bei `encode=false` (Default 1)       |
| `PCM_BITS_PER_SAMPLE`| Adapter  | Erwartete Bit-Tiefe bei `encode=false` (Default 16)      |

## Bazarr-Konfiguration

In Bazarr unter **Einstellungen → Subtitles → Whisper Provider**:
- **Endpoint**: `http://<DGX-Spark-IP>:9000`
- Der Adapter implementiert `GET /status`, `POST /asr` und
  `POST /detect-language` kompatibel zur alten ahmetoner-API.
- Bei sehr langen Filmen ggf. das Timeout in Bazarrs eigenen
  Provider-Einstellungen grosszügig setzen, da eine volle GPU-Transkription
  bei langen Filmen mehrere Minuten dauern kann.

## Funktionsweise des Adapters (kurz)

- `encode=true` (Bazarr schickt eine rohe Video-/Audiodatei): Der Adapter
  normalisiert die Daten selbst per `ffmpeg` zu 16-kHz-Mono-WAV.
- `encode=false` (Bazarr hat die Audiospur bereits selbst zu rohem PCM
  kodiert): Der Adapter verpackt die PCM-Daten direkt mit einem WAV-Header,
  ohne nochmal `ffmpeg` zu bemühen (kein gültiger Container vorhanden, den
  `ffmpeg` demuxen könnte).
- Antworten von `whisperx` werden immer als JSON angefordert und vom
  Adapter selbst ins von Bazarr erwartete Format (SRT/VTT/TXT) umgewandelt.

## Troubleshooting

- **`upstream_unreachable` in `/status`**: Adapter und `whisperx` müssen im
  selben Docker-Netzwerk (`whisper`) laufen. Prüfen mit
  `docker network inspect whisper`.
- **`container whisperx is unhealthy` beim ersten Deploy**: siehe
  "Erster Start dauert länger als erwartet" oben.
- **`ffmpeg normalisation failed: ... Invalid data found`**: Tritt nur im
  `encode=true`-Pfad auf und bedeutet meist, dass die hochgeladene Datei
  unvollständig oder in einem nicht dekodierbaren Format ankam.
- Fehlerdetails stehen immer im Adapter-Log
  (`docker logs whisper-bazarr-adapter`) – jede `4xx`/`5xx`-Antwort wird
  dort mit der genauen Ursache geloggt.