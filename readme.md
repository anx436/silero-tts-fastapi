# Silero v5.5 OpenAI-Compatible TTS Server

A lightweight, local, high-performance Russian Text-to-Speech API server powered by **Silero v5.5**.

Exposes OpenAI-compatible `/v1/audio/speech` and `/v1/audio/voices` endpoints with continuous sentence-by-sentence **MP3 streaming**, making it a drop-in local backend for browser extensions, reader apps, and LLM voice pipelines.

## Features

- **OpenAI Compatible:** Implements `POST /v1/audio/speech` and `GET /v1/audio/voices`.
- **Continuous MP3 Streaming:** Encodes and streams speech chunk-by-chunk using `lameenc`—starts playback in milliseconds even for long articles.
- **Fast on CPU**
- **CORS Enabled:** Ready out of the box for browser extensions and frontends.

---

## Quickstart

### 1. Installation

Using [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`:

```bash
# With uv:
uv venv
uv pip install fastapi uvicorn torch soundfile lameenc numpy

# Or with pip:
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn torch soundfile lameenc numpy
```

### 2. Start the Server

```bash
python server.py
```
*(The 50 MB Silero v5.5 model will be downloaded automatically on first run).*

To customize the port or threads:
```bash
PORT=8881 THREADS=4 python server.py
```

---

## API Usage

### Test with `curl`

```bash
curl -X POST http://localhost:8881/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Ты правда думаешь, что этот замок **очень** старый?", "voice": "xenia"}' \
  --output test.wav
```

### Query Available Voices

```bash
curl http://localhost:8881/v1/audio/voices
```

## License

MIT
