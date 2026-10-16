import io
import os
import re
import typing
from typing import Optional

import lameenc
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(
    title="Silero v5.5 OpenAI-Compatible TTS Server",
    description="Local, lightweight Russian text-to-speech server with continuous MP3 streaming.",
    version="1.0.0",
)

# --- 1. CORS Middleware (Required for browser extensions & web clients) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Model Initialization ---
device = torch.device("cpu")
torch.set_num_threads(int(os.environ.get("THREADS", 4)))

MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
MODEL_PATH = os.environ.get("MODEL_PATH", "v5_5_ru.pt")

if not os.path.exists(MODEL_PATH):
    print(f"Downloading Silero v5.5 Russian model to {MODEL_PATH}...")
    torch.hub.download_url_to_file(MODEL_URL, MODEL_PATH)

print(f"Loading Silero v5.5 model from {MODEL_PATH}...")
model = torch.package.PackageImporter(MODEL_PATH).load_pickle("tts_models", "model")
model.to(device)

# Available Silero v5 Russian voices
AVAILABLE_VOICES = [
    {"id": "xenia", "name": "Xenia (Female)"},
    {"id": "kseniya", "name": "Kseniya (Female)"},
    {"id": "baya", "name": "Baya (Female)"},
    {"id": "aidar", "name": "Aidar (Male)"},
    {"id": "eugene", "name": "Eugene (Male)"},
]
VALID_VOICE_NAMES = {v["id"] for v in AVAILABLE_VOICES}


# --- 3. Request Schema ---
class SpeechRequest(BaseModel):
    model: Optional[str] = "silero-v5.5"
    input: str
    voice: Optional[str] = "xenia"
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    stream: Optional[bool] = True


# --- 4. Health & Discovery Endpoints ---
@app.get("/")
@app.get("/health")
def health():
    return {"status": "healthy", "model": "silero-v5"}


@app.get("/v1/models")
def list_models():
    return {
        "data": [
            {
                "id": "silero-v5.5",
                "object": "model",
                "owned_by": "silero",
            }
        ]
    }


@app.get("/v1/audio/voices")
@app.get("/v1/voices")
def get_voices():
    return {"voices": AVAILABLE_VOICES}


# --- 5. Chunking & Streaming Audio Generator ---
def split_into_sentences(text: str, max_chars: int = 250) -> list[str]:
    """Splits text into natural sentences to avoid Silero's internal buffer limits."""
    raw_sentences = re.split(r"(?<=[.!?…\n])\s+", text.strip())
    chunks = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        while len(s) > max_chars:
            split_idx = s[:max_chars].rfind(",")
            if split_idx == -1:
                split_idx = s[:max_chars].rfind(" ")
            if split_idx == -1:
                split_idx = max_chars
            chunks.append(s[:split_idx].strip())
            s = s[split_idx:].strip()
        if s:
            chunks.append(s)
    return chunks


def audio_stream_generator(sentences: list[str], speaker: str) -> typing.Iterator[bytes]:
    """Synthesizes text sentence-by-sentence and yields continuous MP3 stream bytes."""
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(48000)
    encoder.set_channels(1)
    encoder.set_quality(4)

    for sentence in sentences:
        try:
            audio = model.apply_tts(
                text=sentence,
                speaker=speaker,
                sample_rate=48000,
                put_accent=True,
                put_yo=True,
            )
            pcm = (audio.numpy() * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
            mp3_chunk = encoder.encode(pcm)
            if mp3_chunk:
                yield bytes(mp3_chunk)
        except Exception as e:
            print(f"[Warn] Synthesis error on segment '{sentence[:30]}...': {e}")
            continue

    tail = encoder.flush()
    if tail:
        yield bytes(tail)


# --- 6. OpenAI Speech Endpoint ---
@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    speaker = req.voice.lower() if req.voice else "xenia"
    if speaker not in VALID_VOICE_NAMES:
        speaker = "xenia"

    sentences = split_into_sentences(req.input)
    if not sentences:
        raise HTTPException(status_code=400, detail="No speakable text provided.")

    return StreamingResponse(
        audio_stream_generator(sentences, speaker),
        media_type="audio/mpeg",
        headers={
            "Transfer-Encoding": "chunked",
            "Content-Type": "audio/mpeg",
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8881))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting Silero-FastAPI server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
