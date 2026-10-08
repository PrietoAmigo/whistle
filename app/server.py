"""HTTP API for Whistle: a recording page for phones, a native endpoint and an OpenAI-compatible one."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .audio import AudioDecodeError, AudioTooLongError, decode
from .transcriber import LANGUAGES, Transcriber, parse_keywords

MAX_AUDIO_SECONDS = float(os.environ.get("WHISTLE_MAX_AUDIO_SECONDS", "3600"))
LANGUAGE_NAMES = {"en": "english", "de": "german", "fr": "french", "es": "spanish",
                  "it": "italian", "nl": "dutch", "pl": "polish"}
OPENAI_FORMATS = ("json", "text", "verbose_json")
STATIC = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at startup, so a broken install fails the container rather than the first request.
    app.state.transcriber = Transcriber()
    yield


app = FastAPI(title="Whistle", summary="Cactus Whistle speech-to-text, on the CPU.", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _language(language: str | None) -> str | None:
    if language and language not in LANGUAGES:
        raise HTTPException(400, f"language must be one of {', '.join(LANGUAGES)}")
    return language or None


def _decode(upload: UploadFile) -> bytes:
    suffix = os.path.splitext(upload.filename or "")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix) as audio:
        shutil.copyfileobj(upload.file, audio)
        audio.flush()
        try:
            return decode(audio.name, MAX_AUDIO_SECONDS)
        except AudioTooLongError as error:
            raise HTTPException(413, str(error)) from None
        except AudioDecodeError as error:
            raise HTTPException(400, f"could not decode the audio: {error}") from None


def _transcribe(request: Request, upload: UploadFile, language: str | None, keywords: list[str],
                word_timestamps: bool) -> dict:
    language = _language(language)
    pcm = _decode(upload)
    try:
        return request.app.state.transcriber.transcribe(pcm, language, keywords, word_timestamps)
    except RuntimeError as error:
        raise HTTPException(500, str(error)) from None


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest():
    return FileResponse(os.path.join(STATIC, "manifest.webmanifest"), media_type="application/manifest+json")


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model": os.path.basename(request.app.state.transcriber.weights)}


@app.post("/transcribe")
def transcribe(
    request: Request,
    file: UploadFile = File(..., description="Any audio or video file ffmpeg can read"),
    language: str | None = Form(None, description="en, de, fr, es, it, nl or pl; detected when omitted"),
    keywords: str | None = Form(None, description="Comma-separated words and names to favour"),
    word_timestamps: bool = Form(False, description="Add each word with its start, end and probability"),
):
    """Returns the text, the language, and the duration of the audio in seconds."""
    return _transcribe(request, file, language, parse_keywords(keywords), word_timestamps)


@app.post("/transcribe/stream")
def transcribe_stream(
    request: Request,
    file: UploadFile = File(..., description="Any audio or video file ffmpeg can read"),
    language: str | None = Form(None, description="en, de, fr, es, it, nl or pl; detected when omitted"),
    keywords: str | None = Form(None, description="Comma-separated words and names to favour"),
    word_timestamps: bool = Form(False, description="Add each word with its start, end and probability"),
):
    """/transcribe as server-sent events: `{"partial": ...}` each second while audio over 30 s is
    transcribed, then `{"result": ...}`, or `{"error": ...}` if the engine fails part way.

    Bytes keep flowing during long transcriptions, so proxies with a read timeout, like
    Cloudflare's 100 s, don't cut them off.
    """
    language = _language(language)
    events = request.app.state.transcriber.events(_decode(file), language, parse_keywords(keywords), word_timestamps)

    def stream():
        try:
            for kind, data in events:
                yield f"data: {json.dumps({kind: data}, ensure_ascii=False)}\n\n"
        except RuntimeError as error:
            yield f"data: {json.dumps({'error': str(error)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/v1/audio/transcriptions")
def openai_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    language: str | None = Form(None),
    prompt: str | None = Form(None, description="Used as comma-separated keywords to favour"),
    response_format: str = Form("json", description="json, text or verbose_json"),
    timestamp_granularities: list[str] = Form([], alias="timestamp_granularities[]"),
):
    """OpenAI-compatible transcription. `model` and `temperature` are accepted and ignored."""
    if response_format not in OPENAI_FORMATS:
        raise HTTPException(400, f"response_format must be one of {', '.join(OPENAI_FORMATS)}")
    words = response_format == "verbose_json" and "word" in timestamp_granularities
    result = _transcribe(request, file, language, parse_keywords(prompt), words)
    if response_format == "text":
        return PlainTextResponse(result["text"])
    if response_format == "json":
        return {"text": result["text"]}
    verbose = {"task": "transcribe", "language": LANGUAGE_NAMES.get(result["language"], result["language"]),
               "duration": result["duration"], "text": result["text"]}
    if words:
        verbose["words"] = [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in result["words"]]
    return verbose
