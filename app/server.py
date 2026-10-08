"""HTTP API for Whistle: a native endpoint and an OpenAI-compatible one."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, RedirectResponse

from .audio import AudioDecodeError, AudioTooLongError, decode
from .transcriber import LANGUAGES, Transcriber, parse_keywords

MAX_AUDIO_SECONDS = float(os.environ.get("WHISTLE_MAX_AUDIO_SECONDS", "3600"))
LANGUAGE_NAMES = {"en": "english", "de": "german", "fr": "french", "es": "spanish",
                  "it": "italian", "nl": "dutch", "pl": "polish"}
OPENAI_FORMATS = ("json", "text", "verbose_json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at startup, so a broken install fails the container rather than the first request.
    app.state.transcriber = Transcriber()
    yield


app = FastAPI(title="Whistle", summary="Cactus Whistle speech-to-text, on the CPU.", lifespan=lifespan)


def _transcribe(request: Request, upload: UploadFile, language: str | None, keywords: list[str],
                word_timestamps: bool) -> dict:
    language = language or None
    if language is not None and language not in LANGUAGES:
        raise HTTPException(400, f"language must be one of {', '.join(LANGUAGES)}")
    suffix = os.path.splitext(upload.filename or "")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix) as audio:
        shutil.copyfileobj(upload.file, audio)
        audio.flush()
        try:
            pcm = decode(audio.name, MAX_AUDIO_SECONDS)
        except AudioTooLongError as error:
            raise HTTPException(413, str(error)) from None
        except AudioDecodeError as error:
            raise HTTPException(400, f"could not decode the audio: {error}") from None
    try:
        return request.app.state.transcriber.transcribe(pcm, language, keywords, word_timestamps)
    except RuntimeError as error:
        raise HTTPException(500, str(error)) from None


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


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
