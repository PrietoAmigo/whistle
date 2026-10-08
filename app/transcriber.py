"""Whistle speech-to-text on top of the cactus-needle engine."""

from __future__ import annotations

import re
import threading

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 4  # mono float32
MAX_PASS_SECONDS = 30  # the longest clip Whistle transcribes in one pass
LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")


def parse_keywords(text: str | None) -> list[str]:
    """Comma- or newline-separated words and names to favour, e.g. "Siobhan, Krzysztof"."""
    return [word.strip() for word in re.split(r"[,\n]", text or "") if word.strip()]


class Transcriber:
    """One Whistle model per process. The engine is not thread-safe, so calls are serialised."""

    def __init__(self, weights: str | None = None):
        import needle

        self._model = needle.Whistle(weights=weights)
        self._lock = threading.Lock()

    @property
    def weights(self) -> str:
        return self._model.weights

    def transcribe(self, pcm: bytes, language: str | None = None, keywords: list[str] | None = None,
                   word_timestamps: bool = False) -> dict:
        """Transcribe 16 kHz mono float32 PCM."""
        *_, (_, result) = self.events(pcm, language, keywords, word_timestamps)
        return result

    def events(self, pcm: bytes, language: str | None = None, keywords: list[str] | None = None,
               word_timestamps: bool = False):
        """Transcribe 16 kHz mono float32 PCM, yielding ("partial", {...}) as it goes, then ("result", {...}).

        Clips up to 30 s go through the model in one pass. Longer audio is fed to
        Whistle's streaming decoder one second at a time, which has no length limit;
        each second yields the text it committed and how far the decoder has got.
        """
        keywords = keywords or None
        duration = round(len(pcm) / BYTES_PER_SECOND, 3)
        with self._lock:
            if len(pcm) <= MAX_PASS_SECONDS * BYTES_PER_SECOND:
                result = self._model.transcribe(pcm, language=language, keywords=keywords,
                                                word_timestamps=word_timestamps)
            else:
                chunks = (pcm[i:i + BYTES_PER_SECOND] for i in range(0, len(pcm), BYTES_PER_SECOND))
                texts, words, detected = [], [], None
                for step in self._model.stream(chunks, language=language, keywords=keywords):
                    if step["text"]:
                        texts.append(step["text"])
                    words.extend(step["words"])
                    detected = step["language"] or detected
                    yield "partial", {"text": step["text"], "seconds": step["received"], "duration": duration}
                result = {"text": " ".join(texts), "language": detected or ""}
                if word_timestamps:
                    result["words"] = words
        result["duration"] = duration
        yield "result", result
