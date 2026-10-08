"""Decode any audio or video file to the 16 kHz mono float32 PCM Whistle takes."""

from __future__ import annotations

import subprocess

from .transcriber import BYTES_PER_SECOND, SAMPLE_RATE


class AudioDecodeError(ValueError):
    pass


class AudioTooLongError(ValueError):
    pass


def decode(path: str, max_seconds: float | None = None) -> bytes:
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path]
    if max_seconds:
        # Decode one second past the limit: enough to tell the file is too long
        # without holding all of it in memory.
        command += ["-t", str(max_seconds + 1)]
    command += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    process = subprocess.run(command, capture_output=True)
    if process.returncode != 0:
        # ffmpeg's last line says what went wrong; the ones before name temp files and decoder internals.
        lines = process.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioDecodeError(lines[-1] if lines else "ffmpeg could not decode the audio")
    pcm = process.stdout
    if max_seconds and len(pcm) > max_seconds * BYTES_PER_SECOND:
        raise AudioTooLongError(f"audio is longer than the {max_seconds:g} s limit")
    return pcm
