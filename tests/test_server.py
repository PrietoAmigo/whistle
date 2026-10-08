"""HTTP and audio handling, with a stand-in for the native Whistle engine."""

import math
import struct
import sys
import types
import wave

import pytest
from fastapi.testclient import TestClient

from app import cli, server
from app.transcriber import BYTES_PER_SECOND


class FakeWhistle:
    calls = []

    def __init__(self, weights=None):
        self.weights = weights or "/opt/whistle/whistle.cact"

    def transcribe(self, audio, language=None, keywords=None, word_timestamps=False):
        FakeWhistle.calls.append(("transcribe", len(audio), language, keywords))
        result = {"text": "turn off the kitchen lights", "language": language or "en", "ttft_ms": 11.1,
                  "decode_tps": 1319.0}
        if word_timestamps:
            result["words"] = [{"word": "turn", "start": 0.1, "end": 0.3, "probability": 0.98}]
        return result

    def stream(self, chunks, language=None, keywords=None):
        sizes = [len(chunk) for chunk in chunks]
        FakeWhistle.calls.append(("stream", sizes, language, keywords))
        for second in range(len(sizes)):
            yield {"text": f"w{second}", "words": [{"word": f"w{second}", "start": second, "end": second + 0.5,
                                                   "probability": 0.9}],
                   "pending": "", "language": "de", "received": second + 1, "pass_ms": 5}
        yield {"text": "", "words": [], "pending": "", "language": "", "received": len(sizes), "pass_ms": 0}


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch):
    FakeWhistle.calls = []
    monkeypatch.setitem(sys.modules, "needle", types.SimpleNamespace(Whistle=FakeWhistle))


@pytest.fixture
def client():
    with TestClient(server.app) as client:
        yield client


def write_wav(path, seconds, rate=44100, channels=2):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        frames = (struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) * channels
                  for i in range(int(seconds * rate)))
        out.writeframes(b"".join(frames))
    return path


def post(client, url, path, **data):
    with open(path, "rb") as audio:
        return client.post(url, files={"file": (path.name, audio)}, data=data)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model": "whistle.cact"}


def test_audio_is_resampled_to_16khz_mono_float32(client, tmp_path):
    response = post(client, "/transcribe", write_wav(tmp_path / "clip.wav", 2), language="de",
                    keywords="Siobhan, Krzysztof", word_timestamps="true")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"] == "turn off the kitchen lights"
    assert body["words"][0]["word"] == "turn"
    assert body["duration"] == pytest.approx(2, abs=0.01)
    kind, size, language, keywords = FakeWhistle.calls[0]
    assert (kind, language, keywords) == ("transcribe", "de", ["Siobhan", "Krzysztof"])
    assert size == pytest.approx(2 * BYTES_PER_SECOND, abs=64)


def test_audio_over_30_seconds_is_streamed_one_second_at_a_time(client, tmp_path):
    response = post(client, "/transcribe", write_wav(tmp_path / "long.wav", 32.5, rate=16000, channels=1),
                    word_timestamps="true")
    assert response.status_code == 200, response.text
    body = response.json()
    kind, sizes, _, _ = FakeWhistle.calls[0]
    assert kind == "stream"
    assert sizes == [BYTES_PER_SECOND] * 32 + [BYTES_PER_SECOND // 2]
    assert body["text"] == " ".join(f"w{i}" for i in range(33))
    assert body["language"] == "de"
    assert len(body["words"]) == 33
    assert body["duration"] == 32.5


def test_rejects_what_is_not_audio(client, tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("not audio")
    response = post(client, "/transcribe", junk)
    assert response.status_code == 400
    assert "could not decode" in response.json()["detail"]


def test_rejects_an_unsupported_language(client, tmp_path):
    response = post(client, "/transcribe", write_wav(tmp_path / "clip.wav", 1), language="ja")
    assert response.status_code == 400
    assert FakeWhistle.calls == []


def test_rejects_audio_over_the_limit(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MAX_AUDIO_SECONDS", 2.0)
    response = post(client, "/transcribe", write_wav(tmp_path / "clip.wav", 4, rate=16000, channels=1))
    assert response.status_code == 413
    assert FakeWhistle.calls == []


def test_openai_formats(client, tmp_path):
    clip = write_wav(tmp_path / "clip.wav", 1)
    url = "/v1/audio/transcriptions"
    assert post(client, url, clip, model="whisper-1").json() == {"text": "turn off the kitchen lights"}
    text = post(client, url, clip, response_format="text")
    assert text.headers["content-type"].startswith("text/plain")
    assert text.text == "turn off the kitchen lights"
    verbose = post(client, url, clip, response_format="verbose_json", language="en", prompt="Siobhan",
                   **{"timestamp_granularities[]": "word"}).json()
    assert verbose["task"] == "transcribe"
    assert verbose["language"] == "english"
    assert verbose["words"] == [{"word": "turn", "start": 0.1, "end": 0.3}]
    assert FakeWhistle.calls[-1][3] == ["Siobhan"]
    assert post(client, url, clip, response_format="srt").status_code == 400


def test_cli(tmp_path, capsys):
    clip = write_wav(tmp_path / "clip.wav", 1)
    assert cli.main([str(clip), "--language", "fr", "--word-timestamps"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "turn off the kitchen lights"
    assert "turn" in out[1]
    assert FakeWhistle.calls[0][2] == "fr"


def test_cli_reports_files_it_cannot_read(tmp_path, capsys):
    clip = write_wav(tmp_path / "clip.wav", 1)
    assert cli.main([str(tmp_path / "missing.wav"), str(clip), "--json"]) == 1
    captured = capsys.readouterr()
    assert "missing.wav" in captured.err
    assert '"file":' in captured.out
