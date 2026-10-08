"""HTTP and audio handling, with a stand-in for the native Whistle engine."""

import json
import math
import struct
import sys
import types
import wave
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import cli, server
from app.storage import timestamp
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
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISTLE_DB", str(tmp_path / "data" / "whistle.db"))
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


def events(response):
    assert response.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[len("data: "):]) for line in response.text.split("\n\n") if line]


def test_stream_sends_the_result_of_a_short_clip(client, tmp_path):
    response = post(client, "/transcribe/stream", write_wav(tmp_path / "clip.wav", 2), keywords="Siobhan")
    assert response.status_code == 200, response.text
    [message] = events(response)
    assert message["result"]["text"] == "turn off the kitchen lights"
    assert FakeWhistle.calls[0][3] == ["Siobhan"]


def test_stream_reports_progress_on_long_audio(client, tmp_path):
    response = post(client, "/transcribe/stream", write_wav(tmp_path / "long.wav", 31, rate=16000, channels=1))
    assert response.status_code == 200, response.text
    *partials, last = events(response)
    assert [p["partial"]["text"] for p in partials] == [f"w{i}" for i in range(31)] + [""]
    assert partials[0]["partial"] == {"text": "w0", "seconds": 1, "duration": 31.0}
    assert last["result"]["text"] == " ".join(f"w{i}" for i in range(31))


def test_stream_rejects_bad_input_with_a_status(client, tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("not audio")
    assert post(client, "/transcribe/stream", junk).status_code == 400
    assert post(client, "/transcribe/stream", write_wav(tmp_path / "clip.wav", 1), language="ja").status_code == 400


def test_serves_the_recording_page(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "<title>Whistle</title>" in page.text
    for asset in ("static/app.js", "static/style.css", "manifest.webmanifest", "static/icon-180.png"):
        assert f'"{asset}"' in page.text
        assert client.get("/" + asset).status_code == 200, asset
    manifest = client.get("/manifest.webmanifest").json()
    for icon in manifest["icons"]:
        assert client.get("/" + icon["src"]).status_code == 200, icon["src"]


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


def test_timestamps_are_stored_as_utc_to_the_second():
    assert timestamp("2026-10-08T23:10:03+01:00") == "2026-10-08T22:10:03Z"
    assert timestamp("2026-10-08T22:10:03.456Z") == "2026-10-08T22:10:03Z"
    assert timestamp("2026-10-08T22:10:03") == "2026-10-08T22:10:03Z"
    now = datetime.fromisoformat(timestamp())
    assert abs((datetime.now(timezone.utc) - now).total_seconds()) < 5
    with pytest.raises(ValueError):
        timestamp("yesterday")


def test_transcripts_are_stored_with_when_they_were_recorded(client, tmp_path):
    clip = write_wav(tmp_path / "clip.wav", 1)
    first = post(client, "/transcribe", clip, recorded_at="2026-10-08T23:10:03.456+01:00",
                 word_timestamps="true").json()
    assert first["recorded_at"] == "2026-10-08T22:10:03Z"
    assert first["text"] == "turn off the kitchen lights"
    unstamped = post(client, "/transcribe", clip).json()
    assert abs((datetime.now(timezone.utc) - datetime.fromisoformat(unstamped["recorded_at"])).total_seconds()) < 5
    older = post(client, "/transcribe", clip, recorded_at="2025-01-01T09:00:00Z").json()

    listed = client.get("/transcripts").json()
    assert [t["id"] for t in listed] == [unstamped["id"], first["id"], older["id"]]
    assert listed[1] == {"id": first["id"], "recorded_at": "2026-10-08T22:10:03Z",
                         "text": "turn off the kitchen lights", "language": "en", "duration": 1.0,
                         "words": [{"word": "turn", "start": 0.1, "end": 0.3, "probability": 0.98}]}
    assert "words" not in listed[0]
    assert [t["id"] for t in client.get("/transcripts?limit=1&offset=1").json()] == [first["id"]]
    assert client.get(f"/transcripts/{first['id']}").json() == listed[1]


def test_transcripts_can_be_deleted(client, tmp_path):
    stored = post(client, "/transcribe", write_wav(tmp_path / "clip.wav", 1)).json()
    assert client.delete(f"/transcripts/{stored['id']}").status_code == 204
    assert client.get(f"/transcripts/{stored['id']}").status_code == 404
    assert client.delete(f"/transcripts/{stored['id']}").status_code == 404
    assert client.get("/transcripts").json() == []


def test_rejects_a_recorded_at_that_is_not_a_time(client, tmp_path):
    clip = write_wav(tmp_path / "clip.wav", 1)
    for url in ("/transcribe", "/transcribe/stream"):
        response = post(client, url, clip, recorded_at="yesterday")
        assert response.status_code == 400
        assert "ISO 8601" in response.json()["detail"]
    assert FakeWhistle.calls == []


def test_the_stream_and_openai_endpoints_store_too(client, tmp_path):
    clip = write_wav(tmp_path / "clip.wav", 1)
    [message] = events(post(client, "/transcribe/stream", clip, recorded_at="2026-10-08T22:10:03Z"))
    assert message["result"]["recorded_at"] == "2026-10-08T22:10:03Z"
    post(client, "/v1/audio/transcriptions", clip)
    listed = client.get("/transcripts").json()
    assert len(listed) == 2
    assert message["result"]["id"] in [t["id"] for t in listed]


def test_transcripts_survive_a_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISTLE_DB", str(tmp_path / "whistle.db"))
    clip = write_wav(tmp_path / "clip.wav", 1)
    with TestClient(server.app) as client:
        stored = post(client, "/transcribe", clip).json()
    with TestClient(server.app) as client:
        assert client.get("/transcripts").json()[0]["id"] == stored["id"]
