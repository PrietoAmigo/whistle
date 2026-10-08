# whistle

[Cactus Whistle](https://cactuscompute.com/blog/whistle) speech-to-text in a Docker container, with an HTTP API (including an OpenAI-compatible endpoint) and a CLI.

Whistle is a 16.9 MB speech-to-text model that runs on the CPU, no GPU needed. It handles English, German, French, Spanish, Italian, Dutch and Polish, and can return word timestamps and favour keywords you pass in. It runs on Cactus Compute's [`cactus-needle`](https://pypi.org/project/cactus-needle/) engine. The image downloads the engine and the `whistle.cact` weights from Hugging Face at build time, so the container itself runs offline.

## Run

```sh
docker compose up -d --build
```

or

```sh
docker build -t whistle .
docker run -d -p 8000:8000 --name whistle whistle
```

Then open <http://localhost:8000/docs> to try it from the browser.

The build fetches the engine for the machine's architecture, `linux/amd64` or `linux/arm64` (Apple Silicon, Raspberry Pi with a 64-bit OS), from [Cactus-Compute/needle3](https://huggingface.co/Cactus-Compute/needle3), and the weights from [Cactus-Compute/whistle](https://huggingface.co/Cactus-Compute/whistle).

## HTTP API

Uploads can be any audio or video file ffmpeg reads (wav, mp3, m4a, ogg, opus, webm, flac, mp4, …). They are converted to 16 kHz mono before transcription.

### `POST /transcribe`

```sh
curl http://localhost:8000/transcribe \
  -F file=@clip.wav \
  -F language=en \
  -F "keywords=Siobhan, Krzysztof" \
  -F word_timestamps=true
```

| Field | |
| --- | --- |
| `file` | The audio (required) |
| `language` | `en`, `de`, `fr`, `es`, `it`, `nl` or `pl`; detected when omitted |
| `keywords` | Comma-separated names and product words to favour |
| `word_timestamps` | `true` to add each word with its `start`, `end` and `probability` |

```json
{
  "text": "turn off the kitchen lights",
  "language": "en",
  "ttft_ms": 11.1,
  "decode_tps": 1319.0,
  "duration": 2.4,
  "words": [{"word": "turn", "start": 0.12, "end": 0.31, "probability": 0.98}]
}
```

Silence and steady noise return an empty `text` and `language`.

### `POST /v1/audio/transcriptions` (OpenAI compatible)

This is a drop-in replacement for OpenAI's transcription endpoint, so tools that speak that API (Open WebUI, dictation apps, the OpenAI SDKs) can use Whistle by changing the base URL.

```sh
curl http://localhost:8000/v1/audio/transcriptions -F file=@clip.wav -F model=whisper-1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
with open("clip.wav", "rb") as audio:
    print(client.audio.transcriptions.create(model="whistle", file=audio).text)
```

`response_format` can be `json` (the default), `text` or `verbose_json`. `verbose_json` with `timestamp_granularities[]=word` adds word timestamps. `prompt` is used as comma-separated keywords. `model` and `temperature` are accepted and ignored.

### `GET /health`

Returns `{"status": "ok", "model": "whistle.cact"}`. The image's Docker `HEALTHCHECK` calls this endpoint.

## CLI

Mount a folder and transcribe files without starting the server:

```sh
docker run --rm -v "$PWD:/data" whistle transcribe /data/meeting.m4a
docker run --rm -v "$PWD:/data" whistle transcribe /data/*.wav --language de --word-timestamps
docker run --rm -v "$PWD:/data" whistle transcribe /data/clip.mp3 --json
```

`--keywords "Siobhan, Krzysztof"` favours names, and `--json` prints the full result as one JSON object per file.

## Configuration

| Variable | Default | |
| --- | --- | --- |
| `PORT` | `8000` | Port the API listens on inside the container |
| `WHISTLE_WORKERS` | `1` | Server processes. Each loads its own copy of the model (~17 MB) and handles one transcription at a time |
| `WHISTLE_MAX_AUDIO_SECONDS` | `3600` | Longer uploads are rejected with `413`; `0` removes the limit |
| `NEEDLE_TELEMETRY`, `DO_NOT_TRACK` | `0`, `1` | `cactus-needle` sends anonymous usage counts by default. The image turns this off; set `NEEDLE_TELEMETRY=1` and `DO_NOT_TRACK=` to turn it back on |

## Notes

- Whistle transcribes up to 30 seconds in one pass. Longer audio is fed to Whistle's streaming decoder one second at a time, which has no length limit. `ttft_ms` and `decode_tps` are only reported for single-pass clips.
- The engine isn't thread-safe, so each worker runs one transcription at a time. To run requests in parallel, raise `WHISTLE_WORKERS` or start more containers.
- `cactus-needle` pins the engine and weights versions it downloads, so to move to a newer Whistle, bump it in `requirements.txt` and rebuild.

## Development

```sh
pip install -r requirements-dev.txt
pytest
```

The tests replace the native engine with a stand-in, so they need only `ffmpeg` on the `PATH`, not the model.

Whistle and the needle engine are made by [Cactus Compute](https://cactuscompute.com). `cactus-needle` is Apache-2.0, and the weights' terms are on the [model card](https://huggingface.co/Cactus-Compute/whistle).
