# whistle

[Cactus Whistle](https://cactuscompute.com/blog/whistle) speech-to-text in a Docker container, with a recording page for your phone, an HTTP API (including an OpenAI-compatible endpoint) and a CLI. Every transcript is stored on the server along with the time its audio was recorded.

Whistle is a 16.9 MB speech-to-text model that runs on the CPU, no GPU needed. It handles English, German, French, Spanish, Italian, Dutch and Polish, and can return word timestamps and favour keywords you pass in. It runs on Cactus Compute's [`cactus-needle`](https://pypi.org/project/cactus-needle/) engine. The image downloads the engine and the `whistle.cact` weights from Hugging Face at build time, so the container itself runs offline.

## Run

```sh
docker compose up -d --build
```

Then open <http://localhost:8000> on the server for the recording page, or <http://localhost:8000/docs> for the API.

The build fetches the engine for the machine's architecture, `linux/amd64` or `linux/arm64` (Apple Silicon, Raspberry Pi with a 64-bit OS), from [Cactus-Compute/needle3](https://huggingface.co/Cactus-Compute/needle3), and the weights from [Cactus-Compute/whistle](https://huggingface.co/Cactus-Compute/whistle).

## Record from your phone over the VPN

Browsers only allow the microphone on HTTPS (or `localhost`). Over the VPN, `http://<server>:8000` lets you upload recordings but not record. The `https` profile adds [Caddy](https://caddyserver.com) on port 8443 with a certificate from its own local CA, which works for a bare WireGuard IP and needs no domain.

1. `cp .env.example .env` and set `WHISTLE_HOST` to the address the phone will open, such as the server's WireGuard IP.
2. `docker compose --profile https up -d --build`
3. With the phone on the VPN, open `https://<WHISTLE_HOST>:8443`. The browser warns about the certificate the first time. Tap through it (Advanced → Proceed) and recording works.

To get rid of the warning and add the page to your home screen as an app, trust Caddy's root certificate on the phone:

```sh
docker compose cp https:/data/caddy/pki/authorities/local/root.crt whistle-root.crt
```

Copy `whistle-root.crt` to the phone, then:

- **Android**: Settings → Security → More security settings → Encryption & credentials → Install a certificate → CA certificate.
- **iPhone**: open the file, install it under Settings → Profile Downloaded, then turn it on in Settings → General → About → Certificate Trust Settings.

Anyone holding that CA's key can issue certificates the phone will trust, so keep the `caddy-data` volume private.

There's no login, so anyone who can reach the ports can use Whistle. Docker's published ports bypass ufw and firewalld, so on a server with a public IP, set `BIND_ADDRESS` in `.env` to the WireGuard IP to listen on the VPN only.

### The recording page

- Tap the button to record and tap again to stop. A ring follows your voice, so you can see the microphone is picking you up, and the screen stays on while recording.
- Long recordings show the transcript building up second by second.
- Transcripts are [stored on the server](#stored-transcripts) and listed newest first, each with the date and time it was recorded, in the phone's time zone. They appear on any device that opens the page. Copy, share and delete work on each one, and a failed upload keeps its audio, so you can retry it.
- Settings (the sliders icon) let you force a language or add keywords such as names; both are remembered.
- "Upload a recording" transcribes an existing file, and works over plain HTTP too.

### Cloudflare Tunnel (optional)

To reach it from outside the VPN instead:

1. In Cloudflare Zero Trust, go to Networks → Tunnels, create a tunnel, and copy its token into `CLOUDFLARE_TUNNEL_TOKEN` in `.env`.
2. Give the tunnel a public hostname whose service is `http://whistle:8000`.
3. `docker compose --profile tunnel up -d --build`
4. Put the hostname behind Cloudflare Access (Access → Applications → Self-hosted, allowing only your email). Otherwise the whole internet can use it.

Cloudflare provides the HTTPS, so recording works without the `https` profile. Its free plan limits uploads to 100 MB. The page streams progress while it transcribes, so long recordings don't run into Cloudflare's 100-second response timeout.

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
| `recorded_at` | When the audio was recorded, ISO 8601 (`2026-10-08T22:10:03Z`, or with an offset such as `+02:00`); defaults to now |

The transcript is stored, and the response is the stored record:

```json
{
  "id": 42,
  "recorded_at": "2026-10-08T22:10:03Z",
  "text": "turn off the kitchen lights",
  "language": "en",
  "ttft_ms": 11.1,
  "decode_tps": 1319.0,
  "duration": 2.4,
  "words": [{"word": "turn", "start": 0.12, "end": 0.31, "probability": 0.98}]
}
```

Silence and steady noise return an empty `text` and `language`.

### `POST /transcribe/stream`

This takes the same fields as `/transcribe` and answers with server-sent events. While audio longer than 30 s is transcribed, it sends a `{"partial": {"text", "seconds", "duration"}}` event each second, then `{"result": {...}}` with the same object `/transcribe` returns. If the engine fails part way, the last event is `{"error": "..."}` instead. The recording page uses this endpoint.

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

### Stored transcripts

Every transcription, from any endpoint, is saved to a SQLite database at `/data/whistle.db`, on the `whistle-data` volume in compose. Each one gets an `id` and a `recorded_at` time in UTC, to the second:

- For recordings made on the page, that's when you tapped record.
- For uploaded files, it's the file's last-modified time.
- For API calls, it's the `recorded_at` you send, or the time of the request.

| | |
| --- | --- |
| `GET /transcripts?limit=100&offset=0` | Stored transcripts, the most recently recorded first |
| `GET /transcripts/{id}` | One transcript |
| `DELETE /transcripts/{id}` | Delete one |

```sh
curl http://localhost:8000/transcripts?limit=1
# [{"id": 42, "recorded_at": "2026-10-08T22:10:03Z", "text": "turn off the kitchen lights", "language": "en", "duration": 2.4}]
```

`words` is included when the transcript was made with word timestamps. To back up the database, copy it out with `docker compose cp whistle:/data/whistle.db .`. If you mount a host folder at `/data` instead of the volume, it must be writable by the container's user, uid 10001 (`sudo chown 10001 <folder>`).

### `GET /health`

Returns `{"status": "ok", "model": "whistle.cact"}`. The image's Docker `HEALTHCHECK` calls this endpoint.

## CLI

Mount a folder and transcribe files without starting the server:

```sh
docker run --rm -v "$PWD:/audio" whistle transcribe /audio/meeting.m4a
docker run --rm -v "$PWD:/audio" whistle transcribe /audio/one.wav /audio/two.wav --language de --word-timestamps
docker run --rm -v "$PWD:/audio" whistle transcribe /audio/clip.mp3 --json
```

`--keywords "Siobhan, Krzysztof"` favours names, and `--json` prints the full result as one JSON object per file. The CLI prints transcripts and doesn't store them.

## Configuration

| Variable | Default | |
| --- | --- | --- |
| `PORT` | `8000` | Port the API listens on inside the container |
| `BIND_ADDRESS` | `0.0.0.0` | Host address compose publishes ports 8000 and 8443 on; set in `.env` |
| `WHISTLE_WORKERS` | `1` | Server processes. Each loads its own copy of the model (~17 MB) and handles one transcription at a time |
| `WHISTLE_MAX_AUDIO_SECONDS` | `3600` | Longer uploads are rejected with `413`; `0` removes the limit |
| `WHISTLE_DB` | `/data/whistle.db` | The transcripts database |
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

## License

This repository's own code is MIT (see [LICENSE](LICENSE)). It contains no code or weights from [Cactus Compute](https://cactuscompute.com), who make Whistle; the image downloads those when it's built, and they come under their own terms:

- `cactus-needle`, the Python package, is Apache-2.0 ([cactus-compute/needle](https://github.com/cactus-compute/needle)).
- The native engine comes from [Cactus-Compute/needle3](https://huggingface.co/Cactus-Compute/needle3) and the weights from [Cactus-Compute/whistle](https://huggingface.co/Cactus-Compute/whistle). Their licenses are on those Hugging Face pages.

Check those terms before you publish the built image or use it commercially.
