"""transcribe FILE...: Whistle speech-to-text from the command line."""

from __future__ import annotations

import argparse
import json
import sys

from .audio import AudioDecodeError, decode
from .transcriber import LANGUAGES, Transcriber, parse_keywords


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="transcribe", description="Transcribe audio or video files with Whistle.")
    parser.add_argument("files", nargs="+", help="Any audio or video file ffmpeg can read")
    parser.add_argument("--language", choices=LANGUAGES, help="Force the language (default: detect it)")
    parser.add_argument("--keywords", default="", help="Comma-separated words and names to favour")
    parser.add_argument("--word-timestamps", action="store_true", help="Show each word's start, end and probability")
    parser.add_argument("--json", action="store_true", help="Print the full result as one JSON object per file")
    args = parser.parse_args(argv)

    transcriber = Transcriber()
    failed = False
    for path in args.files:
        try:
            result = transcriber.transcribe(decode(path), args.language, parse_keywords(args.keywords),
                                            args.word_timestamps)
        except (AudioDecodeError, RuntimeError) as error:
            print(f"{path}: {error}", file=sys.stderr)
            failed = True
            continue
        if args.json:
            print(json.dumps({"file": path, **result}, ensure_ascii=False))
            continue
        print(f"{path}: {result['text']}" if len(args.files) > 1 else result["text"])
        for word in result.get("words", []):
            print(f"  {word['start']:7.2f} - {word['end']:7.2f}  {word['word']:<20} {word['probability']:.2f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
