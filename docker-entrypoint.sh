#!/bin/sh
set -e

case "$1" in
    serve)
        shift
        exec uvicorn app.server:app --host 0.0.0.0 --port "${PORT:-8000}" --workers "${WHISTLE_WORKERS:-1}" "$@"
        ;;
    transcribe)
        shift
        exec python -m app.cli "$@"
        ;;
esac

exec "$@"
