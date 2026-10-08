# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NEEDLE_TELEMETRY=0 \
    DO_NOT_TRACK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt


# The engine for this CPU architecture (linux x86_64 or arm64) and the 16.9 MB
# whistle.cact weights, both from Hugging Face. They are baked into the image,
# so the container runs offline.
FROM base AS model
RUN needle fetch --generation 3 --out /opt/whistle \
    && needle download whistle --out /opt/whistle


FROM base

COPY --from=model /opt/whistle /opt/whistle
COPY app ./app
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

ENV NEEDLE3_LIB_PATH=/opt/whistle/libneedle.so \
    NEEDLE_WHISTLE_WEIGHTS=/opt/whistle/whistle.cact \
    HF_HUB_OFFLINE=1 \
    PORT=8000 \
    WHISTLE_DB=/data/whistle.db

# Transcripts are stored in /data: mount a volume there to keep them.
RUN useradd --system --create-home --uid 10001 whistle \
    && mkdir /data && chown whistle /data
USER whistle

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'], timeout=4)"

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["serve"]
