FROM ghcr.io/astral-sh/uv:0.11.19 AS uv
FROM python:3.11-slim
COPY --from=uv /uv /uvx /bin/
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=3 -o Acquire::https::Timeout=30 update \
    && apt-get -o Acquire::Retries=3 -o Acquire::https::Timeout=30 install -y --no-install-recommends ffmpeg espeak libsndfile1 fonts-dejavu-core
WORKDIR /app
ENV UV_LINK_MODE=copy PYTHONUNBUFFERED=1 HF_HOME=/app/data/huggingface EMBEDDING_DEVICE=cpu
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
COPY app ./app
COPY demo ./demo
EXPOSE 8000
CMD ["uv", "run", "--frozen", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
