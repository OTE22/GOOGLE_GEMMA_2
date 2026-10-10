# Validation record · October 10, 2026

## Passed

- Initialized the Python project with `uv init backend`; declared dependencies with `uv add` and `uv add --dev`; generated `backend/uv.lock`.
- Synchronized a separate uv-managed Python 3.11.15 test environment using the lockfile, excluding PyTorch/torchvision while their download is pending. This environment is for API/storage/media tests, not native inference.
- Backend: **8 tests passed**. Checks cover finite normalized 768-dimensional vector validation, persistent mixed-modality Chroma storage and cosine ranking, modality filters, keyword comparison, duplicate ingestion, API type/size/empty-input validation, the two-retrieval agent limit, guide structure, real demo decoding, mono 16 kHz audio conversion, and long-audio timestamps.
- Frontend: **4 tests passed**, covering all five pages, learning controls and theme, real upload request structure and job polling, filtered text search request/results, and cited chat sources/stages. API responses and the diagram component are explicit test fixtures; this is not real-model or browser visual validation.
- Frontend TypeScript checking and production build passed. Lint passed. The build reports a bundle-size advisory for the chart/diagram dependencies.
- Generated all eight demo files: two readable PDFs, three original illustrations, two actual synthesized WAV recordings, and one playable 12-second MP4 slideshow. Speech scripts and captions are documented in the manifest.
- Live backend health, model status, empty collection statistics, and API documentation respond correctly on port **8001**, used because port 8000 is occupied. The frontend proxy on port 3000 reaches this server.
- Frontend and guide return HTTP 200. Guide has exactly five A4 sheet sections and explicit page breaks, including no trailing break after sheet five.
- CPU and GPU Compose configurations validate. Docker engine access works. The frontend Docker image built successfully with `npm ci`, TypeScript checking, Vite, and Nginx.
- Verified the official model card, Transformers API, Sentence Transformers input formats, and published prompt configuration. The model's Hub API confirms `gated: false`. Exact model ID is enforced; no substitute embedding models are used.

## Not yet verified

- Full PyTorch environment installation and native model loading. The original uv download is still pending on the large PyTorch wheel.
- Real 768-dimensional embeddings for every modality, real mixed-modality ingestion, semantic ranking, image/audio-to-text retrieval, and live similarity measurements. The integration command is `uv run python -m app.validate`; these must not be inferred from the unit tests.
- Model weights download: configuration files and both tokenizer files are now cached (12 of 13 requested files). Large-file HTTPS requests repeatedly timed out during TLS handshakes; the supported Xet transport is downloading the remaining weights. No certificate checks were disabled.
- GPU inference and complete backend Docker image/runtime. Docker source uses `uv sync --frozen --no-dev`; this has not yet been verified by running the full backend image.
- Live OpenRouter completion: no valid API key was provided. Tests verify graph behavior with a clearly identified client test double. Local search has no dependency on OpenRouter.
- Visual browser testing and physical print pagination: this session exposes no browser surfaces (`apps: []`, `browsers: []`). Automated DOM/component, type, HTTP, and print-structure checks passed, but are not a substitute for a visual browser/print check.

## Reproduce

From `backend`: `uv sync --frozen`, `uv run pytest -q`, `uv run python -m app.validate`.
From `frontend`: `npm ci`, `npm test`, `npm run build`, `npm run lint`.
With a backend key configured, send an actual question to `/api/chat` and inspect citations and stages.

If port 8000 is busy, run `uv run uvicorn app.main:app --port 8001`, and set `BACKEND_URL=http://127.0.0.1:8001` in the frontend terminal before `npm run dev`. Default deployment ports remain 3000/8000.
