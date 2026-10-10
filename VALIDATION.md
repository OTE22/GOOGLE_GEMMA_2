# Validation record · October 10, 2026

## Passed

- Initialized the Python project with `uv init backend`; declared dependencies with `uv add` and `uv add --dev`; generated `backend/uv.lock`.
- Synchronized the complete normal backend environment with `uv sync --frozen`, using uv-managed Python 3.11.15. Native imports succeed: torch 2.14.1+cpu, torchvision 0.29.1+cpu, Transformers 5.19.0, and Sentence Transformers 6.1.0. CUDA is unavailable on this host.
- Backend: **9 tests passed** in the complete environment. Checks cover finite normalized 768-dimensional vector validation, persistent mixed-modality Chroma storage and cosine ranking with explicit test vectors, modality filters, keyword comparison, duplicate ingestion, API type/size/empty-input validation, the two-retrieval agent limit, guide structure, real demo decoding, mono 16 kHz audio conversion, long-audio timestamps, and native media input contracts.
- Official native processors passed with actual text, image, waveform, and decoded video inputs: 6, 277, 421, and 1538 tokens respectively. The video contains 12 sampled frames. Measured tensor shapes and preprocessing durations are recorded in `backend/processor-validation.json`. These are preprocessing checks, not encoder outputs or embedding latency measurements. The adapter uses the Sentence Transformers 6.1 `.model` API and documented waveform/frame-array inputs.
- Frontend: **4 tests passed**, covering all five pages, learning controls and theme, real upload request structure and job polling, filtered text search request/results, and cited chat sources/stages. API responses and the diagram component are explicit test fixtures; this is not real-model or browser visual validation.
- Frontend TypeScript checking and production build passed. Lint passed. The build reports a bundle-size advisory for the chart/diagram dependencies.
- Generated all eight demo files: two readable PDFs, three original illustrations, two actual synthesized WAV recordings, and one playable 12-second MP4 slideshow. Speech scripts and captions are documented in the manifest.
- Live backend health, model status, empty collection statistics, and API documentation respond correctly on port **8001**, used because port 8000 is occupied. The frontend proxy on port 3000 reaches this server.
- Frontend and guide return HTTP 200. Guide has exactly five A4 sheet sections and explicit page breaks, including no trailing break after sheet five.
- CPU and GPU Compose configurations validate. Docker engine access works. The frontend Docker image built successfully with `npm ci`, TypeScript checking, Vite, and Nginx.
- Verified the official model card, Transformers API, Sentence Transformers input formats, and published prompt configuration. The model's Hub API confirms `gated: false`. Exact model ID is enforced; no substitute embedding models are used.

## Not yet verified

- Native model loading. An actual cached integration attempt (`HF_HUB_OFFLINE=1 uv run --frozen python -m app.validate`) failed because the checkpoint weights are not yet cached. The complete Python environment is installed; this failure is specifically the missing model weights.
- Real 768-dimensional embeddings for every modality, real mixed-modality ingestion, semantic ranking, image/audio-to-text retrieval, and live similarity measurements. The integration command is `uv run python -m app.validate`; these must not be inferred from the unit tests.
- Model weights download: configuration files and both tokenizer files are cached (12 of 13 requested files). Large-file HTTPS requests repeatedly timed out during TLS handshakes; the supported Xet transport still has an incomplete weights download. No certificate checks were disabled.
- GPU inference and complete backend Docker image/runtime. Docker source uses `uv sync --frozen --no-dev`; this has not yet been verified by running the full backend image.
- Live OpenRouter completion: no valid API key was provided. Tests verify graph behavior with a clearly identified client test double. Local search has no dependency on OpenRouter.
- Visual browser testing and physical print pagination: this session exposes no browser surfaces (`apps: []`, `browsers: []`). Automated DOM/component, type, HTTP, and print-structure checks passed, but are not a substitute for a visual browser/print check.

## Reproduce

From `backend`: `uv sync --frozen`, `uv run pytest -q`, `uv run python -m app.validate_processors` (cached processor files required), `uv run python -m app.validate` (full checkpoint required).
From `frontend`: `npm ci`, `npm test`, `npm run build`, `npm run lint`.
With a backend key configured, send an actual question to `/api/chat` and inspect citations and stages.

If port 8000 is busy, run `uv run uvicorn app.main:app --port 8001`, and set `BACKEND_URL=http://127.0.0.1:8001` in the frontend terminal before `npm run dev`. Default deployment ports remain 3000/8000.
