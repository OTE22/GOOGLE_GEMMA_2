# One Space · EmbeddingGemma 2 Lab

A local-first scientific demo: native text, image, audio and video embeddings share one normalized 768-dimensional Chroma collection. React provides five pages: overview, uploads, cross-modal search, cited RAG, and an interactive learning lab. No alternative embedding models, generated search results, or invented scores are used.

## Start locally

Prerequisites: uv 0.11.19+, Python 3.11 (managed by uv), Node.js 24+, FFmpeg and FFprobe on PATH. Windows demo speech uses System.Speech; Linux demo generation needs `espeak`.

From the project root in PowerShell:

```powershell
Copy-Item .env.example .env
cd backend
uv sync --frozen
uv run python -m app.demo
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open http://localhost:3000. Backend: http://localhost:8000; API docs: http://localhost:8000/docs. Click **Load model** or run your first search to download and cache the full model; then **Load demo dataset**. The initial download and CPU inference can take time. Model status reports failures and installed library versions.

If port 8000 is occupied, use `--port 8001` for Uvicorn and set `$env:BACKEND_URL='http://127.0.0.1:8001'` before starting the frontend. Docker defaults remain 3000/8000 on localhost.

Python project initialization used `uv init backend`; dependencies are maintained with `uv add` and `uv add --dev`. Always use `uv sync` and `uv run`. Commit `backend/pyproject.toml` and `backend/uv.lock` together after dependency changes. Frontend dependencies use npm and `package-lock.json`.

## Docker

```powershell
Copy-Item .env.example .env
docker compose up --build -d
```

The backend image installs dependencies with `uv sync --frozen --no-dev` and executes through `uv run --frozen`. Docker caches uv and npm downloads across rebuilds; Debian dependencies use HTTPS with bounded retries. Three named volumes persist vectors, originals, and weights across restarts. Generated demo media is included in the image. To regenerate inside the container: `docker compose exec backend uv run --frozen python -m app.demo`.

This deployment is CPU-only. Torch and torchvision use the explicit official `https://download.pytorch.org/whl/cpu` index in `pyproject.toml` and `uv.lock`. Docker and the default Compose service enforce `EMBEDDING_DEVICE=cpu`; inference uses float32. Use the base Compose file alone. The retained GPU override is unused and incompatible with these CPU-only wheels.

## Native API and scientific choices

Verified references on October 10, 2026:

- [Official model card](https://huggingface.co/google/embeddinggemma-2)
- [Google model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2)
- [Transformers model API](https://huggingface.co/docs/transformers/model_doc/embedding_gemma2)
- [Sentence Transformers input formats](https://sbert.net/docs/input_formats.html)
- [OpenRouter free routing](https://openrouter.ai/openrouter/free)

The locked stack includes Sentence Transformers 6.1+ and Transformers 5.19+. Model input dictionaries use `image`, `audio`, and `video`; text uses `SearchQuery`, `Document`, or symmetric `SentenceSimilarity`. All embeddings are validated for dimension, finiteness and nonzero norm, then normalized. The complete vision and audio encoders are checked at load time. Parameter counts and device are measured after loading, rather than shown as measurements before loading.

Text is split into overlapping passages (1400 characters, 200 overlap), with PDF page references. Audio is converted to mono 16 kHz and split at 60 seconds, then supplied as actual waveform arrays. Video is split at 30 seconds; PyAV supplies one RGB frame per second plus original frame metadata to the model's native video pipeline. Further processor sampling is disabled. This documented frame-array input avoids the platform-specific TorchCodec/FFmpeg shared-library requirement for path decoding. Video embeddings represent visual frames; audio is a separate native input. Originals retain their audio. Media timestamps link to the original file. Scanned PDF OCR is not included.

Chroma collection: `embeddinggemma2_multimodal`, cosine distance, 768 dimensions. Original bytes live outside Chroma; SHA-256 hashes produce stable file and chunk IDs. Re-uploading identical content upserts those IDs. Chroma omits unavailable metadata fields because it does not accept null metadata; clients can interpret missing page/time fields as null. Query media is saved locally but is not automatically indexed. Failed-upload originals remain available on disk for troubleshooting.

Progress uses local background threads and polling, intentionally without a job broker. Pending jobs do not survive a restart; indexed files do. This is a single-user local demonstration, without authentication or production-grade ingestion isolation. Uploads validate extension and size first; parsers validate actual format while processing.

## Evidence chat and privacy

Set `OPENROUTER_API_KEY` in `.env` on the backend only. `openrouter/free` is the default free routing model; availability and rate limits can vary. Local embeddings and search work without this key. Chat sends the question and retrieved readable excerpts externally. It never sends embedding vectors as though they were media content.

LangGraph retrieves, checks readable evidence availability, optionally rewrites and searches again, and generates a cited answer. Maximum two retrieval attempts. The evidence check is deliberately conservative and simple: absence of any readable passages triggers a retry; it is not a claim of measured semantic sufficiency. The LLM is instructed to abstain when passages do not establish the answer and treat evidence as untrusted data.

Demo images are original illustrations. Audio is actual synthesized speech with its exact script recorded as evidence. The short video is a real MP4 slideshow with a documented timeline and safety briefing audio. These are generated fixtures, not field recordings or empirical benchmarks. Captions and scripts supplement RAG only; native media, not these captions, is embedded.

## Validation

```powershell
cd backend
uv run pytest -q
uv run python -m app.validate_processors
uv run python -m app.validate
cd ../frontend
npm run build
npm run lint
```

The unit tests use a clearly identified test embedding engine to verify API behavior and storage, never as an application fallback. `app.validate_processors` checks actual native preprocessing using cached official processor files; it does not load the encoders. `app.validate` loads the actual model, ingests all eight demo files, checks a shared multimodal collection, and writes actual search rankings and timings to `validation-results.json`. It also runs image-to-text and audio-to-text retrieval. A valid OpenRouter key is required to validate live chat. See `VALIDATION.md` for actual execution results and remaining limitations.

The printable guide is at `/guide.html`, with five fixed A4 sheets, explicit page breaks, and no trailing sixth page. Print at A4, 100% scale, browser headers/footers off.
