import hashlib
import json
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from .config import Settings
from .embedding import EmbeddingService
from .ingestion import EXTENSIONS, ingest, parts
from .store import VectorStore
from .rag import chat


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=24000)
    modality: Literal["text", "image", "audio", "video"] | None = None
    limit: int = Field(default=12, ge=1, le=50)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


def create_app(settings=None, engine=None, store=None):
    settings = settings or Settings()
    settings.prepare()
    engine = engine or EmbeddingService(settings)
    store = store or VectorStore(settings.chroma_path)
    app = FastAPI(title="EmbeddingGemma 2 · One Space", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:5173"],
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    app.mount("/media", StaticFiles(directory=settings.upload_dir), name="media")
    jobs = {}
    job_lock = threading.Lock()

    def guarded(operation):
        try:
            return operation()
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc

    async def save(file):
        name = Path((file.filename or "file").replace("\\", "/")).name
        suffix = Path(name).suffix.lower()
        if suffix not in EXTENSIONS:
            raise HTTPException(415, "Supported: PDF, TXT, JPG, PNG, MP3, WAV, MP4")
        temp = Path(settings.upload_dir) / f"{uuid.uuid4().hex}{suffix}"
        size = 0
        digest = hashlib.sha256()
        try:
            with temp.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_mb * 1024 * 1024:
                        raise HTTPException(413, "File exceeds upload size limit")
                    digest.update(chunk)
                    output.write(chunk)
            if size == 0:
                raise HTTPException(422, "File is empty")
            destination = temp.with_name(digest.hexdigest() + suffix)
            if not destination.exists():
                temp.replace(destination)
            return destination, name
        finally:
            temp.unlink(missing_ok=True)

    def new_job(path, name, evidence=None):
        job_id = uuid.uuid4().hex
        with job_lock:
            # Keep transient progress bounded; indexed metadata lives in Chroma.
            if len(jobs) > 200:
                for key in list(jobs):
                    if jobs[key]["stage"] in ("Searchable", "Error"):
                        del jobs[key]
            jobs[job_id] = {"id": job_id, "file_name": name, "stage": "Upload"}
        def progress(stage):
            with job_lock:
                jobs[job_id]["stage"] = stage
        def work():
            try:
                result = ingest(path, name, engine, store, progress, evidence)
                with job_lock:
                    jobs[job_id].update(result, stage="Searchable")
            except Exception as exc:
                with job_lock:
                    jobs[job_id].update(stage="Error", error=str(exc))
        threading.Thread(target=work, daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/health")
    def health():
        return {"status": "ok", "inference": engine.state}

    @app.get("/api/model/status")
    def model_status():
        return engine.status()

    @app.post("/api/model/load")
    def model_load():
        return guarded(lambda: (engine.load(), engine.status())[1])

    @app.get("/api/files")
    def files():
        return store.files()

    @app.get("/api/stats")
    def stats():
        files = store.files()
        return {"files": len(files), "vectors": store.collection.count(),
                "modalities": {m: sum(f["vectors"] for f in files if f["modality"] == m) for m in ("text", "image", "audio", "video")}}

    @app.post("/api/upload", status_code=202)
    async def upload(file: UploadFile = File()):
        path, name = await save(file)
        return new_job(path, name)

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        with job_lock:
            if job_id not in jobs:
                raise HTTPException(404, "Unknown job; progress is transient across restarts. Check indexed files.")
            return dict(jobs[job_id])

    @app.post("/api/embed/text")
    def embed_text(request: TextRequest):
        return guarded(lambda: {"embedding": engine.embed_text(request.text), "dimensions": 768})

    def media_handler(kind):
        async def handler(file: UploadFile = File()):
            path, _ = await save(file)
            if EXTENSIONS[path.suffix] != kind:
                raise HTTPException(415, f"Expected {kind}")
            def run():
                with tempfile.TemporaryDirectory() as directory:
                    return {"dimensions": 768, "segments": [
                        {"embedding": getattr(engine, f"embed_{kind}")(value), **position}
                        for value, position in parts(path, Path(directory))]}
            return await run_in_threadpool(guarded, run)
        return handler
    for kind in ("image", "audio", "video"):
        app.add_api_route(f"/api/embed/{kind}", media_handler(kind), methods=["POST"])

    @app.post("/api/search/text")
    def search_text(request: TextRequest):
        def run():
            started = time.perf_counter()
            results = store.search(engine.embed_query(request.text), request.limit, request.modality)
            return {"results": results, "keyword_results": store.keyword(request.text), "seconds": time.perf_counter() - started}
        return guarded(run)

    def search_handler(kind):
        async def handler(file: UploadFile = File(), modality: str | None = Form(default=None)):
            if modality not in (None, "text", "image", "audio", "video"):
                raise HTTPException(422, "Invalid modality")
            path, _ = await save(file)
            if EXTENSIONS[path.suffix] != kind:
                raise HTTPException(415, f"Expected {kind}")
            def run():
                started = time.perf_counter()
                with tempfile.TemporaryDirectory() as directory:
                    values = list(parts(path, Path(directory)))
                    if len(values) != 1:
                        raise ValueError("Audio queries must be at most 60 seconds")
                    vector = getattr(engine, f"embed_{kind}")(values[0][0])
                    return {"results": store.search(vector, 12, modality), "seconds": time.perf_counter() - started, "keyword_results": []}
            return await run_in_threadpool(guarded, run)
        return handler
    for kind in ("image", "audio"):
        app.add_api_route(f"/api/search/{kind}", search_handler(kind), methods=["POST"])

    @app.post("/api/experiment")
    def experiment(texts: list[str]):
        if not 2 <= len(texts) <= 6 or any(not t.strip() or len(t) > 4000 for t in texts):
            raise HTTPException(422, "Supply 2–6 nonempty texts, each at most 4000 characters")
        def run():
            import numpy as np
            started = time.perf_counter()
            vectors = np.array([engine.encode(t, "SentenceSimilarity") for t in texts])
            return {"similarities": (vectors @ vectors.T).tolist(), "seconds": time.perf_counter() - started}
        return guarded(run)

    @app.post("/api/chat")
    def chat_endpoint(request: ChatRequest):
        return guarded(lambda: chat(request.question, engine, store, settings))

    @app.post("/api/demo/load", status_code=202)
    def demo():
        root = Path(__file__).resolve().parents[1] / "demo"
        manifest = root / "manifest.json"
        if not manifest.exists():
            raise HTTPException(409, "Generate the dataset first: uv run python -m app.demo")
        import shutil
        result = []
        for entry in json.loads(manifest.read_text()):
            source = root / entry["file"]
            destination = Path(settings.upload_dir) / (hashlib.sha256(source.read_bytes()).hexdigest() + source.suffix)
            shutil.copyfile(source, destination)
            result.append(new_job(destination, source.name, entry.get("evidence")))
        return {"jobs": result}
    return app


app = create_app()
