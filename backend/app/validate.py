"""Real-model integration validation; writes measured results only."""
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
from .config import Settings
from .embedding import EmbeddingService
from .ingestion import ingest
from .store import VectorStore


def validate():
    settings = Settings()
    settings.prepare()
    engine = EmbeddingService(settings)
    engine.load()
    root = Path(__file__).resolve().parents[1] / "demo"
    store = VectorStore(settings.chroma_path)
    records = []
    for entry in json.loads((root / "manifest.json").read_text()):
        source = root / entry["file"]
        target = Path(settings.upload_dir) / (hashlib.sha256(source.read_bytes()).hexdigest() + source.suffix)
        shutil.copyfile(source, target)
        records.append(ingest(target, source.name, engine, store, evidence=entry.get("evidence")))
    searches = []
    for query in ("Find information about worker safety.", "Find documents or media related to company vehicles.", "Find information about employee training."):
        started = time.perf_counter()
        results = store.search(engine.embed_query(query))
        searches.append({"query": query, "seconds": time.perf_counter() - started, "results": results})
    for modality, file in (("image", "protective-equipment.png"), ("audio", "safety-briefing.wav")):
        vector = getattr(engine, f"embed_{modality}")(root / file)
        assert len(vector) == 768 and np.isfinite(vector).all()
        searches.append({"query_modality": modality, "file": file, "results": store.search(vector, modality="text")})
    assert {f["modality"] for f in store.files()} == {"text", "image", "audio", "video"}
    report = {"model": engine.status(), "indexing": records, "searches": searches}
    Path("validation-results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"model": report["model"], "indexed_files": len(records), "searches": len(searches)}, indent=2))


if __name__ == "__main__":
    validate()
