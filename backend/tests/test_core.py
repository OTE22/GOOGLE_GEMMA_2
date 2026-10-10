import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.embedding import normalized
from app.config import Settings
from app.store import VectorStore
from app.ingestion import ingest


def test_vector_validation():
    assert np.linalg.norm(normalized(np.ones(768))) == pytest.approx(1)
    for invalid in (np.ones(10), np.zeros(768), np.full(768, np.nan)):
        with pytest.raises(ValueError):
            normalized(invalid)


def test_unified_collection_persists_and_filters(tmp_path):
    store = VectorStore(str(tmp_path / "chroma"))
    rows = []
    for i, modality in enumerate(("text", "image", "audio", "video")):
        vector = np.zeros(768); vector[i] = 1
        rows.append({"id": modality, "embedding": vector.tolist(), "text": "helmet safety" if i == 0 else "",
                     "metadata": {"file_id": modality, "file_name": modality, "modality": modality}})
    store.add(rows)
    second = VectorStore(str(tmp_path / "chroma"))
    assert len(second.files()) == 4
    found = second.search(rows[0]["embedding"])
    assert found[0]["id"] == "text"
    assert found[0]["similarity"] == pytest.approx(1)
    assert second.search(rows[0]["embedding"], modality="audio")[0]["id"] == "audio"
    assert second.keyword("helmet")[0]["id"] == "text"


def test_text_ingestion_idempotence(tmp_path):
    class TestEngine:
        def embed_text(self, value):
            return normalized(np.ones(768))
    path = tmp_path / "policy.txt"
    path.write_text("Safety equipment must be inspected daily. " * 100)
    store = VectorStore(str(tmp_path / "chroma"))
    first = ingest(path, path.name, TestEngine(), store)
    ingest(path, path.name, TestEngine(), store)
    assert store.collection.count() == first["vectors"]
    result = store.search(normalized(np.ones(768)))[0]
    assert result["page"] == 0
    assert result["file_name"] == "policy.txt"


def test_api_validation_and_empty_collection(tmp_path):
    from app.main import create_app
    class TestEngine:
        state = "not_loaded"
        def status(self):
            return {"state": self.state}
        def embed_query(self, text):
            return normalized(np.ones(768))
    settings = Settings(chroma_path=str(tmp_path / "chroma"), upload_dir=str(tmp_path / "uploads"),
                        hf_home=str(tmp_path / "hf"), max_upload_mb=1)
    client = TestClient(create_app(settings, TestEngine()))
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/files").json() == []
    assert client.post("/api/search/text", json={"text": "safety"}).json()["results"] == []
    assert client.post("/api/search/text", json={"text": ""}).status_code == 422
    assert client.post("/api/upload", files={"file": ("x.exe", b"invalid")}).status_code == 415
    assert client.post("/api/upload", files={"file": ("x.txt", b"a" * (1024 * 1024 + 1))}).status_code == 413
    assert client.post("/api/upload", files={"file": ("x.txt", b"")}).status_code == 422
    assert client.post("/api/chat", json={"question": "Safety?"}).status_code == 422
