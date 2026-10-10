import importlib.metadata
import threading
import time
from pathlib import Path
import numpy as np


def normalized(vector):
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.shape != (768,) or not np.isfinite(value).all():
        raise ValueError("Model must produce a finite 768-dimensional vector")
    norm = np.linalg.norm(value)
    if norm == 0:
        raise ValueError("Model returned a zero vector")
    return (value / norm).tolist()


class EmbeddingService:
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.lock = threading.RLock()
        self.state = "not_loaded"
        self.error = None
        self.device = None
        self.load_seconds = None
        self.parameter_count = None

    def load(self):
        with self.lock:
            if self.model is not None:
                return
            self.state = "loading"
            started = time.perf_counter()
            try:
                import torch
                from sentence_transformers import SentenceTransformer
                self.device = ("cuda" if torch.cuda.is_available() else "cpu") if self.settings.embedding_device == "auto" else self.settings.embedding_device
                dtype = torch.bfloat16 if self.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
                model = SentenceTransformer(self.settings.embedding_model, device=self.device,
                    cache_folder=self.settings.hf_home, model_kwargs={"torch_dtype": dtype})
                if model.get_sentence_embedding_dimension() != 768:
                    raise RuntimeError("Incompatible embedding dimension")
                config = model[0].auto_model.config
                if getattr(config, "vision_config", None) is None or getattr(config, "audio_config", None) is None:
                    raise RuntimeError("Full vision and audio encoders are required")
                self.parameter_count = sum(p.numel() for p in model.parameters())
                self.model = model
                self.state, self.error = "ready", None
                self.load_seconds = time.perf_counter() - started
            except Exception as exc:
                self.state, self.error = "error", str(exc)
                raise RuntimeError(f"EmbeddingGemma 2 could not load: {exc}") from exc

    def encode(self, value, prompt=None):
        with self.lock:
            self.load()
            kwargs = {"normalize_embeddings": True, "show_progress_bar": False}
            if prompt:
                kwargs["prompt_name"] = prompt
            return normalized(self.model.encode(value, **kwargs))

    def embed_text(self, text):
        return self.encode(text, "Document")

    def embed_query(self, query):
        return self.encode(query, "SearchQuery")

    def embed_image(self, image_path):
        return self.encode({"image": str(Path(image_path).resolve())})

    def embed_audio(self, audio_path):
        return self.encode({"audio": str(Path(audio_path).resolve())})

    def embed_video(self, video_path):
        return self.encode({"video": str(Path(video_path).resolve())})

    def status(self):
        versions = {}
        for package in ("sentence-transformers", "transformers", "torch"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        return {"model": self.settings.embedding_model, "state": self.state, "error": self.error,
                "device": self.device, "dimensions": 768, "parameter_count": self.parameter_count,
                "modalities": ["text", "image", "audio", "video"], "versions": versions,
                "load_seconds": self.load_seconds,
                "warning": "CPU inference can be slow, especially for audio and video." if self.device == "cpu" else None}
