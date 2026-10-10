from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")
    embedding_model: str = "google/embeddinggemma-2"
    embedding_dimension: int = 768
    embedding_device: str = "auto"
    chroma_path: str = "./data/chroma"
    upload_dir: str = "./data/uploads"
    hf_home: str = "./data/huggingface"
    max_upload_mb: int = 100
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openrouter/free"

    def prepare(self):
        if self.embedding_model != "google/embeddinggemma-2" or self.embedding_dimension != 768:
            raise ValueError("This demo requires google/embeddinggemma-2 at 768 dimensions")
        for directory in (self.chroma_path, self.upload_dir, self.hf_home):
            Path(directory).mkdir(parents=True, exist_ok=True)
