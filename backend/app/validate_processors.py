"""Check the official native processors without claiming encoder inference."""
import json
import time
from pathlib import Path
import soundfile as sf
from PIL import Image
from transformers import AutoProcessor
from .config import Settings
from .embedding import decoded_video
from transformers.video_utils import VideoMetadata


def validate_processors():
    settings = Settings()
    processor = AutoProcessor.from_pretrained(settings.embedding_model, cache_dir=settings.hf_home, local_files_only=True)
    demo = Path(__file__).resolve().parents[1] / "demo"
    audio, rate = sf.read(demo / "safety-briefing.wav", dtype="float32")
    video = decoded_video(demo / "safety-training.mp4")
    requests = {
        "text": {"text": "Workers wearing safety equipment"},
        "image": {"images": Image.open(demo / "protective-equipment.png")},
        "audio": {"audio": audio, "sampling_rate": rate},
        "video": {"videos": video["array"], "video_metadata": VideoMetadata(**video["video_metadata"]),
                  "do_sample_frames": False},
    }
    report = {"processor": type(processor).__name__, "modalities": {}}
    for modality, request in requests.items():
        started = time.perf_counter()
        processed = processor(**request, return_tensors="pt")
        shapes = {name: list(value.shape) for name, value in processed.items() if hasattr(value, "shape")}
        tokens = processed["input_ids"].shape[-1]
        assert 0 < tokens <= 8192
        report["modalities"][modality] = {"tokens": tokens, "shapes": shapes, "seconds": time.perf_counter() - started}
        print(f"{modality}: {tokens} tokens; {shapes}", flush=True)
    Path("processor-validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    validate_processors()
