from pathlib import Path
import numpy as np
import soundfile as sf
import pymupdf
from app.ingestion import parts, duration
from app.embedding import decoded_video, EmbeddingService
from app.config import Settings


def test_audio_preprocessing_normalizes_and_preserves_timestamps(tmp_path):
    source = tmp_path / "long.wav"
    wave = np.sin(np.arange(65 * 22050) * 2 * np.pi * 440 / 22050).astype(np.float32) * .1
    sf.write(source, np.column_stack((wave, wave)), 22050)
    clips = list(parts(source, tmp_path))
    assert len(clips) == 2
    assert clips[0][1] == {"start_time": 0., "end_time": 60.}
    assert clips[1][1] == {"start_time": 60., "end_time": 65.}
    for path, position in clips:
        info = sf.info(path)
        assert info.samplerate == 16000
        assert info.channels == 1
        assert info.duration == position["end_time"] - position["start_time"]


def test_generated_demo_contains_actual_readable_and_playable_content(tmp_path):
    root = Path(__file__).resolve().parents[1] / "demo"
    with pymupdf.open(root / "workplace-safety.pdf") as document:
        assert "hard hat" in document[0].get_text()
    for name in ("safety-briefing.wav", "vehicle-training.wav"):
        audio, rate = sf.read(root / name)
        assert rate == 16000 and audio.ndim == 1
        assert len(audio) > 16000
        assert np.max(np.abs(audio)) > .01
    assert duration(root / "safety-training.mp4") == 12
    clips = list(parts(root / "safety-training.mp4", tmp_path))
    assert len(clips) == 1
    assert clips[0][1] == {"start_time": 0., "end_time": 12.}
    video = decoded_video(root / "safety-training.mp4")
    assert video["array"].shape == (12, 600, 960, 3)
    assert video["video_metadata"]["fps"] == 24
    assert video["video_metadata"]["frames_indices"] == list(range(0, 288, 24))


def test_service_passes_native_media_arrays_without_text_prompts():
    class SpyModel:
        def encode(self, value, **kwargs):
            self.value, self.kwargs = value, kwargs
            return np.ones(768)
    engine = EmbeddingService(Settings())
    engine.model = SpyModel()
    root = Path(__file__).resolve().parents[1] / "demo"
    engine.embed_audio(root / "safety-briefing.wav")
    assert engine.model.value["audio"]["sampling_rate"] == 16000
    assert "prompt_name" not in engine.model.kwargs
    engine.embed_video(root / "safety-training.mp4")
    assert engine.model.value["video"]["array"].shape[0] == 12
    assert engine.model.kwargs["processing_kwargs"] == {"video": {"do_sample_frames": False}}
    assert "prompt_name" not in engine.model.kwargs
