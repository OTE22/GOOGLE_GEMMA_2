from pathlib import Path
import numpy as np
import soundfile as sf
import pymupdf
from app.ingestion import parts, duration


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
