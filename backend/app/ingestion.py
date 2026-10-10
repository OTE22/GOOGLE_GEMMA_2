import hashlib
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import pymupdf
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter

EXTENSIONS = {".pdf": "text", ".txt": "text", ".jpg": "image", ".jpeg": "image", ".png": "image",
              ".mp3": "audio", ".wav": "audio", ".mp4": "video"}


def ffmpeg(*args):
    if not shutil.which("ffmpeg"):
        raise ValueError("FFmpeg is required for audio/video processing")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *map(str, args)], check=True,
                   capture_output=True, timeout=300)


def duration(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                            check=True, capture_output=True, text=True, timeout=30)
    return float(result.stdout.strip())


def parts(path, scratch):
    kind = EXTENSIONS[path.suffix.lower()]
    if kind == "text":
        splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=200)
        if path.suffix.lower() == ".pdf":
            with pymupdf.open(path) as pdf:
                pages = [(i + 1, page.get_text()) for i, page in enumerate(pdf)]
        else:
            pages = [(0, path.read_text(encoding="utf-8"))]
        for page, content in pages:
            for chunk in splitter.split_text(content):
                yield chunk, {"page": page}
    elif kind == "image":
        with Image.open(path) as image:
            image.verify()
        yield path, {}
    else:
        length = duration(path)
        # 30 frames at 1 FPS use about 4200 tokens; 60 s audio uses about 1500.
        segment = 30 if kind == "video" else 60
        for index, start in enumerate(range(0, max(1, int(length + .999)), segment)):
            end = min(start + segment, length)
            output = scratch / f"segment-{index}{'.mp4' if kind == 'video' else '.wav'}"
            args = ["-ss", start, "-i", path, "-t", end - start]
            if kind == "audio":
                args += ["-vn", "-ac", 1, "-ar", 16000, "-c:a", "pcm_s16le"]
            else:
                args += ["-an", "-vf", "fps=1,scale=640:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
            ffmpeg(*args, output)
            yield output, {"start_time": float(start), "end_time": float(end)}


def ingest(path, name, engine, store, progress=lambda stage: None, evidence=None):
    started = time.perf_counter()
    file_id = hashlib.sha256(path.read_bytes()).hexdigest()
    modality = EXTENSIONS[path.suffix.lower()]
    progress("Process")
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for i, (value, position) in enumerate(parts(path, Path(directory))):
            progress("Embed")
            vector = getattr(engine, f"embed_{modality}")(value)
            metadata = {"file_id": file_id, "file_name": name, "modality": modality,
                        "source_path": path.name, "url": f"/media/{path.name}", **position}
            if evidence:
                metadata["evidence_origin"] = "demo generation script"
            text = str(value) if modality == "text" else (evidence or "")
            rows.append({"id": f"{file_id}:{i}", "embedding": vector, "text": text, "metadata": metadata})
    if not rows:
        raise ValueError("No extractable text. Scanned PDFs require OCR.")
    progress("Index")
    store.add(rows)
    elapsed = time.perf_counter() - started
    for row in rows:
        row["metadata"]["index_seconds"] = elapsed
    store.collection.update(ids=[row["id"] for row in rows], metadatas=[row["metadata"] for row in rows])
    progress("Searchable")
    return {"file_id": file_id, "file_name": name, "vectors": len(rows), "seconds": elapsed, "modality": modality}
