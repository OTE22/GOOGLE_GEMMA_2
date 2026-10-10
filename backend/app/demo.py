"""Generate original illustrated workplace fixtures and spoken recordings locally."""
import json
import platform
import subprocess
from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw, ImageFont
from .ingestion import ffmpeg

ROOT = Path(__file__).resolve().parents[1] / "demo"
SCRIPTS = {
    "safety-briefing.wav": "Welcome to the workplace safety briefing. Wear your hard hat and high visibility vest in the loading area. Check protective equipment before every shift. Report hazards to your supervisor. Employee safety training is held every Monday.",
    "vehicle-training.wav": "Company vehicle training begins with a daily inspection. Check the tires, mirrors, lights, and brakes. Wear your seat belt. Only trained employees may drive company vehicles. New drivers must complete supervised training before their first journey.",
}


def generate():
    ROOT.mkdir(exist_ok=True)
    manifest = []
    policies = {
        "workplace-safety.pdf": ("Workplace safety policy", "All employees must wear a hard hat and high visibility vest in the loading area. Inspect protective equipment before every shift. Report hazards to the supervisor. Safety training takes place every Monday. New staff attend induction before entering the warehouse."),
        "company-vehicles.pdf": ("Company vehicle and training policy", "Only trained, authorized employees may operate company vehicles. Inspect tires, mirrors, lights and brakes daily. Seat belts are mandatory. New drivers complete supervised training. Annual refresher training covers safe loading, pedestrian awareness and emergency procedures."),
    }
    for name, (title, text) in policies.items():
        doc = pymupdf.open()
        page = doc.new_page()
        page.draw_rect(pymupdf.Rect(0, 0, 595, 110), fill=(.16, .13, .29))
        page.insert_text((45, 55), "NORTHSTAR / PEOPLE & OPERATIONS", fontsize=12, color=(1, 1, 1))
        page.insert_text((45, 155), title, fontsize=24)
        page.insert_textbox(pymupdf.Rect(45, 195, 545, 690), text, fontsize=15, lineheight=1.8)
        page.insert_text((45, 760), "Illustrative company policy · Generated demonstration material", fontsize=10)
        doc.save(ROOT / name); doc.close()
        manifest.append({"file": name})
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default(size=28)
    for i, name in enumerate(("protective-equipment.png", "company-vehicle.png", "employee-training.png")):
        image = Image.new("RGB", (960, 600), "#eef0f5")
        d = ImageDraw.Draw(image)
        d.rectangle((0, 450, 960, 600), fill="#d2d9df")
        d.text((40, 30), ["Safety at every shift", "Company vehicle inspection", "Employee safety training"][i], fill="#332856", font=font)
        if i == 0:
            d.rectangle((100, 200, 300, 450), fill="#bdc8d0")
            d.rectangle((125, 235, 275, 280), fill="#8b9aa6")
            d.ellipse((460, 150, 580, 270), fill="#d7a979")
            d.pieslice((450, 120, 590, 220), 180, 360, fill="#f8c53c")
            d.rectangle((445, 180, 595, 195), fill="#f8c53c")
            d.polygon([(460, 270), (580, 270), (630, 440), (410, 440)], fill="#e9cc39")
            d.rectangle((445, 330, 590, 348), fill="#f8f9d2")
            d.line((460, 440, 450, 555), fill="#39495e", width=32)
            d.line((580, 440, 590, 555), fill="#39495e", width=32)
            caption = "Original illustration: warehouse worker wearing a yellow hard hat and high visibility vest beside storage shelves."
        elif i == 1:
            d.rounded_rectangle((120, 230, 670, 450), 22, fill="#667cad")
            d.polygon([(670, 270), (800, 300), (850, 450), (670, 450)], fill="#667cad")
            d.rectangle((700, 310, 790, 370), fill="#cfe4ef")
            for x in (250, 710):
                d.ellipse((x - 48, 410, x + 48, 506), fill="#303846")
                d.ellipse((x - 22, 436, x + 22, 480), fill="#adb6c2")
            d.text((190, 300), "NORTHSTAR", fill="white", font=font)
            caption = "Original illustration: a blue Northstar company delivery truck parked for daily inspection."
        else:
            d.rectangle((220, 120, 760, 330), fill="#fff")
            d.rectangle((240, 140, 740, 310), outline="#9984ca", width=4)
            d.text((290, 195), "SAFETY TRAINING", fill="#7157d8", font=font)
            for x in (260, 460, 660):
                d.ellipse((x, 330, x + 70, 400), fill="#cfa078")
                d.rectangle((x - 15, 400, x + 85, 510), fill="#556880")
            caption = "Original illustration: three employees seated in a training room facing a workplace safety training board."
        image.save(ROOT / name)
        manifest.append({"file": name, "evidence": caption})
    for name, script in SCRIPTS.items():
        raw = ROOT / ("raw-" + name)
        if platform.system() == "Windows":
            # Fixed internal strings; never interpolate user-provided content into PowerShell.
            escaped_path = str(raw.resolve()).replace("'", "''")
            escaped_text = script.replace("'", "''")
            command = f"Add-Type -AssemblyName System.Speech; $speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speaker.SetOutputToWaveFile('{escaped_path}'); $speaker.Speak('{escaped_text}'); $speaker.Dispose()"
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command], check=True, timeout=60)
        else:
            subprocess.run(["espeak", "-w", str(raw), script], check=True, timeout=60)
        ffmpeg("-i", raw, "-ac", 1, "-ar", 16000, ROOT / name)
        raw.unlink()
        manifest.append({"file": name, "evidence": "Verified synthesis script: " + script})
    # A real short video assembled from the generated illustrations, with spoken safety audio.
    concat = ROOT / "frames.txt"
    concat.write_text("".join(f"file '{name}'\nduration 4\n" for name in ("protective-equipment.png", "company-vehicle.png", "employee-training.png")) + "file 'employee-training.png'\n")
    ffmpeg("-f", "concat", "-safe", 0, "-i", concat, "-i", ROOT / "safety-briefing.wav", "-t", 12,
           "-vf", "fps=24", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", ROOT / "safety-training.mp4")
    concat.unlink()
    manifest.append({"file": "safety-training.mp4", "evidence": "Generated slideshow: 0–4 seconds warehouse worker in protective equipment; 4–8 seconds company delivery truck; 8–12 seconds employees in safety training. Audio uses the safety briefing synthesis script."})
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(manifest)} genuine demonstration files in {ROOT}")


if __name__ == "__main__":
    generate()
