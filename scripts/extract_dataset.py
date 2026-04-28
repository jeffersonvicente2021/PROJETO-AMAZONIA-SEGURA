import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVENTS_DIR = ROOT / "data" / "events"
RAW_IMAGES_DIR = ROOT / "data" / "datasets" / "raw" / "images"

RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

copied = 0
if EVENTS_DIR.exists():
    for day_dir in EVENTS_DIR.iterdir():
        if not day_dir.is_dir():
            continue
        images_dir = day_dir / "images"
        if images_dir.exists():
            for image in images_dir.glob("snapshot_*_HQ.jpg"):
                shutil.copy2(image, RAW_IMAGES_DIR / image.name)
                copied += 1

print(f"[OK] Copiadas {copied} imagens para {RAW_IMAGES_DIR}")
print("Agora voce pode abrir essa pasta no LabelImg para anotar.")
