from pathlib import Path
import random
import shutil


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "datasets" / "barcos_canoas_tefe"

train_img = DATASET_ROOT / "train" / "images"
train_lbl = DATASET_ROOT / "train" / "labels"
valid_img = DATASET_ROOT / "valid" / "images"
valid_lbl = DATASET_ROOT / "valid" / "labels"

valid_img.mkdir(parents=True, exist_ok=True)
valid_lbl.mkdir(parents=True, exist_ok=True)

valid_count = len(list(valid_img.glob("*.jpg"))) + len(list(valid_img.glob("*.png")))
if valid_count > 0:
    print(f"valid ja tem {valid_count} imagens. Nada a fazer.")
else:
    imgs = sorted(train_img.glob("*.jpg")) + sorted(train_img.glob("*.png"))
    assert imgs, f"Nao ha imagens em {train_img}."
    random.seed(42)
    sample = random.sample(imgs, max(1, int(0.1 * len(imgs))))

    moved = 0
    for img in sample:
        lbl = train_lbl / f"{img.stem}.txt"
        if lbl.exists():
            shutil.move(str(img), str(valid_img / img.name))
            shutil.move(str(lbl), str(valid_lbl / lbl.name))
            moved += 1

    print(f"Movidos {moved} pares para valid/.")
