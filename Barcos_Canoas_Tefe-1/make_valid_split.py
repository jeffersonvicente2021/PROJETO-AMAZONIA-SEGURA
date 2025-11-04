import random, shutil
from pathlib import Path

root = Path("Barcos_Canoas_Tefe-1")
train_img = root / "train" / "images"
train_lbl = root / "train" / "labels"
valid_img = root / "valid" / "images"
valid_lbl = root / "valid" / "labels"

valid_img.mkdir(parents=True, exist_ok=True)
valid_lbl.mkdir(parents=True, exist_ok=True)

imgs = sorted(list(train_img.glob("*.jpg")) + list(train_img.glob("*.png")))
random.seed(42)
take = max(1, int(0.1 * len(imgs)))  # 10% das imagens
sample = random.sample(imgs, take)

moved = 0
for img in sample:
    lbl = train_lbl / (img.stem + ".txt")
    if lbl.exists():
        shutil.move(str(img), str(valid_img / img.name))
        shutil.move(str(lbl), str(valid_lbl / lbl.name))
        moved += 1

print(f"Criados {moved} pares imagem/label em 'valid/'.")
