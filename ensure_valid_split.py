from pathlib import Path
import random, shutil

root = Path("Barcos_Canoas_Tefe-1")
train_img = root/"train"/"images"
train_lbl = root/"train"/"labels"
valid_img = root/"valid"/"images"
valid_lbl = root/"valid"/"labels"

# cria pastas se não existirem
valid_img.mkdir(parents=True, exist_ok=True)
valid_lbl.mkdir(parents=True, exist_ok=True)

# se valid já tiver arquivos, não faz nada
valid_count = len(list(valid_img.glob("*.jpg"))) + len(list(valid_img.glob("*.png")))
if valid_count > 0:
    print(f"valid já tem {valid_count} imagens. Nada a fazer.")
else:
    imgs = sorted(train_img.glob("*.jpg")) + sorted(train_img.glob("*.png"))
    assert imgs, "Não há imagens em train/images."
    random.seed(42)
    take = max(1, int(0.1 * len(imgs)))  # 10%
    sample = random.sample(imgs, take)

    moved = 0
    for img in sample:
        lbl = train_lbl / (img.stem + ".txt")
        if lbl.exists():
            shutil.move(str(img), str(valid_img / img.name))
            shutil.move(str(lbl), str(valid_lbl / lbl.name))
            moved += 1

    print(f"Movidos {moved} pares para valid/.")
