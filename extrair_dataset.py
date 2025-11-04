# extrair_dataset.py
import os
import shutil
from pathlib import Path
import datetime

# Pastas
BASE = Path("data_events")
DATASET = Path("dataset/images/raw")

# Cria pasta de destino
DATASET.mkdir(parents=True, exist_ok=True)

# Contagem
copiados = 0

# Percorre todos os dias dentro de data_events
for day_dir in BASE.iterdir():
    if not day_dir.is_dir():
        continue
    
    images_dir = day_dir / "images"
    if images_dir.exists():
        for img in images_dir.glob("snapshot_*_HQ.jpg"):
            destino = DATASET / img.name
            shutil.copy2(img, destino)
            copiados += 1

print(f"[OK] Copiadas {copiados} imagens para {DATASET}")
print("Agora você pode abrir essa pasta no LabelImg para anotar.")
