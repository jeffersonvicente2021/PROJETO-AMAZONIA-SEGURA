import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "data" / "datasets"

load_dotenv(ROOT / ".env")

api_key = os.getenv("ROBOFLOW_API_KEY")
workspace = os.getenv("ROBOFLOW_WORKSPACE")
project = os.getenv("ROBOFLOW_PROJECT")
version = int(os.getenv("ROBOFLOW_VERSION", "1"))

assert api_key, "ROBOFLOW_API_KEY ausente no .env"
assert workspace, "ROBOFLOW_WORKSPACE ausente no .env"
assert project, "ROBOFLOW_PROJECT ausente no .env"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(DATASETS_DIR)

print(f"Usando workspace='{workspace}', project='{project}', version={version}")
rf = Roboflow(api_key=api_key)
dataset = rf.workspace(workspace).project(project).version(version).download("yolov8")
print("Dataset baixado em:", dataset.location)
