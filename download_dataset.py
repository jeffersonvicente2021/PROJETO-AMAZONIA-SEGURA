import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()  # carrega .env

API_KEY   = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
PROJECT   = os.getenv("ROBOFLOW_PROJECT")
VERSION   = int(os.getenv("ROBOFLOW_VERSION", "1"))

assert API_KEY,   "ROBOFLOW_API_KEY ausente no .env"
assert WORKSPACE, "ROBOFLOW_WORKSPACE ausente no .env"
assert PROJECT,   "ROBOFLOW_PROJECT ausente no .env"

print(f"Usando workspace='{WORKSPACE}', project='{PROJECT}', version={VERSION}")

print("loading Roboflow workspace...")
rf = Roboflow(api_key=API_KEY)
ws = rf.workspace(WORKSPACE)

print("loading Roboflow project...")
prj = ws.project(PROJECT)

print("downloading dataset...")
ds = prj.version(VERSION).download("yolov8")  # mude o formato se quiser
print("Dataset baixado em:", ds.location)
