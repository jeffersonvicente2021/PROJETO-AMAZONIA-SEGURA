import os
import cv2
import time
import json
import math
import sqlite3
from collections import deque
from datetime import datetime, time as dtime
from dataclasses import dataclass, field
from urllib.parse import quote
import numpy as np
import subprocess, shutil  # opcional (recompressão com ffmpeg)

# =================== CONFIG ===================
@dataclass
class Config:
    IP: str = "IP"
    USER: str = "login"
    PASSWORD: str = "Senha"  # ideal: ler de variável de ambiente
    PORT: int = 554
    CHANNEL_DET: str = "201"  # substream pra detecção
    CHANNEL_HQ: str = "101"   # mainstream pra snapshot/qualidade
    SAVE_ROOT: str = "./data_events"
    MIN_CONF: float = 0.25
    CLASSES_INTERESSE: tuple = ()
    FPS_TARGET: int = 15
    PRE_EVENT_SECONDS: int = 3
    POST_EVENT_SECONDS: int = 4
    EVENT_MIN_DURATION: float = 1.5
    MAX_EVENT_DURATION: float = 9.0
    SNAPSHOT_HQ: bool = True
    DRAW: bool = True
    SHOW_WINDOW: bool = True
    MODEL_NAME: str = r"..\runs\detect\train3\weights\best.pt"
    ROI_POLY: list = None  # visão inteira (None)

    # >>> NOVOS CAMPOS para reduzir o tamanho dos arquivos <<<
    OUT_SIZE: tuple = (640, 360)   # resolução de saída do vídeo de evento
    OUT_FPS: int = 8               # FPS do arquivo salvo
    FRAME_STRIDE: int = 2          # pula N-1 frames de entrada (2 = processa 1 e pula 1)
    JPEG_QUALITY: int = 60         # qualidade do snapshot HQ (menor = arquivo menor)

CFG = Config()

def build_rtsp(ip, user, pwd, port, channel):
    user_q = quote(user, safe='')
    pwd_q  = quote(pwd,  safe='')
    return f"rtsp://{user_q}:{pwd_q}@{ip}:{port}/Streaming/Channels/{channel}"

RTSP_DET = build_rtsp(CFG.IP, CFG.USER, CFG.PASSWORD, CFG.PORT, CFG.CHANNEL_DET)
RTSP_HQ  = build_rtsp(CFG.IP, CFG.USER, CFG.PASSWORD, CFG.PORT, CFG.CHANNEL_HQ)
print("[DIAG] RTSP_DET =", RTSP_DET)
print("[DIAG] RTSP_HQ  =", RTSP_HQ)
print("[DIAG] MIN_CONF =", CFG.MIN_CONF)
print("[DIAG] CLASSES_INTERESSE =", CFG.CLASSES_INTERESSE)

def ensure_dirs(base):
    today = datetime.now().strftime("%Y-%m-%d")
    paths = {
        "base":   os.path.join(base, today),
        "videos": os.path.join(base, today, "videos"),
        "images": os.path.join(base, today, "images"),
        "meta":   os.path.join(base, today, "meta")
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

PATHS = ensure_dirs(CFG.SAVE_ROOT)
DB_PATH = os.path.join(CFG.SAVE_ROOT, "events.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_start TEXT,
            ts_end   TEXT,
            duration REAL,
            classes  TEXT,
            conf_max REAL,
            video_path    TEXT,
            snapshot_path TEXT,
            meta_path     TEXT
        )
    """)
    con.commit()
    con.close()

init_db()

# =================== YOLO ===================
try:
    from ultralytics import YOLO
    _yolo_model = YOLO(CFG.MODEL_NAME)
    YOLO_READY = True
except Exception as e:
    print("[WARN] Falha ao carregar YOLO:", e)
    _yolo_model = None
    YOLO_READY = False

def now_ts():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def mask_roi(frame):
    if CFG.ROI_POLY is None:
        return frame
    h, w = frame.shape[:2]
    pts = np.array([[int(x*w), int(y*h)] for x, y in CFG.ROI_POLY], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)

# =================== RECORDER (atualizado para gravar leve) ===================
class EventRecorder:
    def __init__(self, fps, pre_s=5, post_s=5):
        self.fps_in = max(1, int(fps))
        self.fps_out = max(1, int(CFG.OUT_FPS))
        self.pre = deque(maxlen=self.fps_out * max(1, int(pre_s)))
        self.recording = False
        self.writer = None
        self.video_path = None
        self.ts_start = None
        self.classes_seen = set()
        self.conf_max = 0.0
        self.last_detection_time = 0.0
        self.event_start_time_monotonic = None
        self.post_s = max(1, int(post_s))

    def _resize(self, frame):
        return cv2.resize(frame, CFG.OUT_SIZE, interpolation=cv2.INTER_AREA)

    def _new_writer(self, frame):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # simples e compatível
        fname = f"event_{now_ts()}.mp4"
        self.video_path = os.path.join(PATHS["videos"], fname)
        w, h = CFG.OUT_SIZE
        self.writer = cv2.VideoWriter(self.video_path, fourcc, self.fps_out, (w, h))
        # grava o buffer pré-evento já redimensionado
        for pf in list(self.pre):
            self.writer.write(pf)
        self.ts_start = datetime.now().isoformat(timespec="seconds")
        self.event_start_time_monotonic = time.monotonic()

    def start_or_extend(self, frame, classes, conf):
        self.last_detection_time = time.monotonic()
        self.classes_seen.update(classes)
        self.conf_max = max(self.conf_max, conf)
        if not self.recording:
            self._new_writer(frame)
            self.recording = True

    def step(self, frame):
        if not self.recording:
            self.pre.append(self._resize(frame.copy()))
            return None
        self.writer.write(self._resize(frame))
        elapsed = time.monotonic() - self.last_detection_time
        total = time.monotonic() - (self.event_start_time_monotonic or time.monotonic())
        if elapsed >= self.post_s or total >= CFG.MAX_EVENT_DURATION:
            self.close()
            return "closed"
        return None

    def close(self):
        if self.writer is not None:
            self.writer.release()
        self.recording = False

# (opcional) recompressão com ffmpeg para bitrate fixo (arquivos ainda menores)
def recompress_with_ffmpeg(in_path, out_br_kbps=800):
    out_path = in_path.replace(".mp4", f"_c{out_br_kbps}k.mp4")
    if shutil.which("ffmpeg") is None:
        return in_path  # sem ffmpeg instalado
    cmd = [
        "ffmpeg","-y","-i", in_path,
        "-c:v","libx264","-b:v",f"{out_br_kbps}k","-preset","veryfast",
        "-movflags","faststart","-c:a","aac","-b:a","96k",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def grab_hq_snapshot(rtsp_hq):
    cap = cv2.VideoCapture(rtsp_hq, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[WARN] Não abriu HQ stream para snapshot")
        return None
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        path = os.path.join(PATHS["images"], f"snapshot_{now_ts()}_HQ.jpg")
        # >>> compacta o JPEG do snapshot
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, CFG.JPEG_QUALITY])
        return path
    return None

def save_meta(meta):
    path = os.path.join(PATHS["meta"], f"meta_{now_ts()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path

def insert_db(row):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO events (ts_start, ts_end, duration, classes, conf_max, video_path, snapshot_path, meta_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (row["ts_start"], row["ts_end"], row["duration"], row["classes"], row["conf_max"],
          row["video_path"], row["snapshot_path"], row["meta_path"]))
    con.commit()
    con.close()

def draw_dets(frame, dets):
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        lbl = f"{d['cls']} {d['conf']:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, lbl, (x1, max(15, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# >>> yolo_detect com mais pixels e NMS agnóstico
def yolo_detect(model, frame):
    results = model.predict(
        source=frame,
        imgsz=960,           # mais pixels p/ objetos pequenos; se puder use 1280
        conf=CFG.MIN_CONF,
        iou=0.55,
        agnostic_nms=True,
        verbose=False
    )
    dets = []
    if not results:
        return dets
    r0 = results[0]
    names = r0.names
    if r0.boxes is None:
        return dets
    for b in r0.boxes:
        cls_id = int(b.cls[0].item())
        cls_name = names.get(cls_id, str(cls_id))
        conf = float(b.conf[0].item())
        xyxy = b.xyxy[0].tolist()
        if not CFG.CLASSES_INTERESSE or cls_name in CFG.CLASSES_INTERESSE:
            dets.append({"cls": cls_name, "conf": conf, "xyxy": xyxy})
    return dets

# =================== LOOP PRINCIPAL ===================
def main():
    if not YOLO_READY:
        print("[ERRO] YOLO não está pronto.")
        return

    cap = cv2.VideoCapture(RTSP_DET, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[ERRO] Não abriu RTSP:", RTSP_DET)
        return

    fps_read = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps_read) if fps_read and fps_read > 0 else CFG.FPS_TARGET
    recorder = EventRecorder(fps=fps, pre_s=CFG.PRE_EVENT_SECONDS, post_s=CFG.POST_EVENT_SECONDS)

    # Exemplo de ROI (opcional): só metade inferior do frame (ajuste ao seu cenário)
    # CFG.ROI_POLY = [(0.0, 0.45), (1.0, 0.45), (1.0, 1.0), (0.0, 1.0)]

    hora_ini = dtime(5, 30)   # 05:30
    hora_fim = dtime(19, 0)   # 19:00

    while True:
        agora = datetime.now().time()
        if not (hora_ini <= agora <= hora_fim):
            print("[INFO] Fora do horário de monitoramento. Dormindo...")
            time.sleep(60)
            continue

        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame inválido. Reconectando...")
            cap.release()
            time.sleep(1.0)
            cap = cv2.VideoCapture(RTSP_DET, cv2.CAP_FFMPEG)
            continue

        # >>> pular frames de entrada para reduzir processamento
        for _ in range(max(0, CFG.FRAME_STRIDE - 1)):
            cap.grab()

        frame = mask_roi(frame)

        # detecção
        dets = yolo_detect(_yolo_model, frame)

        # >>> anti-ruído: confiança mínima + área mínima
        if dets:
            classes = [d["cls"] for d in dets]
            conf_max = max(d["conf"] for d in dets)
            areas = [(d["xyxy"][2]-d["xyxy"][0]) * (d["xyxy"][3]-d["xyxy"][1]) for d in dets]
            if conf_max >= max(0.28, CFG.MIN_CONF) and max(areas) >= 14*14:
                recorder.start_or_extend(frame, classes, conf_max)

        status = recorder.step(frame)

        vis = frame.copy()
        if CFG.DRAW and dets:
            vis = draw_dets(vis, dets)
        cv2.putText(
            vis,
            f"Detecções: {len(dets)}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        if CFG.SHOW_WINDOW:
            cv2.imshow("Monitor Rio - MVP", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if status == "closed":
            ts_end = datetime.now().isoformat(timespec="seconds")
            try:
                dt_start = datetime.fromisoformat(recorder.ts_start)
                dt_end = datetime.fromisoformat(ts_end)
                duration = (dt_end - dt_start).total_seconds()
            except Exception:
                duration = 0.0

            snapshot_path = grab_hq_snapshot(RTSP_HQ) if CFG.SNAPSHOT_HQ else None

            meta = {
                "ts_start": recorder.ts_start,
                "ts_end": ts_end,
                "duration": duration,
                "classes": sorted(list(recorder.classes_seen)),
                "conf_max": recorder.conf_max,
                "video_path": recorder.video_path,
                "snapshot_path": snapshot_path,
                "channel_det": CFG.CHANNEL_DET,
                "channel_hq": CFG.CHANNEL_HQ,
                "ip": CFG.IP
            }
            meta_path = save_meta(meta)
            meta["meta_path"] = meta_path

            insert_db({
                "ts_start": meta["ts_start"],
                "ts_end": meta["ts_end"],
                "duration": meta["duration"],
                "classes": ",".join(meta["classes"]),
                "conf_max": meta["conf_max"],
                "video_path": meta["video_path"],
                "snapshot_path": meta["snapshot_path"] or "",
                "meta_path": meta["meta_path"]
            })

            # (opcional) recomprimir para bitrate alvo
            # new_path = recompress_with_ffmpeg(meta["video_path"], out_br_kbps=700)

            recorder = EventRecorder(fps=fps, pre_s=CFG.PRE_EVENT_SECONDS, post_s=CFG.POST_EVENT_SECONDS)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
