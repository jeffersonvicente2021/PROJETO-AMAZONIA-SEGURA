
import os
import cv2
import time
import json
import math
import sqlite3
from collections import deque, Counter
from datetime import datetime, time as dtime
from dataclasses import dataclass, field
from urllib.parse import quote
import numpy as np

# =================== CONFIG ===================
@dataclass
class Config:
    IP: str = "192.168.100.251"
    USER: str = "admin"
    PASSWORD: str = "S3lv409052021@@"
    PORT: int = 554
    CHANNEL_DET: str = "201"  # substream pra detecção (se falhar, teste '102')
    CHANNEL_HQ: str = "101"   # mainstream pra snapshot/qualidade
    SAVE_ROOT: str = "./data_events"
    MIN_CONF: float = 0.25
    CLASSES_INTERESSE: tuple = ()  # vazio = aceita todas as classes
    FPS_TARGET: int = 15
    PRE_EVENT_SECONDS: int = 5
    POST_EVENT_SECONDS: int = 7
    EVENT_MIN_DURATION: float = 1.5
    MAX_EVENT_DURATION: float = 60.0
    SNAPSHOT_HQ: bool = True
    DRAW: bool = True
    SHOW_WINDOW: bool = True
    MODEL_NAME: str = r"..\runs\detect\train3\weights\best.pt"
    ROI_POLY: list = None  # visão inteira (None)

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
    cur.execute("""        CREATE TABLE IF NOT EXISTS events (            id INTEGER PRIMARY KEY AUTOINCREMENT,            ts_start TEXT,            ts_end   TEXT,            duration REAL,            classes  TEXT,            conf_max REAL,            video_path    TEXT,            snapshot_path TEXT,            meta_path     TEXT        )    """ )
    con.commit()
    con.close()

init_db()

# =================== YOLO ===================
try:
    from ultralytics import YOLO
    _yolo_model = YOLO(CFG.MODEL_NAME)
    YOLO_READY = True
    try:
        _names = _yolo_model.names if hasattr(_yolo_model, "names") else (_yolo_model.model.names if hasattr(_yolo_model, "model") else {})
        print("[DIAG] YOLO carregado. Classes no modelo:", _names)
    except Exception as e:
        print("[DIAG] Falha ao ler nomes do modelo:", e)
except Exception as e:
    print("[WARN] Falha ao carregar YOLO:", e)
    _yolo_model = None
    YOLO_READY = False
    print("[DIAG] YOLO_READY = False (modelo não carregado)")

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

class EventRecorder:
    def __init__(self, fps, pre_s=5, post_s=5):
        self.fps = max(1, int(fps))
        self.pre = deque(maxlen=self.fps * max(1, int(pre_s)))
        self.recording = False
        self.writer = None
        self.video_path = None
        self.ts_start = None
        self.classes_seen = set()
        self.conf_max = 0.0
        self.last_detection_time = 0.0
        self.event_start_time_monotonic = None
        self.post_s = max(1, int(post_s))

    def _new_writer(self, frame):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fname = f"event_{now_ts()}.mp4"
        self.video_path = os.path.join(PATHS["videos"], fname)
        h, w = frame.shape[:2]
        self.writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (w, h))
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
            self.pre.append(frame.copy())
            return None
        self.writer.write(frame)
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
        cv2.imwrite(path, frame)
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
    cur.execute("""        INSERT INTO events (ts_start, ts_end, duration, classes, conf_max, video_path, snapshot_path, meta_path)        VALUES (?, ?, ?, ?, ?, ?, ?, ?)    """, (row["ts_start"], row["ts_end"], row["duration"], row["classes"], row["conf_max"],          row["video_path"], row["snapshot_path"], row["meta_path"]))
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

def draw_hud(img, dets, recorder, fps_ema=None):
    counts = Counter([d["cls"] for d in dets]) if dets else {}
    total = sum(counts.values())
    lines = []

    status = "REC" if recorder.recording else "IDLE"
    if recorder.recording and recorder.event_start_time_monotonic:
        dur = time.monotonic() - recorder.event_start_time_monotonic
        lines.append(f"Status: {status} ({dur:04.1f}s)")
    else:
        lines.append(f"Status: {status}")

    if fps_ema is not None:
        lines[-1] += f"  |  FPS: {fps_ema:04.1f}"

    if total > 0:
        parts = [f"{k}:{v}" for k, v in counts.items()]
        lines.append("Detecções: " + ", ".join(parts) + f"  (total={total})")
        lines.append(f"conf_max(frame): {max(d['conf'] for d in dets):.2f}")
    else:
        lines.append("Detecções: 0")

    y = 22
    for t in lines:
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        y += 22
    return img

def yolo_detect(model, frame):
    results = model.predict(source=frame, imgsz=640, conf=CFG.MIN_CONF, verbose=False)
    dets = []
    if not results:
        return dets
    r0 = results[0]
    names = r0.names
    if r0.boxes is None:
        return dets
    for b in r0.boxes:
        cls_id = int(float(b.cls[0]))
        cls_name = names.get(cls_id, str(cls_id))
        conf = float(b.conf[0])
        xyxy = b.xyxy[0].tolist()
        # aceita todas as classes quando CLASSES_INTERESSE estiver vazio
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
        print("[DICA] Em Hik/Intelbras, o substream do canal 1 geralmente é '102'. Teste CHANNEL_DET='102'.")
        return

    # teste inicial de leitura (descobre problema de canal)
    print("[DIAG] Tentando ler 60 frames do RTSP_DET...")
    ok_frames = 0
    for i in range(60):
        ret, f = cap.read()
        if not ret or f is None:
            time.sleep(0.05)
            continue
        ok_frames += 1
    print(f"[DIAG] Frames lidos no teste inicial: {ok_frames}/60")
    if ok_frames < 5:
        print("[ERRO] Quase nenhum frame lido do substream. Troque CHANNEL_DET para '102' e rode novamente.")
        return
    cap.release()

    cap = cv2.VideoCapture(RTSP_DET, cv2.CAP_FFMPEG)
    fps_read = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps_read) if fps_read and fps_read > 0 else CFG.FPS_TARGET
    recorder = EventRecorder(fps=fps, pre_s=CFG.PRE_EVENT_SECONDS, post_s=CFG.POST_EVENT_SECONDS)

    # janela de monitoramento (desative durante testes noturnos, se quiser)
    hora_ini = dtime(5, 30)   # 05:30
    hora_fim = dtime(19, 0)   # 19:00

    prev = time.time()
    fps_ema = None

    while True:
        agora = datetime.now().time()
        if not (hora_ini <= agora <= hora_fim):
            # fora do horário, só espera um pouco
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

        # FPS instantâneo (EMA)
        now = time.time()
        inst = 1.0 / max(1e-6, (now - prev))
        prev = now
        fps_ema = inst if fps_ema is None else (0.9 * fps_ema + 0.1 * inst)

        frame = mask_roi(frame)

        # detecção
        dets = yolo_detect(_yolo_model, frame)
        if dets:
            cl = [d["cls"] for d in dets]
            print(f"[DIAG] dets={len(dets)} classes={cl} conf_max={max(d['conf'] for d in dets):.2f}")
            classes = cl
            conf_max = max(d["conf"] for d in dets)
            recorder.start_or_extend(frame, classes, conf_max)

        status = recorder.step(frame)

        vis = frame.copy()
        if CFG.DRAW and dets:
            vis = draw_dets(vis, dets)

        # HUD sempre visível
        vis = draw_hud(vis, dets, recorder, fps_ema=fps_ema)

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

            recorder = EventRecorder(fps=fps, pre_s=CFG.PRE_EVENT_SECONDS, post_s=CFG.POST_EVENT_SECONDS)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
