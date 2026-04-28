from collections import Counter

import cv2
import numpy as np

from .config import Config


def load_yolo_model(model_path):
    try:
        from ultralytics import YOLO

        return YOLO(str(model_path)), True
    except Exception as exc:
        print("[WARN] Falha ao carregar YOLO:", exc)
        return None, False


def mask_roi(frame, roi_poly):
    if roi_poly is None:
        return frame
    h, w = frame.shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in roi_poly], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)


def yolo_detect(model, frame, config: Config):
    results = model.predict(
        source=frame,
        imgsz=960,
        conf=config.min_conf,
        iou=0.55,
        agnostic_nms=True,
        verbose=False,
    )
    dets = []
    if not results:
        return dets

    r0 = results[0]
    names = r0.names
    if r0.boxes is None:
        return dets

    for box in r0.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = names.get(cls_id, str(cls_id))
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        if not config.classes_interesse or cls_name in config.classes_interesse:
            dets.append({"cls": cls_name, "conf": conf, "xyxy": xyxy})
    return dets


def draw_dets(frame, dets):
    for det in dets:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        label = f"{det['cls']} {det['conf']:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def draw_hud(frame, dets, recorder, fps_ema=None):
    counts = Counter(det["cls"] for det in dets) if dets else {}
    total = sum(counts.values())
    status = "REC" if recorder.recording else "IDLE"
    lines = [f"Status: {status}"]
    if fps_ema is not None:
        lines[0] += f" | FPS: {fps_ema:04.1f}"
    if total:
        parts = [f"{name}:{count}" for name, count in counts.items()]
        lines.append("Deteccoes: " + ", ".join(parts) + f" (total={total})")
        lines.append(f"conf_max(frame): {max(det['conf'] for det in dets):.2f}")
    else:
        lines.append("Deteccoes: 0")

    y = 22
    for text in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 22
    return frame
