import time
from datetime import datetime, time as dtime
from pathlib import Path
import sys

import cv2

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.camera import build_streams, grab_hq_snapshot, open_capture, reconnect_capture
    from src.config import build_config, ensure_event_dirs
    from src.database import init_db, insert_event, save_meta
    from src.detector import draw_dets, draw_hud, load_yolo_model, mask_roi, yolo_detect
    from src.recorder import EventRecorder
else:
    from .camera import build_streams, grab_hq_snapshot, open_capture, reconnect_capture
    from .config import build_config, ensure_event_dirs
    from .database import init_db, insert_event, save_meta
    from .detector import draw_dets, draw_hud, load_yolo_model, mask_roi, yolo_detect
    from .recorder import EventRecorder


def _parse_time(value: str) -> dtime:
    hour, minute = value.split(":", maxsplit=1)
    return dtime(int(hour), int(minute))


def _should_start_event(dets, min_conf: float) -> tuple[bool, list[str], float]:
    if not dets:
        return False, [], 0.0
    classes = [det["cls"] for det in dets]
    conf_max = max(det["conf"] for det in dets)
    areas = [(det["xyxy"][2] - det["xyxy"][0]) * (det["xyxy"][3] - det["xyxy"][1]) for det in dets]
    return conf_max >= max(0.28, min_conf) and max(areas) >= 14 * 14, classes, conf_max


def main() -> None:
    config = build_config()
    paths = ensure_event_dirs(config.save_root)
    init_db(config.db_path)

    rtsp_det, rtsp_hq = build_streams(config)
    print("[DIAG] RTSP_DET =", rtsp_det)
    print("[DIAG] RTSP_HQ  =", rtsp_hq)
    print("[DIAG] MIN_CONF =", config.min_conf)
    print("[DIAG] CLASSES_INTERESSE =", config.classes_interesse)
    print("[DIAG] MODEL_NAME =", config.model_name)

    model, yolo_ready = load_yolo_model(config.model_name)
    if not yolo_ready:
        print("[ERRO] YOLO nao esta pronto.")
        return

    cap = open_capture(rtsp_det)
    if not cap.isOpened():
        print("[ERRO] Nao abriu RTSP:", rtsp_det)
        return

    fps_read = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps_read) if fps_read and fps_read > 0 else config.fps_target
    recorder = EventRecorder(
        config=config,
        paths=paths,
        fps=fps,
        pre_s=config.pre_event_seconds,
        post_s=config.post_event_seconds,
    )
    monitor_start = _parse_time(config.monitor_start)
    monitor_end = _parse_time(config.monitor_end)
    fps_ema = None
    last_frame_time = time.monotonic()

    while True:
        now_time = datetime.now().time()
        if not (monitor_start <= now_time <= monitor_end):
            print("[INFO] Fora do horario de monitoramento. Dormindo...")
            time.sleep(60)
            continue

        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame invalido. Reconectando...")
            cap.release()
            cap = reconnect_capture(rtsp_det)
            continue

        for _ in range(max(0, config.frame_stride - 1)):
            cap.grab()

        current_time = time.monotonic()
        frame_fps = 1.0 / max(current_time - last_frame_time, 1e-6)
        fps_ema = frame_fps if fps_ema is None else (0.9 * fps_ema + 0.1 * frame_fps)
        last_frame_time = current_time

        frame = mask_roi(frame, config.roi_poly)
        dets = yolo_detect(model, frame, config)

        should_start, classes, conf_max = _should_start_event(dets, config.min_conf)
        if should_start:
            recorder.start_or_extend(frame, classes, conf_max)

        status = recorder.step(frame)

        vis = frame.copy()
        if config.draw and dets:
            vis = draw_dets(vis, dets)
        if config.hud:
            vis = draw_hud(vis, dets, recorder, fps_ema=fps_ema)
        else:
            cv2.putText(
                vis,
                f"Deteccoes: {len(dets)}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if config.show_window:
            cv2.imshow("Monitor Rio - MVP", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if status == "closed":
            ts_end = datetime.now().isoformat(timespec="seconds")
            try:
                dt_start = datetime.fromisoformat(recorder.ts_start)
                dt_end = datetime.fromisoformat(ts_end)
                duration = (dt_end - dt_start).total_seconds()
            except Exception:
                duration = 0.0

            snapshot_path = (
                grab_hq_snapshot(rtsp_hq, paths["images"], config.jpeg_quality)
                if config.snapshot_hq
                else None
            )
            meta = {
                "ts_start": recorder.ts_start,
                "ts_end": ts_end,
                "duration": duration,
                "classes": sorted(recorder.classes_seen),
                "conf_max": recorder.conf_max,
                "video_path": recorder.video_path,
                "snapshot_path": snapshot_path,
                "channel_det": config.channel_det,
                "channel_hq": config.channel_hq,
                "ip": config.ip,
            }
            meta_path = save_meta(paths["meta"], meta)
            meta["meta_path"] = meta_path

            insert_event(
                config.db_path,
                {
                    "ts_start": meta["ts_start"],
                    "ts_end": meta["ts_end"],
                    "duration": meta["duration"],
                    "classes": ",".join(meta["classes"]),
                    "conf_max": meta["conf_max"],
                    "video_path": meta["video_path"],
                    "snapshot_path": meta["snapshot_path"] or "",
                    "meta_path": meta["meta_path"],
                },
            )
            recorder = EventRecorder(
                config=config,
                paths=paths,
                fps=fps,
                pre_s=config.pre_event_seconds,
                post_s=config.post_event_seconds,
            )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
