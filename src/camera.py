import time
from pathlib import Path
from urllib.parse import quote

import cv2

from .config import Config, now_ts


def build_rtsp(ip: str, user: str, password: str, port: int, channel: str) -> str:
    user_q = quote(user, safe="")
    password_q = quote(password, safe="")
    return f"rtsp://{user_q}:{password_q}@{ip}:{port}/Streaming/Channels/{channel}"


def build_streams(config: Config) -> tuple[str, str]:
    rtsp_det = build_rtsp(config.ip, config.user, config.password, config.port, config.channel_det)
    rtsp_hq = build_rtsp(config.ip, config.user, config.password, config.port, config.channel_hq)
    return rtsp_det, rtsp_hq


def open_capture(rtsp_url: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)


def reconnect_capture(rtsp_url: str, delay_seconds: float = 1.0) -> cv2.VideoCapture:
    time.sleep(delay_seconds)
    return open_capture(rtsp_url)


def grab_hq_snapshot(rtsp_hq: str, images_dir: Path, jpeg_quality: int) -> str | None:
    cap = open_capture(rtsp_hq)
    if not cap.isOpened():
        print("[WARN] Nao abriu HQ stream para snapshot")
        return None

    frame = None
    ret = False
    for _ in range(6):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.05)
    cap.release()

    if ret and frame is not None:
        path = images_dir / f"snapshot_{now_ts()}_HQ.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        return str(path)
    return None
