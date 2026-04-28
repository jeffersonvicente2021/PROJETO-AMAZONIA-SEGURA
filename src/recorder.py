import shutil
import subprocess
import time
from collections import deque

import cv2

from .config import Config, now_ts


class EventRecorder:
    def __init__(self, config: Config, paths: dict, fps: int, pre_s: int = 5, post_s: int = 5):
        self.config = config
        self.paths = paths
        self.fps_in = max(1, int(fps))
        self.fps_out = max(1, int(config.out_fps))
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
        return cv2.resize(frame, self.config.out_size, interpolation=cv2.INTER_AREA)

    def _new_writer(self, frame) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_path = str(self.paths["videos"] / f"event_{now_ts()}.mp4")
        width, height = self.config.out_size
        self.writer = cv2.VideoWriter(self.video_path, fourcc, self.fps_out, (width, height))
        for pre_frame in list(self.pre):
            self.writer.write(pre_frame)
        self.ts_start = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.event_start_time_monotonic = time.monotonic()

    def start_or_extend(self, frame, classes, conf) -> None:
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
        if elapsed >= self.post_s or total >= self.config.max_event_duration:
            self.close()
            return "closed"
        return None

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self.recording = False


def recompress_with_ffmpeg(in_path: str, out_br_kbps: int = 800) -> str:
    out_path = in_path.replace(".mp4", f"_c{out_br_kbps}k.mp4")
    if shutil.which("ffmpeg") is None:
        return in_path
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        in_path,
        "-c:v",
        "libx264",
        "-b:v",
        f"{out_br_kbps}k",
        "-preset",
        "veryfast",
        "-movflags",
        "faststart",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        out_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return out_path
