import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional until dependencies are installed
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
EVENTS_DIR = DATA_DIR / "events"
REPORTS_DIR = DATA_DIR / "reports"
RUNS_DIR = DATA_DIR / "runs"
MODELS_DIR = DATA_DIR / "models"
DEFAULT_MODEL_PATH = RUNS_DIR / "detect" / "train3" / "weights" / "best.pt"


def load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "sim", "on"}


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _env_size(name: str, default: tuple[int, int]) -> tuple[int, int]:
    value = os.getenv(name)
    if not value:
        return default
    parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        return default
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return default


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value) if value else default
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


@dataclass
class Config:
    ip: str = field(default_factory=lambda: os.getenv("IP", "192.168.100.10"))
    user: str = field(default_factory=lambda: os.getenv("USER", "admin"))
    password: str = field(default_factory=lambda: os.getenv("PASSWORD", "S3lv409052021@@"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "554")))
    channel_det: str = field(default_factory=lambda: os.getenv("CHANNEL_DET", "201"))
    channel_hq: str = field(default_factory=lambda: os.getenv("CHANNEL_HQ", "101"))
    save_root: Path = field(default_factory=lambda: _env_path("SAVE_ROOT", EVENTS_DIR))
    reports_root: Path = field(default_factory=lambda: _env_path("REPORTS_ROOT", REPORTS_DIR))
    model_name: Path = field(default_factory=lambda: _env_path("MODEL_NAME", DEFAULT_MODEL_PATH))
    min_conf: float = field(default_factory=lambda: float(os.getenv("MIN_CONF", "0.25")))
    classes_interesse: tuple[str, ...] = field(default_factory=lambda: _env_tuple("CLASSES_INTERESSE", ()))
    fps_target: int = field(default_factory=lambda: int(os.getenv("FPS_TARGET", "15")))
    pre_event_seconds: int = field(default_factory=lambda: int(os.getenv("PRE_EVENT_SECONDS", "3")))
    post_event_seconds: int = field(default_factory=lambda: int(os.getenv("POST_EVENT_SECONDS", "4")))
    event_min_duration: float = field(default_factory=lambda: float(os.getenv("EVENT_MIN_DURATION", "1.5")))
    max_event_duration: float = field(default_factory=lambda: float(os.getenv("MAX_EVENT_DURATION", "9.0")))
    snapshot_hq: bool = field(default_factory=lambda: _env_bool("SNAPSHOT_HQ", True))
    draw: bool = field(default_factory=lambda: _env_bool("DRAW", True))
    show_window: bool = field(default_factory=lambda: _env_bool("SHOW_WINDOW", True))
    hud: bool = field(default_factory=lambda: _env_bool("HUD", False))
    roi_poly: list[tuple[float, float]] | None = None
    out_size: tuple[int, int] = field(default_factory=lambda: _env_size("OUT_SIZE", (640, 360)))
    out_fps: int = field(default_factory=lambda: int(os.getenv("OUT_FPS", "8")))
    frame_stride: int = field(default_factory=lambda: int(os.getenv("FRAME_STRIDE", "2")))
    jpeg_quality: int = field(default_factory=lambda: int(os.getenv("JPEG_QUALITY", "60")))
    monitor_start: str = field(default_factory=lambda: os.getenv("MONITOR_START", "05:30"))
    monitor_end: str = field(default_factory=lambda: os.getenv("MONITOR_END", "18:30"))

    @property
    def db_path(self) -> Path:
        return self.save_root / "events.db"


def build_config() -> Config:
    load_environment()
    return Config()


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_event_dirs(base: Path) -> dict[str, Path]:
    today = datetime.now().strftime("%Y-%m-%d")
    paths = {
        "base": base / today,
        "videos": base / today / "videos",
        "images": base / today / "images",
        "meta": base / today / "meta",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
