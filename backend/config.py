from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    data_dir: Path
    index_db_path: Path
    frontend_dist_dir: Path
    host: str
    port: int
    log_level: str
    auto_transcribe: bool
    default_quick_mode: bool
    enable_thumbnail_capture: bool
    vision_provider: str
    anthropic_api_key: str | None
    gemini_api_key: str | None
    vision_model: str
    vision_max_image_edge_px: int
    vision_timeout_sec: float
    gemini_max_attempts: int
    yolo_world_model: str
    yolo_world_confidence: float
    yolo_world_max_objects: int
    thumbnail_cache_version: str
    thumbnail_source_width: int
    thumbnail_source_height: int
    thumbnail_jpeg_quality: int


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_vision_provider() -> str:
    explicit = os.getenv("RESOLVE_CLIP_SEARCH_VISION_PROVIDER")
    if explicit:
        return explicit.strip().lower()
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "heuristic"


def _default_vision_model(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "gemini":
        return "gemini-2.5-flash"
    if normalized == "anthropic":
        return "claude-sonnet-4-6"
    return "heuristic"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv(ROOT_DIR / ".env")
    vision_provider = _default_vision_provider()

    data_dir = Path(
        os.getenv(
            "RESOLVE_CLIP_SEARCH_DATA_DIR",
            str(ROOT_DIR / ".resolve-clip-search"),
        )
    ).expanduser()

    index_db_path = Path(
        os.getenv(
            "RESOLVE_CLIP_SEARCH_INDEX_DB",
            str(data_dir / "resolve_clip_search.sqlite3"),
        )
    ).expanduser()

    return Settings(
        root_dir=ROOT_DIR,
        data_dir=data_dir,
        index_db_path=index_db_path,
        frontend_dist_dir=ROOT_DIR / "frontend" / "dist",
        host=os.getenv("RESOLVE_CLIP_SEARCH_HOST", "127.0.0.1"),
        port=int(os.getenv("RESOLVE_CLIP_SEARCH_PORT", "8000")),
        log_level=os.getenv("RESOLVE_CLIP_SEARCH_LOG_LEVEL", "info"),
        auto_transcribe=_env_bool("RESOLVE_CLIP_SEARCH_AUTO_TRANSCRIBE", False),
        default_quick_mode=_env_bool("RESOLVE_CLIP_SEARCH_QUICK_MODE", False),
        enable_thumbnail_capture=_env_bool(
            "RESOLVE_CLIP_SEARCH_ENABLE_THUMBNAILS", False
        ),
        vision_provider=vision_provider,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        vision_model=os.getenv(
            "RESOLVE_CLIP_SEARCH_VISION_MODEL",
            _default_vision_model(vision_provider),
        ),
        vision_max_image_edge_px=int(
            os.getenv("RESOLVE_CLIP_SEARCH_VISION_MAX_IMAGE_EDGE_PX", "768")
        ),
        vision_timeout_sec=float(
            os.getenv("RESOLVE_CLIP_SEARCH_VISION_TIMEOUT_SEC", "60")
        ),
        gemini_max_attempts=max(
            int(os.getenv("RESOLVE_CLIP_SEARCH_GEMINI_MAX_ATTEMPTS", "2")),
            1,
        ),
        yolo_world_model=os.getenv(
            "RESOLVE_CLIP_SEARCH_YOLO_WORLD_MODEL",
            "yolov8s-worldv2.pt",
        ),
        yolo_world_confidence=float(
            os.getenv("RESOLVE_CLIP_SEARCH_YOLO_WORLD_CONFIDENCE", "0.18")
        ),
        yolo_world_max_objects=int(
            os.getenv("RESOLVE_CLIP_SEARCH_YOLO_WORLD_MAX_OBJECTS", "14")
        ),
        thumbnail_cache_version=os.getenv(
            "RESOLVE_CLIP_SEARCH_THUMBNAIL_CACHE_VERSION",
            "v2",
        ),
        thumbnail_source_width=int(
            os.getenv("RESOLVE_CLIP_SEARCH_THUMBNAIL_SOURCE_WIDTH", "160")
        ),
        thumbnail_source_height=int(
            os.getenv("RESOLVE_CLIP_SEARCH_THUMBNAIL_SOURCE_HEIGHT", "104")
        ),
        thumbnail_jpeg_quality=int(
            os.getenv("RESOLVE_CLIP_SEARCH_THUMBNAIL_JPEG_QUALITY", "12")
        ),
    )
