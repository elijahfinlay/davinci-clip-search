from __future__ import annotations

import base64
import shutil
import subprocess
import threading
from pathlib import Path

from backend.config import Settings

from .index_store import IndexStore


class ThumbnailService:
    def __init__(self, *, settings: Settings, store: IndexStore) -> None:
        self.settings = settings
        self.store = store
        self._lock = threading.Lock()
        self._ffmpeg = shutil.which("ffmpeg")

    def get_or_create_thumbnail(self, clip_id: str) -> bytes | None:
        clip = self.store.get_clip(clip_id)
        if not clip:
            return None

        cached = clip.get("thumbnail_data")
        if cached and self._is_current_thumbnail(cached):
            return self._decode_data_uri(cached)

        file_path = (clip.get("file_path") or "").strip()
        if not file_path or not Path(file_path).exists() or not self._ffmpeg:
            return None

        with self._lock:
            clip = self.store.get_clip(clip_id)
            if not clip:
                return None
            cached = clip.get("thumbnail_data")
            if cached and self._is_current_thumbnail(cached):
                return self._decode_data_uri(cached)

            jpeg = self._extract_preview_bytes(file_path)
            if not jpeg:
                return None

            self.store.update_clip_thumbnail(clip_id, self._encode_data_uri(jpeg))
            return jpeg

    def _encode_data_uri(self, image_bytes: bytes) -> str:
        return (
            f"data:image/jpeg;preview={self.settings.thumbnail_cache_version};base64,"
            + base64.b64encode(image_bytes).decode("ascii")
        )

    @staticmethod
    def _decode_data_uri(data_uri: str) -> bytes | None:
        if not data_uri.startswith("data:image"):
            return None
        _, _, encoded = data_uri.partition(",")
        if not encoded:
            return None
        return base64.b64decode(encoded)

    def _is_current_thumbnail(self, data_uri: str) -> bool:
        return f"preview={self.settings.thumbnail_cache_version}" in data_uri

    def _extract_preview_bytes(self, file_path: str) -> bytes | None:
        for timestamp in ("1.0", "0.0"):
            result = subprocess.run(
                [
                    self._ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    timestamp,
                    "-i",
                    file_path,
                    "-frames:v",
                    "1",
                    "-vf",
                    (
                        f"scale={self.settings.thumbnail_source_width}:"
                        f"{self.settings.thumbnail_source_height}:"
                        "force_original_aspect_ratio=increase,"
                        f"crop={self.settings.thumbnail_source_width}:"
                        f"{self.settings.thumbnail_source_height}"
                    ),
                    "-q:v",
                    str(self.settings.thumbnail_jpeg_quality),
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "mjpeg",
                    "-",
                ],
                capture_output=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
        return None
