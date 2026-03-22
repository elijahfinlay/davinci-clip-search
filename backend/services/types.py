from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MarkerInfo:
    frame: int
    color: str = ""
    name: str = ""
    note: str = ""
    duration: float = 0.0
    custom_data: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "color": self.color,
            "name": self.name,
            "note": self.note,
            "duration": self.duration,
            "custom_data": self.custom_data,
        }


@dataclass(slots=True)
class VisualDescription:
    frame_offset_sec: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_offset_sec": self.frame_offset_sec,
            "description": self.description,
        }


@dataclass(slots=True)
class VisionAnalysis:
    summary: str
    tags: list[str] = field(default_factory=list)
    frame_descriptions: list[VisualDescription] = field(default_factory=list)
    clip_type_hint: str | None = None
    cache_signature: str = "heuristic:v1"
    provider: str = "heuristic"
    model: str | None = None


@dataclass(slots=True)
class ClipRecord:
    clip_id: str
    content_signature: str
    vision_cache_signature: str
    project_uid: str
    timeline_uid: str
    timeline_name: str
    timeline_index: int
    clip_name: str
    file_path: str
    file_name: str
    track: int
    track_name: str
    item_index: int
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_seconds: float
    fps: float
    start_timecode: str
    end_timecode: str
    source_in: str | None
    source_out: str | None
    resolution: str | None
    codec: str | None
    clip_color: str | None
    clip_type: str
    has_audio: bool
    description: str
    transcript: str | None
    tags: list[str] = field(default_factory=list)
    markers: list[MarkerInfo] = field(default_factory=list)
    timeline_markers: list[MarkerInfo] = field(default_factory=list)
    visual_descriptions: list[VisualDescription] = field(default_factory=list)
    searchable_text: str = ""
    source_signature: str = ""
    thumbnail_data: str | None = None


@dataclass(slots=True)
class TimelineRecord:
    timeline_uid: str
    project_uid: str
    timeline_name: str
    timeline_index: int
    clip_count: int


@dataclass(slots=True)
class ProjectIndexMeta:
    project_uid: str
    project_name: str
    indexed_at: str
    clip_count: int
    timeline_count: int
    signature_hash: str | None
    quick_mode: bool


@dataclass(slots=True)
class ReindexState:
    running: bool = False
    progress: float = 0.0
    message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    last_error: str | None = None
    current_timeline: str | None = None
    processed_clips: int = 0
    total_clips: int = 0
    active_clip_index: int = 0
    active_clip_name: str | None = None
    quick_mode: bool = False
    latest_clip: dict[str, Any] | None = None
    latest_clip_stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "progress": self.progress,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_error": self.last_error,
            "current_timeline": self.current_timeline,
            "processed_clips": self.processed_clips,
            "total_clips": self.total_clips,
            "active_clip_index": self.active_clip_index,
            "active_clip_name": self.active_clip_name,
            "quick_mode": self.quick_mode,
            "latest_clip": self.latest_clip,
            "latest_clip_stage": self.latest_clip_stage,
        }
