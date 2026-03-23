from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchResultModel(BaseModel):
    id: str
    filename: str
    timeline: str
    timecode: str
    duration: str
    track: int
    description: str
    tags: list[str] = Field(default_factory=list)
    type: str
    fps: str | None = None
    resolution: str | None = None
    thumbnail: str | None = None


class SearchResponseModel(BaseModel):
    query: str
    filter: str = "All"
    scope: str = "all"
    total: int
    results: list[SearchResultModel] = Field(default_factory=list)


class CoverageModel(BaseModel):
    label: str
    indexed: int = 0
    total: int = 0
    missing: int = 0
    complete: bool = False


class TimelineOptionModel(BaseModel):
    timeline_name: str
    timeline_uid: str | None = None
    total: int = 0
    indexed: int = 0
    missing: int = 0
    complete: bool = False
    current: bool = False


class IndexStatsModel(BaseModel):
    project_name: str | None = None
    total: int = 0
    timelines: int = 0
    last_indexed: str | None = None
    available_types: list[str] = Field(default_factory=list)
    is_stale: bool = False
    quick_mode: bool = False
    storage_format: str = "sqlite"
    storage_path: str | None = None
    loaded_from_disk: bool = False
    project_coverage: CoverageModel | None = None
    current_timeline_coverage: CoverageModel | None = None
    timeline_options: list[TimelineOptionModel] = Field(default_factory=list)


class ReindexStateModel(BaseModel):
    running: bool = False
    enrichment_running: bool = False
    progress: float = 0.0
    enrichment_progress: float = 0.0
    message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    last_error: str | None = None
    current_timeline: str | None = None
    processed_clips: int = 0
    total_clips: int = 0
    enriched_clips: int = 0
    total_enrichment_clips: int = 0
    active_clip_index: int = 0
    active_clip_name: str | None = None
    quick_mode: bool = False
    latest_clip: SearchResultModel | None = None
    latest_clip_stage: str | None = None


class StatusResponseModel(BaseModel):
    connected: bool
    message: str
    project_name: str | None = None
    current_timeline: str | None = None
    current_timeline_uid: str | None = None
    current_page: str | None = None
    index: IndexStatsModel = Field(default_factory=IndexStatsModel)
    reindex: ReindexStateModel = Field(default_factory=ReindexStateModel)


class JumpRequestModel(BaseModel):
    clip_id: str


class JumpResponseModel(BaseModel):
    success: bool = True
    clip_id: str
    timeline_name: str
    start_timecode: str


class ReindexRequestModel(BaseModel):
    timeline_uids: list[str] | None = None
    timeline_names: list[str] | None = None
    quick_mode: bool | None = None


class HealthModel(BaseModel):
    status: Literal["ok"] = "ok"
