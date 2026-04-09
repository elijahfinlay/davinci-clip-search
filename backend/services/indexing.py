from __future__ import annotations

import hashlib
import json
import queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from backend.config import Settings

from .index_store import IndexStore
from .resolve_api import ResolveFacade, ResolveConnectionError, safe_call
from .search import canonical_clip_type, format_fps, human_duration
from .timecode import timeline_frame_to_timecode
from .types import (
    ClipRecord,
    MarkerInfo,
    ProjectIndexMeta,
    ReindexState,
    TimelineRecord,
    VisionAnalysis,
    VisualDescription,
)
from .vision import build_visual_analyzer


ProgressCallback = Callable[[ReindexState], None]
PartialClipCallback = Callable[[ClipRecord, str], None]
SessionActionCallback = Callable[[Any, Any], None]
CancellationCallback = Callable[[], bool]


@dataclass(slots=True)
class EnrichmentTask:
    clip: ClipRecord


@dataclass(slots=True)
class ClipBuildResult:
    clip: ClipRecord
    enrichment_task: EnrichmentTask | None = None


@dataclass(slots=True)
class IndexBuildResult:
    project_meta: ProjectIndexMeta
    enrichment_tasks: list[EnrichmentTask] = field(default_factory=list)


class IndexingCancelledError(Exception):
    pass


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        value = item.strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def parse_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def lookup_value(mapping: dict[str, Any], *keys: str) -> Any:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def normalize_markers(marker_map: dict[Any, Any]) -> list[MarkerInfo]:
    markers: list[MarkerInfo] = []
    for frame, marker in sorted(marker_map.items(), key=lambda item: float(item[0])):
        payload = marker or {}
        markers.append(
            MarkerInfo(
                frame=int(float(frame)),
                color=str(payload.get("color", "")),
                name=str(payload.get("name", "")),
                note=str(payload.get("note", "")),
                duration=float(payload.get("duration", 0.0) or 0.0),
                custom_data=str(payload.get("customData", "")),
            )
        )
    return markers


def nearby_markers(
    timeline_markers: list[MarkerInfo],
    *,
    start_frame: int,
    end_frame: int,
    padding: int = 12,
) -> list[MarkerInfo]:
    matches: list[MarkerInfo] = []
    for marker in timeline_markers:
        if start_frame - padding <= marker.frame <= end_frame + padding:
            matches.append(marker)
    return matches


def infer_clip_type(
    *,
    clip_name: str,
    timeline_name: str,
    track_name: str,
    tags: list[str],
    transcript: str | None,
    file_path: str,
) -> str:
    haystack = " ".join(
        [clip_name, timeline_name, track_name, " ".join(tags), transcript or "", file_path]
    ).lower()
    if any(token in haystack for token in ["dji", "drone", "fpv", "aerial"]):
        return "drone"
    if any(token in haystack for token in ["interview", "dialogue", "talking head", "vox", "testimonial"]):
        return "interview"
    if any(token in haystack for token in ["handheld", "gimbal", "steadicam", "shoulder"]):
        return "handheld"
    return "handheld"


def extract_keyword_tags(
    *,
    clip_name: str,
    timeline_name: str,
    track_name: str,
    file_path: str,
    markers: list[MarkerInfo],
    transcript: str | None,
) -> list[str]:
    combined = " ".join(
        [
            clip_name,
            timeline_name,
            track_name,
            file_path,
            transcript or "",
            " ".join(
                " ".join([marker.color, marker.name, marker.note])
                for marker in markers
            ),
        ]
    )
    tokens = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", combined.lower())
        if token not in {"track", "timeline", "clip", "video", "audio", "take"}
    ]
    return dedupe(tokens[:40])


def build_description(
    *,
    clip_name: str,
    clip_type: str,
    timeline_name: str,
    track_name: str,
    tags: list[str],
    transcript: str | None,
    visual_descriptions: list[VisualDescription],
) -> str:
    if visual_descriptions:
        return visual_descriptions[0].description
    if transcript:
        excerpt = transcript.strip().replace("\n", " ")
        return excerpt[:140]
    prominent_tags = ", ".join(tags[:4]) if tags else "untagged"
    return f"{clip_type.title()} clip from {timeline_name} / {track_name} with {prominent_tags}"


def looks_like_heuristic_visual(
    *,
    description: str,
    visual_descriptions: list[VisualDescription],
    clip_type: str,
    file_path: str,
) -> bool:
    summary = description.strip()
    if not summary:
        return False
    if not summary.startswith(f"{clip_type.title()} clip with "):
        return False
    if file_path and not summary.endswith(f". Source: {Path(file_path).name}"):
        return False
    if not visual_descriptions:
        return True
    return len(visual_descriptions) == 1 and visual_descriptions[0].description.strip() == summary


def build_searchable_text(
    *,
    clip_name: str,
    file_name: str,
    file_path: str,
    timeline_name: str,
    track_name: str,
    description: str,
    transcript: str | None,
    tags: list[str],
    clip_type: str,
    markers: list[MarkerInfo],
    timeline_markers: list[MarkerInfo],
    visual_descriptions: list[VisualDescription],
    codec: str | None,
    resolution: str | None,
) -> str:
    marker_bits = []
    for marker in markers + timeline_markers:
        marker_bits.extend([marker.color, marker.name, marker.note, marker.custom_data])

    visual_bits = [item.description for item in visual_descriptions]
    chunks = [
        clip_name,
        file_name,
        file_path,
        timeline_name,
        track_name,
        description,
        transcript or "",
        " ".join(tags),
        clip_type,
        " ".join(marker_bits),
        " ".join(visual_bits),
        codec or "",
        resolution or "",
    ]
    return " ".join(chunk for chunk in chunks if chunk).strip()


def build_content_signature(*, file_name: str, duration_frames: int) -> str:
    normalized_name = file_name.strip().lower() if file_name else ""
    return f"{normalized_name}|{max(duration_frames, 0)}"


def build_source_signature(
    *,
    clip_name: str,
    file_path: str,
    timeline_name: str,
    track_index: int,
    start_frame: int,
    end_frame: int,
    duration_frames: int,
    fps: float,
    resolution: str | None,
    codec: str | None,
    markers: list[MarkerInfo],
    timeline_markers: list[MarkerInfo],
    transcript: str | None,
    vision_cache_signature: str,
) -> str:
    payload = {
        "clip_name": clip_name,
        "file_path": file_path,
        "timeline_name": timeline_name,
        "track": track_index,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "duration_frames": duration_frames,
        "fps": fps,
        "resolution": resolution,
        "codec": codec,
        "markers": [marker.to_dict() for marker in markers],
        "timeline_markers": [marker.to_dict() for marker in timeline_markers],
        "transcript": transcript,
        "vision_cache_signature": vision_cache_signature,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def clip_record_to_result(clip: ClipRecord) -> dict[str, Any]:
    thumbnail = None
    if clip.thumbnail_data or clip.file_path:
        thumbnail = f"/api/clips/{clip.clip_id}/thumbnail"
    return {
        "id": clip.clip_id,
        "filename": clip.file_name or clip.clip_name,
        "timeline": clip.timeline_name,
        "timecode": clip.start_timecode,
        "duration": human_duration(clip.duration_seconds),
        "track": clip.track,
        "description": clip.description or clip.clip_name,
        "tags": clip.tags,
        "type": canonical_clip_type(clip.clip_type),
        "fps": format_fps(clip.fps),
        "resolution": clip.resolution,
        "thumbnail": thumbnail,
    }


class IndexingService:
    def __init__(
        self,
        *,
        settings: Settings,
        store: IndexStore,
        resolve: ResolveFacade,
    ) -> None:
        self.settings = settings
        self.store = store
        self.resolve = resolve
        self.visual_analyzer = build_visual_analyzer(settings)

    def _supports_background_enrichment(self, *, quick_mode: bool) -> bool:
        if quick_mode:
            return False
        return bool(self.visual_analyzer.background_enrichment_enabled())

    def _apply_analysis_to_clip(
        self,
        *,
        clip: ClipRecord,
        analysis: VisionAnalysis,
    ) -> ClipRecord:
        clip_type = analysis.clip_type_hint or clip.clip_type
        tags = dedupe(clip.tags + analysis.tags + [clip_type])
        visual_descriptions = analysis.frame_descriptions or clip.visual_descriptions
        description = analysis.summary.strip() or clip.description
        if not description:
            description = build_description(
                clip_name=clip.clip_name,
                clip_type=clip_type,
                timeline_name=clip.timeline_name,
                track_name=clip.track_name,
                tags=tags,
                transcript=clip.transcript,
                visual_descriptions=visual_descriptions,
            )

        searchable_text = build_searchable_text(
            clip_name=clip.clip_name,
            file_name=clip.file_name,
            file_path=clip.file_path,
            timeline_name=clip.timeline_name,
            track_name=clip.track_name,
            description=description,
            transcript=clip.transcript,
            tags=tags,
            clip_type=clip_type,
            markers=clip.markers,
            timeline_markers=clip.timeline_markers,
            visual_descriptions=visual_descriptions,
            codec=clip.codec,
            resolution=clip.resolution,
        )
        source_signature = build_source_signature(
            clip_name=clip.clip_name,
            file_path=clip.file_path,
            timeline_name=clip.timeline_name,
            track_index=clip.track,
            start_frame=clip.start_frame,
            end_frame=clip.end_frame,
            duration_frames=clip.duration_frames,
            fps=clip.fps,
            resolution=clip.resolution,
            codec=clip.codec,
            markers=clip.markers,
            timeline_markers=clip.timeline_markers,
            transcript=clip.transcript,
            vision_cache_signature=analysis.cache_signature,
        )

        return ClipRecord(
            clip_id=clip.clip_id,
            content_signature=clip.content_signature,
            vision_cache_signature=analysis.cache_signature,
            project_uid=clip.project_uid,
            timeline_uid=clip.timeline_uid,
            timeline_name=clip.timeline_name,
            timeline_index=clip.timeline_index,
            clip_name=clip.clip_name,
            file_path=clip.file_path,
            file_name=clip.file_name,
            track=clip.track,
            track_name=clip.track_name,
            item_index=clip.item_index,
            start_frame=clip.start_frame,
            end_frame=clip.end_frame,
            duration_frames=clip.duration_frames,
            duration_seconds=clip.duration_seconds,
            fps=clip.fps,
            start_timecode=clip.start_timecode,
            end_timecode=clip.end_timecode,
            source_in=clip.source_in,
            source_out=clip.source_out,
            resolution=clip.resolution,
            codec=clip.codec,
            clip_color=clip.clip_color,
            clip_type=clip_type,
            has_audio=clip.has_audio,
            description=description,
            transcript=clip.transcript,
            detected_objects=analysis.detected_objects or clip.detected_objects,
            tags=tags,
            markers=clip.markers,
            timeline_markers=clip.timeline_markers,
            visual_descriptions=visual_descriptions,
            searchable_text=searchable_text,
            source_signature=source_signature,
            thumbnail_data=clip.thumbnail_data,
        )

    def enrich_clip(self, clip: ClipRecord) -> ClipRecord | None:
        analysis = self.visual_analyzer.enrich(
            clip_name=clip.clip_name,
            clip_type=clip.clip_type,
            tags=clip.tags,
            file_path=clip.file_path,
            duration_seconds=clip.duration_seconds,
            detected_objects=clip.detected_objects,
        )
        if not analysis:
            return None
        return self._apply_analysis_to_clip(clip=clip, analysis=analysis)

    def build_index(
        self,
        *,
        timeline_uids: list[str] | None,
        timeline_names: list[str] | None,
        quick_mode: bool,
        progress_callback: ProgressCallback | None = None,
        session_action_callback: SessionActionCallback | None = None,
        cancellation_callback: CancellationCallback | None = None,
    ) -> IndexBuildResult:
        signature = self.resolve.compute_project_signature()
        target_uids = set(timeline_uids or [])
        target_names = set(timeline_names or [])
        replace_timeline_uids = {
            timeline["timeline_uid"]
            for timeline in signature["timelines"]
            if (
                (not target_uids and not target_names)
                or timeline["timeline_uid"] in target_uids
                or timeline["timeline_name"] in target_names
            )
        }
        existing_cache_by_clip_id = self.store.get_existing_cache(
            signature["project_uid"],
            timeline_uids=replace_timeline_uids if (target_uids or target_names) else None,
        )
        existing_cache_by_content_signature: dict[str, list[dict[str, Any]]] = {}
        for cached in existing_cache_by_clip_id.values():
            content_signature = str(cached.get("content_signature") or "").strip()
            if not content_signature:
                continue
            existing_cache_by_content_signature.setdefault(content_signature, []).append(
                cached
            )
        total_target_clips = sum(
            timeline["clip_count"]
            for timeline in signature["timelines"]
            if (
                (not target_uids and not target_names)
                or timeline["timeline_uid"] in target_uids
                or timeline["timeline_name"] in target_names
            )
        )
        enrichment_tasks: list[EnrichmentTask] = []

        state = ReindexState(
            running=True,
            progress=0.0,
            message="Preparing index",
            started_at=now_iso(),
            processed_clips=0,
            total_clips=total_target_clips,
            quick_mode=quick_mode,
        )
        if progress_callback:
            progress_callback(state)

        run_indexed_at = state.started_at or now_iso()
        project_meta = ProjectIndexMeta(
            project_uid=signature["project_uid"],
            project_name=signature["project_name"],
            indexed_at=run_indexed_at,
            clip_count=signature["clip_count"],
            timeline_count=signature["timeline_count"],
            signature_hash=signature["signature_hash"],
            quick_mode=quick_mode,
        )
        self.store.upsert_project_meta(project_meta)

        def _index(_resolve: Any, _project_manager: Any, project: Any) -> None:
            seen_clip_ids: set[str] = set()
            seen_timeline_uids: set[str] = set()

            timeline_count = safe_call(project.GetTimelineCount, default=0) or 0
            current_indexed = 0
            for timeline_index in range(1, timeline_count + 1):
                if cancellation_callback and cancellation_callback():
                    raise IndexingCancelledError("Indexing stopped by user.")
                timeline = safe_call(project.GetTimelineByIndex, timeline_index)
                if not timeline:
                    continue
                timeline_name = safe_call(timeline.GetName) or f"Timeline {timeline_index}"
                timeline_uid = safe_call(timeline.GetUniqueId) or hashlib.sha1(
                    f"{signature['project_uid']}::{timeline_name}".encode("utf-8")
                ).hexdigest()
                if (
                    target_uids or target_names
                ) and timeline_uid not in target_uids and timeline_name not in target_names:
                    continue
                state.current_timeline = timeline_name
                state.message = f'Indexing "{timeline_name}"'
                if progress_callback:
                    progress_callback(state)
                if session_action_callback:
                    session_action_callback(_resolve, project)

                timeline_markers = normalize_markers(
                    safe_call(timeline.GetMarkers, default={}) or {}
                )
                timeline_start_frame = int(safe_call(timeline.GetStartFrame, default=0) or 0)
                timeline_start_tc = safe_call(timeline.GetStartTimecode, default="00:00:00:00") or "00:00:00:00"
                timeline_fps = parse_float(
                    safe_call(project.GetSetting, "timelineFrameRate", default=None),
                    24.0,
                )
                drop_frame = bool(
                    str(
                        safe_call(project.GetSetting, "timelineDropFrameTimecode", default="0")
                    ).lower()
                    in {"1", "true", "yes"}
                    or ";" in timeline_start_tc
                )
                self.store.upsert_timeline(
                    TimelineRecord(
                        timeline_uid=timeline_uid,
                        project_uid=signature["project_uid"],
                        timeline_name=timeline_name,
                        timeline_index=timeline_index,
                        clip_count=0,
                    )
                )

                track_count = safe_call(timeline.GetTrackCount, "video", default=0) or 0
                timeline_clip_count = 0
                for track_index in range(1, track_count + 1):
                    if cancellation_callback and cancellation_callback():
                        raise IndexingCancelledError("Indexing stopped by user.")
                    track_name = safe_call(
                        timeline.GetTrackName,
                        "video",
                        track_index,
                        default=f"V{track_index}",
                    ) or f"V{track_index}"
                    items = safe_call(
                        timeline.GetItemListInTrack,
                        "video",
                        track_index,
                        default=[],
                    ) or []
                    ordered_items = sorted(
                        list(items),
                        key=lambda item: safe_call(item.GetStart, default=0) or 0,
                    )
                    for item_index, item in enumerate(ordered_items, start=1):
                        if cancellation_callback and cancellation_callback():
                            raise IndexingCancelledError("Indexing stopped by user.")
                        clip_label = safe_call(item.GetName) or f"Clip {item_index}"
                        state.current_timeline = timeline_name
                        state.active_clip_index = current_indexed + 1
                        state.active_clip_name = clip_label
                        state.message = (
                            f'Detecting objects {current_indexed + 1}/{total_target_clips}: {clip_label}'
                        )
                        if progress_callback:
                            progress_callback(state)

                        build_result = self._build_clip_record(
                            project=project,
                            signature=signature,
                            timeline=timeline,
                            timeline_uid=timeline_uid,
                            timeline_name=timeline_name,
                            timeline_index=timeline_index,
                            timeline_start_frame=timeline_start_frame,
                            timeline_start_tc=timeline_start_tc,
                            timeline_fps=timeline_fps,
                            drop_frame=drop_frame,
                            track_index=track_index,
                            track_name=track_name,
                            item_index=item_index,
                            item=item,
                            timeline_markers=timeline_markers,
                            existing_cache_by_clip_id=existing_cache_by_clip_id,
                            existing_cache_by_content_signature=existing_cache_by_content_signature,
                            quick_mode=quick_mode,
                        )
                        clip = build_result.clip
                        self.store.upsert_clip(clip, indexed_at=run_indexed_at)
                        seen_clip_ids.add(clip.clip_id)
                        if build_result.enrichment_task:
                            enrichment_tasks.append(build_result.enrichment_task)
                        timeline_clip_count += 1
                        current_indexed += 1
                        state.processed_clips = current_indexed
                        if total_target_clips:
                            state.progress = min(current_indexed / total_target_clips, 1.0)
                        state.message = f'Indexed {current_indexed}/{total_target_clips} clips'
                        state.latest_clip = clip_record_to_result(clip)
                        state.latest_clip_stage = "Completed clip"
                        if progress_callback:
                            progress_callback(state)
                        if session_action_callback:
                            session_action_callback(_resolve, project)

                if cancellation_callback and cancellation_callback():
                    raise IndexingCancelledError("Indexing stopped by user.")

                self.store.upsert_timeline(
                    TimelineRecord(
                        timeline_uid=timeline_uid,
                        project_uid=signature["project_uid"],
                        timeline_name=timeline_name,
                        timeline_index=timeline_index,
                        clip_count=timeline_clip_count,
                    )
                )
                seen_timeline_uids.add(timeline_uid)

            if cancellation_callback and cancellation_callback():
                raise IndexingCancelledError("Indexing stopped by user.")
            self.store.cleanup_index_scope(
                project_uid=signature["project_uid"],
                keep_clip_ids=seen_clip_ids,
                keep_timeline_uids=seen_timeline_uids,
                target_timeline_uids=(
                    replace_timeline_uids if (target_uids or target_names) else None
                ),
            )
            self.store.finalize_project_meta(
                ProjectIndexMeta(
                    project_uid=signature["project_uid"],
                    project_name=signature["project_name"],
                    indexed_at=now_iso(),
                    clip_count=signature["clip_count"],
                    timeline_count=signature["timeline_count"],
                    signature_hash=signature["signature_hash"],
                    quick_mode=quick_mode,
                )
            )

        self.resolve.with_project(_index)
        return IndexBuildResult(
            project_meta=project_meta,
            enrichment_tasks=enrichment_tasks,
        )

    def _build_clip_record(
        self,
        *,
        project: Any,
        signature: dict[str, Any],
        timeline: Any,
        timeline_uid: str,
        timeline_name: str,
        timeline_index: int,
        timeline_start_frame: int,
        timeline_start_tc: str,
        timeline_fps: float,
        drop_frame: bool,
        track_index: int,
        track_name: str,
        item_index: int,
        item: Any,
        timeline_markers: list[MarkerInfo],
        existing_cache_by_clip_id: dict[str, dict[str, Any]],
        existing_cache_by_content_signature: dict[str, list[dict[str, Any]]],
        quick_mode: bool,
    ) -> ClipBuildResult:
        media_pool_item = safe_call(item.GetMediaPoolItem)
        clip_props = safe_call(media_pool_item.GetClipProperty, default={}) if media_pool_item else {}
        clip_props = clip_props or {}
        item_props = safe_call(item.GetProperty, default={}) or {}
        metadata = safe_call(media_pool_item.GetMetadata, default={}) if media_pool_item else {}
        metadata = metadata or {}

        start_frame = int(safe_call(item.GetStart, default=0) or 0)
        duration_frames = int(safe_call(item.GetDuration, default=0) or 0)
        end_frame = int(safe_call(item.GetEnd, default=start_frame + duration_frames) or (start_frame + duration_frames))

        clip_name = safe_call(item.GetName)
        if not clip_name and media_pool_item:
            clip_name = safe_call(media_pool_item.GetName)
        clip_name = clip_name or f"Clip {item_index}"
        media_id = safe_call(media_pool_item.GetMediaId) if media_pool_item else None
        item_uid = safe_call(item.GetUniqueId)
        clip_id = item_uid or hashlib.sha1(
            f"{timeline_uid}:{track_index}:{item_index}:{media_id}:{start_frame}:{clip_name}".encode("utf-8")
        ).hexdigest()

        file_path = str(
            lookup_value(clip_props, "File Path", "Clip Path", "Source File", "FilePath")
            or lookup_value(metadata, "File Path", "Clip Path", "Source File")
            or ""
        )
        file_name = Path(file_path).name if file_path else clip_name
        content_signature = build_content_signature(
            file_name=file_name,
            duration_frames=duration_frames,
        )

        fps = parse_float(
            lookup_value(
                clip_props,
                "FPS",
                "Clip Frame Rate",
                "Video Frame Rate",
                "Frame Rate",
            ),
            timeline_fps or 24.0,
        )

        tc_start = timeline_frame_to_timecode(
            timeline_start_frame=timeline_start_frame,
            timeline_start_timecode=timeline_start_tc,
            frame=start_frame,
            timeline_fps=timeline_fps,
            drop_frame=drop_frame,
        )
        tc_end = timeline_frame_to_timecode(
            timeline_start_frame=timeline_start_frame,
            timeline_start_timecode=timeline_start_tc,
            frame=end_frame,
            timeline_fps=timeline_fps,
            drop_frame=drop_frame,
        )

        source_in = lookup_value(item_props, "Start", "Source Start Timecode")
        source_out = lookup_value(item_props, "End", "Source End Timecode")
        resolution = (
            lookup_value(clip_props, "Resolution")
            or self._resolution_from_props(clip_props)
        )
        codec = lookup_value(clip_props, "Codec", "Video Codec")
        clip_color = safe_call(item.GetClipColor)
        if not clip_color and media_pool_item:
            clip_color = safe_call(media_pool_item.GetClipColor)
        transcript = (
            lookup_value(clip_props, "Transcription", "Transcript")
            or lookup_value(metadata, "Transcription", "Transcript")
        )
        if (
            not transcript
            and self.settings.auto_transcribe
            and media_pool_item
            and safe_call(media_pool_item.TranscribeAudio, default=False)
        ):
            transcript = (
                lookup_value(
                    safe_call(media_pool_item.GetClipProperty, default={}) or {},
                    "Transcription",
                    "Transcript",
                )
                or transcript
            )

        item_markers = normalize_markers(safe_call(item.GetMarkers, default={}) or {})
        media_markers = normalize_markers(
            safe_call(media_pool_item.GetMarkers, default={}) if media_pool_item else {}
            or {}
        )
        all_markers = item_markers + media_markers
        near_markers = nearby_markers(
            timeline_markers,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        tags = extract_keyword_tags(
            clip_name=clip_name,
            timeline_name=timeline_name,
            track_name=track_name,
            file_path=file_path,
            markers=all_markers + near_markers,
            transcript=transcript,
        )
        clip_type = infer_clip_type(
            clip_name=clip_name,
            timeline_name=timeline_name,
            track_name=track_name,
            tags=tags,
            transcript=transcript,
            file_path=file_path,
        )
        tags = dedupe(tags + [clip_type])

        background_enrichment = self._supports_background_enrichment(quick_mode=quick_mode)
        desired_vision_signature = (
            "quick:v1" if quick_mode else self.visual_analyzer.cache_signature()
        )
        desired_local_vision_signature = (
            desired_vision_signature
            if not background_enrichment
            else self.visual_analyzer.local_cache_signature()
        )
        analysis_duration_seconds = (max(duration_frames, 0) / fps) if fps else 0.0
        desired_source_signature = build_source_signature(
            clip_name=clip_name,
            file_path=file_path,
            timeline_name=timeline_name,
            track_index=track_index,
            start_frame=start_frame,
            end_frame=end_frame,
            duration_frames=duration_frames,
            fps=fps,
            resolution=str(resolution) if resolution is not None else None,
            codec=str(codec) if codec is not None else None,
            markers=all_markers,
            timeline_markers=near_markers,
            transcript=str(transcript) if transcript is not None else None,
            vision_cache_signature=desired_vision_signature,
        )
        desired_local_source_signature = build_source_signature(
            clip_name=clip_name,
            file_path=file_path,
            timeline_name=timeline_name,
            track_index=track_index,
            start_frame=start_frame,
            end_frame=end_frame,
            duration_frames=duration_frames,
            fps=fps,
            resolution=str(resolution) if resolution is not None else None,
            codec=str(codec) if codec is not None else None,
            markers=all_markers,
            timeline_markers=near_markers,
            transcript=str(transcript) if transcript is not None else None,
            vision_cache_signature=desired_local_vision_signature,
        )

        has_audio = any(
            lookup_value(clip_props, key)
            for key in ("Audio Codec", "Audio Channels", "Sample Rate", "Audio Bit Depth")
        )

        def compose_clip_record(
            *,
            clip_type_value: str,
            tags_value: list[str],
            visual_descriptions_value: list[VisualDescription],
            description_value: str,
            transcript_value: str | None,
            source_signature_value: str,
            vision_cache_signature_value: str,
            thumbnail_data_value: str | None,
            detected_objects_value: list[str],
        ) -> ClipRecord:
            resolved_description = description_value.strip()
            if not resolved_description:
                resolved_description = build_description(
                    clip_name=clip_name,
                    clip_type=clip_type_value,
                    timeline_name=timeline_name,
                    track_name=track_name,
                    tags=tags_value,
                    transcript=transcript_value,
                    visual_descriptions=visual_descriptions_value,
                )

            searchable_text = build_searchable_text(
                clip_name=clip_name,
                file_name=file_name,
                file_path=file_path,
                timeline_name=timeline_name,
                track_name=track_name,
                description=resolved_description,
                transcript=transcript_value,
                tags=tags_value,
                clip_type=clip_type_value,
                markers=all_markers,
                timeline_markers=near_markers,
                visual_descriptions=visual_descriptions_value,
                codec=codec,
                resolution=resolution,
            )

            return ClipRecord(
                clip_id=clip_id,
                content_signature=content_signature,
                vision_cache_signature=vision_cache_signature_value,
                project_uid=signature["project_uid"],
                timeline_uid=timeline_uid,
                timeline_name=timeline_name,
                timeline_index=timeline_index,
                clip_name=clip_name,
                file_path=file_path,
                file_name=file_name,
                track=track_index,
                track_name=track_name,
                item_index=item_index,
                start_frame=start_frame,
                end_frame=end_frame,
                duration_frames=max(duration_frames, 0),
                duration_seconds=analysis_duration_seconds,
                fps=fps,
                start_timecode=tc_start,
                end_timecode=tc_end,
                source_in=str(source_in) if source_in is not None else None,
                source_out=str(source_out) if source_out is not None else None,
                resolution=str(resolution) if resolution is not None else None,
                codec=str(codec) if codec is not None else None,
                clip_color=str(clip_color) if clip_color is not None else None,
                clip_type=clip_type_value,
                has_audio=bool(has_audio),
                description=resolved_description,
                transcript=str(transcript_value) if transcript_value is not None else None,
                detected_objects=detected_objects_value,
                tags=tags_value,
                markers=all_markers,
                timeline_markers=near_markers,
                visual_descriptions=visual_descriptions_value,
                searchable_text=searchable_text,
                source_signature=source_signature_value,
                thumbnail_data=thumbnail_data_value,
            )

        cached = existing_cache_by_clip_id.get(clip_id)
        if cached and cached.get("content_signature") != content_signature:
            cached = None
        if not cached:
            cached_candidates = existing_cache_by_content_signature.get(
                content_signature,
                [],
            )
            cached = cached_candidates[0] if cached_candidates else None
        visual_descriptions: list[VisualDescription]
        description: str
        thumbnail_data: str | None = cached.get("thumbnail_data") if cached else None
        effective_vision_signature = desired_vision_signature
        detected_objects: list[str] = []
        cached_visual_descriptions = [
            VisualDescription(
                frame_offset_sec=float(item["frame_offset_sec"]),
                description=str(item["description"]),
            )
        for item in json.loads(cached["visual_descriptions_json"])
        ] if cached else []
        cached_description = str(cached.get("description") or "") if cached else ""
        cached_vision_signature = str(cached.get("vision_cache_signature") or "") if cached else ""
        cached_detected_objects = (
            [str(item) for item in json.loads(cached.get("detected_objects_json") or "[]")]
            if cached else []
        )
        cached_has_stale_heuristic_visual = (
            bool(cached)
            and not quick_mode
            and desired_vision_signature != "heuristic:v2"
            and looks_like_heuristic_visual(
                description=cached_description,
                visual_descriptions=cached_visual_descriptions,
                clip_type=str(cached.get("clip_type") or clip_type),
                file_path=file_path,
            )
        )
        cached_analysis_reusable = (
            bool(cached)
            and not quick_mode
            and cached_vision_signature == desired_vision_signature
            and not cached_has_stale_heuristic_visual
        )
        cached_local_analysis_reusable = (
            bool(cached)
            and background_enrichment
            and cached_vision_signature == desired_local_vision_signature
            and str(cached.get("source_signature") or "") == desired_local_source_signature
        )

        if (
            cached
            and (
                cached["source_signature"] == desired_source_signature
                or cached_analysis_reusable
            )
        ):
            visual_descriptions = cached_visual_descriptions
            description = cached_description
            transcript = transcript or cached.get("transcript")
            thumbnail_data = cached.get("thumbnail_data")
            clip_type = str(cached.get("clip_type") or clip_type)
            tags = dedupe(tags + json.loads(cached["tags_json"]))
            detected_objects = cached_detected_objects
            if cached_vision_signature:
                effective_vision_signature = cached_vision_signature
            source_signature = str(
                cached.get("source_signature") or desired_source_signature
            )
            return ClipBuildResult(
                clip=compose_clip_record(
                    clip_type_value=clip_type,
                    tags_value=tags,
                    visual_descriptions_value=visual_descriptions,
                    description_value=description,
                    transcript_value=transcript,
                    source_signature_value=source_signature,
                    vision_cache_signature_value=effective_vision_signature,
                    thumbnail_data_value=thumbnail_data,
                    detected_objects_value=detected_objects,
                )
            )
        if cached_local_analysis_reusable:
            visual_descriptions = cached_visual_descriptions
            description = cached_description
            transcript = transcript or cached.get("transcript")
            thumbnail_data = cached.get("thumbnail_data")
            clip_type = str(cached.get("clip_type") or clip_type)
            tags = dedupe(tags + json.loads(cached["tags_json"]))
            detected_objects = cached_detected_objects
            source_signature = desired_local_source_signature
            local_clip = compose_clip_record(
                clip_type_value=clip_type,
                tags_value=tags,
                visual_descriptions_value=visual_descriptions,
                description_value=description,
                transcript_value=transcript,
                source_signature_value=source_signature,
                vision_cache_signature_value=desired_local_vision_signature,
                thumbnail_data_value=thumbnail_data,
                detected_objects_value=detected_objects,
            )
            return ClipBuildResult(
                clip=local_clip,
                enrichment_task=(
                    EnrichmentTask(clip=local_clip)
                    if file_path
                    else None
                ),
            )
        else:
            visual_descriptions = []
            description = ""
            transcript = transcript or (cached.get("transcript") if cached else None)
            actual_vision_signature = desired_vision_signature
            source_signature = desired_source_signature
            if not quick_mode:
                if background_enrichment:
                    analysis = self.visual_analyzer.analyze_local(
                        clip_name=clip_name,
                        clip_type=clip_type,
                        tags=tags,
                        file_path=file_path,
                        duration_seconds=analysis_duration_seconds,
                    )
                    actual_vision_signature = analysis.cache_signature
                    source_signature = desired_local_source_signature
                else:
                    analysis = self.visual_analyzer.analyze(
                        clip_name=clip_name,
                        clip_type=clip_type,
                        tags=tags,
                        file_path=file_path,
                        duration_seconds=analysis_duration_seconds,
                    )
                visual_descriptions = analysis.frame_descriptions
                actual_vision_signature = analysis.cache_signature
                detected_objects = analysis.detected_objects
                if analysis.clip_type_hint:
                    clip_type = analysis.clip_type_hint
                tags = dedupe(tags + analysis.tags + [clip_type])
                description = analysis.summary.strip()
            else:
                tags = dedupe(tags + [clip_type])

            if not description:
                description = build_description(
                    clip_name=clip_name,
                    clip_type=clip_type,
                    timeline_name=timeline_name,
                    track_name=track_name,
                    tags=tags,
                    transcript=transcript,
                    visual_descriptions=visual_descriptions,
                )

            source_signature = build_source_signature(
                clip_name=clip_name,
                file_path=file_path,
                timeline_name=timeline_name,
                track_index=track_index,
                start_frame=start_frame,
                end_frame=end_frame,
                duration_frames=duration_frames,
                fps=fps,
                resolution=str(resolution) if resolution is not None else None,
                codec=str(codec) if codec is not None else None,
                markers=all_markers,
                timeline_markers=near_markers,
                transcript=str(transcript) if transcript is not None else None,
                vision_cache_signature=actual_vision_signature,
            )
            effective_vision_signature = actual_vision_signature
            built_clip = compose_clip_record(
                clip_type_value=clip_type,
                tags_value=tags,
                visual_descriptions_value=visual_descriptions,
                description_value=description,
                transcript_value=transcript,
                source_signature_value=source_signature,
                vision_cache_signature_value=effective_vision_signature,
                thumbnail_data_value=thumbnail_data,
                detected_objects_value=detected_objects,
            )
            return ClipBuildResult(
                clip=built_clip,
                enrichment_task=(
                    EnrichmentTask(clip=built_clip)
                    if background_enrichment and file_path
                    else None
                ),
            )

    @staticmethod
    def _resolution_from_props(clip_props: dict[str, Any]) -> str | None:
        width = lookup_value(clip_props, "Resolution Width", "Width")
        height = lookup_value(clip_props, "Resolution Height", "Height")
        if width and height:
            return f"{width}x{height}"
        return None


class ReindexCoordinator:
    def __init__(self, indexing_service: IndexingService) -> None:
        self.indexing_service = indexing_service
        self._state = ReindexState()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._enrichment_worker: threading.Thread | None = None
        self._listeners: set[queue.Queue[dict[str, Any]]] = set()
        self._jump_requests: queue.Queue[PendingJumpRequest] = queue.Queue()
        self._cancel_requested = threading.Event()

    def snapshot(self) -> ReindexState:
        with self._lock:
            return ReindexState(**self._state.to_dict())

    def start(
        self,
        *,
        timeline_uids: list[str] | None,
        timeline_names: list[str] | None,
        quick_mode: bool,
    ) -> ReindexState:
        with self._lock:
            if self._state.running or self._state.enrichment_running:
                return ReindexState(**self._state.to_dict())

            self._cancel_requested.clear()
            self._state = ReindexState(
                running=True,
                enrichment_running=False,
                progress=0.0,
                enrichment_progress=0.0,
                message="Starting reindex",
                started_at=now_iso(),
                finished_at=None,
                last_error=None,
                current_timeline=None,
                processed_clips=0,
                total_clips=0,
                enriched_clips=0,
                total_enrichment_clips=0,
                active_clip_index=0,
                active_clip_name=None,
                quick_mode=quick_mode,
            )
            self._broadcast_locked()

            self._worker = threading.Thread(
                target=self._run,
                kwargs={
                    "timeline_uids": timeline_uids,
                    "timeline_names": timeline_names,
                    "quick_mode": quick_mode,
                },
                daemon=True,
            )
            self._worker.start()
            return ReindexState(**self._state.to_dict())

    def cancel(self) -> ReindexState:
        with self._lock:
            if not self._state.running and not self._state.enrichment_running:
                return ReindexState(**self._state.to_dict())

            self._cancel_requested.set()
            active_message = (
                "Stopping after current clip..."
                if self._state.running
                else "Stopping Gemini enrichment after current clip..."
            )
            self._state.message = active_message
            self._broadcast_locked()
            return ReindexState(**self._state.to_dict())

    def _is_cancel_requested(self) -> bool:
        return self._cancel_requested.is_set()

    def request_jump(self, clip: dict[str, Any], *, timeout_sec: float = 20.0) -> dict[str, Any]:
        request = PendingJumpRequest(clip=clip)
        self._jump_requests.put(request)
        if not request.done.wait(timeout=timeout_sec):
            raise ResolveConnectionError(
                "Jump request timed out while indexing. Try again in a moment."
            )
        if request.error:
            raise request.error
        if not request.result:
            raise ResolveConnectionError("Jump request did not return a result.")
        return request.result

    def _update(self, state: ReindexState) -> None:
        with self._lock:
            self._state = ReindexState(**state.to_dict())
            self._broadcast_locked()

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        listener: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=8)
        with self._lock:
            self._listeners.add(listener)
            self._push_state(listener, self._state.to_dict())
        return listener

    def unsubscribe(self, listener: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._listeners.discard(listener)

    def _broadcast_locked(self) -> None:
        payload = self._state.to_dict()
        stale: list[queue.Queue[dict[str, Any]]] = []
        for listener in self._listeners:
            try:
                self._push_state(listener, payload)
            except Exception:
                stale.append(listener)
        for listener in stale:
            self._listeners.discard(listener)

    @staticmethod
    def _push_state(listener: queue.Queue[dict[str, Any]], payload: dict[str, Any]) -> None:
        while True:
            try:
                listener.put_nowait(payload)
                return
            except queue.Full:
                try:
                    listener.get_nowait()
                except queue.Empty:
                    return

    def _run(
        self,
        *,
        timeline_uids: list[str] | None,
        timeline_names: list[str] | None,
        quick_mode: bool,
    ) -> None:
        try:
            build_result = self.indexing_service.build_index(
                timeline_uids=timeline_uids,
                timeline_names=timeline_names,
                quick_mode=quick_mode,
                progress_callback=self._update,
                session_action_callback=self._service_pending_jump_requests,
                cancellation_callback=self._is_cancel_requested,
            )
            with self._lock:
                cancelled = self._cancel_requested.is_set()
                self._state.running = False
                self._state.progress = 1.0 if not cancelled else min(self._state.progress, 0.999)
                self._state.active_clip_index = 0
                self._state.active_clip_name = None
                if cancelled:
                    self._state.enrichment_running = False
                    self._state.enrichment_progress = 0.0
                    self._state.message = "Indexing stopped"
                    self._state.finished_at = now_iso()
                    self._broadcast_locked()
                elif build_result.enrichment_tasks:
                    self._state.enrichment_running = True
                    self._state.enrichment_progress = 0.0
                    self._state.enriched_clips = 0
                    self._state.total_enrichment_clips = len(build_result.enrichment_tasks)
                    self._state.message = (
                        f"Index ready, Gemini enriching 0/{len(build_result.enrichment_tasks)} clips"
                    )
                    self._broadcast_locked()
                    self._enrichment_worker = threading.Thread(
                        target=self._run_enrichment,
                        kwargs={
                            "tasks": build_result.enrichment_tasks,
                            "project_meta": build_result.project_meta,
                        },
                        daemon=True,
                    )
                    self._enrichment_worker.start()
                else:
                    self._state.message = "Index complete"
                    self._state.finished_at = now_iso()
                    self._broadcast_locked()
            self._fail_pending_jump_requests(
                ResolveConnectionError("Indexing finished before the jump request was handled.")
            )
        except IndexingCancelledError:
            with self._lock:
                self._state.running = False
                self._state.enrichment_running = False
                self._state.message = "Indexing stopped"
                self._state.finished_at = now_iso()
                self._state.active_clip_index = 0
                self._state.active_clip_name = None
                self._broadcast_locked()
            self._fail_pending_jump_requests(
                ResolveConnectionError("Indexing was stopped before the jump request was handled.")
            )
        except Exception as exc:
            with self._lock:
                self._state.running = False
                self._state.enrichment_running = False
                self._state.last_error = str(exc)
                self._state.message = "Indexing failed"
                self._state.finished_at = now_iso()
                self._state.active_clip_index = 0
                self._state.active_clip_name = None
                self._broadcast_locked()
            self._fail_pending_jump_requests(
                exc if isinstance(exc, ResolveConnectionError) else ResolveConnectionError(str(exc))
            )

    def _enrich_single(self, task: EnrichmentTask) -> tuple[EnrichmentTask, ClipRecord | None, str | None]:
        try:
            enriched = self.indexing_service.enrich_clip(task.clip)
            return (task, enriched, None)
        except Exception as exc:
            return (task, None, str(exc))

    def _run_enrichment(
        self,
        *,
        tasks: list[EnrichmentTask],
        project_meta: ProjectIndexMeta,
    ) -> None:
        workers = getattr(self.indexing_service.settings, "enrichment_workers", 4)
        try:
            completed_count = 0
            total = len(tasks)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {}
                for task in tasks:
                    if self._is_cancel_requested():
                        break
                    future = pool.submit(self._enrich_single, task)
                    futures[future] = task

                for future in as_completed(futures):
                    if self._is_cancel_requested():
                        for pending in futures:
                            pending.cancel()
                        break

                    task, enriched_clip, error = future.result()
                    completed_count += 1

                    if error:
                        with self._lock:
                            self._state.last_error = error

                    if enriched_clip:
                        self.indexing_service.store.upsert_clip(
                            enriched_clip,
                            indexed_at=now_iso(),
                        )
                        with self._lock:
                            self._state.latest_clip = clip_record_to_result(enriched_clip)
                            self._state.latest_clip_stage = "Gemini enrichment complete"

                    with self._lock:
                        clip_name = task.clip.file_name or task.clip.clip_name
                        self._state.enriched_clips = completed_count
                        self._state.total_enrichment_clips = total
                        self._state.enrichment_progress = min(completed_count / total, 1.0)
                        self._state.active_clip_index = completed_count
                        self._state.active_clip_name = clip_name
                        self._state.message = f"Gemini enriched {completed_count}/{total} clips"
                        self._broadcast_locked()

            if self._is_cancel_requested():
                with self._lock:
                    self._state.enrichment_running = False
                    self._state.message = "Gemini enrichment stopped"
                    self._state.finished_at = now_iso()
                    self._state.active_clip_index = 0
                    self._state.active_clip_name = None
                    self._broadcast_locked()
                return

            self.indexing_service.store.finalize_project_meta(
                ProjectIndexMeta(
                    project_uid=project_meta.project_uid,
                    project_name=project_meta.project_name,
                    indexed_at=now_iso(),
                    clip_count=project_meta.clip_count,
                    timeline_count=project_meta.timeline_count,
                    signature_hash=project_meta.signature_hash,
                    quick_mode=project_meta.quick_mode,
                )
            )
            with self._lock:
                self._state.enrichment_running = False
                self._state.enrichment_progress = 1.0 if tasks else 0.0
                self._state.message = "Index and Gemini enrichment complete"
                self._state.finished_at = now_iso()
                self._state.active_clip_index = 0
                self._state.active_clip_name = None
                self._broadcast_locked()
        except Exception as exc:
            with self._lock:
                self._state.enrichment_running = False
                self._state.last_error = str(exc)
                self._state.message = "Gemini enrichment failed"
                self._state.finished_at = now_iso()
                self._state.active_clip_index = 0
                self._state.active_clip_name = None
                self._broadcast_locked()

    def _service_pending_jump_requests(self, resolve: Any, project: Any) -> None:
        while True:
            try:
                request = self._jump_requests.get_nowait()
            except queue.Empty:
                return

            try:
                request.result = self.indexing_service.resolve.jump_to_clip_in_project(
                    resolve=resolve,
                    project=project,
                    clip_id=request.clip["clip_id"],
                    timeline_uid=request.clip["timeline_uid"],
                    timeline_name=request.clip["timeline_name"],
                    start_timecode=request.clip["start_timecode"],
                    clip_name=request.clip.get("clip_name"),
                    file_path=request.clip.get("file_path"),
                    duration_frames=request.clip.get("duration_frames"),
                    track_index=request.clip.get("track"),
                )
            except Exception as exc:
                request.error = (
                    exc if isinstance(exc, ResolveConnectionError) else ResolveConnectionError(str(exc))
                )
            finally:
                request.done.set()

    def _fail_pending_jump_requests(self, error: ResolveConnectionError) -> None:
        while True:
            try:
                request = self._jump_requests.get_nowait()
            except queue.Empty:
                return
            request.error = error
            request.done.set()


@dataclass(slots=True)
class PendingJumpRequest:
    clip: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: ResolveConnectionError | None = None
