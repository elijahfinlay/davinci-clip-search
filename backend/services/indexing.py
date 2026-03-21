from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from backend.config import Settings

from .index_store import IndexStore
from .resolve_api import ResolveFacade, ResolveConnectionError, safe_call
from .timecode import timeline_frame_to_timecode
from .types import (
    ClipRecord,
    MarkerInfo,
    ProjectIndexMeta,
    ReindexState,
    TimelineRecord,
    VisualDescription,
)
from .vision import build_visual_analyzer


ProgressCallback = Callable[[ReindexState], None]


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

    def build_index(
        self,
        *,
        timeline_uids: list[str] | None,
        timeline_names: list[str] | None,
        quick_mode: bool,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
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

        def _index(_resolve: Any, _project_manager: Any, project: Any) -> None:
            clip_records: list[ClipRecord] = []
            timeline_records: list[TimelineRecord] = []

            timeline_count = safe_call(project.GetTimelineCount, default=0) or 0
            current_indexed = 0
            for timeline_index in range(1, timeline_count + 1):
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

                track_count = safe_call(timeline.GetTrackCount, "video", default=0) or 0
                timeline_clip_count = 0
                for track_index in range(1, track_count + 1):
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
                        clip = self._build_clip_record(
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
                        clip_records.append(clip)
                        timeline_clip_count += 1
                        current_indexed += 1
                        state.processed_clips = current_indexed
                        if total_target_clips:
                            state.progress = min(current_indexed / total_target_clips, 1.0)
                        state.message = f'Indexed {current_indexed}/{total_target_clips} clips'
                        if progress_callback:
                            progress_callback(state)

                timeline_records.append(
                    TimelineRecord(
                        timeline_uid=timeline_uid,
                        project_uid=signature["project_uid"],
                        timeline_name=timeline_name,
                        timeline_index=timeline_index,
                        clip_count=timeline_clip_count,
                    )
                )

            project_meta = ProjectIndexMeta(
                project_uid=signature["project_uid"],
                project_name=signature["project_name"],
                indexed_at=now_iso(),
                clip_count=signature["clip_count"],
                timeline_count=signature["timeline_count"],
                signature_hash=signature["signature_hash"],
                quick_mode=quick_mode,
            )

            self.store.replace_index(
                project_meta,
                timeline_records,
                clip_records,
                replace_timeline_uids=replace_timeline_uids if (target_uids or target_names) else None,
            )

        self.resolve.with_project(_index)

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
    ) -> ClipRecord:
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

        desired_vision_signature = (
            "quick:v1" if quick_mode else self.visual_analyzer.cache_signature()
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
            vision_cache_signature=desired_vision_signature,
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
        cached_visual_descriptions = [
            VisualDescription(
                frame_offset_sec=float(item["frame_offset_sec"]),
                description=str(item["description"]),
            )
            for item in json.loads(cached["visual_descriptions_json"])
        ] if cached else []
        cached_description = str(cached.get("description") or "") if cached else ""
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

        if (
            cached
            and cached["source_signature"] == source_signature
            and not cached_has_stale_heuristic_visual
        ):
            visual_descriptions = cached_visual_descriptions
            description = cached_description
            transcript = transcript or cached.get("transcript")
            thumbnail_data = cached.get("thumbnail_data")
            clip_type = str(cached.get("clip_type") or clip_type)
            tags = dedupe(tags + json.loads(cached["tags_json"]))
        else:
            visual_descriptions = []
            description = ""
            transcript = transcript or (cached.get("transcript") if cached else None)
            actual_vision_signature = desired_vision_signature
            if not quick_mode:
                analysis = self.visual_analyzer.analyze(
                    clip_name=clip_name,
                    clip_type=clip_type,
                    tags=tags,
                    file_path=file_path,
                    duration_seconds=(max(duration_frames, 0) / fps) if fps else 0.0,
                )
                visual_descriptions = analysis.frame_descriptions
                actual_vision_signature = analysis.cache_signature
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

        searchable_text = build_searchable_text(
            clip_name=clip_name,
            file_name=file_name,
            file_path=file_path,
            timeline_name=timeline_name,
            track_name=track_name,
            description=description,
            transcript=transcript,
            tags=tags,
            clip_type=clip_type,
            markers=all_markers,
            timeline_markers=near_markers,
            visual_descriptions=visual_descriptions,
            codec=codec,
            resolution=resolution,
        )

        has_audio = any(
            lookup_value(clip_props, key)
            for key in ("Audio Codec", "Audio Channels", "Sample Rate", "Audio Bit Depth")
        )

        return ClipRecord(
            clip_id=clip_id,
            content_signature=content_signature,
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
            duration_seconds=(max(duration_frames, 0) / fps) if fps else 0.0,
            fps=fps,
            start_timecode=tc_start,
            end_timecode=tc_end,
            source_in=str(source_in) if source_in is not None else None,
            source_out=str(source_out) if source_out is not None else None,
            resolution=str(resolution) if resolution is not None else None,
            codec=str(codec) if codec is not None else None,
            clip_color=str(clip_color) if clip_color is not None else None,
            clip_type=clip_type,
            has_audio=bool(has_audio),
            description=description,
            transcript=str(transcript) if transcript is not None else None,
            tags=tags,
            markers=all_markers,
            timeline_markers=near_markers,
            visual_descriptions=visual_descriptions,
            searchable_text=searchable_text,
            source_signature=source_signature,
            thumbnail_data=thumbnail_data,
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
            if self._state.running:
                return ReindexState(**self._state.to_dict())

            self._state = ReindexState(
                running=True,
                progress=0.0,
                message="Starting reindex",
                started_at=now_iso(),
                finished_at=None,
                last_error=None,
                current_timeline=None,
                processed_clips=0,
                total_clips=0,
                quick_mode=quick_mode,
            )

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

    def _update(self, state: ReindexState) -> None:
        with self._lock:
            self._state = ReindexState(**state.to_dict())

    def _run(
        self,
        *,
        timeline_uids: list[str] | None,
        timeline_names: list[str] | None,
        quick_mode: bool,
    ) -> None:
        try:
            self.indexing_service.build_index(
                timeline_uids=timeline_uids,
                timeline_names=timeline_names,
                quick_mode=quick_mode,
                progress_callback=self._update,
            )
            with self._lock:
                self._state.running = False
                self._state.progress = 1.0
                self._state.message = "Index complete"
                self._state.finished_at = now_iso()
        except Exception as exc:
            with self._lock:
                self._state.running = False
                self._state.last_error = str(exc)
                self._state.message = "Indexing failed"
                self._state.finished_at = now_iso()
