from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .timecode import timecode_to_frames, timeline_frame_to_timecode


class ResolveConnectionError(RuntimeError):
    """Raised when Resolve or its scripting API cannot be reached."""


def safe_call(func: Callable[..., Any], *args: Any, default: Any = None) -> Any:
    try:
        return func(*args)
    except Exception:
        return default


def _parse_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def _lookup_value(mapping: dict[str, Any], *keys: str) -> Any:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def _normalize_path(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip().replace("\\", "/").lower()


def _default_paths() -> tuple[str | None, str | None]:
    system = platform.system()
    if system == "Darwin":
        return (
            "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting",
            "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so",
        )
    if system == "Windows":
        return (
            os.path.join(
                os.getenv("PROGRAMDATA", ""),
                "Blackmagic Design",
                "DaVinci Resolve",
                "Support",
                "Developer",
                "Scripting",
            ),
            r"C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll",
        )
    if system == "Linux":
        return (
            "/opt/resolve/Developer/Scripting",
            "/opt/resolve/libs/Fusion/fusionscript.so",
        )
    return None, None


def _bootstrap_resolve_module() -> Any:
    script_api, script_lib = _default_paths()
    if script_api and "RESOLVE_SCRIPT_API" not in os.environ:
        os.environ["RESOLVE_SCRIPT_API"] = script_api
    if script_lib and "RESOLVE_SCRIPT_LIB" not in os.environ:
        os.environ["RESOLVE_SCRIPT_LIB"] = script_lib

    resolved_api = os.getenv("RESOLVE_SCRIPT_API")
    if resolved_api:
        modules_path = str(Path(resolved_api) / "Modules")
        if modules_path not in sys.path:
            sys.path.append(modules_path)

    try:
        return importlib.import_module("DaVinciResolveScript")
    except ImportError as exc:  # pragma: no cover - requires local Resolve install
        raise ResolveConnectionError(
            "Unable to import DaVinci Resolve's scripting API. "
            "Check Resolve Preferences > General > External Scripting and confirm "
            "RESOLVE_SCRIPT_API / RESOLVE_SCRIPT_LIB point to your Resolve install."
        ) from exc


@dataclass(slots=True)
class ResolveStatus:
    connected: bool
    message: str
    project_name: str | None = None
    project_uid: str | None = None
    current_timeline_name: str | None = None
    current_timeline_uid: str | None = None
    current_page: str | None = None
    version_string: str | None = None


class ResolveFacade:
    def __init__(self) -> None:
        self._lock = threading.RLock()

    def _connect(self) -> tuple[Any, Any, Any]:
        module = _bootstrap_resolve_module()
        resolve = module.scriptapp("Resolve")
        if not resolve:
            raise ResolveConnectionError(
                "DaVinci Resolve is not running or scripting access is disabled."
            )

        project_manager = resolve.GetProjectManager()
        if not project_manager:
            raise ResolveConnectionError("Resolve is running, but the project manager is unavailable.")

        project = project_manager.GetCurrentProject()
        if not project:
            raise ResolveConnectionError("Resolve is connected, but no project is currently open.")

        return resolve, project_manager, project

    def with_project(self, callback: Callable[[Any, Any, Any], Any]) -> Any:
        with self._lock:
            resolve, project_manager, project = self._connect()
            return callback(resolve, project_manager, project)

    def get_status(self) -> ResolveStatus:
        try:
            def _read(resolve: Any, _project_manager: Any, project: Any) -> ResolveStatus:
                current_timeline = safe_call(project.GetCurrentTimeline)
                return ResolveStatus(
                    connected=True,
                    message="Resolve connected",
                    project_name=safe_call(project.GetName),
                    project_uid=safe_call(project.GetUniqueId),
                    current_timeline_name=safe_call(current_timeline.GetName) if current_timeline else None,
                    current_timeline_uid=safe_call(current_timeline.GetUniqueId) if current_timeline else None,
                    current_page=safe_call(resolve.GetCurrentPage),
                    version_string=safe_call(resolve.GetVersionString),
                )

            return self.with_project(_read)
        except ResolveConnectionError as exc:
            return ResolveStatus(connected=False, message=str(exc))

    def compute_project_signature(self) -> dict[str, Any]:
        def _build(_resolve: Any, _project_manager: Any, project: Any) -> dict[str, Any]:
            timelines: list[dict[str, Any]] = []
            total_clips = 0
            timeline_count = safe_call(project.GetTimelineCount, default=0) or 0
            for timeline_index in range(1, timeline_count + 1):
                timeline = safe_call(project.GetTimelineByIndex, timeline_index)
                if not timeline:
                    continue
                track_summaries: list[dict[str, Any]] = []
                track_count = safe_call(timeline.GetTrackCount, "video", default=0) or 0
                timeline_clip_count = 0
                for track_index in range(1, track_count + 1):
                    items = safe_call(
                        timeline.GetItemListInTrack,
                        "video",
                        track_index,
                        default=[],
                    ) or []
                    item_count = len(items)
                    timeline_clip_count += item_count
                    track_summaries.append(
                        {
                            "track": track_index,
                            "track_name": safe_call(
                                timeline.GetTrackName,
                                "video",
                                track_index,
                                default=f"V{track_index}",
                            )
                            or f"V{track_index}",
                            "count": item_count,
                        }
                    )

                timeline_uid = safe_call(timeline.GetUniqueId) or hashlib.sha1(
                    f"{safe_call(project.GetUniqueId)}::{safe_call(timeline.GetName)}".encode("utf-8")
                ).hexdigest()
                total_clips += timeline_clip_count
                timelines.append(
                    {
                        "timeline_uid": timeline_uid,
                        "timeline_name": safe_call(timeline.GetName) or f"Timeline {timeline_index}",
                        "timeline_index": timeline_index,
                        "track_summaries": track_summaries,
                        "clip_count": timeline_clip_count,
                    }
                )

            payload = {
                "project_uid": safe_call(project.GetUniqueId) or safe_call(project.GetName),
                "project_name": safe_call(project.GetName),
                "timeline_count": len(timelines),
                "clip_count": total_clips,
                "timelines": timelines,
            }
            payload["signature_hash"] = hashlib.sha1(
                json.dumps(payload, sort_keys=True).encode("utf-8")
            ).hexdigest()
            return payload

        return self.with_project(_build)

    def jump_to_clip(
        self,
        *,
        clip_id: str,
        timeline_uid: str | None,
        timeline_name: str,
        start_timecode: str,
        clip_name: str | None = None,
        file_path: str | None = None,
        duration_frames: int | None = None,
        track_index: int | None = None,
    ) -> dict[str, Any]:
        def _jump(resolve: Any, _project_manager: Any, project: Any) -> dict[str, Any]:
            timeline_count = safe_call(project.GetTimelineCount, default=0) or 0
            preferred_timelines: list[tuple[Any, str | None, str]] = []
            other_timelines: list[tuple[Any, str | None, str]] = []
            fallback_timeline = None
            for timeline_index in range(1, timeline_count + 1):
                timeline = safe_call(project.GetTimelineByIndex, timeline_index)
                if not timeline:
                    continue
                candidate_uid = safe_call(timeline.GetUniqueId)
                candidate_name = safe_call(timeline.GetName) or f"Timeline {timeline_index}"
                if timeline_uid and candidate_uid == timeline_uid:
                    preferred_timelines.insert(0, (timeline, candidate_uid, candidate_name))
                    fallback_timeline = timeline
                    continue
                if candidate_name == timeline_name:
                    preferred_timelines.append((timeline, candidate_uid, candidate_name))
                    fallback_timeline = fallback_timeline or timeline
                    continue
                other_timelines.append((timeline, candidate_uid, candidate_name))

            if not fallback_timeline and preferred_timelines:
                fallback_timeline = preferred_timelines[0][0]

            if not fallback_timeline:
                raise ResolveConnectionError(
                    f'Unable to find timeline "{timeline_name}" in the current project.'
                )

            resolved_target = self._find_live_clip_location(
                project=project,
                timelines=preferred_timelines + other_timelines,
                clip_id=clip_id,
                clip_name=clip_name,
                file_path=file_path,
                duration_frames=duration_frames,
                track_index=track_index,
                indexed_start_timecode=start_timecode,
            )

            target_timeline = resolved_target["timeline"] if resolved_target else fallback_timeline
            target_timeline_name = (
                resolved_target["timeline_name"] if resolved_target else timeline_name
            )
            target_timecode = (
                resolved_target["start_timecode"] if resolved_target else start_timecode
            )

            if not safe_call(project.SetCurrentTimeline, target_timeline, default=False):
                raise ResolveConnectionError(
                    f'Failed to switch Resolve to timeline "{target_timeline_name}".'
                )

            safe_call(resolve.OpenPage, "edit", default=False)
            if not safe_call(
                target_timeline.SetCurrentTimecode,
                target_timecode,
                default=False,
            ):
                raise ResolveConnectionError(
                    f'Failed to move the playhead to {target_timecode} in "{target_timeline_name}".'
                )

            return {
                "timeline_name": target_timeline_name,
                "start_timecode": target_timecode,
            }

        return self.with_project(_jump)

    def _find_live_clip_location(
        self,
        *,
        project: Any,
        timelines: list[tuple[Any, str | None, str]],
        clip_id: str,
        clip_name: str | None,
        file_path: str | None,
        duration_frames: int | None,
        track_index: int | None,
        indexed_start_timecode: str,
    ) -> dict[str, Any] | None:
        normalized_clip_name = (clip_name or "").strip().lower()
        normalized_file_path = _normalize_path(file_path)
        best_fallback: tuple[tuple[int, int, int], dict[str, Any]] | None = None

        for timeline, candidate_uid, candidate_name in timelines:
            track_count = safe_call(timeline.GetTrackCount, "video", default=0) or 0
            for candidate_track_index in range(1, track_count + 1):
                items = safe_call(
                    timeline.GetItemListInTrack,
                    "video",
                    candidate_track_index,
                    default=[],
                ) or []
                for item in items:
                    item_uid = safe_call(item.GetUniqueId)
                    if item_uid and item_uid == clip_id:
                        start_timecode = self._timeline_item_start_timecode(
                            project=project,
                            timeline=timeline,
                            item=item,
                        )
                        return {
                            "timeline": timeline,
                            "timeline_uid": candidate_uid,
                            "timeline_name": candidate_name,
                            "start_timecode": start_timecode,
                        }

                    fallback_score = self._fallback_match_score(
                        item=item,
                        expected_clip_name=normalized_clip_name,
                        expected_file_path=normalized_file_path,
                        expected_duration_frames=duration_frames,
                        expected_track_index=track_index,
                        candidate_track_index=candidate_track_index,
                    )
                    if fallback_score <= 0:
                        continue

                    start_timecode = self._timeline_item_start_timecode(
                        project=project,
                        timeline=timeline,
                        item=item,
                    )
                    frame_delta = self._frame_delta(
                        indexed_timecode=indexed_start_timecode,
                        candidate_timecode=start_timecode,
                        fps=_parse_float(
                            safe_call(project.GetSetting, "timelineFrameRate", default=None),
                            24.0,
                        ),
                    )
                    ranking = (
                        fallback_score,
                        -frame_delta,
                        -candidate_track_index,
                    )
                    candidate = {
                        "timeline": timeline,
                        "timeline_uid": candidate_uid,
                        "timeline_name": candidate_name,
                        "start_timecode": start_timecode,
                    }
                    if not best_fallback or ranking > best_fallback[0]:
                        best_fallback = (ranking, candidate)

        return best_fallback[1] if best_fallback else None

    @staticmethod
    def _frame_delta(*, indexed_timecode: str, candidate_timecode: str, fps: float) -> int:
        try:
            return abs(
                timecode_to_frames(candidate_timecode, fps)
                - timecode_to_frames(indexed_timecode, fps)
            )
        except Exception:
            return 0

    @staticmethod
    def _fallback_match_score(
        *,
        item: Any,
        expected_clip_name: str,
        expected_file_path: str,
        expected_duration_frames: int | None,
        expected_track_index: int | None,
        candidate_track_index: int,
    ) -> int:
        media_pool_item = safe_call(item.GetMediaPoolItem)
        clip_props = safe_call(media_pool_item.GetClipProperty, default={}) if media_pool_item else {}
        metadata = safe_call(media_pool_item.GetMetadata, default={}) if media_pool_item else {}
        item_name = (
            safe_call(item.GetName)
            or (safe_call(media_pool_item.GetName) if media_pool_item else None)
            or ""
        ).strip().lower()
        item_file_path = _normalize_path(
            str(
                _lookup_value(clip_props, "File Path", "Clip Path", "Source File", "FilePath")
                or _lookup_value(metadata or {}, "File Path", "Clip Path", "Source File")
                or ""
            )
        )
        item_duration_frames = int(safe_call(item.GetDuration, default=0) or 0)

        score = 0
        if expected_file_path and item_file_path == expected_file_path:
            score += 100
        if expected_clip_name and item_name == expected_clip_name:
            score += 20
        if expected_duration_frames is not None and item_duration_frames == expected_duration_frames:
            score += 12
        if (
            expected_track_index is not None
            and expected_track_index > 0
            and expected_track_index == candidate_track_index
        ):
            # Same-track candidates are slightly more likely to be the original clip.
            score += 3
        return score

    @staticmethod
    def _timeline_item_start_timecode(*, project: Any, timeline: Any, item: Any) -> str:
        timeline_start_frame = int(safe_call(timeline.GetStartFrame, default=0) or 0)
        timeline_start_tc = (
            safe_call(timeline.GetStartTimecode, default="00:00:00:00")
            or "00:00:00:00"
        )
        fps = _parse_float(
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
        start_frame = int(safe_call(item.GetStart, default=0) or 0)
        return timeline_frame_to_timecode(
            timeline_start_frame=timeline_start_frame,
            timeline_start_timecode=timeline_start_tc,
            frame=start_frame,
            timeline_fps=fps,
            drop_frame=drop_frame,
        )
