from __future__ import annotations

import ast
import base64
import io
import json
import logging
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from backend.config import Settings

from .types import VisionAnalysis, VisualDescription

try:  # pragma: no cover - exercised in runtime environment
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - safe fallback when dependency missing
    Anthropic = None

try:  # pragma: no cover - exercised in runtime environment
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - safe fallback when dependency missing
    genai = None
    genai_types = None

try:  # pragma: no cover - exercised in runtime environment
    import numpy as np
    from PIL import Image
    from ultralytics import YOLOWorld
except ImportError:  # pragma: no cover - safe fallback when dependency missing
    np = None
    Image = None
    YOLOWorld = None


LOGGER = logging.getLogger(__name__)

LEGACY_VISION_SYSTEM_PROMPT = """
You generate concise visual search metadata for DaVinci Resolve timeline clips.
Use only what is visible in the provided frames. Do not identify people by name.
Return valid JSON only, with no markdown fences or extra commentary.
""".strip()

LEGACY_VISION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
        },
        "clip_type_hint": {"type": "string"},
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "frame_offset_sec": {"type": "number"},
                    "description": {"type": "string"},
                },
                "required": ["frame_offset_sec", "description"],
            },
        },
    },
    "required": ["summary", "tags", "clip_type_hint", "frames"],
}

GEMINI_GUIDED_SYSTEM_PROMPT = """
You refine clip search metadata for a single video frame.
Use the provided object detections as already-known facts, and only add what is clearly visible in the frame.
Do not identify people by name.
Return valid JSON only, with no markdown fences or extra commentary.
""".strip()

GEMINI_GUIDED_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "shot_type": {"type": "string"},
        "camera_movement": {"type": "string"},
        "lighting": {"type": "string"},
        "additional_subjects_or_objects": {
            "type": "array",
            "items": {"type": "string"},
        },
        "clip_type_hint": {"type": "string"},
    },
    "required": [
        "shot_type",
        "camera_movement",
        "lighting",
        "additional_subjects_or_objects",
        "clip_type_hint",
    ],
}

SHORT_CLIP_SINGLE_FRAME_THRESHOLD_SEC = 15.0
LONG_CLIP_FRAME_STEP_SEC = 10.0
LEGACY_MULTI_FRAME_SIGNATURE = "middle-short_or_midpoint-10s-segments:v1"
YOLO_WORLD_FRAME_SIGNATURE = "three-even-segment-midpoints:v1"
GEMINI_SINGLE_FRAME_SIGNATURE = "single-middle-frame:v1"
YOLO_GEMINI_PIPELINE_SIGNATURE = "yolo-world-3f-plus-gemini-1f:v1"
YOLO_PARTIAL_STAGE_SIGNATURE = "yolo-objects-partial:v1"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = item.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _canonical_clip_type(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"ground", "handheld"}:
        return "handheld"
    if normalized in {"drone", "interview"}:
        return normalized
    return None


def _extract_json_block(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Vision response did not contain a JSON object.")
    block = cleaned[start : end + 1]
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(block)
        if not isinstance(parsed, dict):
            raise ValueError("Vision response JSON object could not be parsed.")
        return parsed


GUIDED_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "shot_type": ("shot_type", "shot type"),
    "camera_movement": ("camera_movement", "camera movement"),
    "lighting": ("lighting",),
    "additional_subjects_or_objects": (
        "additional_subjects_or_objects",
        "additional subjects or objects",
        "additional subjects",
        "additional objects",
    ),
    "clip_type_hint": ("clip_type_hint", "clip type hint"),
}

GUIDED_FIELD_ALIAS_TO_KEY = {
    alias.lower(): key
    for key, aliases in GUIDED_FIELD_ALIASES.items()
    for alias in aliases
}

GUIDED_FIELD_PATTERN = re.compile(
    r'(?is)(?:^|[\n\r,{])\s*[-*]?\s*["\']?(?P<label>'
    + "|".join(
        sorted(
            (re.escape(alias) for alias in GUIDED_FIELD_ALIAS_TO_KEY),
            key=len,
            reverse=True,
        )
    )
    + r')["\']?\s*[:=-]'
)


def _clean_guided_scalar(value: str) -> str:
    cleaned = value.strip().strip(",").strip()
    if "," in cleaned:
        cleaned = cleaned.split(",", 1)[0]
    cleaned = cleaned.strip("{}[]")
    cleaned = cleaned.strip().strip("\"'`")
    cleaned = cleaned.strip()
    if cleaned.lower() in {"none", "n/a", "null"}:
        return ""
    return cleaned


def _split_guided_list(value: str) -> list[str]:
    cleaned = value.strip().strip(",").strip()
    if not cleaned:
        return []

    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(cleaned)
            except Exception:
                parsed = None
        if isinstance(parsed, list):
            return _dedupe(
                [
                    _clean_guided_scalar(str(item))
                    for item in parsed
                    if _clean_guided_scalar(str(item))
                ]
            )

    tokens = [
        _clean_guided_scalar(token)
        for token in re.split(r"[\n,;]", cleaned)
    ]
    normalized = [
        re.sub(r"^[-*]\s*", "", token).strip()
        for token in tokens
        if token and token.lower() not in {"none", "n/a", "null", "unknown"}
    ]
    return _dedupe([token for token in normalized if token])


def _extract_guided_payload(text: str) -> dict[str, Any]:
    try:
        return _extract_json_block(text)
    except Exception:
        pass

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    matches = list(GUIDED_FIELD_PATTERN.finditer(cleaned))
    if not matches:
        raise ValueError("Vision response did not contain a JSON object.")

    raw_fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = GUIDED_FIELD_ALIAS_TO_KEY[match.group("label").lower()]
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        raw_value = cleaned[match.end() : value_end].strip().strip(",").strip()
        raw_fields.setdefault(key, raw_value)

    payload = {
        "shot_type": _clean_guided_scalar(raw_fields.get("shot_type", "")),
        "camera_movement": _clean_guided_scalar(raw_fields.get("camera_movement", "")),
        "lighting": _clean_guided_scalar(raw_fields.get("lighting", "")),
        "additional_subjects_or_objects": _split_guided_list(
            raw_fields.get("additional_subjects_or_objects", "")
        ),
        "clip_type_hint": _clean_guided_scalar(raw_fields.get("clip_type_hint", "")),
    }
    if not any(
        [
            payload.get("shot_type"),
            payload.get("camera_movement"),
            payload.get("lighting"),
            payload.get("additional_subjects_or_objects"),
            payload.get("clip_type_hint"),
        ]
    ):
        raise ValueError("Vision response did not contain a complete guided payload.")
    return payload


def _legacy_multi_frame_offsets(duration_seconds: float) -> list[float]:
    if duration_seconds <= 0:
        return [0.0]
    if duration_seconds <= SHORT_CLIP_SINGLE_FRAME_THRESHOLD_SEC:
        return [round(duration_seconds / 2, 2)]

    offsets: list[float] = []
    segment_start = 0.0
    while segment_start < duration_seconds:
        segment_end = min(segment_start + LONG_CLIP_FRAME_STEP_SEC, duration_seconds)
        midpoint = segment_start + ((segment_end - segment_start) / 2)
        offsets.append(round(midpoint, 2))
        segment_start += LONG_CLIP_FRAME_STEP_SEC

    return offsets


def _single_middle_offset(duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    return round(duration_seconds / 2, 2)


def _yolo_world_offsets(duration_seconds: float) -> list[float]:
    if duration_seconds <= 0:
        return [0.0]
    candidates = [
        round(duration_seconds / 6, 2),
        round(duration_seconds / 2, 2),
        round((duration_seconds * 5) / 6, 2),
    ]
    return _dedupe_float_offsets(candidates)


def _dedupe_float_offsets(offsets: list[float]) -> list[float]:
    seen: set[float] = set()
    ordered: list[float] = []
    for offset in offsets:
        normalized = round(max(offset, 0.0), 2)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered or [0.0]


def _closest_frame(
    frames: list["ExtractedFrame"],
    *,
    target_offset_sec: float,
) -> "ExtractedFrame" | None:
    if not frames:
        return None
    return min(
        frames,
        key=lambda frame: abs(frame.frame_offset_sec - target_offset_sec),
    )


@dataclass(slots=True)
class ExtractedFrame:
    frame_offset_sec: float
    image_bytes: bytes


def _build_legacy_analysis_prompt(frames: list[ExtractedFrame]) -> str:
    offsets = [frame.frame_offset_sec for frame in frames]
    return (
        "These images are chronological frames from a single video clip. "
        f"The frame offsets in seconds are: {offsets}. "
        "Return valid JSON with this exact shape: "
        '{"summary":"...",'
        '"tags":["tag"],'
        '"clip_type_hint":"drone|ground|interview|unknown",'
        '"frames":[{"frame_offset_sec":0.0,"description":"..."}]}. '
        "Requirements: summary must be one sentence under 140 characters; "
        "tags must be 6-10 lower-case search terms; "
        "each frame description must be one sentence under 160 characters and mention only clearly visible shot type, subject, movement, lighting, and mood."
    )


def _build_gemini_guided_prompt(detected_objects: list[str]) -> str:
    object_text = ", ".join(detected_objects) if detected_objects else "none"
    return (
        f"Objects detected: [{object_text}]. "
        "Now describe only: shot type, camera movement, lighting, and any notable subjects or objects not already listed. "
        "Return valid JSON with this exact shape: "
        '{"shot_type":"...",'
        '"camera_movement":"...",'
        '"lighting":"...",'
        '"additional_subjects_or_objects":["..."],'
        '"clip_type_hint":"drone|ground|interview|unknown"}. '
        "Requirements: use short lower-case phrases, do not repeat objects already listed, do not speculate, and return JSON only."
    )


def _build_gemini_guided_repair_prompt(detected_objects: list[str]) -> str:
    object_text = ", ".join(detected_objects) if detected_objects else "none"
    return (
        f"Objects detected: [{object_text}]. "
        "Return exactly these five lines and nothing else: "
        "shot type: ... ; "
        "camera movement: ... ; "
        "lighting: ... ; "
        "additional subjects or objects: item 1, item 2 ; "
        "clip type hint: drone|ground|interview|unknown. "
        "Use short lower-case phrases, do not repeat already listed objects, and if something is unclear write unknown."
    )


def _extract_frames(
    *,
    ffmpeg_binary: str | None,
    settings: Settings,
    file_path: Path,
    offsets: list[float],
) -> list[ExtractedFrame]:
    if not ffmpeg_binary:
        return []

    frames: list[ExtractedFrame] = []
    scale_filter = (
        "scale="
        f"'if(gt(iw,ih),{settings.vision_max_image_edge_px},-2)':"
        f"'if(gt(iw,ih),-2,{settings.vision_max_image_edge_px})'"
    )

    for offset in _dedupe_float_offsets(offsets):
        result = subprocess.run(
            [
                ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(offset),
                "-i",
                str(file_path),
                "-frames:v",
                "1",
                "-vf",
                scale_filter,
                "-q:v",
                "5",
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
            frames.append(
                ExtractedFrame(
                    frame_offset_sec=offset,
                    image_bytes=result.stdout,
                )
            )
    return frames


def _analysis_from_legacy_response(
    text: str,
    *,
    clip_type: str,
    fallback: "HeuristicVisualAnalyzer",
    fallback_tags: list[str],
    frames: list[ExtractedFrame],
    provider: str,
    model: str | None,
    cache_signature: str,
) -> VisionAnalysis:
    payload = _extract_json_block(text)
    summary = str(payload.get("summary") or "").strip()
    if not summary:
        summary = " ".join(
            str(item.get("description", "")).strip()
            for item in payload.get("frames", [])
        ).strip()
    if not summary:
        summary = fallback.analyze(
            clip_name="",
            clip_type=clip_type,
            tags=fallback_tags,
            file_path="",
            duration_seconds=0,
        ).summary

    frame_descriptions: list[VisualDescription] = []
    raw_frames = payload.get("frames", [])
    if isinstance(raw_frames, list):
        for index, frame in enumerate(raw_frames[: len(frames)]):
            description = str(frame.get("description") or "").strip()
            offset = frame.get("frame_offset_sec", frames[index].frame_offset_sec)
            try:
                offset_value = float(offset)
            except (TypeError, ValueError):
                offset_value = frames[index].frame_offset_sec
            if description:
                frame_descriptions.append(
                    VisualDescription(
                        frame_offset_sec=offset_value,
                        description=description,
                    )
                )

    if not frame_descriptions:
        frame_descriptions = [
            VisualDescription(
                frame_offset_sec=frame.frame_offset_sec,
                description=summary,
            )
            for frame in frames
        ]

    payload_tags = payload.get("tags", [])
    tags = _dedupe(
        [
            *fallback_tags,
            *([str(item) for item in payload_tags] if isinstance(payload_tags, list) else []),
        ]
    )
    if len(tags) > 16:
        tags = tags[:16]

    return VisionAnalysis(
        summary=summary,
        tags=tags,
        frame_descriptions=frame_descriptions,
        detected_objects=[],
        clip_type_hint=_canonical_clip_type(payload.get("clip_type_hint"))
        or _canonical_clip_type(clip_type),
        cache_signature=cache_signature,
        provider=provider,
        model=model,
    )


def _normalize_detection_label(value: str) -> str:
    normalized = re.sub(r"[_-]+", " ", value).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _merge_guided_summary(
    *,
    clip_type: str,
    detected_objects: list[str],
    shot_type: str,
    camera_movement: str,
    lighting: str,
    additional_subjects_or_objects: list[str],
) -> str:
    subjects = _dedupe(
        [
            *detected_objects,
            *additional_subjects_or_objects,
        ]
    )

    visual_bits = [
        bit
        for bit in [
            shot_type.strip().lower(),
            camera_movement.strip().lower(),
            lighting.strip().lower(),
        ]
        if bit and bit not in {"unknown", "none", "n/a"}
    ]
    visual_summary = ", ".join(visual_bits) if visual_bits else f"{clip_type.lower()} clip"

    if subjects:
        return f"{', '.join(subjects[:8])}. {visual_summary}."
    return f"{visual_summary}."


def _guided_analysis_from_response(
    text: str,
    *,
    clip_type: str,
    fallback: "HeuristicVisualAnalyzer",
    fallback_tags: list[str],
    detected_objects: list[str],
    frame: ExtractedFrame,
    provider: str,
    model: str | None,
    cache_signature: str,
) -> VisionAnalysis:
    payload = _extract_guided_payload(text)
    shot_type = str(payload.get("shot_type") or "").strip().lower()
    camera_movement = str(payload.get("camera_movement") or "").strip().lower()
    lighting = str(payload.get("lighting") or "").strip().lower()
    raw_additional = payload.get("additional_subjects_or_objects", [])
    additional_subjects_or_objects = _dedupe(
        [
            str(item)
            for item in raw_additional
            if str(item).strip()
        ]
        if isinstance(raw_additional, list)
        else []
    )

    summary = _merge_guided_summary(
        clip_type=clip_type,
        detected_objects=detected_objects,
        shot_type=shot_type,
        camera_movement=camera_movement,
        lighting=lighting,
        additional_subjects_or_objects=additional_subjects_or_objects,
    )

    if not summary.strip():
        summary = fallback.analyze(
            clip_name="",
            clip_type=clip_type,
            tags=fallback_tags + detected_objects,
            file_path="",
            duration_seconds=0,
        ).summary

    tags = _dedupe(
        [
            *fallback_tags,
            *detected_objects,
            shot_type,
            camera_movement,
            lighting,
            *additional_subjects_or_objects,
            str(payload.get("clip_type_hint") or ""),
        ]
    )
    if len(tags) > 20:
        tags = tags[:20]

    return VisionAnalysis(
        summary=summary,
        tags=tags,
        frame_descriptions=[
            VisualDescription(
                frame_offset_sec=frame.frame_offset_sec,
                description=summary,
            )
        ],
        detected_objects=detected_objects,
        clip_type_hint=_canonical_clip_type(payload.get("clip_type_hint"))
        or _canonical_clip_type(clip_type),
        cache_signature=cache_signature,
        provider=provider,
        model=model,
    )


def _fallback_analysis(
    fallback: "HeuristicVisualAnalyzer",
    *,
    clip_name: str,
    clip_type: str,
    tags: list[str],
    file_path: str,
    duration_seconds: float,
) -> VisionAnalysis:
    return fallback.analyze(
        clip_name=clip_name,
        clip_type=clip_type,
        tags=tags,
        file_path=file_path,
        duration_seconds=duration_seconds,
    )


def _object_aware_fallback_analysis(
    fallback: "HeuristicVisualAnalyzer",
    *,
    clip_name: str,
    clip_type: str,
    tags: list[str],
    file_path: str,
    duration_seconds: float,
    detected_objects: list[str],
    frame_offset_sec: float,
) -> VisionAnalysis:
    fallback_result = fallback.analyze(
        clip_name=clip_name,
        clip_type=clip_type,
        tags=tags + detected_objects,
        file_path=file_path,
        duration_seconds=duration_seconds,
    )
    summary = fallback_result.summary.rstrip(".")
    if detected_objects:
        summary = f"{', '.join(_dedupe(detected_objects)[:6])}. {summary}."
    else:
        summary = f"{summary}."
    return VisionAnalysis(
        summary=summary,
        tags=_dedupe(fallback_result.tags + detected_objects),
        frame_descriptions=[
            VisualDescription(frame_offset_sec=frame_offset_sec, description=summary)
        ],
        detected_objects=detected_objects,
        clip_type_hint=fallback_result.clip_type_hint,
        cache_signature=fallback_result.cache_signature,
        provider=fallback_result.provider,
        model=fallback_result.model,
    )


def _detected_objects_analysis(
    *,
    clip_type: str,
    fallback_tags: list[str],
    detected_objects: list[str],
    frame_offset_sec: float,
    cache_signature: str,
    model: str | None,
) -> VisionAnalysis:
    summary = _merge_guided_summary(
        clip_type=clip_type,
        detected_objects=detected_objects,
        shot_type="",
        camera_movement="",
        lighting="",
        additional_subjects_or_objects=[],
    )
    return VisionAnalysis(
        summary=summary,
        tags=_dedupe([*fallback_tags, *detected_objects]),
        frame_descriptions=[
            VisualDescription(
                frame_offset_sec=frame_offset_sec,
                description=summary,
            )
        ],
        detected_objects=detected_objects,
        clip_type_hint=_canonical_clip_type(clip_type),
        cache_signature=cache_signature,
        provider="yolo-world",
        model=model,
    )


class BaseVisualAnalyzer:
    def cache_signature(self) -> str:
        raise NotImplementedError

    def local_cache_signature(self) -> str:
        return self.cache_signature()

    def background_enrichment_enabled(self) -> bool:
        return False

    def analyze(
        self,
        *,
        clip_name: str,
        clip_type: str,
        tags: list[str],
        file_path: str,
        duration_seconds: float,
        partial_callback: Callable[[VisionAnalysis], None] | None = None,
    ) -> VisionAnalysis:
        raise NotImplementedError

    def analyze_local(
        self,
        *,
        clip_name: str,
        clip_type: str,
        tags: list[str],
        file_path: str,
        duration_seconds: float,
    ) -> VisionAnalysis:
        return self.analyze(
            clip_name=clip_name,
            clip_type=clip_type,
            tags=tags,
            file_path=file_path,
            duration_seconds=duration_seconds,
        )

    def enrich(
        self,
        *,
        clip_name: str,
        clip_type: str,
        tags: list[str],
        file_path: str,
        duration_seconds: float,
        detected_objects: list[str],
    ) -> VisionAnalysis | None:
        return None


class HeuristicVisualAnalyzer(BaseVisualAnalyzer):
    def cache_signature(self) -> str:
        return "heuristic:v2"

    def analyze(
        self,
        *,
        clip_name: str,
        clip_type: str,
        tags: list[str],
        file_path: str,
        duration_seconds: float,
        partial_callback: Callable[[VisionAnalysis], None] | None = None,
    ) -> VisionAnalysis:
        description_tags = ", ".join(tags[:6]) if tags else clip_type
        summary = f"{clip_type.title()} clip with {description_tags}"
        if file_path:
            summary = f"{summary}. Source: {Path(file_path).name}"
        return VisionAnalysis(
            summary=summary,
            tags=_dedupe(tags[:12]),
            frame_descriptions=[
                VisualDescription(frame_offset_sec=0.0, description=summary)
            ],
            detected_objects=[],
            clip_type_hint=_canonical_clip_type(clip_type),
            cache_signature=self.cache_signature(),
            provider="heuristic",
            model=None,
        )


class YoloWorldObjectDetector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ffmpeg = shutil.which("ffmpeg")
        self._model_lock = threading.Lock()
        self._model = None

    def available(self) -> bool:
        return bool(self._ffmpeg and YOLOWorld and np is not None and Image is not None)

    def cache_signature(self) -> str:
        if not self.available():
            return "yolo-world:unavailable"
        return (
            f"yolo-world:{self.settings.yolo_world_model}:"
            f"{YOLO_WORLD_FRAME_SIGNATURE}:"
            f"{self.settings.yolo_world_confidence}:"
            f"{self.settings.yolo_world_max_objects}:v1"
        )

    def detect(self, *, file_path: str, duration_seconds: float) -> list[str]:
        frames = self.sample_frames(
            file_path=file_path,
            duration_seconds=duration_seconds,
        )
        return self.detect_from_frames(frames=frames)

    def sample_frames(self, *, file_path: str, duration_seconds: float) -> list[ExtractedFrame]:
        if not self.available() or not file_path:
            return []

        file = Path(file_path)
        if not file.exists():
            return []

        return _extract_frames(
            ffmpeg_binary=self._ffmpeg,
            settings=self.settings,
            file_path=file,
            offsets=_yolo_world_offsets(duration_seconds),
        )

    def detect_from_frames(self, *, frames: list[ExtractedFrame]) -> list[str]:
        if not self.available() or not frames:
            return []

        model = self._get_model()
        if not model:
            return []

        frame_images: list[Any] = []
        for frame in frames:
            try:
                image = Image.open(io.BytesIO(frame.image_bytes)).convert("RGB")
            except Exception as exc:  # pragma: no cover - runtime dependency
                LOGGER.warning("YOLO World frame decode failed: %s", exc)
                continue
            frame_images.append(np.array(image))

        if not frame_images:
            return []

        try:
            results = model.predict(
                source=frame_images,
                conf=self.settings.yolo_world_confidence,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - runtime dependency
            LOGGER.warning("YOLO World batch inference failed: %s", exc)
            return []

        score_by_label: dict[str, float] = {}
        count_by_label: dict[str, int] = {}

        for result in results or []:
            labels = self._labels_from_prediction_result(result)
            for label, confidence in labels:
                count_by_label[label] = count_by_label.get(label, 0) + 1
                score_by_label[label] = max(score_by_label.get(label, 0.0), confidence)

        ranked = sorted(
            score_by_label,
            key=lambda label: (
                -count_by_label.get(label, 0),
                -score_by_label.get(label, 0.0),
                label,
            ),
        )
        return ranked[: self.settings.yolo_world_max_objects]

    def _get_model(self) -> Any | None:
        if not self.available():
            return None

        with self._model_lock:
            if self._model is None:
                try:
                    self._model = YOLOWorld(self.settings.yolo_world_model)
                except Exception as exc:  # pragma: no cover - runtime dependency
                    LOGGER.warning("YOLO World initialization failed: %s", exc)
                    self._model = False
            return None if self._model is False else self._model

    def _labels_from_prediction_result(self, result: Any) -> list[tuple[str, float]]:
        labels: list[tuple[str, float]] = []
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        classes = getattr(boxes, "cls", None)
        confidences = getattr(boxes, "conf", None)
        if classes is None:
            return labels
        class_ids = classes.tolist()
        confidence_values = confidences.tolist() if confidences is not None else [0.0] * len(class_ids)
        for index, class_id in enumerate(class_ids):
            label = _normalize_detection_label(str(names.get(int(class_id), class_id)))
            if not label:
                continue
            labels.append((label, float(confidence_values[index])))
        return labels


class AnthropicVisionAnalyzer(BaseVisualAnalyzer):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._fallback = HeuristicVisualAnalyzer()
        self._ffmpeg = shutil.which("ffmpeg")
        self._client = (
            Anthropic(
                api_key=settings.anthropic_api_key,
                timeout=settings.vision_timeout_sec,
            )
            if Anthropic and settings.anthropic_api_key
            else None
        )

    def cache_signature(self) -> str:
        if not self._client or not self._ffmpeg:
            return self._fallback.cache_signature()
        return (
            f"anthropic:{self.settings.vision_model}:"
            f"{LEGACY_MULTI_FRAME_SIGNATURE}:"
            f"{self.settings.vision_max_image_edge_px}:v2"
        )

    def analyze(
        self,
        *,
        clip_name: str,
        clip_type: str,
        tags: list[str],
        file_path: str,
        duration_seconds: float,
        partial_callback: Callable[[VisionAnalysis], None] | None = None,
    ) -> VisionAnalysis:
        if not self._client or not self._ffmpeg or not file_path:
            return _fallback_analysis(
                self._fallback,
                clip_name=clip_name,
                clip_type=clip_type,
                tags=tags,
                file_path=file_path,
                duration_seconds=duration_seconds,
            )

        file = Path(file_path)
        if not file.exists():
            return _fallback_analysis(
                self._fallback,
                clip_name=clip_name,
                clip_type=clip_type,
                tags=tags,
                file_path=file_path,
                duration_seconds=duration_seconds,
            )

        frames = _extract_frames(
            ffmpeg_binary=self._ffmpeg,
            settings=self.settings,
            file_path=file,
            offsets=_legacy_multi_frame_offsets(duration_seconds),
        )
        if not frames:
            return _fallback_analysis(
                self._fallback,
                clip_name=clip_name,
                clip_type=clip_type,
                tags=tags,
                file_path=file_path,
                duration_seconds=duration_seconds,
            )

        try:
            cache_signature = self.cache_signature()
            response = self._client.messages.create(
                model=self.settings.vision_model,
                max_tokens=700,
                temperature=0,
                system=LEGACY_VISION_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": self._build_content(frames),
                    }
                ],
            )
            text = "".join(
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            )
            return _analysis_from_legacy_response(
                text,
                clip_type=clip_type,
                fallback=self._fallback,
                fallback_tags=tags,
                frames=frames,
                provider="anthropic",
                model=self.settings.vision_model,
                cache_signature=cache_signature,
            )
        except Exception as exc:  # pragma: no cover - depends on external API/runtime
            LOGGER.warning("Claude vision fallback for %s: %s", clip_name, exc)
            return _fallback_analysis(
                self._fallback,
                clip_name=clip_name,
                clip_type=clip_type,
                tags=tags,
                file_path=file_path,
                duration_seconds=duration_seconds,
            )

    def _build_content(self, frames: list[ExtractedFrame]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for frame in frames:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(frame.image_bytes).decode("ascii"),
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": _build_legacy_analysis_prompt(frames),
            }
        )
        return content


def _extract_gemini_response_text(response: Any) -> str:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return json.dumps(parsed)

    text = getattr(response, "text", "") or ""
    if text:
        return text

    candidates = getattr(response, "candidates", None) or []
    return "".join(
        getattr(part, "text", "")
        for candidate in candidates
        for part in getattr(getattr(candidate, "content", None), "parts", []) or []
    )


class GeminiVisionAnalyzer(BaseVisualAnalyzer):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._fallback = HeuristicVisualAnalyzer()
        self._ffmpeg = shutil.which("ffmpeg")
        self._detector = YoloWorldObjectDetector(settings)
        self._client = (
            genai.Client(
                api_key=settings.gemini_api_key,
                http_options=genai_types.HttpOptions(
                    timeout=max(int(settings.vision_timeout_sec * 1000), 10000),
                ),
            )
            if genai and genai_types and settings.gemini_api_key
            else None
        )

    def cache_signature(self) -> str:
        if not self._client or not self._ffmpeg:
            return self._fallback.cache_signature()
        return (
            f"gemini-yolo-world:{self.settings.vision_model}:"
            f"{YOLO_GEMINI_PIPELINE_SIGNATURE}:"
            f"{self._detector.cache_signature()}:"
            f"{GEMINI_SINGLE_FRAME_SIGNATURE}:"
            f"{self.settings.vision_max_image_edge_px}:v1"
        )

    def local_cache_signature(self) -> str:
        return f"{self.cache_signature()}:{YOLO_PARTIAL_STAGE_SIGNATURE}"

    def background_enrichment_enabled(self) -> bool:
        return bool(self._client and self._ffmpeg and self._detector.available())

    def analyze_local(
        self,
        *,
        clip_name: str,
        clip_type: str,
        tags: list[str],
        file_path: str,
        duration_seconds: float,
    ) -> VisionAnalysis:
        if not self.background_enrichment_enabled() or not file_path:
            return _fallback_analysis(
                self._fallback,
                clip_name=clip_name,
                clip_type=clip_type,
                tags=tags,
                file_path=file_path,
                duration_seconds=duration_seconds,
            )

        file = Path(file_path)
        if not file.exists():
            return _fallback_analysis(
                self._fallback,
                clip_name=clip_name,
                clip_type=clip_type,
                tags=tags,
                file_path=file_path,
                duration_seconds=duration_seconds,
            )

        sampled_frames = self._detector.sample_frames(
            file_path=file_path,
            duration_seconds=duration_seconds,
        )
        detected_objects = self._detector.detect_from_frames(frames=sampled_frames)
        middle_offset = _single_middle_offset(duration_seconds)
        frame = _closest_frame(sampled_frames, target_offset_sec=middle_offset)

        if detected_objects:
            return _detected_objects_analysis(
                clip_type=clip_type,
                fallback_tags=tags,
                detected_objects=detected_objects,
                frame_offset_sec=frame.frame_offset_sec if frame else middle_offset,
                cache_signature=self.local_cache_signature(),
                model=self.settings.yolo_world_model,
            )

        fallback_result = _fallback_analysis(
            self._fallback,
            clip_name=clip_name,
            clip_type=clip_type,
            tags=tags,
            file_path=file_path,
            duration_seconds=duration_seconds,
        )
        return VisionAnalysis(
            summary=fallback_result.summary,
            tags=fallback_result.tags,
            frame_descriptions=[
                VisualDescription(
                    frame_offset_sec=frame.frame_offset_sec if frame else middle_offset,
                    description=fallback_result.summary,
                )
            ],
            detected_objects=[],
            clip_type_hint=fallback_result.clip_type_hint,
            cache_signature=self.local_cache_signature(),
            provider="yolo-world",
            model=self.settings.yolo_world_model,
        )

    def enrich(
        self,
        *,
        clip_name: str,
        clip_type: str,
        tags: list[str],
        file_path: str,
        duration_seconds: float,
        detected_objects: list[str],
    ) -> VisionAnalysis | None:
        if not self.background_enrichment_enabled() or not file_path:
            return None

        file = Path(file_path)
        if not file.exists():
            return None

        middle_offset = _single_middle_offset(duration_seconds)
        middle_frames = _extract_frames(
            ffmpeg_binary=self._ffmpeg,
            settings=self.settings,
            file_path=file,
            offsets=[middle_offset],
        )
        frame = middle_frames[0] if middle_frames else None
        if frame is None:
            return None

        cache_signature = self.cache_signature()
        last_exc: Exception | None = None
        for attempt in range(self.settings.gemini_max_attempts):
            try:
                prompt = _build_gemini_guided_prompt(detected_objects)
                if attempt:
                    prompt += " Previous response was invalid or incomplete. Return only one complete JSON object."
                response = self._client.models.generate_content(
                    model=self.settings.vision_model,
                    contents=[
                        genai_types.Part.from_bytes(
                            data=frame.image_bytes,
                            mime_type="image/jpeg",
                        ),
                        genai_types.Part.from_text(text=prompt),
                    ],
                    config=genai_types.GenerateContentConfig(
                        systemInstruction=GEMINI_GUIDED_SYSTEM_PROMPT,
                        temperature=0,
                        maxOutputTokens=600,
                        responseMimeType="application/json",
                        responseSchema=GEMINI_GUIDED_RESPONSE_SCHEMA,
                    ),
                )
                text = _extract_gemini_response_text(response)
                return _guided_analysis_from_response(
                    text,
                    clip_type=clip_type,
                    fallback=self._fallback,
                    fallback_tags=tags,
                    detected_objects=detected_objects,
                    frame=frame,
                    provider="gemini",
                    model=self.settings.vision_model,
                    cache_signature=cache_signature,
                )
            except Exception as exc:  # pragma: no cover - depends on external API/runtime
                last_exc = exc

        try:
            repair_response = self._client.models.generate_content(
                model=self.settings.vision_model,
                contents=[
                    genai_types.Part.from_bytes(
                        data=frame.image_bytes,
                        mime_type="image/jpeg",
                    ),
                    genai_types.Part.from_text(
                        text=_build_gemini_guided_repair_prompt(detected_objects)
                    ),
                ],
                config=genai_types.GenerateContentConfig(
                    systemInstruction=GEMINI_GUIDED_SYSTEM_PROMPT,
                    temperature=0,
                    maxOutputTokens=180,
                    responseMimeType="text/plain",
                ),
            )
            return _guided_analysis_from_response(
                _extract_gemini_response_text(repair_response),
                clip_type=clip_type,
                fallback=self._fallback,
                fallback_tags=tags,
                detected_objects=detected_objects,
                frame=frame,
                provider="gemini",
                model=self.settings.vision_model,
                cache_signature=cache_signature,
            )
        except Exception as exc:  # pragma: no cover - depends on external API/runtime
            last_exc = exc

        LOGGER.warning("Gemini enrichment fallback for %s: %s", clip_name, last_exc)
        return None

    def analyze(
        self,
        *,
        clip_name: str,
        clip_type: str,
        tags: list[str],
        file_path: str,
        duration_seconds: float,
        partial_callback: Callable[[VisionAnalysis], None] | None = None,
    ) -> VisionAnalysis:
        local_analysis = self.analyze_local(
            clip_name=clip_name,
            clip_type=clip_type,
            tags=tags,
            file_path=file_path,
            duration_seconds=duration_seconds,
        )
        if partial_callback:
            partial_callback(local_analysis)
        enrichment = self.enrich(
            clip_name=clip_name,
            clip_type=clip_type,
            tags=tags,
            file_path=file_path,
            duration_seconds=duration_seconds,
            detected_objects=local_analysis.detected_objects,
        )
        return enrichment or local_analysis


def build_visual_analyzer(settings: Settings) -> BaseVisualAnalyzer:
    provider = settings.vision_provider.strip().lower()
    if provider == "gemini":
        return GeminiVisionAnalyzer(settings)
    if provider == "anthropic":
        return AnthropicVisionAnalyzer(settings)
    if provider != "heuristic":
        LOGGER.warning("Unknown vision provider '%s'; falling back to heuristic mode.", provider)
    return HeuristicVisualAnalyzer()
