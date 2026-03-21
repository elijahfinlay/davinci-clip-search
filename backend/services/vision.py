from __future__ import annotations

import base64
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


LOGGER = logging.getLogger(__name__)

VISION_SYSTEM_PROMPT = """
You generate concise visual search metadata for DaVinci Resolve timeline clips.
Use only what is visible in the provided frames. Do not identify people by name.
Return valid JSON only, with no markdown fences or extra commentary.
""".strip()

VISION_RESPONSE_SCHEMA: dict[str, Any] = {
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

SHORT_CLIP_SINGLE_FRAME_THRESHOLD_SEC = 15.0
LONG_CLIP_FRAME_STEP_SEC = 10.0
FRAME_SAMPLING_SIGNATURE = "middle-short_or_midpoint-10s-segments:v1"


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
    return json.loads(cleaned[start : end + 1])


def _frame_offsets(duration_seconds: float) -> list[float]:
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


@dataclass(slots=True)
class ExtractedFrame:
    frame_offset_sec: float
    image_bytes: bytes


def _build_analysis_prompt(frames: list[ExtractedFrame]) -> str:
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


def _extract_frames(
    *,
    ffmpeg_binary: str | None,
    settings: Settings,
    file_path: Path,
    duration_seconds: float,
) -> list[ExtractedFrame]:
    if not ffmpeg_binary:
        return []

    offsets = _frame_offsets(duration_seconds)
    frames: list[ExtractedFrame] = []
    scale_filter = (
        "scale="
        f"'if(gt(iw,ih),{settings.vision_max_image_edge_px},-2)':"
        f"'if(gt(iw,ih),-2,{settings.vision_max_image_edge_px})'"
    )

    for offset in offsets:
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


def _analysis_from_response(
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


class BaseVisualAnalyzer:
    def cache_signature(self) -> str:
        raise NotImplementedError

    def analyze(
        self,
        *,
        clip_name: str,
        clip_type: str,
        tags: list[str],
        file_path: str,
        duration_seconds: float,
    ) -> VisionAnalysis:
        raise NotImplementedError


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
            clip_type_hint=_canonical_clip_type(clip_type),
            cache_signature=self.cache_signature(),
            provider="heuristic",
            model=None,
        )


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
            f"{FRAME_SAMPLING_SIGNATURE}:"
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
            duration_seconds=duration_seconds,
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
                system=VISION_SYSTEM_PROMPT,
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
            return _analysis_from_response(
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
                "text": _build_analysis_prompt(frames),
            }
        )
        return content


class GeminiVisionAnalyzer(BaseVisualAnalyzer):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._fallback = HeuristicVisualAnalyzer()
        self._ffmpeg = shutil.which("ffmpeg")
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
            f"gemini:{self.settings.vision_model}:"
            f"{FRAME_SAMPLING_SIGNATURE}:"
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
            duration_seconds=duration_seconds,
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

        cache_signature = self.cache_signature()
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                prompt = _build_analysis_prompt(frames)
                if attempt:
                    prompt += " Previous response was invalid or incomplete. Return only one complete JSON object."
                response = self._client.models.generate_content(
                    model=self.settings.vision_model,
                    contents=[
                        *[
                            genai_types.Part.from_bytes(
                                data=frame.image_bytes,
                                mime_type="image/jpeg",
                            )
                            for frame in frames
                        ],
                        genai_types.Part.from_text(text=prompt),
                    ],
                    config=genai_types.GenerateContentConfig(
                        systemInstruction=VISION_SYSTEM_PROMPT,
                        temperature=0,
                        maxOutputTokens=1200,
                        responseMimeType="application/json",
                        responseSchema=VISION_RESPONSE_SCHEMA,
                    ),
                )
                parsed = getattr(response, "parsed", None)
                text = json.dumps(parsed) if parsed is not None else (getattr(response, "text", "") or "")
                if not text:
                    candidates = getattr(response, "candidates", None) or []
                    text = "".join(
                        getattr(part, "text", "")
                        for candidate in candidates
                        for part in getattr(getattr(candidate, "content", None), "parts", []) or []
                    )
                return _analysis_from_response(
                    text,
                    clip_type=clip_type,
                    fallback=self._fallback,
                    fallback_tags=tags,
                    frames=frames,
                    provider="gemini",
                    model=self.settings.vision_model,
                    cache_signature=cache_signature,
                )
            except Exception as exc:  # pragma: no cover - depends on external API/runtime
                last_exc = exc

        LOGGER.warning("Gemini vision fallback for %s: %s", clip_name, last_exc)
        return _fallback_analysis(
            self._fallback,
            clip_name=clip_name,
            clip_type=clip_type,
            tags=tags,
            file_path=file_path,
            duration_seconds=duration_seconds,
        )


def build_visual_analyzer(settings: Settings) -> BaseVisualAnalyzer:
    provider = settings.vision_provider.strip().lower()
    if provider == "gemini":
        return GeminiVisionAnalyzer(settings)
    if provider == "anthropic":
        return AnthropicVisionAnalyzer(settings)
    if provider != "heuristic":
        LOGGER.warning("Unknown vision provider '%s'; falling back to heuristic mode.", provider)
    return HeuristicVisualAnalyzer()
