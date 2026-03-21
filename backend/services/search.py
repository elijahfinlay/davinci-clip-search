from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from .index_store import IndexStore


FIXED_FILTERS = ["All", "Drone", "Ground", "Interview"]
CLIP_TYPE_ALIASES = {"ground": "handheld"}
CLIP_TYPE_LABELS = {"handheld": "Ground"}


def canonical_clip_type(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    return CLIP_TYPE_ALIASES.get(normalized, normalized)


def display_clip_type(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    return CLIP_TYPE_LABELS.get(normalized, normalized.title())


def human_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{round(seconds * 1000)}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = seconds % 60
    return f"{minutes}m {remainder:.1f}s"


def format_fps(value: float | None) -> str | None:
    if value is None:
        return None
    if abs(value - round(value)) < 0.001:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


@dataclass(slots=True)
class ParsedQuery:
    raw: str
    text_query: str = ""
    terms: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    min_duration: float | None = None
    max_duration: float | None = None
    timeline_query: str | None = None
    folder_query: str | None = None
    marker_color: str | None = None
    marker_text: str | None = None
    track: int | None = None


class SearchService:
    def __init__(self, store: IndexStore) -> None:
        self.store = store

    def build_filter_options(self) -> list[str]:
        stats = self.store.get_stats()
        dynamic = [
            display_clip_type(item)
            for item in stats["available_types"]
            if display_clip_type(item) not in FIXED_FILTERS
        ]
        filters = [item for item in FIXED_FILTERS]
        filters.extend(sorted(dynamic))
        return filters

    def parse_query(self, raw_query: str) -> ParsedQuery:
        working = raw_query.strip()
        parsed = ParsedQuery(raw=raw_query)

        duration_patterns = [
            (r"longer than\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", "min_duration"),
            (r"over\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", "min_duration"),
            (r"shorter than\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", "max_duration"),
            (r"under\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", "max_duration"),
        ]
        for pattern, field_name in duration_patterns:
            match = re.search(pattern, working, flags=re.IGNORECASE)
            if match:
                setattr(parsed, field_name, float(match.group(1)))
                working = working.replace(match.group(0), " ")

        match = re.search(r"\btrack\s+(\d+)\b", working, flags=re.IGNORECASE)
        if match:
            parsed.track = int(match.group(1))
            working = working.replace(match.group(0), " ")

        match = re.search(r"\bin\s+timeline\s+([a-z0-9 _-]+)", working, flags=re.IGNORECASE)
        if match:
            parsed.timeline_query = match.group(1).strip()
            working = working.replace(match.group(0), " ")

        match = re.search(r"\bfrom\s+the\s+([a-z0-9 _-]+)\s+folder\b", working, flags=re.IGNORECASE)
        if match:
            parsed.folder_query = match.group(1).strip()
            working = working.replace(match.group(0), " ")

        match = re.search(r"\b(?:with|has)\s+([a-z]+)\s+markers?\b", working, flags=re.IGNORECASE)
        if match:
            parsed.marker_color = match.group(1).strip()
            working = working.replace(match.group(0), " ")

        match = re.search(r"\bmarked\s+([a-z0-9 _-]+)", working, flags=re.IGNORECASE)
        if match:
            parsed.marker_text = match.group(1).strip()
            working = working.replace(match.group(0), " ")

        parsed.phrases = re.findall(r'"([^"]+)"', working)
        working = re.sub(r'"[^"]+"', " ", working)

        try:
            tokens = shlex.split(working)
        except ValueError:
            tokens = working.split()

        parsed.terms = [normalize_text(token) for token in tokens if normalize_text(token)]
        parsed.text_query = " ".join([phrase for phrase in parsed.phrases] + parsed.terms).strip()
        return parsed

    def search(
        self,
        raw_query: str,
        *,
        clip_type: str = "All",
        scope: str = "all",
        timeline_uid: str | None = None,
        timeline_name: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        parsed = self.parse_query(raw_query)
        rows = self.store.get_search_rows(
            clip_type=canonical_clip_type(clip_type),
            timeline_uid=timeline_uid if scope == "current" else None,
            timeline_name=timeline_name if scope == "current" else None,
        )

        ranked: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            if not self._passes_filters(row, parsed):
                continue
            score = self._score_row(row, parsed)
            if parsed.text_query and score <= 0:
                continue
            ranked.append((score, row))

        ranked.sort(
            key=lambda item: (
                -item[0],
                item[1]["timeline_index"],
                item[1]["track"],
                item[1]["start_frame"],
            )
        )

        if limit is None:
            selected_rows = ranked
        else:
            selected_rows = ranked[:limit]

        results = [self._to_result(row) for _, row in selected_rows]
        return {
            "query": raw_query,
            "filter": clip_type,
            "scope": scope,
            "total": len(ranked),
            "results": results,
        }

    def _passes_filters(self, row: dict[str, Any], parsed: ParsedQuery) -> bool:
        if parsed.min_duration is not None and row["duration_seconds"] < parsed.min_duration:
            return False
        if parsed.max_duration is not None and row["duration_seconds"] > parsed.max_duration:
            return False
        if parsed.track is not None and row["track"] != parsed.track:
            return False
        if parsed.timeline_query:
            if parsed.timeline_query.lower() not in row["timeline_name"].lower():
                return False
        if parsed.folder_query:
            if parsed.folder_query.lower() not in (row.get("file_path") or "").lower():
                return False
        if parsed.marker_color:
            markers_blob = " ".join(
                " ".join(
                    [
                        marker.get("color", ""),
                        marker.get("name", ""),
                        marker.get("note", ""),
                    ]
                )
                for marker in row["markers"] + row["timeline_markers"]
            ).lower()
            if parsed.marker_color.lower() not in markers_blob:
                return False
        if parsed.marker_text:
            marker_blob = " ".join(
                " ".join(
                    [
                        marker.get("color", ""),
                        marker.get("name", ""),
                        marker.get("note", ""),
                    ]
                )
                for marker in row["markers"] + row["timeline_markers"]
            ).lower()
            if parsed.marker_text.lower() not in marker_blob:
                return False
        return True

    def _score_row(self, row: dict[str, Any], parsed: ParsedQuery) -> float:
        if not parsed.text_query:
            return 1.0

        searchable_text = normalize_text(row["searchable_text"])
        description = normalize_text(row["description"] or "")
        clip_name = normalize_text(row["clip_name"])
        timeline_name = normalize_text(row["timeline_name"])
        tags = {normalize_text(tag) for tag in row["tags"]}
        phrases = [normalize_text(item) for item in parsed.phrases if normalize_text(item)]

        score = 0.0
        for phrase in phrases:
            if phrase and phrase in searchable_text:
                score += 10.0
            elif phrase and phrase in description:
                score += 8.0

        term_hits = 0
        for term in parsed.terms:
            if not term:
                continue
            if term in clip_name:
                score += 6.0
                term_hits += 1
                continue
            if any(term == tag or term in tag for tag in tags):
                score += 5.0
                term_hits += 1
                continue
            if term in description:
                score += 4.5
                term_hits += 1
                continue
            if term in searchable_text:
                score += 3.0
                term_hits += 1
                continue
            if term in timeline_name:
                score += 2.0
                term_hits += 1

        if parsed.terms:
            score += (term_hits / max(len(parsed.terms), 1)) * 3.0

        return score

    def _to_result(self, row: dict[str, Any]) -> dict[str, Any]:
        thumbnail = None
        if row.get("thumbnail_data") or row.get("file_path"):
            thumbnail = f"/api/clips/{row['clip_id']}/thumbnail"

        return {
            "id": row["clip_id"],
            "filename": row["file_name"] or row["clip_name"],
            "timeline": row["timeline_name"],
            "timecode": row["start_timecode"],
            "duration": human_duration(row["duration_seconds"]),
            "track": row["track"],
            "description": row["description"] or row["clip_name"],
            "tags": row["tags"],
            "type": canonical_clip_type(row["clip_type"]),
            "fps": format_fps(row["fps"]),
            "resolution": row["resolution"],
            "thumbnail": thumbnail,
        }
