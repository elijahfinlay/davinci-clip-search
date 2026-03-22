from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .types import ClipRecord, ProjectIndexMeta, TimelineRecord


class IndexStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS projects (
                    project_uid TEXT PRIMARY KEY,
                    project_name TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    clip_count INTEGER NOT NULL DEFAULT 0,
                    timeline_count INTEGER NOT NULL DEFAULT 0,
                    signature_hash TEXT,
                    quick_mode INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS timelines (
                    timeline_uid TEXT PRIMARY KEY,
                    project_uid TEXT NOT NULL,
                    timeline_name TEXT NOT NULL,
                    timeline_index INTEGER NOT NULL,
                    clip_count INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(project_uid) REFERENCES projects(project_uid) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS clips (
                    clip_id TEXT PRIMARY KEY,
                    content_signature TEXT,
                    vision_cache_signature TEXT,
                    project_uid TEXT NOT NULL,
                    timeline_uid TEXT NOT NULL,
                    timeline_name TEXT NOT NULL,
                    timeline_index INTEGER NOT NULL,
                    clip_name TEXT NOT NULL,
                    file_path TEXT,
                    file_name TEXT,
                    track INTEGER NOT NULL,
                    track_name TEXT,
                    item_index INTEGER NOT NULL,
                    start_frame INTEGER NOT NULL,
                    end_frame INTEGER NOT NULL,
                    duration_frames INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    fps REAL NOT NULL,
                    start_timecode TEXT NOT NULL,
                    end_timecode TEXT,
                    source_in TEXT,
                    source_out TEXT,
                    resolution TEXT,
                    codec TEXT,
                    clip_color TEXT,
                    clip_type TEXT NOT NULL,
                    has_audio INTEGER NOT NULL DEFAULT 0,
                    description TEXT,
                    transcript TEXT,
                    tags_json TEXT NOT NULL,
                    markers_json TEXT NOT NULL,
                    timeline_markers_json TEXT NOT NULL,
                    visual_descriptions_json TEXT NOT NULL,
                    searchable_text TEXT NOT NULL,
                    source_signature TEXT NOT NULL,
                    thumbnail_data TEXT,
                    indexed_at TEXT NOT NULL,
                    FOREIGN KEY(project_uid) REFERENCES projects(project_uid) ON DELETE CASCADE,
                    FOREIGN KEY(timeline_uid) REFERENCES timelines(timeline_uid) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_clips_project ON clips(project_uid);
                CREATE INDEX IF NOT EXISTS idx_clips_timeline ON clips(timeline_uid);
                CREATE INDEX IF NOT EXISTS idx_clips_type ON clips(clip_type);
                CREATE INDEX IF NOT EXISTS idx_clips_start ON clips(start_frame);
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(clips)").fetchall()
            }
            if "content_signature" not in columns:
                conn.execute("ALTER TABLE clips ADD COLUMN content_signature TEXT")
            if "vision_cache_signature" not in columns:
                conn.execute("ALTER TABLE clips ADD COLUMN vision_cache_signature TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_clips_content_signature ON clips(content_signature)"
            )
            conn.execute(
                """
                UPDATE clips
                SET content_signature =
                    lower(trim(COALESCE(NULLIF(file_name, ''), clip_name)))
                    || '|'
                    || CAST(CASE WHEN duration_frames < 0 THEN 0 ELSE duration_frames END AS TEXT)
                WHERE content_signature IS NULL OR content_signature = ''
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self) -> Iterable[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def get_latest_project_meta(self) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM projects ORDER BY indexed_at DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def get_stats(self) -> dict[str, Any]:
        project = self.get_latest_project_meta()
        if not project:
            return {
                "project_name": None,
                "total": 0,
                "timelines": 0,
                "last_indexed": None,
                "available_types": [],
                "quick_mode": False,
                "signature_hash": None,
                "storage_format": "sqlite",
                "storage_path": str(self.db_path),
                "loaded_from_disk": False,
            }

        with self._connection() as conn:
            available_types = [
                row["clip_type"]
                for row in conn.execute(
                    """
                    SELECT DISTINCT clip_type
                    FROM clips
                    WHERE project_uid = ?
                    ORDER BY CASE clip_type
                        WHEN 'drone' THEN 1
                        WHEN 'handheld' THEN 2
                        WHEN 'interview' THEN 3
                        ELSE 4
                    END, clip_type
                    """,
                    (project["project_uid"],),
                ).fetchall()
            ]

        return {
            "project_name": project["project_name"],
            "total": int(project["clip_count"]),
            "timelines": int(project["timeline_count"]),
            "last_indexed": project["indexed_at"],
            "available_types": available_types,
            "quick_mode": bool(project["quick_mode"]),
            "signature_hash": project["signature_hash"],
            "project_uid": project["project_uid"],
            "storage_format": "sqlite",
            "storage_path": str(self.db_path),
            "loaded_from_disk": True,
        }

    def get_existing_cache(
        self,
        project_uid: str,
        timeline_uids: set[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        query = """
            SELECT clip_id, content_signature, source_signature, tags_json,
                   vision_cache_signature, visual_descriptions_json, description, transcript, clip_type,
                   thumbnail_data
            FROM clips
            WHERE project_uid = ?
        """
        params: list[Any] = [project_uid]
        if timeline_uids:
            placeholders = ", ".join("?" for _ in timeline_uids)
            query += f" AND timeline_uid IN ({placeholders})"
            params.extend(sorted(timeline_uids))

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()

        cache: dict[str, dict[str, Any]] = {}
        for row in rows:
            cache[row["clip_id"]] = dict(row)
        return cache

    def get_indexed_coverage(self, project_uid: str | None) -> dict[str, Any]:
        if not project_uid:
            return {
                "project_indexed": 0,
                "timeline_counts": {},
                "timeline_name_counts": {},
            }

        with self._connection() as conn:
            total_row = conn.execute(
                """
                SELECT COUNT(*) AS clip_count
                FROM clips
                WHERE project_uid = ?
                """,
                (project_uid,),
            ).fetchone()
            timeline_rows = conn.execute(
                """
                SELECT timeline_uid, timeline_name, COUNT(*) AS clip_count
                FROM clips
                WHERE project_uid = ?
                GROUP BY timeline_uid, timeline_name
                """,
                (project_uid,),
            ).fetchall()

        timeline_counts = {
            row["timeline_uid"]: int(row["clip_count"])
            for row in timeline_rows
        }
        timeline_name_counts = {
            row["timeline_name"]: int(row["clip_count"])
            for row in timeline_rows
        }
        return {
            "project_indexed": int(total_row["clip_count"]) if total_row else 0,
            "timeline_counts": timeline_counts,
            "timeline_name_counts": timeline_name_counts,
        }

    def get_indexed_timelines(self, project_uid: str | None) -> list[dict[str, Any]]:
        if not project_uid:
            return []

        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT timeline_uid, timeline_name, timeline_index, clip_count
                FROM timelines
                WHERE project_uid = ?
                ORDER BY timeline_index ASC, timeline_name ASC
                """,
                (project_uid,),
            ).fetchall()

        return [dict(row) for row in rows]

    def upsert_project_meta(self, project_meta: ProjectIndexMeta) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """
                INSERT INTO projects (
                    project_uid, project_name, indexed_at,
                    clip_count, timeline_count, signature_hash, quick_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_uid) DO UPDATE SET
                    project_name = excluded.project_name,
                    indexed_at = excluded.indexed_at,
                    clip_count = excluded.clip_count,
                    timeline_count = excluded.timeline_count,
                    signature_hash = excluded.signature_hash,
                    quick_mode = excluded.quick_mode
                """,
                (
                    project_meta.project_uid,
                    project_meta.project_name,
                    project_meta.indexed_at,
                    project_meta.clip_count,
                    project_meta.timeline_count,
                    project_meta.signature_hash,
                    int(project_meta.quick_mode),
                ),
            )
            conn.commit()

    def upsert_timeline(self, timeline: TimelineRecord) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """
                INSERT INTO timelines (
                    timeline_uid, project_uid, timeline_name, timeline_index, clip_count
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(timeline_uid) DO UPDATE SET
                    project_uid = excluded.project_uid,
                    timeline_name = excluded.timeline_name,
                    timeline_index = excluded.timeline_index,
                    clip_count = excluded.clip_count
                """,
                (
                    timeline.timeline_uid,
                    timeline.project_uid,
                    timeline.timeline_name,
                    timeline.timeline_index,
                    timeline.clip_count,
                ),
            )
            conn.commit()

    def upsert_clip(self, clip: ClipRecord, *, indexed_at: str) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """
                INSERT INTO clips (
                    clip_id, content_signature, vision_cache_signature, project_uid, timeline_uid, timeline_name, timeline_index,
                    clip_name, file_path, file_name, track, track_name, item_index,
                    start_frame, end_frame, duration_frames, duration_seconds, fps,
                    start_timecode, end_timecode, source_in, source_out, resolution, codec,
                    clip_color, clip_type, has_audio, description, transcript,
                    tags_json, markers_json, timeline_markers_json, visual_descriptions_json,
                    searchable_text, source_signature, thumbnail_data, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(clip_id) DO UPDATE SET
                    content_signature = excluded.content_signature,
                    vision_cache_signature = excluded.vision_cache_signature,
                    project_uid = excluded.project_uid,
                    timeline_uid = excluded.timeline_uid,
                    timeline_name = excluded.timeline_name,
                    timeline_index = excluded.timeline_index,
                    clip_name = excluded.clip_name,
                    file_path = excluded.file_path,
                    file_name = excluded.file_name,
                    track = excluded.track,
                    track_name = excluded.track_name,
                    item_index = excluded.item_index,
                    start_frame = excluded.start_frame,
                    end_frame = excluded.end_frame,
                    duration_frames = excluded.duration_frames,
                    duration_seconds = excluded.duration_seconds,
                    fps = excluded.fps,
                    start_timecode = excluded.start_timecode,
                    end_timecode = excluded.end_timecode,
                    source_in = excluded.source_in,
                    source_out = excluded.source_out,
                    resolution = excluded.resolution,
                    codec = excluded.codec,
                    clip_color = excluded.clip_color,
                    clip_type = excluded.clip_type,
                    has_audio = excluded.has_audio,
                    description = excluded.description,
                    transcript = excluded.transcript,
                    tags_json = excluded.tags_json,
                    markers_json = excluded.markers_json,
                    timeline_markers_json = excluded.timeline_markers_json,
                    visual_descriptions_json = excluded.visual_descriptions_json,
                    searchable_text = excluded.searchable_text,
                    source_signature = excluded.source_signature,
                    thumbnail_data = excluded.thumbnail_data,
                    indexed_at = excluded.indexed_at
                """,
                self._clip_insert_params(clip, indexed_at=indexed_at),
            )
            conn.commit()

    def cleanup_index_scope(
        self,
        *,
        project_uid: str,
        keep_clip_ids: set[str],
        keep_timeline_uids: set[str],
        target_timeline_uids: set[str] | None = None,
    ) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON")

            clip_rows_query = """
                SELECT clip_id
                FROM clips
                WHERE project_uid = ?
            """
            clip_rows_params: list[Any] = [project_uid]
            timeline_rows_query = """
                SELECT timeline_uid
                FROM timelines
                WHERE project_uid = ?
            """
            timeline_rows_params: list[Any] = [project_uid]

            if target_timeline_uids:
                placeholders = ", ".join("?" for _ in target_timeline_uids)
                clip_rows_query += f" AND timeline_uid IN ({placeholders})"
                timeline_rows_query += f" AND timeline_uid IN ({placeholders})"
                sorted_uids = sorted(target_timeline_uids)
                clip_rows_params.extend(sorted_uids)
                timeline_rows_params.extend(sorted_uids)

            existing_clip_ids = {
                row["clip_id"]
                for row in conn.execute(clip_rows_query, clip_rows_params).fetchall()
            }
            stale_clip_ids = sorted(existing_clip_ids - keep_clip_ids)
            for chunk in self._chunked(stale_clip_ids, 400):
                placeholders = ", ".join("?" for _ in chunk)
                conn.execute(
                    f"DELETE FROM clips WHERE clip_id IN ({placeholders})",
                    chunk,
                )

            existing_timeline_uids = {
                row["timeline_uid"]
                for row in conn.execute(
                    timeline_rows_query,
                    timeline_rows_params,
                ).fetchall()
            }
            stale_timeline_uids = sorted(existing_timeline_uids - keep_timeline_uids)
            for chunk in self._chunked(stale_timeline_uids, 200):
                placeholders = ", ".join("?" for _ in chunk)
                conn.execute(
                    f"DELETE FROM timelines WHERE timeline_uid IN ({placeholders})",
                    chunk,
                )

            conn.commit()

    def finalize_project_meta(self, project_meta: ProjectIndexMeta) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            totals = conn.execute(
                """
                SELECT COUNT(*) AS clip_count,
                       COUNT(DISTINCT timeline_uid) AS timeline_count
                FROM clips
                WHERE project_uid = ?
                """,
                (project_meta.project_uid,),
            ).fetchone()
            conn.execute(
                """
                UPDATE projects
                SET indexed_at = ?, clip_count = ?, timeline_count = ?, signature_hash = ?, quick_mode = ?
                WHERE project_uid = ?
                """,
                (
                    project_meta.indexed_at,
                    totals["clip_count"] if totals else 0,
                    totals["timeline_count"] if totals else 0,
                    project_meta.signature_hash,
                    int(project_meta.quick_mode),
                    project_meta.project_uid,
                ),
            )
            conn.commit()

    def replace_index(
        self,
        project_meta: ProjectIndexMeta,
        timeline_records: Iterable[TimelineRecord],
        clip_records: Iterable[ClipRecord],
        *,
        replace_timeline_uids: set[str] | None = None,
    ) -> None:
        timeline_records = list(timeline_records)
        clip_records = list(clip_records)

        with self._connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("BEGIN")
            try:
                if replace_timeline_uids:
                    placeholders = ", ".join("?" for _ in replace_timeline_uids)
                    delete_params = [project_meta.project_uid, *sorted(replace_timeline_uids)]
                    conn.execute(
                        f"""
                        DELETE FROM clips
                        WHERE project_uid = ?
                          AND timeline_uid IN ({placeholders})
                        """,
                        delete_params,
                    )
                    conn.execute(
                        f"""
                        DELETE FROM timelines
                        WHERE project_uid = ?
                          AND timeline_uid IN ({placeholders})
                        """,
                        delete_params,
                    )
                else:
                    conn.execute("DELETE FROM clips")
                    conn.execute("DELETE FROM timelines")
                    conn.execute("DELETE FROM projects")

                conn.execute(
                    """
                    INSERT INTO projects (
                        project_uid, project_name, indexed_at,
                        clip_count, timeline_count, signature_hash, quick_mode
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_uid) DO UPDATE SET
                        project_name = excluded.project_name,
                        indexed_at = excluded.indexed_at,
                        clip_count = excluded.clip_count,
                        timeline_count = excluded.timeline_count,
                        signature_hash = excluded.signature_hash,
                        quick_mode = excluded.quick_mode
                    """,
                    (
                        project_meta.project_uid,
                        project_meta.project_name,
                        project_meta.indexed_at,
                        project_meta.clip_count,
                        project_meta.timeline_count,
                        project_meta.signature_hash,
                        int(project_meta.quick_mode),
                    ),
                )

                conn.executemany(
                    """
                    INSERT INTO timelines (
                        timeline_uid, project_uid, timeline_name, timeline_index, clip_count
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(timeline_uid) DO UPDATE SET
                        project_uid = excluded.project_uid,
                        timeline_name = excluded.timeline_name,
                        timeline_index = excluded.timeline_index,
                        clip_count = excluded.clip_count
                    """,
                    [
                        (
                            timeline.timeline_uid,
                            timeline.project_uid,
                            timeline.timeline_name,
                            timeline.timeline_index,
                            timeline.clip_count,
                        )
                        for timeline in timeline_records
                    ],
                )

                conn.executemany(
                    """
                    INSERT INTO clips (
                        clip_id, content_signature, vision_cache_signature, project_uid, timeline_uid, timeline_name, timeline_index,
                        clip_name, file_path, file_name, track, track_name, item_index,
                        start_frame, end_frame, duration_frames, duration_seconds, fps,
                        start_timecode, end_timecode, source_in, source_out, resolution, codec,
                        clip_color, clip_type, has_audio, description, transcript,
                        tags_json, markers_json, timeline_markers_json, visual_descriptions_json,
                        searchable_text, source_signature, thumbnail_data, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(clip_id) DO UPDATE SET
                        content_signature = excluded.content_signature,
                        vision_cache_signature = excluded.vision_cache_signature,
                        project_uid = excluded.project_uid,
                        timeline_uid = excluded.timeline_uid,
                        timeline_name = excluded.timeline_name,
                        timeline_index = excluded.timeline_index,
                        clip_name = excluded.clip_name,
                        file_path = excluded.file_path,
                        file_name = excluded.file_name,
                        track = excluded.track,
                        track_name = excluded.track_name,
                        item_index = excluded.item_index,
                        start_frame = excluded.start_frame,
                        end_frame = excluded.end_frame,
                        duration_frames = excluded.duration_frames,
                        duration_seconds = excluded.duration_seconds,
                        fps = excluded.fps,
                        start_timecode = excluded.start_timecode,
                        end_timecode = excluded.end_timecode,
                        source_in = excluded.source_in,
                        source_out = excluded.source_out,
                        resolution = excluded.resolution,
                        codec = excluded.codec,
                        clip_color = excluded.clip_color,
                        clip_type = excluded.clip_type,
                        has_audio = excluded.has_audio,
                        description = excluded.description,
                        transcript = excluded.transcript,
                        tags_json = excluded.tags_json,
                        markers_json = excluded.markers_json,
                        timeline_markers_json = excluded.timeline_markers_json,
                        visual_descriptions_json = excluded.visual_descriptions_json,
                        searchable_text = excluded.searchable_text,
                        source_signature = excluded.source_signature,
                        thumbnail_data = excluded.thumbnail_data,
                        indexed_at = excluded.indexed_at
                    """,
                    [
                        self._clip_insert_params(
                            clip,
                            indexed_at=project_meta.indexed_at,
                        )
                        for clip in clip_records
                    ],
                )

                if replace_timeline_uids:
                    totals = conn.execute(
                        """
                        SELECT COUNT(*) AS clip_count,
                               COUNT(DISTINCT timeline_uid) AS timeline_count
                        FROM clips
                        WHERE project_uid = ?
                        """,
                        (project_meta.project_uid,),
                    ).fetchone()
                    conn.execute(
                        """
                        UPDATE projects
                        SET indexed_at = ?, clip_count = ?, timeline_count = ?, signature_hash = ?, quick_mode = ?
                        WHERE project_uid = ?
                        """,
                        (
                            project_meta.indexed_at,
                            totals["clip_count"],
                            totals["timeline_count"],
                            project_meta.signature_hash,
                            int(project_meta.quick_mode),
                            project_meta.project_uid,
                        ),
                    )

                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def _chunked(items: list[str], size: int) -> list[list[str]]:
        return [items[index : index + size] for index in range(0, len(items), size)]

    @staticmethod
    def _clip_insert_params(clip: ClipRecord, *, indexed_at: str) -> tuple[Any, ...]:
        return (
            clip.clip_id,
            clip.content_signature,
            clip.vision_cache_signature,
            clip.project_uid,
            clip.timeline_uid,
            clip.timeline_name,
            clip.timeline_index,
            clip.clip_name,
            clip.file_path,
            clip.file_name,
            clip.track,
            clip.track_name,
            clip.item_index,
            clip.start_frame,
            clip.end_frame,
            clip.duration_frames,
            clip.duration_seconds,
            clip.fps,
            clip.start_timecode,
            clip.end_timecode,
            clip.source_in,
            clip.source_out,
            clip.resolution,
            clip.codec,
            clip.clip_color,
            clip.clip_type,
            int(clip.has_audio),
            clip.description,
            clip.transcript,
            json.dumps(clip.tags),
            json.dumps([marker.to_dict() for marker in clip.markers]),
            json.dumps([marker.to_dict() for marker in clip.timeline_markers]),
            json.dumps([item.to_dict() for item in clip.visual_descriptions]),
            clip.searchable_text,
            clip.source_signature,
            clip.thumbnail_data,
            indexed_at,
        )

    def get_clip(self, clip_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM clips
                WHERE clip_id = ?
                """,
                (clip_id,),
            ).fetchone()
            return self._deserialize_row(row) if row else None

    def update_clip_thumbnail(self, clip_id: str, thumbnail_data: str) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE clips
                SET thumbnail_data = ?
                WHERE clip_id = ?
                """,
                (thumbnail_data, clip_id),
            )
            conn.commit()

    def get_search_rows(
        self,
        *,
        clip_type: str | None = None,
        timeline_uid: str | None = None,
        timeline_name: str | None = None,
    ) -> list[dict[str, Any]]:
        project = self.get_latest_project_meta()
        if not project:
            return []

        query = """
            SELECT *
            FROM clips
            WHERE project_uid = ?
        """
        params: list[Any] = [project["project_uid"]]
        if clip_type and clip_type.lower() != "all":
            query += " AND clip_type = ?"
            params.append(clip_type.lower())
        if timeline_uid:
            query += " AND timeline_uid = ?"
            params.append(timeline_uid)
        elif timeline_name:
            query += " AND timeline_name = ?"
            params.append(timeline_name)
        query += " ORDER BY timeline_index ASC, track ASC, start_frame ASC"

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._deserialize_row(row) for row in rows]

    def _deserialize_row(self, row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            return {}
        data = dict(row)
        data["tags"] = json.loads(data.pop("tags_json"))
        data["markers"] = json.loads(data.pop("markers_json"))
        data["timeline_markers"] = json.loads(data.pop("timeline_markers_json"))
        data["visual_descriptions"] = json.loads(data.pop("visual_descriptions_json"))
        data["has_audio"] = bool(data["has_audio"])
        return data
