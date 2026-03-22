from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.services.index_store import IndexStore
from backend.services.types import ClipRecord, ProjectIndexMeta, TimelineRecord


class TrackingConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.was_closed = False

    def close(self) -> None:
        self.was_closed = True
        super().close()


class IndexStoreConnectionTests(unittest.TestCase):
    def test_store_methods_close_sqlite_connections(self) -> None:
        real_connect = sqlite3.connect
        connections: list[TrackingConnection] = []

        def tracking_connect(*args, **kwargs):
            kwargs["factory"] = TrackingConnection
            conn = real_connect(*args, **kwargs)
            connections.append(conn)
            return conn

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "index.sqlite3"
            store = IndexStore(db_path)

            with mock.patch(
                "backend.services.index_store.sqlite3.connect",
                side_effect=tracking_connect,
            ):
                store.initialize()
                store.get_latest_project_meta()
                store.get_stats()
                store.get_existing_cache("missing-project")
                store.get_indexed_coverage("missing-project")
                store.get_indexed_timelines("missing-project")
                store.get_clip("missing-clip")
                store.get_search_rows()
                project = ProjectIndexMeta(
                    project_uid="project-1",
                    project_name="Test Project",
                    indexed_at="2026-03-21T00:00:00+00:00",
                    clip_count=1,
                    timeline_count=1,
                    signature_hash="sig",
                    quick_mode=False,
                )
                timeline = TimelineRecord(
                    timeline_uid="timeline-1",
                    project_uid="project-1",
                    timeline_name="Timeline 1",
                    timeline_index=1,
                    clip_count=1,
                )
                clip = ClipRecord(
                    clip_id="clip-1",
                    content_signature="clip.mp4|24",
                    vision_cache_signature="vision:v1",
                    project_uid="project-1",
                    timeline_uid="timeline-1",
                    timeline_name="Timeline 1",
                    timeline_index=1,
                    clip_name="clip.mp4",
                    file_path="/tmp/clip.mp4",
                    file_name="clip.mp4",
                    track=1,
                    track_name="V1",
                    item_index=1,
                    start_frame=0,
                    end_frame=24,
                    duration_frames=24,
                    duration_seconds=1.0,
                    fps=24.0,
                    start_timecode="01:00:00:00",
                    end_timecode="01:00:01:00",
                    source_in=None,
                    source_out=None,
                    resolution="1920x1080",
                    codec="h264",
                    clip_color=None,
                    clip_type="handheld",
                    has_audio=True,
                    description="objects: person.",
                    transcript=None,
                    tags=["person"],
                    markers=[],
                    timeline_markers=[],
                    visual_descriptions=[],
                    searchable_text="objects person",
                    source_signature="source:v1",
                    thumbnail_data=None,
                )
                store.upsert_project_meta(project)
                store.upsert_timeline(timeline)
                store.upsert_clip(clip, indexed_at=project.indexed_at)
                store.cleanup_index_scope(
                    project_uid="project-1",
                    keep_clip_ids={"clip-1"},
                    keep_timeline_uids={"timeline-1"},
                )
                store.finalize_project_meta(project)

        self.assertTrue(connections)
        self.assertTrue(all(connection.was_closed for connection in connections))


if __name__ == "__main__":
    unittest.main()
