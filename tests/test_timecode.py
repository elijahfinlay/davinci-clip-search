from __future__ import annotations

import unittest

from backend.services.timecode import timeline_frame_to_timecode, timecode_to_frames


class TimelineTimecodeTests(unittest.TestCase):
    def test_non_drop_frame_timeline_uses_timeline_fps(self) -> None:
        start_tc = "01:00:00:00"
        for source_fps in (24.0, 48.0, 59.94):
            with self.subTest(source_fps=source_fps):
                self.assertEqual(
                    timeline_frame_to_timecode(
                        timeline_start_frame=0,
                        timeline_start_timecode=start_tc,
                        frame=24,
                        timeline_fps=24.0,
                        drop_frame=False,
                    ),
                    "01:00:01:00",
                )

    def test_non_drop_frame_timeline_preserves_frame_offsets(self) -> None:
        self.assertEqual(
            timeline_frame_to_timecode(
                timeline_start_frame=86400,
                timeline_start_timecode="01:00:00:00",
                frame=86437,
                timeline_fps=24.0,
                drop_frame=False,
            ),
            "01:00:01:13",
        )

    def test_drop_frame_timeline_round_trip(self) -> None:
        timecode = timeline_frame_to_timecode(
            timeline_start_frame=0,
            timeline_start_timecode="01:00:00;00",
            frame=1798,
            timeline_fps=29.97,
            drop_frame=True,
        )
        self.assertEqual(timecode, "01:00:59;28")
        self.assertEqual(
            timecode_to_frames(timecode, 29.97, drop_frame=True),
            timecode_to_frames("01:00:00;00", 29.97, drop_frame=True) + 1798,
        )


if __name__ == "__main__":
    unittest.main()
