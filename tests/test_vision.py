from __future__ import annotations

import unittest

from backend.services.vision import (
    ExtractedFrame,
    _guided_analysis_from_response,
    _merge_guided_summary,
    _single_middle_offset,
    _yolo_world_offsets,
)


class VisionPipelineTests(unittest.TestCase):
    def test_yolo_world_offsets_sample_three_frames(self) -> None:
        self.assertEqual(_yolo_world_offsets(60.0), [10.0, 30.0, 50.0])
        self.assertEqual(_yolo_world_offsets(0.0), [0.0])

    def test_middle_offset_uses_clip_midpoint(self) -> None:
        self.assertEqual(_single_middle_offset(18.0), 9.0)
        self.assertEqual(_single_middle_offset(0.0), 0.0)

    def test_merge_guided_summary_includes_objects(self) -> None:
        summary = _merge_guided_summary(
            clip_type="handheld",
            detected_objects=["person", "school bus"],
            shot_type="wide shot",
            camera_movement="slow pan",
            lighting="bright daylight",
            additional_subjects_or_objects=["campus building"],
        )
        self.assertTrue(summary.startswith("objects: person, school bus"))
        self.assertIn("wide shot", summary)
        self.assertIn("person, school bus", summary)
        self.assertIn("campus building", summary)

    def test_guided_analysis_merges_yolo_objects_and_gemini_fields(self) -> None:
        analysis = _guided_analysis_from_response(
            '{"shot_type":"medium shot","camera_movement":"static","lighting":"soft indoor lighting","additional_subjects_or_objects":["podium"],"clip_type_hint":"interview"}',
            clip_type="handheld",
            fallback=None,  # type: ignore[arg-type]
            fallback_tags=["school"],
            detected_objects=["person", "microphone"],
            frame=ExtractedFrame(frame_offset_sec=3.5, image_bytes=b""),
            provider="gemini",
            model="gemini-2.5-flash",
            cache_signature="test-signature",
        )
        self.assertTrue(analysis.summary.startswith("objects: person, microphone"))
        self.assertIn("person", analysis.summary)
        self.assertIn("microphone", analysis.tags)
        self.assertIn("podium", analysis.tags)
        self.assertEqual(analysis.clip_type_hint, "interview")
        self.assertEqual(analysis.frame_descriptions[0].frame_offset_sec, 3.5)


if __name__ == "__main__":
    unittest.main()
