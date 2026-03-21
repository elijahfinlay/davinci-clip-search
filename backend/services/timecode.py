from __future__ import annotations


def nominal_fps(value: float) -> int:
    if abs(value - 23.976) < 0.01:
        return 24
    if abs(value - 29.97) < 0.01:
        return 30
    if abs(value - 59.94) < 0.01:
        return 60
    return int(round(value))


def frames_to_timecode(total_frames: int, fps: float, drop_frame: bool = False) -> str:
    fps_int = nominal_fps(fps)
    if drop_frame and fps_int in {30, 60}:
        drop_frames = 2 if fps_int == 30 else 4
        frames_per_10_minutes = fps_int * 60 * 10 - drop_frames * 9
        frames_per_minute = fps_int * 60 - drop_frames
        actual_frames_per_hour = frames_per_10_minutes * 6
        actual_frames_per_24_hours = actual_frames_per_hour * 24
        total_frames %= actual_frames_per_24_hours

        ten_minute_chunks = total_frames // frames_per_10_minutes
        remaining_frames = total_frames % frames_per_10_minutes
        frame_number = total_frames + drop_frames * 9 * ten_minute_chunks
        if remaining_frames >= drop_frames:
            frame_number += drop_frames * ((remaining_frames - drop_frames) // frames_per_minute)

        seconds_total, frames = divmod(frame_number, fps_int)
        minutes_total, seconds = divmod(seconds_total, 60)
        hours, minutes = divmod(minutes_total, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d};{frames:02d}"

    seconds_total, frames = divmod(total_frames, fps_int)
    minutes_total, seconds = divmod(seconds_total, 60)
    hours, minutes = divmod(minutes_total, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def timecode_to_frames(timecode: str, fps: float, drop_frame: bool = False) -> int:
    cleaned = timecode.strip()
    delimiter = ";" if ";" in cleaned else ":"
    parts = cleaned.replace(";", ":").split(":")
    if len(parts) != 4:
        raise ValueError(f"Invalid timecode: {timecode}")
    hours, minutes, seconds, frames = [int(part) for part in parts]
    fps_int = nominal_fps(fps)

    if drop_frame or delimiter == ";":
        drop_frames = 2 if fps_int == 30 else 4 if fps_int == 60 else 0
        total_minutes = hours * 60 + minutes
        return (
            ((hours * 3600) + (minutes * 60) + seconds) * fps_int
            + frames
            - drop_frames * (total_minutes - total_minutes // 10)
        )

    return ((hours * 3600) + (minutes * 60) + seconds) * fps_int + frames


def timeline_frame_to_timecode(
    *,
    timeline_start_frame: int,
    timeline_start_timecode: str,
    frame: int,
    timeline_fps: float,
    drop_frame: bool = False,
) -> str:
    timeline_base_frames = timecode_to_frames(
        timeline_start_timecode,
        timeline_fps,
        drop_frame=drop_frame,
    )
    return frames_to_timecode(
        timeline_base_frames + (frame - timeline_start_frame),
        timeline_fps,
        drop_frame=drop_frame,
    )
