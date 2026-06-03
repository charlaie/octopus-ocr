from __future__ import annotations

from pathlib import Path

import numpy as np

from octopus_ocr.video import _CandidateFrame, _CoverageSelector, _movement_warning, is_video_path


def test_is_video_path_recognizes_supported_extensions() -> None:
    assert is_video_path(Path("recording.MP4"))
    assert is_video_path(Path("recording.mov"))
    assert is_video_path(Path("recording.m4v"))
    assert not is_video_path(Path("screenshot.PNG"))


def test_coverage_selector_keeps_overlapping_keyframes() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    selector = _CoverageSelector()

    for index, scroll_position in enumerate(range(0, 2100, 200)):
        selector.add(
            _CandidateFrame(
                sequence_index=index,
                video_frame_index=index,
                time_s=float(index),
                image_bgr=image,
                blur_score=float(index),
                scroll_position=float(scroll_position),
                stable=index % 2 == 0,
                coverage_threshold_px=600.0,
                visible_height_px=1400.0,
            )
        )

    selected = selector.finalize()

    assert len(selected) < 11
    assert len(selected) >= 4
    assert selected[0].scroll_position == 0
    assert selected[-1].scroll_position >= 1800
    assert all(
        abs(current.scroll_position - previous.scroll_position) <= 800
        for previous, current in zip(selected, selected[1:])
    )


def test_movement_warning_mentions_possible_skipped_transactions() -> None:
    warning = _movement_warning(Path("video.mp4"), 12.345, 1800.0, 1400.0)

    assert "possible skipped transactions" in warning
    assert "12.35s" in warning
    assert "larger than the visible list height" in warning
