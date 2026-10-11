from __future__ import annotations

from pathlib import Path
import weakref

import cv2
import numpy as np
import pytest

from octopus_ocr.models import BBox
from octopus_ocr.video import (
    _CandidateFrame,
    _CoverageSelector,
    _movement_warning,
    _write_keyframes,
    extract_video_keyframes,
    is_video_path,
)
from octopus_ocr.vision import RowRegion


def test_is_video_path_recognizes_supported_extensions() -> None:
    assert is_video_path(Path("recording.MP4"))
    assert is_video_path(Path("recording.mov"))
    assert is_video_path(Path("recording.m4v"))
    assert not is_video_path(Path("screenshot.PNG"))


def test_coverage_selector_keeps_overlapping_keyframes() -> None:
    selector = _CoverageSelector()

    for index, scroll_position in enumerate(range(0, 2100, 200)):
        selector.add(
            _CandidateFrame(
                sequence_index=index,
                video_frame_index=index,
                blur_score=float(index),
                scroll_position=float(scroll_position),
                stable=index % 2 == 0,
                coverage_threshold_px=600.0,
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


@pytest.mark.parametrize("sample_fps", [0.0, -1.0, float("nan"), float("inf")])
def test_extract_video_keyframes_rejects_invalid_sample_fps(tmp_path: Path, sample_fps: float) -> None:
    with pytest.raises(ValueError, match="finite and greater than zero"):
        extract_video_keyframes(tmp_path / "video.mp4", tmp_path / "frames", sample_fps=sample_fps)


class _FakeCapture:
    def __init__(self, *, opened: bool = True, native_fps: float = 30.0) -> None:
        self.opened = opened
        self.native_fps = native_fps
        self.released = False
        self.remaining_frames = 3

    def isOpened(self) -> bool:
        return self.opened

    def get(self, _property: int) -> float:
        return self.native_fps

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.remaining_frames == 0:
            return False, None
        self.remaining_frames -= 1
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_extract_video_keyframes_releases_capture_when_detection_fails(monkeypatch, tmp_path: Path) -> None:
    capture = _FakeCapture()

    def fail_detection(image):
        raise RuntimeError("detection failed")

    monkeypatch.setattr("octopus_ocr.video.cv2.VideoCapture", lambda _: capture)
    monkeypatch.setattr("octopus_ocr.video.detect_rows", fail_detection)

    with pytest.raises(RuntimeError, match="detection failed"):
        extract_video_keyframes(tmp_path / "video.mp4", tmp_path / "frames")

    assert capture.released


def _row() -> RowRegion:
    bbox = BBox(x=0, y=0, width=64, height=64)
    return RowRegion(bbox, bbox, "transport", bbox, bbox, bbox)


def test_extract_video_keyframes_preserves_sequentially_decoded_pixels(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (64, 64))
    assert writer.isOpened()
    rng = np.random.default_rng(42)
    try:
        for _ in range(16):
            writer.write(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
    finally:
        writer.release()

    expected = []
    capture = cv2.VideoCapture(str(video_path))
    try:
        for index in range(16):
            ok, frame = capture.read()
            assert ok
            if index % 5 == 0:
                expected.append(frame)
    finally:
        capture.release()
    monkeypatch.setattr("octopus_ocr.video.detect_rows", lambda _: [_row()])
    monkeypatch.setattr("octopus_ocr.video._estimate_vertical_motion", lambda *_: 40.0)

    result = extract_video_keyframes(video_path, tmp_path / "frames", sample_fps=2.0, min_rows=1)

    assert result.sampled_frame_count == 4
    assert result.relevant_frame_count == 4
    assert len(result.frames) == 4
    for path, original in zip(result.frames, expected, strict=True):
        assert np.array_equal(cv2.imread(str(path)), original)


def test_extract_video_keyframes_does_not_retain_stationary_frame_pixels(monkeypatch, tmp_path: Path) -> None:
    class TrackingCapture(_FakeCapture):
        def __init__(self) -> None:
            super().__init__(native_fps=5.0)
            self.remaining_frames = 500
            self.references = []

        def read(self):
            # A decoder image from the prior iteration can still be live during the next read.
            assert sum(reference() is not None for reference in self.references) <= 1
            ok, image = super().read()
            if ok:
                self.references.append(weakref.ref(image))
            return ok, image

    captures = [TrackingCapture(), _FakeCapture()]
    pending = iter(captures)
    monkeypatch.setattr("octopus_ocr.video.cv2.VideoCapture", lambda _: next(pending))
    monkeypatch.setattr("octopus_ocr.video.detect_rows", lambda _: [_row()])
    monkeypatch.setattr("octopus_ocr.video._estimate_vertical_motion", lambda *_: 0.0)

    result = extract_video_keyframes(tmp_path / "video.mp4", tmp_path / "frames", min_rows=1)

    assert result.relevant_frame_count == 500
    assert len(result.frames) == 1
    assert all(capture.released for capture in captures)


@pytest.mark.parametrize("failure", ["open", "read", "write"])
def test_write_keyframes_releases_capture_on_failure(monkeypatch, tmp_path: Path, failure: str) -> None:
    capture = _FakeCapture(opened=failure != "open")
    if failure == "read":
        capture.remaining_frames = 0
    monkeypatch.setattr("octopus_ocr.video.cv2.VideoCapture", lambda _: capture)
    if failure == "write":
        monkeypatch.setattr("octopus_ocr.video.cv2.imwrite", lambda *_: False)
    selected = [_CandidateFrame(0, 0, 1.0, 0.0, True, 100.0)]

    with pytest.raises(ValueError):
        _write_keyframes(tmp_path / "source.mp4", tmp_path / "frames", selected)

    assert capture.released


def test_extract_video_keyframes_releases_unopened_capture(monkeypatch, tmp_path: Path) -> None:
    capture = _FakeCapture(opened=False)
    monkeypatch.setattr("octopus_ocr.video.cv2.VideoCapture", lambda _: capture)

    with pytest.raises(ValueError, match="Could not read video"):
        extract_video_keyframes(tmp_path / "video.mp4", tmp_path / "frames")

    assert capture.released


@pytest.mark.parametrize("native_fps", [0.0, float("nan"), float("inf")])
def test_extract_video_keyframes_falls_back_from_invalid_native_fps(monkeypatch, tmp_path: Path, native_fps: float) -> None:
    capture = _FakeCapture(native_fps=native_fps)
    monkeypatch.setattr("octopus_ocr.video.cv2.VideoCapture", lambda _: capture)
    monkeypatch.setattr("octopus_ocr.video.detect_rows", lambda _: [])

    result = extract_video_keyframes(tmp_path / "video.mp4", tmp_path / "frames")

    assert result.sampled_frame_count == 3
    assert result.frames == []
    assert result.warnings
    assert capture.released
