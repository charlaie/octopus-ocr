from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from octopus_ocr.vision import RowRegion, detect_rows

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}
DEFAULT_VIDEO_SAMPLE_FPS = 5.0
MIN_TRANSACTION_ROWS = 8
STABLE_SPEED_PX_PER_SECOND = 250.0
MOTION_SCALE = 0.25


@dataclass(frozen=True)
class VideoExtractionResult:
    frames: list[Path]
    warnings: list[str]
    sampled_frame_count: int
    relevant_frame_count: int


@dataclass(frozen=True)
class _CandidateFrame:
    sequence_index: int
    video_frame_index: int
    time_s: float
    image_bgr: np.ndarray
    blur_score: float
    scroll_position: float
    stable: bool
    coverage_threshold_px: float
    visible_height_px: float


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def extract_video_keyframes(
    video_path: Path,
    frames_dir: Path,
    *,
    sample_fps: float = DEFAULT_VIDEO_SAMPLE_FPS,
    min_rows: int = MIN_TRANSACTION_ROWS,
) -> VideoExtractionResult:
    if sample_fps <= 0:
        raise ValueError("Video sample FPS must be greater than zero.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not read video: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if native_fps <= 0:
        native_fps = sample_fps

    frames_dir.mkdir(parents=True, exist_ok=True)
    selector = _CoverageSelector()
    warnings: list[str] = []
    sampled_frame_count = 0
    relevant_frame_count = 0
    sequence_index = 0
    frame_index = 0
    next_sample_time = 0.0
    previous_motion_image: np.ndarray | None = None
    previous_time_s: float | None = None
    scroll_position = 0.0

    while True:
        ok, image_bgr = cap.read()
        if not ok:
            break

        time_s = frame_index / native_fps
        if time_s + 1e-9 < next_sample_time:
            frame_index += 1
            continue

        sampled_frame_count += 1
        next_sample_time += 1.0 / sample_fps
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rows = detect_rows(image_rgb)
        if len(rows) < min_rows:
            frame_index += 1
            continue

        relevant_frame_count += 1
        motion_image = _motion_image(image_bgr)
        visible_height = _visible_height(rows)
        motion_px: float | None = None
        speed_px_s: float | None = None
        if previous_motion_image is not None and previous_time_s is not None:
            estimated_motion = _estimate_vertical_motion(previous_motion_image, motion_image)
            if estimated_motion is not None:
                motion_px = estimated_motion
                scroll_position += motion_px
                elapsed = max(time_s - previous_time_s, 1e-6)
                speed_px_s = abs(motion_px) / elapsed
                if abs(motion_px) > visible_height:
                    warnings.append(_movement_warning(video_path, time_s, motion_px, visible_height))

        previous_motion_image = motion_image
        previous_time_s = time_s
        candidate = _CandidateFrame(
            sequence_index=sequence_index,
            video_frame_index=frame_index,
            time_s=time_s,
            image_bgr=image_bgr.copy(),
            blur_score=_blur_score(image_bgr),
            scroll_position=scroll_position,
            stable=speed_px_s is None or speed_px_s <= STABLE_SPEED_PX_PER_SECOND,
            coverage_threshold_px=_coverage_threshold(rows),
            visible_height_px=visible_height,
        )
        selector.add(candidate)
        sequence_index += 1
        frame_index += 1

    cap.release()

    selected = selector.finalize()
    frame_paths: list[Path] = []
    for output_index, candidate in enumerate(selected, start=1):
        output_path = frames_dir / f"frame_{output_index:04d}.png"
        if not cv2.imwrite(str(output_path), candidate.image_bgr):
            raise ValueError(f"Could not write extracted video frame: {output_path}")
        frame_paths.append(output_path)

    if sampled_frame_count > 0 and not frame_paths:
        warnings.append(f"{video_path}: no transaction-list keyframes were selected from sampled video frames.")

    return VideoExtractionResult(
        frames=frame_paths,
        warnings=warnings,
        sampled_frame_count=sampled_frame_count,
        relevant_frame_count=relevant_frame_count,
    )


class _CoverageSelector:
    def __init__(self) -> None:
        self._selected: list[_CandidateFrame] = []
        self._bucket: list[_CandidateFrame] = []

    def add(self, candidate: _CandidateFrame) -> None:
        if not self._selected:
            self._selected.append(candidate)
            return

        last_selected = self._selected[-1]
        self._bucket.append(candidate)
        threshold = _threshold_between(last_selected, candidate)
        if abs(candidate.scroll_position - last_selected.scroll_position) < threshold:
            return

        eligible = [
            frame
            for frame in self._bucket
            if abs(frame.scroll_position - last_selected.scroll_position) >= threshold * 0.8
        ]
        choice = _best_keyframe(eligible or self._bucket, last_selected.scroll_position)
        if choice.sequence_index != last_selected.sequence_index:
            self._selected.append(choice)
        self._bucket = [frame for frame in self._bucket if frame.sequence_index > choice.sequence_index]

    def finalize(self) -> list[_CandidateFrame]:
        if not self._selected:
            return []
        if self._bucket:
            last_selected = self._selected[-1]
            last_candidate = self._bucket[-1]
            threshold = _threshold_between(last_selected, last_candidate)
            if abs(last_candidate.scroll_position - last_selected.scroll_position) >= threshold * 0.35:
                eligible = [
                    frame
                    for frame in self._bucket
                    if abs(frame.scroll_position - last_selected.scroll_position) >= threshold * 0.25
                ]
                choice = _best_keyframe(eligible or self._bucket, last_selected.scroll_position)
                if choice.sequence_index != last_selected.sequence_index:
                    self._selected.append(choice)
        return sorted(self._selected, key=lambda frame: frame.sequence_index)


def _best_keyframe(candidates: Sequence[_CandidateFrame], last_scroll_position: float) -> _CandidateFrame:
    return max(
        candidates,
        key=lambda frame: (
            1 if frame.stable else 0,
            frame.blur_score,
            abs(frame.scroll_position - last_scroll_position),
        ),
    )


def _threshold_between(previous: _CandidateFrame, current: _CandidateFrame) -> float:
    return max(previous.coverage_threshold_px, current.coverage_threshold_px)


def _motion_image(image_bgr: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    y1 = int(height * 0.10)
    y2 = int(height * 0.95)
    x1 = int(width * 0.04)
    x2 = int(width * 0.96)
    gray = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (0, 0), fx=MOTION_SCALE, fy=MOTION_SCALE, interpolation=cv2.INTER_AREA).astype(np.float32)


def _estimate_vertical_motion(previous: np.ndarray, current: np.ndarray) -> float | None:
    if previous.shape != current.shape:
        return None
    (_dx, dy), response = cv2.phaseCorrelate(previous, current)
    if response < 0.05:
        return None
    return float(dy / MOTION_SCALE)


def _blur_score(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _coverage_threshold(rows: Sequence[RowRegion]) -> float:
    visible_height = _visible_height(rows)
    spacing = _row_spacing(rows)
    return min(max(spacing * 3.5, visible_height * 0.40), visible_height * 0.50)


def _visible_height(rows: Sequence[RowRegion]) -> float:
    top = min(row.row_bbox.y for row in rows)
    bottom = max(row.row_bbox.y + row.row_bbox.height for row in rows)
    return max(float(bottom - top), 1.0)


def _row_spacing(rows: Sequence[RowRegion]) -> float:
    centers = sorted(row.icon_bbox.y + row.icon_bbox.height / 2 for row in rows)
    if len(centers) < 2:
        return 160.0
    spacings = np.diff(np.array(centers, dtype=np.float64))
    return max(float(np.median(spacings)), 1.0)


def _movement_warning(video_path: Path, time_s: float, motion_px: float, visible_height_px: float) -> str:
    return (
        f"{video_path}: possible skipped transactions near {time_s:.2f}s; "
        f"estimated scroll jump was {abs(motion_px):.0f}px, larger than the visible list height "
        f"({visible_height_px:.0f}px)."
    )
