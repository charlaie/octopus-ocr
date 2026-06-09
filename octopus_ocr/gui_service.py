from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from octopus_ocr.inputs import IMAGE_SUFFIXES, expand_input_paths
from octopus_ocr.models import PipelineResult
from octopus_ocr.ocr import OcrEngineName
from octopus_ocr.pipeline import ProgressCallback, ProgressEvent, run_pipeline
from octopus_ocr.video import DEFAULT_VIDEO_SAMPLE_FPS, extract_video_keyframes, is_video_path


DEFAULT_GUI_OUT_ROOT = Path("out/gui-runs")
DEFAULT_PADDLE_MODEL = "en_PP-OCRv5_mobile_rec"


@dataclass(frozen=True)
class GuiRunResult:
    output_dir: Path
    input_count: int
    detected_rows: int
    exported_transactions: int
    warning_rows: int
    total_seconds: float

    @property
    def output_files(self) -> dict[str, Path]:
        return {
            "transactions": self.output_dir / "transactions.json",
            "review": self.output_dir / "review.csv",
            "monthly_totals": self.output_dir / "monthly_category_totals.csv",
            "ofx": self.output_dir / "actual.ofx",
            "folder": self.output_dir,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "input_count": self.input_count,
            "detected_rows": self.detected_rows,
            "exported_transactions": self.exported_transactions,
            "warning_rows": self.warning_rows,
            "total_seconds": self.total_seconds,
            "output_files": {name: str(path) for name, path in self.output_files.items()},
        }


def create_run_dir(out_root: Path = DEFAULT_GUI_OUT_ROOT, *, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    candidate = out_root / timestamp
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        suffixed = out_root / f"{timestamp}-{suffix}"
        if not suffixed.exists():
            return suffixed
        suffix += 1


def run_gui_pipeline(
    input_paths: list[Path],
    *,
    out_root: Path = DEFAULT_GUI_OUT_ROOT,
    ocr_engine: OcrEngineName = "tesseract",
    paddle_model: str = DEFAULT_PADDLE_MODEL,
    progress: ProgressCallback | None = None,
    video_sample_fps: float = DEFAULT_VIDEO_SAMPLE_FPS,
    now: datetime | None = None,
    pipeline_runner: Callable[..., Any] = run_pipeline,
) -> GuiRunResult:
    expanded_inputs = expand_input_paths(input_paths)
    if not expanded_inputs:
        raise ValueError("Drop at least one supported image, video, or folder.")

    out_dir = create_run_dir(out_root, now=now)
    image_paths = _expand_videos(expanded_inputs, out_dir, progress=progress, video_sample_fps=video_sample_fps)
    if not image_paths:
        raise ValueError("No screenshot images or video keyframes were found.")

    started = perf_counter()
    result = pipeline_runner(
        image_paths,
        out_dir,
        write_debug=True,
        ocr_engine=ocr_engine,
        paddle_model=paddle_model,
        progress=progress,
    )
    total_seconds = perf_counter() - started
    return _summarize_result(result, out_dir, len(expanded_inputs), total_seconds)


def _expand_videos(
    input_paths: list[Path],
    out_dir: Path,
    *,
    progress: ProgressCallback | None,
    video_sample_fps: float,
) -> list[Path]:
    image_paths: list[Path] = []
    for input_path in input_paths:
        if input_path.suffix.lower() in IMAGE_SUFFIXES:
            image_paths.append(input_path)
            continue
        if not is_video_path(input_path):
            continue

        _emit_progress(progress, ProgressEvent("prepare", f"Extracting keyframes from {input_path.name}", output_path=out_dir))
        extraction = extract_video_keyframes(input_path, out_dir / "frames" / _safe_stem(input_path), sample_fps=video_sample_fps)
        image_paths.extend(extraction.frames)
    return image_paths


def _summarize_result(result: PipelineResult, out_dir: Path, input_count: int, total_seconds: float) -> GuiRunResult:
    return GuiRunResult(
        output_dir=out_dir,
        input_count=input_count,
        detected_rows=len(result.candidates),
        exported_transactions=len(result.transactions),
        warning_rows=len([candidate for candidate in result.candidates if candidate.warnings]),
        total_seconds=total_seconds,
    )


def _safe_stem(path: Path) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in path.stem)
    return cleaned.strip("_") or "video"


def _emit_progress(progress: ProgressCallback | None, event: ProgressEvent) -> None:
    if progress is not None:
        progress(event)
