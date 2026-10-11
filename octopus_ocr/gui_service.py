from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol

from octopus_ocr.inputs import expand_input_paths, prepare_image_paths
from octopus_ocr.models import PipelineResult
from octopus_ocr.ocr import OcrEngineName
from octopus_ocr.pipeline import ProgressCallback, ProgressEvent, run_pipeline
from octopus_ocr.video import DEFAULT_VIDEO_SAMPLE_FPS


DEFAULT_GUI_OUT_ROOT = Path("out/gui-runs")
DEFAULT_PADDLE_MODEL = "en_PP-OCRv5_mobile_rec"


class PipelineRunner(Protocol):
    def __call__(
        self, image_paths: list[Path], out_dir: Path, *, write_debug: bool,
        ocr_engine: OcrEngineName, paddle_model: str, progress: ProgressCallback | None,
    ) -> PipelineResult: ...


@dataclass(frozen=True)
class GuiRunResult:
    output_dir: Path
    input_count: int
    detected_rows: int
    exported_transactions: int
    warning_rows: int
    total_seconds: float
    video_warnings: tuple[str, ...] = ()

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
            "video_warnings": list(self.video_warnings),
            "output_files": {name: str(path) for name, path in self.output_files.items()},
        }


def create_run_dir(out_root: Path = DEFAULT_GUI_OUT_ROOT, *, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        candidate = out_root / (timestamp if suffix == 1 else f"{timestamp}-{suffix}")
        try:
            candidate.mkdir()
        except FileExistsError:
            suffix += 1
            continue
        return candidate


def run_gui_pipeline(
    input_paths: list[Path],
    *,
    out_root: Path = DEFAULT_GUI_OUT_ROOT,
    ocr_engine: OcrEngineName = "tesseract",
    paddle_model: str = DEFAULT_PADDLE_MODEL,
    progress: ProgressCallback | None = None,
    video_sample_fps: float = DEFAULT_VIDEO_SAMPLE_FPS,
    now: datetime | None = None,
    pipeline_runner: PipelineRunner = run_pipeline,
) -> GuiRunResult:
    expanded_inputs = expand_input_paths(input_paths)
    if not expanded_inputs:
        raise ValueError("Drop at least one supported image, video, or folder.")

    started = perf_counter()
    out_dir = create_run_dir(out_root, now=now)

    def on_video_start(path: Path) -> None:
        if progress is not None:
            progress(ProgressEvent("prepare", f"Extracting keyframes from {path.name}", output_path=out_dir))

    prepared = prepare_image_paths(
        expanded_inputs, out_dir / "frames", sample_fps=video_sample_fps, on_video_start=on_video_start,
    )
    if not prepared.image_paths:
        raise ValueError("No screenshot images or video keyframes were found.")

    result = pipeline_runner(
        prepared.image_paths,
        out_dir,
        write_debug=True,
        ocr_engine=ocr_engine,
        paddle_model=paddle_model,
        progress=progress,
    )
    total_seconds = perf_counter() - started
    return GuiRunResult(
        output_dir=out_dir,
        input_count=len(expanded_inputs),
        detected_rows=len(result.candidates),
        exported_transactions=len(result.transactions),
        warning_rows=sum(bool(candidate.warnings) for candidate in result.candidates),
        total_seconds=total_seconds,
        video_warnings=tuple(warning for video in prepared.videos for warning in video.extraction.warnings),
    )
