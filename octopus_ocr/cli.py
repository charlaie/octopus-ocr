from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import cast

from octopus_ocr.inputs import prepare_image_paths
from octopus_ocr.models import ProcessTiming
from octopus_ocr.ocr import OcrEngineName, OcrUnavailableError
from octopus_ocr.pipeline import run_pipeline
from octopus_ocr.video import DEFAULT_VIDEO_SAMPLE_FPS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="octopus-ocr",
        description="OCR Octopus transaction screenshots or screen recordings and export JSON, CSV, and OFX files.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Octopus screenshot image or screen recording paths.")
    parser.add_argument("--out", type=Path, default=Path("out"), help="Output directory.")
    parser.add_argument("--no-debug", action="store_true", help="Do not write debug crops/annotated screenshots.")
    parser.add_argument(
        "--ocr-engine",
        choices=["tesseract", "paddle"],
        default="tesseract",
        help="OCR backend to use for cropped text fields. Default: tesseract.",
    )
    parser.add_argument(
        "--paddle-model",
        default="en_PP-OCRv5_mobile_rec",
        help="PaddleOCR TextRecognition model to use with --ocr-engine paddle.",
    )
    parser.add_argument(
        "--video-sample-fps",
        type=float,
        default=DEFAULT_VIDEO_SAMPLE_FPS,
        help=(
            "Frames per second to sample from video inputs before keyframe filtering. "
            f"Default: {DEFAULT_VIDEO_SAMPLE_FPS}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.video_sample_fps) or args.video_sample_fps <= 0:
        parser.error("--video-sample-fps must be finite and greater than zero")

    input_paths = args.inputs
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        parser.error(f"Input file(s) not found: {', '.join(missing)}")

    try:
        prepared = prepare_image_paths(input_paths, args.out / "frames", sample_fps=args.video_sample_fps)
    except ValueError as exc:
        parser.error(str(exc))
    timings: list[ProcessTiming] = []
    for video in prepared.videos:
        timings.append(
            ProcessTiming(
                name=f"extract video keyframes ({video.source_path.name})",
                seconds=video.seconds,
            )
        )
        print(
            "Extracted "
            f"{len(video.extraction.frames)} keyframe(s) from {video.extraction.relevant_frame_count} relevant "
            f"sampled frame(s) in {video.source_path}."
        )

    if not prepared.image_paths:
        parser.error("No screenshot images or video keyframes were found.")

    try:
        result = run_pipeline(
            prepared.image_paths,
            args.out,
            write_debug=not args.no_debug,
            ocr_engine=cast(OcrEngineName, args.ocr_engine),
            paddle_model=args.paddle_model,
        )
    except OcrUnavailableError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    timings.extend(getattr(result, "timings", []))
    for video in prepared.videos:
        for warning in video.extraction.warnings:
            print(f"Warning: {warning}", file=sys.stderr)
    parsed = len([candidate for candidate in result.candidates if not candidate.warnings])
    print(f"Detected {len(result.candidates)} rows, exported {len(result.transactions)} transactions.")
    if parsed != len(result.candidates):
        print(f"{len(result.candidates) - parsed} row(s) have warnings; inspect {args.out / 'review.csv'}.")
    print(f"Wrote {args.out / 'transactions.json'}")
    print(f"Wrote {args.out / 'review.csv'}")
    print(f"Wrote {args.out / 'monthly_category_totals.csv'}")
    print(f"Wrote {args.out / 'actual.ofx'}")
    _print_timings(timings)
    return 0


def _print_timings(timings: list[ProcessTiming]) -> None:
    if not timings:
        return
    label_width = max(len(timing.name) for timing in timings)
    print("Timings:")
    for timing in timings:
        print(f"  {timing.name:<{label_width}}  {_format_duration(timing.seconds)}")
    total_seconds = sum(timing.seconds for timing in timings)
    print(f"  {'total':<{label_width}}  {_format_duration(total_seconds)}")


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:04.1f}s"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
