from __future__ import annotations

import argparse
import sys
from pathlib import Path

from octopus_ocr.ocr import OcrUnavailableError
from octopus_ocr.pipeline import run_pipeline
from octopus_ocr.video import DEFAULT_VIDEO_SAMPLE_FPS, extract_video_keyframes, is_video_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="octopus-ocr",
        description="OCR Octopus transaction screenshots or screen recordings and export JSON, CSV, and OFX files.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Octopus screenshot image or screen recording paths.")
    parser.add_argument("--out", type=Path, default=Path("out"), help="Output directory.")
    parser.add_argument("--no-debug", action="store_true", help="Do not write debug crops/annotated screenshots.")
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
    if args.video_sample_fps <= 0:
        parser.error("--video-sample-fps must be greater than zero")

    input_paths = args.inputs
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        parser.error(f"Input file(s) not found: {', '.join(missing)}")

    image_paths: list[Path] = []
    video_warnings: list[str] = []
    for input_path in input_paths:
        if not is_video_path(input_path):
            image_paths.append(input_path)
            continue

        frame_dir = args.out / "frames" / _safe_stem(input_path)
        try:
            extraction = extract_video_keyframes(input_path, frame_dir, sample_fps=args.video_sample_fps)
        except ValueError as exc:
            parser.error(str(exc))
        image_paths.extend(extraction.frames)
        video_warnings.extend(extraction.warnings)
        print(
            "Extracted "
            f"{len(extraction.frames)} keyframe(s) from {extraction.relevant_frame_count} relevant "
            f"sampled frame(s) in {input_path}."
        )

    if not image_paths:
        parser.error("No screenshot images or video keyframes were found.")

    try:
        result = run_pipeline(image_paths, args.out, write_debug=not args.no_debug)
    except OcrUnavailableError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    for warning in video_warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    parsed = len([candidate for candidate in result.candidates if not candidate.warnings])
    print(f"Detected {len(result.candidates)} rows, exported {len(result.transactions)} transactions.")
    if parsed != len(result.candidates):
        print(f"{len(result.candidates) - parsed} row(s) have warnings; inspect {args.out / 'review.csv'}.")
    print(f"Wrote {args.out / 'transactions.json'}")
    print(f"Wrote {args.out / 'review.csv'}")
    print(f"Wrote {args.out / 'actual.ofx'}")
    return 0


def _safe_stem(path: Path) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in path.stem)
    return cleaned.strip("_") or "video"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
