from __future__ import annotations

import argparse
import sys
from pathlib import Path

from octopus_ocr.ocr import OcrUnavailableError
from octopus_ocr.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="octopus-ocr",
        description="OCR Octopus transaction screenshots and export JSON, CSV, and OFX files.",
    )
    parser.add_argument("images", nargs="+", type=Path, help="Octopus screenshot image paths.")
    parser.add_argument("--out", type=Path, default=Path("out"), help="Output directory.")
    parser.add_argument("--no-debug", action="store_true", help="Do not write debug crops/annotated screenshots.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    image_paths = sorted(args.images)
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        parser.error(f"Input image(s) not found: {', '.join(missing)}")

    try:
        result = run_pipeline(image_paths, args.out, write_debug=not args.no_debug)
    except OcrUnavailableError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    parsed = len([candidate for candidate in result.candidates if not candidate.warnings])
    print(f"Detected {len(result.candidates)} rows, exported {len(result.transactions)} transactions.")
    if parsed != len(result.candidates):
        print(f"{len(result.candidates) - parsed} row(s) have warnings; inspect {args.out / 'review.csv'}.")
    print(f"Wrote {args.out / 'transactions.json'}")
    print(f"Wrote {args.out / 'review.csv'}")
    print(f"Wrote {args.out / 'actual.ofx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
