# Octopus OCR

OCR Octopus transaction screenshots and export:

- `transactions.json`: canonical records with OCR/debug metadata
- `review.csv`: human inspection file
- `monthly_category_totals.csv`: manual verification totals grouped by month and category
- `actual.ofx`: import file for Actual Budget
- `debug/`: cropped OCR fields and annotated screenshots

## Requirements

This project is pinned to Python 3.12. Tesseract is a system dependency:

```bash
brew install tesseract
```

Install Python dependencies with `uv`:

```bash
uv sync --dev
```

## Usage

```bash
uv run octopus-ocr data/*.PNG --out out/
uv run octopus-ocr data/screen-recording.mp4 --out out/ --video-sample-fps 5
```

Inspect `out/review.csv` and `out/monthly_category_totals.csv` before importing `out/actual.ofx` into Actual Budget.

Video input is treated as a screenshot source: the CLI samples the recording, keeps coverage-based keyframes in
`out/frames/`, and then runs the same screenshot OCR pipeline. `--video-sample-fps` controls video sampling before
keyframe filtering; it does not mean every sampled frame is OCRed.

## Notes

The pipeline detects transaction rows and categories from the screenshot layout, then runs Tesseract only on cropped payee/date/amount fields. Top-ups and fare subsidies are exported as inflows when their amount text is positive/green.
