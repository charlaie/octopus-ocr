# Octopus OCR

OCR Octopus transaction screenshots and export:

- `transactions.json`: canonical records with OCR/debug metadata
- `review.csv`: human inspection file
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
```

Inspect `out/review.csv` before importing `out/actual.ofx` into Actual Budget.

## Notes

The pipeline detects transaction rows and categories from the screenshot layout, then runs Tesseract only on cropped payee/date/amount fields. Top-ups are exported as inflows for v1.
