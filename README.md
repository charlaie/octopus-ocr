# Octopus OCR

OCR Octopus transaction screenshots and export:

- `transactions.json`: canonical records with OCR/debug metadata
- `review.csv`: human inspection file, including rows with failed OCR
- `monthly_category_totals.csv`: manual verification totals grouped by month and category
- `actual.ofx`: import file for Actual Budget
- `debug/`: cropped OCR fields and annotated screenshots

## Requirements

This project is pinned to Python 3.12. Tesseract is the default OCR system dependency:

```bash
brew install tesseract
```

Install Python dependencies with `uv`:

```bash
uv sync --dev
```

PaddleOCR is also supported for cleaner digital text recognition. Install it into the project environment when you want
to run with `--ocr-engine paddle`:

```bash
uv pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
uv pip install "paddleocr[all]"
```

The default Paddle recognition model is `en_PP-OCRv5_mobile_rec`, a small English/numeric PP-OCRv5 recognizer. Paddle
mode runs Paddle's detection + recognition pipeline on each detected transaction row, then maps detected text boxes back
to payee, date/time, and amount fields. PaddleOCR downloads model weights on first use; if Hugging Face is unavailable,
set `PADDLE_PDX_MODEL_SOURCE=BOS`.

## Usage

```bash
uv run octopus-ocr data/*.PNG --out out/
uv run octopus-ocr data/screen-recording.mp4 --out out/ --video-sample-fps 5
uv run octopus-ocr data/*.PNG --out out/ --ocr-engine paddle
```

Inspect `out/review.csv` and `out/monthly_category_totals.csv` before importing `out/actual.ofx` into Actual Budget.
Rows with an unparsed payee, date, or amount stay in the review CSV and JSON with their OCR evidence and warnings.
They are excluded from monthly totals and OFX until they can be parsed.

## Local GUI

Launch the drag-and-drop macOS GUI with:

```bash
uv run octopus-ocr-gui
```

The GUI accepts screenshots, screen recordings, and folders. It writes each run to a timestamped folder under
`out/gui-runs/` and shows image/row progress while OCR is running.

Add files to the source queue, choose a recognition engine in the sidebar, then select **Extract transactions**.
You can remove individual files before a run. Paddle's model setting appears when Paddle is selected. When the run
finishes, open the review CSV to check the results, then use the OFX file to import into Actual Budget. Monthly totals,
transaction JSON, and the output folder are also available.

The desktop shell uses pywebview, with macOS WebKit rendering a local HTML/CSS/JavaScript interface. Native file
dialogs and file drops pass paths to the Python bridge, and OCR runs on a background thread. This is sufficient for
the app's file queue, settings, progress, and exports. The UI assets live in `octopus_ocr/assets/ui/` and are bundled
with the Python package. No frontend build or external UI resources are required.

To create a local double-clickable app launcher:

```bash
uv run octopus-ocr-make-app
```

This writes `dist/Octopus OCR.app`, which launches the GUI from this checkout with `uv`.

Video input is treated as a screenshot source: the CLI samples the recording, keeps coverage-based keyframes in
`out/frames/`, and then runs the same screenshot OCR pipeline. `--video-sample-fps` controls video sampling before
keyframe filtering; it does not mean every sampled frame is OCRed.

## Notes

The pipeline detects transaction rows and categories from the screenshot layout. Tesseract runs on cropped
payee/date/amount fields; Paddle runs on whole row crops and maps detected text boxes back to those fields. Top-ups and
fare subsidies are exported as inflows when their amount text is positive/green.
