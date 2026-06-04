from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from octopus_ocr.ocr import OcrUnavailableError, TesseractOcr
from octopus_ocr.pipeline import run_pipeline


class MissingOcr(TesseractOcr):
    def assert_available(self) -> None:
        raise OcrUnavailableError("missing test tesseract")


def test_pipeline_fails_fast_when_ocr_is_missing(tmp_path: Path) -> None:
    with pytest.raises(OcrUnavailableError, match="missing test tesseract"):
        run_pipeline([Path("data/1.PNG")], tmp_path, ocr=MissingOcr())


def test_pipeline_returns_step_timings(monkeypatch, tmp_path: Path) -> None:
    class AvailableOcr(TesseractOcr):
        def assert_available(self) -> None:
            return None

    monkeypatch.setattr("octopus_ocr.pipeline.load_rgb", lambda path: object())
    monkeypatch.setattr("octopus_ocr.pipeline.detect_rows", lambda image: [])
    monkeypatch.setattr("octopus_ocr.pipeline.dedupe_candidates", lambda candidates: [])
    monkeypatch.setattr(
        "octopus_ocr.pipeline.annotate_rows",
        lambda image, rows: SimpleNamespace(save=lambda path: None),
    )

    result = run_pipeline([Path("image-1.png"), Path("image-2.png")], tmp_path, ocr=AvailableOcr())

    timing_names = [timing.name for timing in result.timings]
    assert timing_names == [
        "prepare output directories",
        "initialize OCR engine",
        "load images",
        "detect rows",
        "write debug annotations",
        "dedupe transactions",
        "write review.csv",
        "write actual.ofx",
        "write transactions.json",
    ]
    assert all(timing.seconds >= 0 for timing in result.timings)
