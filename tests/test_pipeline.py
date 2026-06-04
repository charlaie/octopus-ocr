from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from octopus_ocr.models import BBox, OcrField
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


def test_pipeline_exports_fare_subsidy_as_inflow(monkeypatch, tmp_path: Path) -> None:
    class AvailableOcr(TesseractOcr):
        values = {
            "payee": "Fare Subsidy",
            "datetime": "2026-05-16 21:46",
            "amount": "+129.1",
        }

        def assert_available(self) -> None:
            return None

        def read_image(self, image, bbox: BBox, *, mode: str) -> OcrField:
            return OcrField(text=self.values[mode], confidence=99.0, bbox=bbox)

    row = SimpleNamespace(
        row_bbox=BBox(x=40, y=990, width=1060, height=209),
        icon_bbox=BBox(x=72, y=1047, width=96, height=96),
        category="fare subsidy",
        payee_bbox=BBox(x=205, y=1040, width=650, height=56),
        datetime_bbox=BBox(x=205, y=1107, width=430, height=47),
        amount_bbox=BBox(x=850, y=1050, width=215, height=80),
        amount_direction="inflow",
    )
    monkeypatch.setattr("octopus_ocr.pipeline.load_rgb", lambda path: object())
    monkeypatch.setattr("octopus_ocr.pipeline.detect_rows", lambda image: [row])
    monkeypatch.setattr("octopus_ocr.pipeline.to_pil_crop", lambda image, bbox: object())

    result = run_pipeline([Path("fare_subsidy.jpeg")], tmp_path, write_debug=False, ocr=AvailableOcr())

    assert len(result.transactions) == 1
    record = result.transactions[0]
    assert record.payee == "Fare Subsidy"
    assert record.category == "fare subsidy"
    assert record.amount == Decimal("129.1")
    assert record.direction == "inflow"
