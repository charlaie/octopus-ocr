from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from octopus_ocr.models import BBox, OcrField
from octopus_ocr.ocr import OcrFieldName, OcrUnavailableError, TesseractOcr
from octopus_ocr.pipeline import _read_row, run_pipeline
from octopus_ocr.vision import RowRegion


def _row_region() -> RowRegion:
    return RowRegion(
        row_bbox=BBox(x=40, y=990, width=1060, height=209),
        icon_bbox=BBox(x=72, y=1047, width=96, height=96),
        category="transport",
        payee_bbox=BBox(x=205, y=1040, width=650, height=56),
        datetime_bbox=BBox(x=205, y=1107, width=430, height=47),
        amount_bbox=BBox(x=850, y=1050, width=215, height=80),
        amount_direction="outflow",
    )


class _FieldReader(TesseractOcr):
    values = {"payee": "MTR", "datetime": "2026-05-16 21:46", "amount": "-12.0"}

    def assert_available(self) -> None:
        return None

    def read_image(self, image: Image.Image, bbox: BBox, *, mode: str) -> OcrField:
        return OcrField(text=self.values[mode], confidence=99.0, bbox=bbox)


class _RowReader(_FieldReader):
    def read_row_fields(
        self, row_image: Image.Image, row_bbox: BBox, field_bboxes: dict[OcrFieldName, BBox]
    ) -> dict[OcrFieldName, OcrField]:
        return {name: OcrField(text=self.values[name], confidence=99.0, bbox=bbox) for name, bbox in field_bboxes.items()}


@pytest.mark.parametrize("write_debug", [False, True])
@pytest.mark.parametrize("use_row_reader", [False, True])
def test_read_row_only_crops_fields_that_are_needed(monkeypatch, tmp_path: Path, write_debug: bool, use_row_reader: bool) -> None:
    row = _row_region()
    crop_calls: list[BBox] = []

    def crop(image, bbox):
        crop_calls.append(bbox)
        return Image.new("RGB", (8, 8))

    monkeypatch.setattr("octopus_ocr.pipeline.to_pil_crop", crop)
    reader = _RowReader() if use_row_reader else _FieldReader()

    candidate = _read_row(
        np.empty((0, 0, 3)), Path("image.png"), 0, row, reader,
        tmp_path if write_debug else None, debug_label="image",
    )

    expected_crops = [] if use_row_reader and not write_debug else [row.payee_bbox, row.datetime_bbox, row.amount_bbox]
    if use_row_reader:
        expected_crops.append(row.row_bbox)
    assert crop_calls == expected_crops
    assert candidate.parsed_payee == "MTR"
    assert candidate.parsed_amount == Decimal("-12.0")
    assert candidate.warnings == []
    assert len(list(tmp_path.glob("*.png"))) == (3 if write_debug else 0)


def test_read_row_falls_back_for_fields_missing_from_row_result(monkeypatch) -> None:
    class PartialRowReader(_RowReader):
        def read_row_fields(self, row_image, row_bbox, field_bboxes):
            fields = super().read_row_fields(row_image, row_bbox, field_bboxes)
            del fields["amount"]
            return fields

    row = _row_region()
    crop_calls = []
    monkeypatch.setattr(
        "octopus_ocr.pipeline.to_pil_crop",
        lambda image, bbox: crop_calls.append(bbox) or Image.new("RGB", (8, 8)),
    )

    candidate = _read_row(np.empty((0, 0, 3)), Path("image.png"), 0, row, PartialRowReader(), None, debug_label="image")

    assert crop_calls == [row.row_bbox, row.amount_bbox]
    assert candidate.parsed_amount == Decimal("-12.0")


def test_read_row_preserves_warning_when_row_reader_fails(monkeypatch) -> None:
    class FailingRowReader(_RowReader):
        def read_row_fields(self, row_image, row_bbox, field_bboxes):
            raise RuntimeError("failed prediction")

    monkeypatch.setattr("octopus_ocr.pipeline.to_pil_crop", lambda image, bbox: Image.new("RGB", (8, 8)))

    candidate = _read_row(np.empty((0, 0, 3)), Path("image.png"), 0, _row_region(), FailingRowReader(), None, debug_label="image")

    assert candidate.parsed_amount == Decimal("-12.0")
    assert candidate.warnings == ["row OCR failed: failed prediction"]


def test_pipeline_keeps_debug_evidence_for_same_named_inputs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("octopus_ocr.pipeline.load_rgb", lambda path: np.zeros((1, 1, 3), dtype=np.uint8))
    monkeypatch.setattr("octopus_ocr.pipeline.detect_rows", lambda image: [_row_region()])
    monkeypatch.setattr("octopus_ocr.pipeline.to_pil_crop", lambda image, bbox: Image.new("RGB", (8, 8)))
    monkeypatch.setattr("octopus_ocr.pipeline.annotate_rows", lambda image, rows: Image.new("RGB", (8, 8)))
    paths = [tmp_path / "first" / "same.png", tmp_path / "second" / "same.png", tmp_path / "unique.png"]
    out_dir = tmp_path / "out"

    run_pipeline(paths, out_dir, ocr=_FieldReader())

    debug_files = list((out_dir / "debug").glob("*.png"))
    assert len(debug_files) == 12
    assert len(list((out_dir / "debug").glob("same_*_annotated.png"))) == 2
    assert len(list((out_dir / "debug").glob("same_*_00_amount.png"))) == 2
    assert (out_dir / "debug" / "unique_annotated.png").is_file()
    assert (out_dir / "debug" / "unique_00_payee.png").is_file()


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
        "write monthly_category_totals.csv",
        "write actual.ofx",
        "write transactions.json",
    ]
    assert all(timing.seconds >= 0 for timing in result.timings)


def test_pipeline_reports_detection_and_row_progress(monkeypatch, tmp_path: Path) -> None:
    class AvailableOcr(TesseractOcr):
        values = {
            "payee": "MTR",
            "datetime": "2026-05-16 21:46",
            "amount": "-12.0",
        }

        def assert_available(self) -> None:
            return None

        def read_image(self, image, bbox: BBox, *, mode: str) -> OcrField:
            return OcrField(text=self.values[mode], confidence=99.0, bbox=bbox)

    row = SimpleNamespace(
        row_bbox=BBox(x=40, y=990, width=1060, height=209),
        icon_bbox=BBox(x=72, y=1047, width=96, height=96),
        category="transport",
        payee_bbox=BBox(x=205, y=1040, width=650, height=56),
        datetime_bbox=BBox(x=205, y=1107, width=430, height=47),
        amount_bbox=BBox(x=850, y=1050, width=215, height=80),
        amount_direction="outflow",
    )
    detect_calls = {"count": 0}

    def fake_detect_rows(image):
        detect_calls["count"] += 1
        return [row] if detect_calls["count"] == 1 else [row, row]

    events = []
    monkeypatch.setattr("octopus_ocr.pipeline.load_rgb", lambda path: object())
    monkeypatch.setattr("octopus_ocr.pipeline.detect_rows", fake_detect_rows)
    monkeypatch.setattr("octopus_ocr.pipeline.to_pil_crop", lambda image, bbox: object())
    monkeypatch.setattr("octopus_ocr.pipeline.dedupe_candidates", lambda candidates: [])

    run_pipeline(
        [Path("image-1.png"), Path("image-2.png")],
        tmp_path,
        write_debug=False,
        ocr=AvailableOcr(),
        progress=events.append,
    )

    assert [event.message for event in events if event.phase == "detect"] == [
        "Detecting image 1/2",
        "Detecting image 2/2",
        "Detected 3 row(s) across 2 image(s)",
    ]
    completed_row_events = [
        event for event in events if event.phase == "ocr" and event.completed_rows > 0
    ]
    assert completed_row_events[-1].completed_rows == 3
    assert completed_row_events[-1].total_rows == 3
    assert events[-1].phase == "complete"


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
