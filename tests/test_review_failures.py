from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
import pytest

from octopus_ocr.models import BBox, OcrField
from octopus_ocr.pipeline import run_pipeline
from octopus_ocr.vision import RowRegion


@pytest.mark.parametrize(
    ("failed_field", "raw_text", "review_column"),
    [("payee", "", "payee"), ("datetime", "not a date", "datetime"), ("amount", "unreadable", "amount")],
)
def test_failed_rows_remain_reviewable_without_becoming_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_field: str,
    raw_text: str,
    review_column: str,
) -> None:
    bad_source = tmp_path / "bad.png"
    good_source = tmp_path / "good.png"
    for path in (bad_source, good_source):
        Image.new("RGB", (80, 80), "white").save(path)
    row = RowRegion(
        row_bbox=BBox(x=0, y=0, width=80, height=80),
        icon_bbox=BBox(x=0, y=0, width=8, height=8),
        category="transport",
        payee_bbox=BBox(x=8, y=8, width=60, height=12),
        datetime_bbox=BBox(x=8, y=24, width=60, height=12),
        amount_bbox=BBox(x=48, y=40, width=24, height=12),
        amount_direction=None,
    )
    monkeypatch.setattr("octopus_ocr.pipeline.detect_rows", lambda _image: [row])

    class FakeOcr:
        calls = 0

        def assert_available(self) -> None:
            pass

        def read_image(self, image: Image.Image, bbox: BBox, *, mode: str) -> OcrField:
            values = {"payee": "MTR", "datetime": "2026-05-16 21:46", "amount": "-12.0"}
            text = raw_text if self.calls < 3 and mode == failed_field else values[mode]
            self.calls += 1
            return OcrField(text=text, confidence=99.0, bbox=bbox)

    output = tmp_path / "out"
    result = run_pipeline([bad_source, good_source], output, write_debug=False, ocr=FakeOcr())

    assert len(result.candidates) == 2
    assert not result.candidates[0].is_parsed
    assert result.candidates[1].is_parsed
    assert len(result.transactions) == 1
    with (output / "review.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["fitid"] == result.transactions[0].fitid
    partial = rows[1]
    assert partial[review_column] == raw_text
    assert partial["source_images"] == str(bad_source)
    assert partial["row_refs"] == f"{bad_source}#0"
    assert partial["dedupe_status"] == "review"
    assert partial["fitid"] == ""
    assert partial["duplicate_group"] == ""
    assert f"{failed_field} did not parse" in partial["warnings"]
    with (output / "monthly_category_totals.csv").open(encoding="utf-8", newline="") as file:
        monthly = list(csv.DictReader(file))
    assert monthly[-1]["transaction_count"] == "1"
    assert monthly[-1]["total_amount"] == "-12.00"
    assert (output / "actual.ofx").read_text(encoding="utf-8").count("<STMTTRN>") == 1
