from __future__ import annotations

import csv
from decimal import Decimal
from typing import Literal

from octopus_ocr.exporters import write_ofx, write_review_csv
from octopus_ocr.models import TransactionRecord
from octopus_ocr.normalize import parse_datetime


def _record(
    *,
    dedupe_status: Literal["kept", "duplicate", "review"] = "kept",
    warnings: list[str] | None = None,
) -> TransactionRecord:
    when = parse_datetime("2026-05-28 19:06")
    assert when is not None
    return TransactionRecord(
        id="txn_abc",
        fitid="octopus_abc",
        source_images=["data/2.PNG"],
        row_refs=["data/2.PNG#4"],
        duplicate_group="dup_abc",
        dedupe_status=dedupe_status,
        datetime=when,
        payee="Yummy Cat Street Food",
        amount=Decimal("-10.0"),
        direction="outflow",
        category="eat and drink",
        warnings=warnings or [],
    )


def test_write_review_csv(tmp_path) -> None:
    path = tmp_path / "review.csv"
    write_review_csv([_record()], path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["payee"] == "Yummy Cat Street Food"
    assert rows[0]["amount"] == "-10.00"
    assert rows[0]["fitid"] == "octopus_abc"


def test_write_ofx_contains_transaction_fields(tmp_path) -> None:
    path = tmp_path / "actual.ofx"
    write_ofx([_record()], path)
    content = path.read_text(encoding="utf-8")
    assert "<FITID>octopus_abc" in content
    assert "<TRNAMT>-10.00" in content
    assert "<NAME>Yummy Cat Street Food" in content


def test_write_ofx_memo_contains_only_category(tmp_path) -> None:
    path = tmp_path / "actual.ofx"
    write_ofx(
        [
            _record(
                dedupe_status="review",
                warnings=["Matched 2 screenshot rows with the same normalized transaction key."],
            )
        ],
        path,
    )
    content = path.read_text(encoding="utf-8")
    assert "<MEMO>eat and drink" in content
    assert "review" not in content
    assert "Matched 2 screenshot rows" not in content
