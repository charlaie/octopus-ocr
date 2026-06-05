from __future__ import annotations

import csv
from decimal import Decimal
from typing import Literal

from octopus_ocr.exporters import write_monthly_category_csv, write_ofx, write_review_csv
from octopus_ocr.models import Category, TransactionRecord
from octopus_ocr.normalize import parse_datetime


def _record(
    *,
    dedupe_status: Literal["kept", "duplicate", "review"] = "kept",
    warnings: list[str] | None = None,
    datetime_text: str = "2026-05-28 19:06",
    amount: Decimal = Decimal("-10.0"),
    category: Category = "eat and drink",
) -> TransactionRecord:
    when = parse_datetime(datetime_text)
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
        amount=amount,
        direction="inflow" if amount >= 0 else "outflow",
        category=category,
        warnings=warnings or [],
    )


def test_write_review_csv(tmp_path) -> None:
    path = tmp_path / "review.csv"
    write_review_csv([_record()], path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["payee"] == "Yummy Cat Street Food"
    assert rows[0]["amount"] == "-10.00"
    assert rows[0]["fitid"] == "octopus_abc"


def test_write_monthly_category_csv_groups_totals_by_month_and_category(tmp_path) -> None:
    path = tmp_path / "monthly_category_totals.csv"
    write_monthly_category_csv(
        [
            _record(datetime_text="2026-05-28 19:06", amount=Decimal("-10.0"), category="eat and drink"),
            _record(datetime_text="2026-05-27 09:00", amount=Decimal("-4.5"), category="eat and drink"),
            _record(datetime_text="2026-05-03 08:15", amount=Decimal("-12.0"), category="transport"),
            _record(datetime_text="2026-04-30 18:30", amount=Decimal("100.0"), category="top-up"),
        ],
        path,
    )

    rows = list(csv.DictReader(path.open(encoding="utf-8")))

    assert rows == [
        {"month": "2026-05", "category": "transport", "transaction_count": "1", "total_amount": "-12.00"},
        {"month": "2026-05", "category": "eat and drink", "transaction_count": "2", "total_amount": "-14.50"},
        {"month": "2026-05", "category": "TOTAL", "transaction_count": "3", "total_amount": "-26.50"},
        {"month": "2026-04", "category": "top-up", "transaction_count": "1", "total_amount": "100.00"},
        {"month": "2026-04", "category": "TOTAL", "transaction_count": "1", "total_amount": "100.00"},
    ]


def test_write_monthly_category_csv_excludes_duplicates(tmp_path) -> None:
    path = tmp_path / "monthly_category_totals.csv"
    write_monthly_category_csv(
        [
            _record(amount=Decimal("-10.0"), category="eat and drink"),
            _record(amount=Decimal("-10.0"), category="eat and drink", dedupe_status="duplicate"),
            _record(amount=Decimal("-5.0"), category="transport", dedupe_status="review"),
        ],
        path,
    )

    rows = list(csv.DictReader(path.open(encoding="utf-8")))

    assert rows == [
        {"month": "2026-05", "category": "transport", "transaction_count": "1", "total_amount": "-5.00"},
        {"month": "2026-05", "category": "eat and drink", "transaction_count": "1", "total_amount": "-10.00"},
        {"month": "2026-05", "category": "TOTAL", "transaction_count": "2", "total_amount": "-15.00"},
    ]


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
