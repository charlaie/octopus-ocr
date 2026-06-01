from __future__ import annotations

from decimal import Decimal

from octopus_ocr.dedupe import dedupe_candidates
from octopus_ocr.models import BBox, OcrField, TransactionCandidate
from octopus_ocr.normalize import parse_datetime


def _candidate(source: str, row: int) -> TransactionCandidate:
    when = parse_datetime("2026-05-28 19:06")
    assert when is not None
    return TransactionCandidate(
        source_image=source,
        row_index=row,
        row_bbox=BBox(x=0, y=0, width=10, height=10),
        icon_bbox=BBox(x=0, y=0, width=10, height=10),
        category="eat and drink",
        payee=OcrField(text="Yummy Cat Street Food", bbox=BBox(x=0, y=0, width=10, height=10)),
        datetime_field=OcrField(text="2026-05-28 19:06", bbox=BBox(x=0, y=0, width=10, height=10)),
        amount_field=OcrField(text="-10.0", bbox=BBox(x=0, y=0, width=10, height=10)),
        parsed_datetime=when,
        parsed_payee="Yummy Cat Street Food",
        parsed_amount=Decimal("-10.0"),
        direction="outflow",
    )


def test_dedupe_groups_repeated_screenshot_rows_for_review() -> None:
    records = dedupe_candidates([_candidate("data/2.PNG", 4), _candidate("data/3.PNG", 0)])
    assert len(records) == 1
    assert records[0].dedupe_status == "review"
    assert len(records[0].source_images) == 2
    assert records[0].fitid.startswith("octopus_")


def test_dedupe_ignores_unparsed_candidates() -> None:
    candidate = _candidate("data/2.PNG", 4)
    candidate.parsed_amount = None
    assert dedupe_candidates([candidate]) == []
