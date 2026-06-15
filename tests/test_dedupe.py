from __future__ import annotations

from decimal import Decimal

from octopus_ocr.dedupe import dedupe_candidates
from octopus_ocr.models import BBox, OcrField, TransactionCandidate
from octopus_ocr.normalize import parse_datetime


def _candidate(
    source: str,
    row: int,
    *,
    datetime_text: str = "2026-05-28 19:06",
    payee: str = "Yummy Cat Street Food",
    amount: Decimal = Decimal("-10.0"),
) -> TransactionCandidate:
    when = parse_datetime(datetime_text)
    assert when is not None
    return TransactionCandidate(
        source_image=source,
        row_index=row,
        row_bbox=BBox(x=0, y=0, width=10, height=10),
        icon_bbox=BBox(x=0, y=0, width=10, height=10),
        category="eat and drink",
        payee=OcrField(text=payee, bbox=BBox(x=0, y=0, width=10, height=10)),
        datetime_field=OcrField(text=datetime_text, bbox=BBox(x=0, y=0, width=10, height=10)),
        amount_field=OcrField(text=str(amount), bbox=BBox(x=0, y=0, width=10, height=10)),
        parsed_datetime=when,
        parsed_payee=payee,
        parsed_amount=amount,
        direction="inflow" if amount >= 0 else "outflow",
    )


def test_dedupe_groups_repeated_screenshot_rows_for_review() -> None:
    records = dedupe_candidates([_candidate("data/2.PNG", 4), _candidate("data/3.PNG", 0)])
    assert len(records) == 1
    assert records[0].dedupe_status == "review"
    assert len(records[0].source_images) == 2
    assert records[0].fitid.startswith("octopus_")


def test_fitid_is_stable_across_overlapping_runs() -> None:
    first_run = dedupe_candidates(
        [
            _candidate(
                "data/jun10.PNG",
                0,
                datetime_text="2026-06-10 09:00",
                payee="Earlier",
                amount=Decimal("-1.0"),
            ),
            _candidate("data/jun18.PNG", 0, datetime_text="2026-06-18 12:34"),
            _candidate(
                "data/jun19.PNG",
                0,
                datetime_text="2026-06-19 18:00",
                payee="Later",
                amount=Decimal("-2.0"),
            ),
        ]
    )
    second_run = dedupe_candidates(
        [
            _candidate("data/jun18-again.PNG", 0, datetime_text="2026-06-18 12:34"),
            _candidate("data/jun29.PNG", 0, datetime_text="2026-06-29 20:00", payee="Newest", amount=Decimal("-3.0")),
        ]
    )

    first_overlap = next(record for record in first_run if record.datetime.day == 18)
    second_overlap = next(record for record in second_run if record.datetime.day == 18)
    assert first_overlap.fitid == second_overlap.fitid
    assert first_overlap.id == second_overlap.id


def test_dedupe_ignores_unparsed_candidates() -> None:
    candidate = _candidate("data/2.PNG", 4)
    candidate.parsed_amount = None
    assert dedupe_candidates([candidate]) == []
