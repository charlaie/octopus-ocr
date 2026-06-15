from __future__ import annotations

from collections import defaultdict

from octopus_ocr.models import TransactionCandidate, TransactionRecord
from octopus_ocr.normalize import canonical_key, stable_id


def dedupe_candidates(candidates: list[TransactionCandidate]) -> list[TransactionRecord]:
    parsed = [
        candidate
        for candidate in candidates
        if candidate.parsed_payee
        and candidate.parsed_datetime
        and candidate.parsed_amount is not None
        and candidate.direction
    ]

    groups: dict[str, list[TransactionCandidate]] = defaultdict(list)
    for candidate in parsed:
        assert candidate.parsed_payee is not None
        assert candidate.parsed_datetime is not None
        assert candidate.parsed_amount is not None
        assert candidate.direction is not None
        key = canonical_key(
            candidate.parsed_payee,
            candidate.parsed_datetime,
            candidate.parsed_amount,
            candidate.category,
            candidate.direction,
        )
        groups[key].append(candidate)

    records: list[TransactionRecord] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        group.sort(key=lambda item: (item.source_image, item.row_index))
        representative = group[0]
        assert representative.parsed_payee is not None
        assert representative.parsed_datetime is not None
        assert representative.parsed_amount is not None
        assert representative.direction is not None

        duplicate_group = stable_id("dup", key, length=12)
        warnings = sorted({warning for candidate in group for warning in candidate.warnings})
        if len(group) > 1:
            warnings.append(f"Matched {len(group)} screenshot rows with the same normalized transaction key.")

        status = "review" if len(group) > 1 else "kept"
        fitid = stable_id("octopus", key, length=24)
        records.append(
            TransactionRecord(
                id=stable_id("txn", key, length=16),
                fitid=fitid,
                source_images=sorted({candidate.source_image for candidate in group}),
                row_refs=[f"{candidate.source_image}#{candidate.row_index}" for candidate in group],
                duplicate_group=duplicate_group,
                dedupe_status=status,
                datetime=representative.parsed_datetime,
                payee=representative.parsed_payee,
                amount=representative.parsed_amount,
                direction=representative.direction,
                category=representative.category,
                confidence=_average_confidence(group),
                warnings=warnings,
                raw_candidates=group,
            )
        )
    return records


def _average_confidence(candidates: list[TransactionCandidate]) -> float | None:
    values = [
        value
        for candidate in candidates
        for value in [
            candidate.payee.confidence,
            candidate.datetime_field.confidence,
            candidate.amount_field.confidence,
        ]
        if value is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 2)
