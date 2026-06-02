from __future__ import annotations

from decimal import Decimal

from octopus_ocr.normalize import canonical_key, normalize_payee, parse_amount, parse_datetime, stable_id


def test_parse_amount_handles_signs_and_symbols() -> None:
    assert parse_amount("-8.6") == (Decimal("-8.6"), "outflow")
    assert parse_amount("−123.8") == (Decimal("-123.8"), "outflow")
    assert parse_amount("+500.0") == (Decimal("500.0"), "inflow")
    assert parse_amount("HK$ 40.0") == (Decimal("40.0"), "inflow")
    assert parse_amount("20") == (Decimal("2.0"), "inflow")
    assert parse_amount("-820") == (Decimal("-82.0"), "outflow")


def test_parse_datetime_requires_octopus_format() -> None:
    parsed = parse_datetime("2026-05-28 19:06")
    assert parsed is not None
    assert parsed.year == 2026
    assert parsed.month == 5
    assert parsed.day == 28
    assert parsed.hour == 19
    assert parsed.minute == 6
    assert parse_datetime("May 28") is None


def test_canonical_key_normalizes_payee_spacing_and_case() -> None:
    when = parse_datetime("2026-05-28 19:06")
    assert when is not None
    key_a = canonical_key(" Yummy\nCat Street Food ", when, Decimal("-10.0"), "eat and drink", "outflow")
    key_b = canonical_key("yummy cat street food", when, Decimal("-10.0"), "eat and drink", "outflow")
    assert key_a == key_b


def test_stable_id_is_deterministic() -> None:
    assert stable_id("txn", "a", 1) == stable_id("txn", "a", 1)
    assert stable_id("txn", "a", 1) != stable_id("txn", "a", 2)


def test_normalize_payee_collapses_lines() -> None:
    assert normalize_payee("PAPER AND\nCOFFEE LIMITED") == "PAPER AND COFFEE LIMITED"
