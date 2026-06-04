from __future__ import annotations

from octopus_ocr.models import BBox, OcrField
from octopus_ocr.ocr import _choose_amount_ocr_result


def test_amount_ocr_uses_higher_confidence_digit_read_when_primary_is_weak() -> None:
    bbox = BBox(x=0, y=0, width=100, height=50)
    primary = OcrField(text="-4.9", confidence=0.0, bbox=bbox)
    secondary = OcrField(text="79", confidence=47.82, bbox=bbox)

    result = _choose_amount_ocr_result(primary, secondary)

    assert result.text == "-7.9"
    assert result.confidence == 47.82


def test_amount_ocr_keeps_high_confidence_primary_read() -> None:
    bbox = BBox(x=0, y=0, width=100, height=50)
    primary = OcrField(text="-4.9", confidence=88.54, bbox=bbox)
    secondary = OcrField(text="49", confidence=40.23, bbox=bbox)

    result = _choose_amount_ocr_result(primary, secondary)

    assert result.text == "-4.9"
    assert result.confidence == 88.54


def test_amount_ocr_ignores_incomplete_secondary_read() -> None:
    bbox = BBox(x=0, y=0, width=100, height=50)
    primary = OcrField(text="-4.9", confidence=72.0, bbox=bbox)
    secondary = OcrField(text=".9", confidence=12.0, bbox=bbox)

    result = _choose_amount_ocr_result(primary, secondary)

    assert result.text == "-4.9"
