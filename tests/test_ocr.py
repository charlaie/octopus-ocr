from __future__ import annotations

import sys
from types import SimpleNamespace

from PIL import Image

from octopus_ocr.models import BBox, OcrField
from octopus_ocr.ocr import PaddleOcr, _choose_amount_ocr_result, _trim_ocr_whitespace


def test_amount_ocr_uses_higher_confidence_digit_read_when_primary_is_weak() -> None:
    bbox = BBox(x=0, y=0, width=100, height=50)
    primary = OcrField(text="-4.9", confidence=0.0, bbox=bbox)
    secondary = OcrField(text="79", confidence=47.82, bbox=bbox)

    result = _choose_amount_ocr_result(primary, secondary)

    assert result.text == "-7.9"
    assert result.confidence == 47.82


def test_amount_ocr_uses_secondary_when_primary_loses_integer_digits() -> None:
    bbox = BBox(x=0, y=0, width=100, height=50)
    primary = OcrField(text="-.4", confidence=0.0, bbox=bbox)
    secondary = OcrField(text="-74", confidence=44.49, bbox=bbox)

    result = _choose_amount_ocr_result(primary, secondary)

    assert result.text == "-7.4"
    assert result.confidence == 44.49


def test_amount_ocr_uses_secondary_when_primary_drops_leading_digit() -> None:
    bbox = BBox(x=0, y=0, width=100, height=50)
    primary = OcrField(text="-4.0", confidence=0.0, bbox=bbox)
    secondary = OcrField(text="-74.0", confidence=91.57, bbox=bbox)

    result = _choose_amount_ocr_result(primary, secondary)

    assert result.text == "-74.0"
    assert result.confidence == 91.57


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


def test_trim_ocr_whitespace_keeps_text_padding() -> None:
    image = Image.new("RGB", (100, 50), "white")
    for y in range(20, 31):
        for x in range(60, 71):
            image.putpixel((x, y), (190, 0, 0))

    trimmed = _trim_ocr_whitespace(image, padding=5)

    assert trimmed.size == (21, 21)
    assert trimmed.getpixel((5, 5)) == (190, 0, 0)


def test_paddle_ocr_reads_text_recognition_result(monkeypatch) -> None:
    calls = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def predict(self, *, input):
            calls["predict"] = input
            return [{"rec_texts": ["-28.9"], "rec_scores": [0.98765], "rec_boxes": [[0, 0, 8, 8]]}]

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakePaddleOCR))
    bbox = BBox(x=1, y=2, width=3, height=4)

    field = PaddleOcr(model_name="en_PP-OCRv5_mobile_rec").read_image(
        Image.new("RGB", (8, 8)),
        bbox,
        mode="amount",
    )

    assert calls["init"] == {
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    assert calls["predict"]
    assert field == OcrField(text="-28.9", confidence=98.77, bbox=bbox)


def test_paddle_ocr_maps_row_pipeline_text_to_fields(monkeypatch) -> None:
    class FakePaddleOCR:
        def __init__(self, **kwargs):
            pass

        def predict(self, *, input):
            return [
                {
                    "rec_texts": ["PAPER", "R AND", "41", "COFFEE", "E LIMITED", "-82.0", "2026-05-28 13:44"],
                    "rec_scores": [0.999, 0.905, 0.93, 0.922, 0.933, 0.999, 0.998],
                    "rec_boxes": [
                        [269, 66, 496, 127],
                        [469, 66, 663, 128],
                        [82, 130, 159, 229],
                        [267, 151, 540, 212],
                        [516, 151, 833, 214],
                        [1347, 141, 1542, 220],
                        [267, 252, 752, 301],
                    ],
                }
            ]

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakePaddleOCR))
    row_bbox = BBox(x=30, y=1555, width=804, height=180)
    row_image = Image.new("RGB", (1608, 360), "white")

    fields = PaddleOcr().read_row_fields(
        row_image,
        row_bbox,
        {
            "payee": BBox(x=155, y=1584, width=493, height=84),
            "datetime": BBox(x=155, y=1678, width=326, height=34),
            "amount": BBox(x=645, y=1610, width=163, height=60),
        },
    )

    assert fields["payee"].text == "PAPER AND COFFEE LIMITED"
    assert fields["payee"].confidence == 93.98
    assert fields["datetime"].text == "2026-05-28 13:44"
    assert fields["amount"].text == "-82.0"
