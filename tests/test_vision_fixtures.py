from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from octopus_ocr.vision import detect_rows, load_rgb


def test_fixture_screenshots_have_detectable_transaction_rows() -> None:
    paths = sorted(Path("data").glob("*.PNG"))
    assert len(paths) == 5
    rows_by_path = [detect_rows(load_rgb(path)) for path in paths]
    counts = [len(rows) for rows in rows_by_path]
    assert all(count >= 8 for count in counts)
    assert max(counts) >= 10

    categories = [row.category for rows in rows_by_path for row in rows]
    assert "transport" in categories
    assert "eat and drink" in categories
    assert "living and others" in categories
    assert "top-up" in categories


def test_fare_subsidy_plus_icon_creates_own_row() -> None:
    rows = detect_rows(load_rgb(Path("data/fare_subsidy.jpeg")))
    categories = [row.category for row in rows]
    subsidy_index = categories.index("fare subsidy")
    subsidy = rows[subsidy_index]
    following = rows[subsidy_index + 1]

    assert subsidy.amount_direction == "inflow"
    assert 1030 <= subsidy.payee_bbox.y <= 1050
    assert subsidy.payee_bbox.height <= 70
    assert 1095 <= subsidy.datetime_bbox.y <= 1120
    assert subsidy.datetime_bbox.height <= 60
    assert following.row_bbox.y >= subsidy.row_bbox.y + subsidy.row_bbox.height


def test_multiline_payee_uses_lower_date_bbox() -> None:
    rows = detect_rows(load_rgb(Path("data/2.PNG")))
    first_paper = rows[8]
    second_paper = rows[9]

    assert first_paper.category == "eat and drink"
    assert second_paper.category == "eat and drink"
    assert second_paper.row_bbox.height > first_paper.row_bbox.height
    assert second_paper.payee_bbox.height > first_paper.payee_bbox.height
    assert second_paper.datetime_bbox.y > second_paper.payee_bbox.y + second_paper.payee_bbox.height - 5
    assert second_paper.amount_direction == "outflow"
