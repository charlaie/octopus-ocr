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
