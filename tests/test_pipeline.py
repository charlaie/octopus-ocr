from __future__ import annotations

from pathlib import Path

import pytest

from octopus_ocr.ocr import OcrUnavailableError, TesseractOcr
from octopus_ocr.pipeline import run_pipeline


class MissingOcr(TesseractOcr):
    def assert_available(self) -> None:
        raise OcrUnavailableError("missing test tesseract")


def test_pipeline_fails_fast_when_ocr_is_missing(tmp_path: Path) -> None:
    with pytest.raises(OcrUnavailableError, match="missing test tesseract"):
        run_pipeline([Path("data/1.PNG")], tmp_path, ocr=MissingOcr())
