from __future__ import annotations

from octopus_ocr.cli import main
from octopus_ocr.ocr import OcrUnavailableError


def test_cli_reports_missing_ocr(monkeypatch, capsys) -> None:
    def fail_pipeline(*args, **kwargs):
        raise OcrUnavailableError("install tesseract")

    monkeypatch.setattr("octopus_ocr.cli.run_pipeline", fail_pipeline)
    code = main(["data/1.PNG"])
    captured = capsys.readouterr()
    assert code == 2
    assert "install tesseract" in captured.err
