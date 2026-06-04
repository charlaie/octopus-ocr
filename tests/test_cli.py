from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from octopus_ocr.cli import _format_duration, main
from octopus_ocr.models import ProcessTiming
from octopus_ocr.ocr import OcrUnavailableError
from octopus_ocr.video import VideoExtractionResult


def test_cli_reports_missing_ocr(monkeypatch, capsys) -> None:
    def fail_pipeline(*args, **kwargs):
        raise OcrUnavailableError("install tesseract")

    monkeypatch.setattr("octopus_ocr.cli.run_pipeline", fail_pipeline)
    code = main(["data/1.PNG"])
    captured = capsys.readouterr()
    assert code == 2
    assert "install tesseract" in captured.err


def test_cli_expands_video_inputs(monkeypatch, tmp_path: Path, capsys) -> None:
    video_path = tmp_path / "screen recording.mp4"
    video_path.write_bytes(b"not a real video because extraction is stubbed")
    extracted_frame = tmp_path / "frame_0001.png"
    calls = {}

    def fake_extract(video_path_arg, frames_dir, *, sample_fps):
        calls["extract"] = (video_path_arg, frames_dir, sample_fps)
        return VideoExtractionResult(
            frames=[extracted_frame],
            warnings=["possible skipped transactions"],
            sampled_frame_count=10,
            relevant_frame_count=8,
        )

    def fake_pipeline(image_paths, out_dir, *, write_debug=True):
        calls["pipeline"] = (image_paths, out_dir, write_debug)
        return SimpleNamespace(candidates=[], transactions=[])

    monkeypatch.setattr("octopus_ocr.cli.extract_video_keyframes", fake_extract)
    monkeypatch.setattr("octopus_ocr.cli.run_pipeline", fake_pipeline)

    out_dir = tmp_path / "out"
    code = main(["data/1.PNG", str(video_path), "--out", str(out_dir), "--video-sample-fps", "3", "--no-debug"])
    captured = capsys.readouterr()

    assert code == 0
    assert calls["extract"] == (video_path, out_dir / "frames" / "screen_recording", 3.0)
    assert calls["pipeline"] == ([Path("data/1.PNG"), extracted_frame], out_dir, False)
    assert "Extracted 1 keyframe(s)" in captured.out
    assert "Timings:" in captured.out
    assert "extract video keyframes (screen recording.mp4)" in captured.out
    assert "Warning: possible skipped transactions" in captured.err


def test_cli_prints_pipeline_timings(monkeypatch, capsys) -> None:
    def fake_pipeline(*args, **kwargs):
        return SimpleNamespace(
            candidates=[],
            transactions=[],
            timings=[
                ProcessTiming(name="load images", seconds=0.25),
                ProcessTiming(name="OCR rows", seconds=1.5),
            ],
        )

    monkeypatch.setattr("octopus_ocr.cli.run_pipeline", fake_pipeline)

    code = main(["data/1.PNG"])
    captured = capsys.readouterr()

    assert code == 0
    assert "Timings:" in captured.out
    assert "load images" in captured.out
    assert "250 ms" in captured.out
    assert "OCR rows" in captured.out
    assert "1.50 s" in captured.out
    assert "total" in captured.out
    assert "1.75 s" in captured.out


def test_format_duration() -> None:
    assert _format_duration(0.1234) == "123 ms"
    assert _format_duration(3.456) == "3.46 s"
    assert _format_duration(65.0) == "1m 05.0s"


def test_cli_rejects_non_positive_video_sample_fps() -> None:
    with pytest.raises(SystemExit):
        main(["data/1.PNG", "--video-sample-fps", "0"])


def test_cli_reports_video_extraction_errors(monkeypatch, tmp_path: Path, capsys) -> None:
    video_path = tmp_path / "bad.mp4"
    video_path.write_bytes(b"not a valid video")

    def fail_extract(*args, **kwargs):
        raise ValueError("Could not read video")

    monkeypatch.setattr("octopus_ocr.cli.extract_video_keyframes", fail_extract)

    with pytest.raises(SystemExit):
        main([str(video_path)])

    captured = capsys.readouterr()
    assert "Could not read video" in captured.err
