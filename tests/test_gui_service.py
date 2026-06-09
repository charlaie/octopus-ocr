from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from octopus_ocr.app_bundle import make_dev_app
from octopus_ocr.gui_service import create_run_dir, run_gui_pipeline


def test_create_run_dir_uses_timestamp_and_suffixes_existing_runs(tmp_path: Path) -> None:
    now = datetime(2026, 6, 9, 14, 5, 6)
    existing = tmp_path / "20260609-140506"
    existing.mkdir()

    assert create_run_dir(tmp_path, now=now) == tmp_path / "20260609-140506-2"


def test_run_gui_pipeline_expands_inputs_and_returns_summary(tmp_path: Path) -> None:
    folder = tmp_path / "drop"
    folder.mkdir()
    second = folder / "02.png"
    first = folder / "01.png"
    first.write_bytes(b"x")
    second.write_bytes(b"x")
    calls = {}

    def fake_pipeline(image_paths, out_dir, *, write_debug, ocr_engine, paddle_model, progress):
        calls["pipeline"] = (image_paths, out_dir, write_debug, ocr_engine, paddle_model, progress)
        return SimpleNamespace(
            candidates=[SimpleNamespace(warnings=[]), SimpleNamespace(warnings=["check amount"])],
            transactions=[object()],
        )

    result = run_gui_pipeline(
        [folder],
        out_root=tmp_path / "runs",
        ocr_engine="paddle",
        paddle_model="test-model",
        now=datetime(2026, 6, 9, 14, 5, 6),
        pipeline_runner=fake_pipeline,
    )

    assert calls["pipeline"] == (
        [first, second],
        tmp_path / "runs" / "20260609-140506",
        True,
        "paddle",
        "test-model",
        None,
    )
    assert result.output_dir == tmp_path / "runs" / "20260609-140506"
    assert result.input_count == 2
    assert result.detected_rows == 2
    assert result.exported_transactions == 1
    assert result.warning_rows == 1
    assert result.output_files["review"] == result.output_dir / "review.csv"


def test_make_dev_app_writes_launcher_bundle(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    app_path = tmp_path / "Octopus OCR.app"
    rendered_sizes: list[int] = []
    rendered_sources: list[Path] = []

    def fake_icon_renderer(svg_path: Path, size: int) -> bytes:
        rendered_sources.append(svg_path)
        rendered_sizes.append(size)
        return b"fake-png"

    make_dev_app(app_path, repo_root=repo_root, icon_renderer=fake_icon_renderer)

    launcher = app_path / "Contents" / "MacOS" / "Octopus OCR"
    info = app_path / "Contents" / "Info.plist"
    icon = app_path / "Contents" / "Resources" / "AppIcon.icns"
    svg = app_path / "Contents" / "Resources" / "OctopusOCR.svg"
    assert launcher.exists()
    assert info.exists()
    assert icon.read_bytes().startswith(b"icns")
    assert "<svg" in svg.read_text(encoding="utf-8")
    assert "<key>CFBundleIconFile</key>" in info.read_text(encoding="utf-8")
    assert rendered_sizes == [16, 32, 64, 128, 256, 512, 1024]
    assert all(path.name == "octopus_ocr_icon.svg" for path in rendered_sources)
    assert "uv run octopus-ocr-gui" in launcher.read_text(encoding="utf-8")
    assert str(repo_root) in launcher.read_text(encoding="utf-8")
    assert launcher.stat().st_mode & 0o111
