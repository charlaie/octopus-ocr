from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from octopus_ocr import app_bundle
from octopus_ocr.app_bundle import make_dev_app


@pytest.mark.parametrize("repo_name", ["repo with spaces", "repo's \"quoted\" $PROJECT_HOME `pwd` \\name\nnext"])
def test_launcher_preserves_literal_repository_and_search_paths(tmp_path: Path, repo_name: str) -> None:
    repo_root = tmp_path / repo_name
    repo_root.mkdir()
    bin_dir = tmp_path / "tools $PROJECT_HOME `pwd` 'quoted'"
    bin_dir.mkdir()
    uv_stub = bin_dir / "uv"
    uv_stub.write_text("#!/bin/sh\nprintf '%s\\n' \"$PWD\" \"$*\"\n", encoding="utf-8")
    uv_stub.chmod(0o755)
    app_path = tmp_path / "Octopus OCR.app"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("PATH", str(bin_dir))
        make_dev_app(app_path, repo_root=repo_root, icon_renderer=lambda _path, _size: b"png")

    process = subprocess.run(
        ["/bin/sh", str(app_path / "Contents" / "MacOS" / "Octopus OCR")],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert process.stdout == f"{repo_root}\nrun octopus-ocr-gui\n"


@pytest.mark.parametrize("argv", [None, []])
def test_bundle_entrypoint_distinguishes_process_arguments_from_explicit_empty_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str] | None,
) -> None:
    custom_path = tmp_path / "Custom.app"
    calls: list[Path] = []

    def fake_make_dev_app(path: Path, *, repo_root: Path) -> None:
        calls.append(path)

    monkeypatch.setattr("sys.argv", ["octopus-ocr-make-app", str(custom_path)])
    monkeypatch.setattr(app_bundle, "make_dev_app", fake_make_dev_app)

    assert app_bundle.main(argv) == 0
    default_path = Path(app_bundle.__file__).resolve().parents[1] / "dist" / "Octopus OCR.app"
    assert calls == [custom_path if argv is None else default_path]
