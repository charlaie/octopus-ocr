from pathlib import Path
import runpy

import pytest


@pytest.mark.parametrize("exit_code", [0, 2])
def test_script_preserves_cli_exit_status(monkeypatch: pytest.MonkeyPatch, exit_code: int) -> None:
    monkeypatch.setattr("octopus_ocr.cli.main", lambda: exit_code)

    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"), run_name="__main__")

    assert raised.value.code == exit_code
