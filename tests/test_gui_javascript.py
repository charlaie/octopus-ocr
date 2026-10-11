from pathlib import Path
import shutil
import subprocess

import pytest


def test_gui_state_contract_in_javascript() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the desktop JavaScript regression tests")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("gui_state.test.js"))],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
