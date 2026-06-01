from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from octopus_ocr.models import BBox, OcrField


class OcrUnavailableError(RuntimeError):
    pass


class TesseractOcr:
    def __init__(self, executable: str = "tesseract") -> None:
        self.executable = executable

    def assert_available(self) -> None:
        if shutil.which(self.executable) is None:
            raise OcrUnavailableError(
                f"Could not find '{self.executable}'. Install it with 'brew install tesseract'."
            )

    def read_image(self, image: Image.Image, bbox: BBox, *, mode: str) -> OcrField:
        self.assert_available()
        config = self._config_for(mode)
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "crop.png"
            output_base = Path(tmpdir) / "out"
            image.save(image_path)
            cmd = [self.executable, str(image_path), str(output_base), "--psm", config["psm"], "-l", "eng"]
            for key, value in config.get("vars", {}).items():
                cmd.extend(["-c", f"{key}={value}"])
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or "Tesseract failed")
            text = output_base.with_suffix(".txt").read_text(encoding="utf-8").strip()
        return OcrField(text=text, confidence=None, bbox=bbox)

    def _config_for(self, mode: str) -> dict[str, object]:
        if mode == "amount":
            return {
                "psm": "7",
                "vars": {
                    "tessedit_char_whitelist": "0123456789.+-−",
                },
            }
        if mode == "datetime":
            return {
                "psm": "7",
                "vars": {
                    "tessedit_char_whitelist": "0123456789-: /.",
                },
            }
        return {"psm": "6"}
