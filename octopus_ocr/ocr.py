from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import NotRequired, TypedDict

from PIL import Image

from octopus_ocr.models import BBox, OcrField


class OcrUnavailableError(RuntimeError):
    pass


class TesseractConfig(TypedDict):
    psm: str
    vars: NotRequired[dict[str, str]]


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
            if mode == "amount":
                primary = self._read_tsv_stdout(image_path, bbox, config)
                secondary = self._read_tsv_stdout(
                    image_path,
                    bbox,
                    {
                        "psm": "8",
                        "vars": config.get("vars", {}),
                    },
                )
                return _choose_amount_ocr_result(primary, secondary)
            text = self._read_text_file(image_path, output_base, config)
        return OcrField(text=text, confidence=None, bbox=bbox)

    def _read_text_file(self, image_path: Path, output_base: Path, config: TesseractConfig) -> str:
        cmd = [self.executable, str(image_path), str(output_base), "--psm", config["psm"], "-l", "eng"]
        for key, value in config.get("vars", {}).items():
            cmd.extend(["-c", f"{key}={value}"])
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "Tesseract failed")
        return output_base.with_suffix(".txt").read_text(encoding="utf-8").strip()

    def _read_tsv_stdout(self, image_path: Path, bbox: BBox, config: TesseractConfig) -> OcrField:
        cmd = [self.executable, str(image_path), "stdout", "--psm", config["psm"], "-l", "eng"]
        for key, value in config.get("vars", {}).items():
            cmd.extend(["-c", f"{key}={value}"])
        cmd.append("tsv")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "Tesseract failed")

        words: list[str] = []
        confidences: list[float] = []
        for line in proc.stdout.splitlines()[1:]:
            columns = line.split("\t")
            if len(columns) < 12 or columns[0] != "5":
                continue
            text = columns[11].strip()
            if not text:
                continue
            words.append(text)
            try:
                confidence = float(columns[10])
            except ValueError:
                continue
            if confidence >= 0:
                confidences.append(confidence)

        confidence_value = None
        if confidences:
            confidence_value = round(sum(confidences) / len(confidences), 2)
        return OcrField(text=" ".join(words).strip(), confidence=confidence_value, bbox=bbox)

    def _config_for(self, mode: str) -> TesseractConfig:
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


def _choose_amount_ocr_result(primary: OcrField, secondary: OcrField) -> OcrField:
    primary_digits = _digits(primary.text)
    secondary_digits = _digits(secondary.text)
    if not secondary_digits:
        return primary
    if not primary_digits:
        return _format_secondary_amount(primary.text, secondary)
    if primary_digits == secondary_digits:
        return primary
    if len(primary_digits) != len(secondary_digits):
        return primary
    if primary.confidence is None or secondary.confidence is None:
        return primary
    if primary.confidence > 20 or secondary.confidence < primary.confidence + 20:
        return primary
    return _format_secondary_amount(primary.text, secondary)


def _format_secondary_amount(primary_text: str, secondary: OcrField) -> OcrField:
    digits = _digits(secondary.text)
    sign = "-" if "-" in primary_text or "−" in primary_text or secondary.text.strip().startswith("-") else ""
    decimal_places = _primary_decimal_places(primary_text)
    if decimal_places is not None and len(digits) > decimal_places:
        text = f"{sign}{digits[:-decimal_places]}.{digits[-decimal_places:]}"
    else:
        text = f"{sign}{digits}"
    return OcrField(text=text, confidence=secondary.confidence, bbox=secondary.bbox)


def _digits(text: str) -> str:
    return "".join(char for char in text if char.isdigit())


def _primary_decimal_places(text: str) -> int | None:
    if "." not in text:
        return None
    after_decimal = text.rsplit(".", maxsplit=1)[1]
    places = sum(1 for char in after_decimal if char.isdigit())
    return places or None
