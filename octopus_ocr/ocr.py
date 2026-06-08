from __future__ import annotations

from dataclasses import dataclass
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal, NotRequired, Protocol, TypedDict, cast

from PIL import Image

from octopus_ocr.models import BBox, OcrField


class OcrUnavailableError(RuntimeError):
    pass


class TesseractConfig(TypedDict):
    psm: str
    vars: NotRequired[dict[str, str]]


@dataclass(frozen=True)
class _PaddleTextLine:
    text: str
    confidence: float | None
    box: tuple[float, float, float, float]


OcrEngineName = Literal["tesseract", "paddle"]


class OcrEngine(Protocol):
    def assert_available(self) -> None: ...

    def read_image(self, image: Image.Image, bbox: BBox, *, mode: str) -> OcrField: ...


def create_ocr_engine(name: OcrEngineName, *, paddle_model: str = "en_PP-OCRv5_mobile_rec") -> OcrEngine:
    if name == "tesseract":
        return TesseractOcr()
    if name == "paddle":
        return PaddleOcr(model_name=paddle_model)
    raise ValueError(f"Unsupported OCR engine: {name}")


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
            _trim_ocr_whitespace(image).save(image_path)
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


def _trim_ocr_whitespace(image: Image.Image, *, padding: int = 12, threshold: int = 245) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    assert pixels is not None
    left = width
    top = height
    right = -1
    bottom = -1
    for y in range(height):
        for x in range(width):
            red, green, blue = cast(tuple[int, int, int], pixels[x, y])
            if min(red, green, blue) >= threshold:
                continue
            left = min(left, x)
            top = min(top, y)
            right = max(right, x)
            bottom = max(bottom, y)

    if right < left or bottom < top:
        return image

    crop_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding + 1),
        min(height, bottom + padding + 1),
    )
    if crop_box == (0, 0, width, height):
        return image
    return image.crop(crop_box)


class PaddleOcr:
    def __init__(self, *, model_name: str = "en_PP-OCRv5_mobile_rec", engine: str | None = None) -> None:
        self.model_name = model_name
        self.engine = engine
        self._model: Any | None = None

    def assert_available(self) -> None:
        self._load_model()

    def read_image(self, image: Image.Image, bbox: BBox, *, mode: str) -> OcrField:
        lines = self._read_lines(image, mode=mode)
        return _field_from_paddle_lines(lines, bbox)

    def read_row_fields(
        self,
        row_image: Image.Image,
        row_bbox: BBox,
        field_bboxes: dict[str, BBox],
    ) -> dict[str, OcrField]:
        lines = self._read_lines(row_image, mode="row")
        fields: dict[str, OcrField] = {}
        for name, field_bbox in field_bboxes.items():
            field_box = _field_bbox_in_row_image(field_bbox, row_bbox, row_image.size)
            field_lines = _lines_intersecting_box(lines, field_box)
            fields[name] = _field_from_paddle_lines(field_lines, field_bbox)
        return fields

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OcrUnavailableError(
                "Could not import PaddleOCR. Install PaddlePaddle and PaddleOCR, then rerun with "
                "'--ocr-engine paddle'. See README for the install commands."
            ) from exc

        kwargs: dict[str, Any] = {
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": self.model_name,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if self.engine is not None:
            kwargs["engine"] = self.engine
        try:
            self._model = PaddleOCR(**kwargs)
        except Exception as exc:  # noqa: BLE001 - dependency/model setup failures should be user-readable.
            raise OcrUnavailableError(f"Could not initialize PaddleOCR model '{self.model_name}': {exc}") from exc
        return self._model

    def _read_lines(self, image: Image.Image, *, mode: str) -> list[_PaddleTextLine]:
        model = self._load_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / f"{mode}.png"
            image.save(image_path)
            output = model.predict(input=str(image_path))
        return _extract_paddle_lines(output)


def _extract_paddle_lines(output: Any) -> list[_PaddleTextLine]:
    records = list(output) if isinstance(output, (list, tuple)) else [output]
    lines: list[_PaddleTextLine] = []
    for record in records:
        data = _paddle_record_data(record)
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        boxes = _paddle_boxes(data.get("rec_boxes", []))
        if not isinstance(texts, list):
            continue
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip() or index >= len(boxes):
                continue
            score = scores[index] if isinstance(scores, list) and index < len(scores) else None
            confidence = float(score) * 100 if isinstance(score, (int, float)) else None
            lines.append(_PaddleTextLine(text=text.strip(), confidence=confidence, box=boxes[index]))
    return lines


def _paddle_boxes(raw_boxes: Any) -> list[tuple[float, float, float, float]]:
    if hasattr(raw_boxes, "tolist"):
        raw_boxes = raw_boxes.tolist()
    boxes: list[tuple[float, float, float, float]] = []
    if not isinstance(raw_boxes, list):
        return boxes
    for raw_box in raw_boxes:
        if not isinstance(raw_box, list | tuple) or len(raw_box) < 4:
            continue
        try:
            left = float(raw_box[0])
            top = float(raw_box[1])
            right = float(raw_box[2])
            bottom = float(raw_box[3])
        except (TypeError, ValueError):
            continue
        boxes.append((left, top, right, bottom))
    return boxes


def _field_bbox_in_row_image(field_bbox: BBox, row_bbox: BBox, row_image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    scale_x = row_image_size[0] / row_bbox.width
    scale_y = row_image_size[1] / row_bbox.height
    left = (field_bbox.x - row_bbox.x) * scale_x
    top = (field_bbox.y - row_bbox.y) * scale_y
    right = left + field_bbox.width * scale_x
    bottom = top + field_bbox.height * scale_y
    return left, top, right, bottom


def _lines_intersecting_box(
    lines: list[_PaddleTextLine],
    box: tuple[float, float, float, float],
) -> list[_PaddleTextLine]:
    selected = [
        line
        for line in lines
        if _intersection_area(line.box, box) / max(1.0, _box_area(line.box)) >= 0.35
        or _box_center_in_box(line.box, box)
    ]
    return sorted(selected, key=lambda line: (_box_center(line.box)[1], _box_center(line.box)[0]))


def _field_from_paddle_lines(lines: list[_PaddleTextLine], bbox: BBox) -> OcrField:
    text = _join_paddle_lines(lines)
    confidences = [line.confidence for line in lines if line.confidence is not None]
    confidence = round(sum(confidences) / len(confidences), 2) if confidences else None
    return OcrField(text=text, confidence=confidence, bbox=bbox)


def _join_paddle_lines(lines: list[_PaddleTextLine]) -> str:
    rows: list[list[_PaddleTextLine]] = []
    for line in sorted(lines, key=lambda item: (_box_center(item.box)[1], _box_center(item.box)[0])):
        if rows and _vertical_overlap_ratio(rows[-1][-1].box, line.box) >= 0.5:
            rows[-1].append(line)
        else:
            rows.append([line])

    row_texts = [_join_paddle_row_fragments(row) for row in rows]
    return " ".join(text for text in row_texts if text).strip()


def _join_paddle_row_fragments(lines: list[_PaddleTextLine]) -> str:
    row_text = ""
    previous_box: tuple[float, float, float, float] | None = None
    for line in sorted(lines, key=lambda item: _box_center(item.box)[0]):
        text = line.text
        if not row_text or previous_box is None:
            row_text = text
            previous_box = line.box
            continue

        overlaps_previous = min(previous_box[2], line.box[2]) > max(previous_box[0], line.box[0])
        if overlaps_previous and row_text[-1:].casefold() == text[:1].casefold():
            row_text += text[1:]
        elif overlaps_previous:
            row_text += text
        else:
            row_text += f" {text}"
        previous_box = (
            min(previous_box[0], line.box[0]),
            min(previous_box[1], line.box[1]),
            max(previous_box[2], line.box[2]),
            max(previous_box[3], line.box[3]),
        )
    return row_text.strip()


def _vertical_overlap_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    top = max(first[1], second[1])
    bottom = min(first[3], second[3])
    if bottom <= top:
        return 0
    shorter_height = max(1.0, min(first[3] - first[1], second[3] - second[1]))
    return (bottom - top) / shorter_height


def _intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0
    return (right - left) * (bottom - top)


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _box_center_in_box(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> bool:
    center_x, center_y = _box_center(inner)
    return outer[0] <= center_x <= outer[2] and outer[1] <= center_y <= outer[3]


def _extract_paddle_text(output: Any) -> tuple[str, float | None]:
    records = list(output) if isinstance(output, (list, tuple)) else [output]
    texts: list[str] = []
    scores: list[float] = []
    for record in records:
        data = _paddle_record_data(record)
        text = data.get("rec_text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
        score = data.get("rec_score")
        if isinstance(score, (int, float)):
            scores.append(float(score) * 100)

    confidence = round(sum(scores) / len(scores), 2) if scores else None
    return " ".join(texts).strip(), confidence


def _paddle_record_data(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        value = record.get("res", record)
        return value if isinstance(value, dict) else {}
    res = getattr(record, "res", None)
    if isinstance(res, dict):
        return res
    json_value = getattr(record, "json", None)
    if callable(json_value):
        value = json_value()
        if isinstance(value, dict):
            nested = value.get("res", value)
            return cast(dict[str, Any], nested) if isinstance(nested, dict) else {}
    return {}


def _choose_amount_ocr_result(primary: OcrField, secondary: OcrField) -> OcrField:
    primary_digits = _digits(primary.text)
    secondary_digits = _digits(secondary.text)
    if not secondary_digits:
        return primary
    if not primary_digits:
        return _format_secondary_amount(primary.text, secondary)
    if _primary_is_missing_integer_digits(primary) and len(secondary_digits) > len(primary_digits):
        if primary.confidence is None or secondary.confidence is None:
            return primary
        if primary.confidence <= 20 and secondary.confidence >= primary.confidence + 20:
            return _format_secondary_amount(primary.text, secondary)
    if primary_digits == secondary_digits:
        return primary
    if len(primary_digits) != len(secondary_digits):
        if _secondary_preserves_primary_suffix(primary, secondary, primary_digits, secondary_digits):
            return _format_secondary_amount(primary.text, secondary)
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


def _primary_is_missing_integer_digits(primary: OcrField) -> bool:
    text = primary.text.replace("−", "-").replace(" ", "")
    return bool(re.search(r"^[+-]?\.\d+$", text))


def _secondary_preserves_primary_suffix(
    primary: OcrField,
    secondary: OcrField,
    primary_digits: str,
    secondary_digits: str,
) -> bool:
    if not primary_digits or not secondary_digits.endswith(primary_digits):
        return False
    if primary.confidence is None or secondary.confidence is None:
        return False
    return primary.confidence <= 20 and secondary.confidence >= primary.confidence + 20


def _primary_decimal_places(text: str) -> int | None:
    if "." not in text:
        return None
    after_decimal = text.rsplit(".", maxsplit=1)[1]
    places = sum(1 for char in after_decimal if char.isdigit())
    return places or None
