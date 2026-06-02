from __future__ import annotations

from datetime import datetime
from pathlib import Path

from octopus_ocr.dedupe import dedupe_candidates
from octopus_ocr.exporters import write_json, write_ofx, write_review_csv
from octopus_ocr.models import BBox, OcrField, PipelineResult, TransactionCandidate
from octopus_ocr.normalize import normalize_payee, parse_amount, parse_datetime
from octopus_ocr.ocr import TesseractOcr
from octopus_ocr.vision import annotate_rows, detect_rows, load_rgb, to_pil_crop


def run_pipeline(
    image_paths: list[Path],
    out_dir: Path,
    *,
    write_debug: bool = True,
    ocr: TesseractOcr | None = None,
) -> PipelineResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    if write_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    ocr_engine = ocr or TesseractOcr()
    if hasattr(ocr_engine, "assert_available"):
        ocr_engine.assert_available()
    candidates: list[TransactionCandidate] = []
    for image_path in image_paths:
        image_rgb = load_rgb(image_path)
        rows = detect_rows(image_rgb)
        if write_debug:
            annotate_rows(image_rgb, rows).save(debug_dir / f"{image_path.stem}_annotated.png")
        for row_index, row in enumerate(rows):
            candidate = _read_row(image_rgb, image_path, row_index, row, ocr_engine, debug_dir if write_debug else None)
            candidates.append(candidate)

    records = dedupe_candidates(candidates)
    result = PipelineResult(
        generated_at=datetime.now(),
        input_images=[str(path) for path in image_paths],
        candidates=candidates,
        transactions=records,
    )
    write_json(result, out_dir / "transactions.json")
    write_review_csv(records, out_dir / "review.csv")
    write_ofx(records, out_dir / "actual.ofx")
    return result


def _read_row(
    image_rgb,
    image_path: Path,
    row_index: int,
    row,
    ocr_engine: TesseractOcr,
    debug_dir: Path | None,
) -> TransactionCandidate:
    warnings: list[str] = []
    fields: dict[str, OcrField] = {}
    for name, bbox, mode in [
        ("payee", row.payee_bbox, "payee"),
        ("datetime", row.datetime_bbox, "datetime"),
        ("amount", row.amount_bbox, "amount"),
    ]:
        crop = to_pil_crop(image_rgb, bbox)
        if debug_dir:
            crop.save(debug_dir / f"{image_path.stem}_{row_index:02d}_{name}.png")
        try:
            fields[name] = ocr_engine.read_image(crop, bbox, mode=mode)
        except Exception as exc:  # noqa: BLE001 - preserve partial row evidence for review.
            warnings.append(f"{name} OCR failed: {exc}")
            fields[name] = OcrField(text="", confidence=None, bbox=bbox)

    payee = normalize_payee(fields["payee"].text)
    parsed_datetime = parse_datetime(fields["datetime"].text)
    parsed_amount, direction = parse_amount(fields["amount"].text)
    if parsed_amount is not None and row.amount_direction is not None:
        color_signed_amount = abs(parsed_amount) if row.amount_direction == "inflow" else -abs(parsed_amount)
        if color_signed_amount != parsed_amount:
            warnings.append(f"amount sign corrected from OCR using {row.amount_direction} text color")
        parsed_amount = color_signed_amount
        direction = row.amount_direction
    if not payee:
        warnings.append("payee did not parse")
    if parsed_datetime is None:
        warnings.append("datetime did not parse")
    if parsed_amount is None:
        warnings.append("amount did not parse")
    if row.category == "top-up" and parsed_amount is not None and parsed_amount < 0:
        warnings.append("top-up icon detected with negative amount")
    if row.category == "top-up" and parsed_amount is not None and parsed_amount >= 0:
        direction = "inflow"

    return TransactionCandidate(
        source_image=str(image_path),
        row_index=row_index,
        row_bbox=row.row_bbox,
        icon_bbox=row.icon_bbox,
        category=row.category,
        payee=fields["payee"],
        datetime_field=fields["datetime"],
        amount_field=fields["amount"],
        parsed_datetime=parsed_datetime,
        parsed_payee=payee or None,
        parsed_amount=parsed_amount,
        direction=direction,
        warnings=warnings,
    )
