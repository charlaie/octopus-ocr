from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator, Literal, cast

from octopus_ocr.dedupe import dedupe_candidates
from octopus_ocr.exporters import write_json, write_monthly_category_csv, write_ofx, write_review_csv
from octopus_ocr.models import BBox, OcrField, PipelineResult, ProcessTiming, TransactionCandidate
from octopus_ocr.normalize import normalize_payee, parse_amount, parse_datetime
from octopus_ocr.ocr import OcrEngine, OcrEngineName, create_ocr_engine
from octopus_ocr.vision import RowRegion, annotate_rows, detect_rows, load_rgb, to_pil_crop


ProgressPhase = Literal["prepare", "detect", "ocr", "export", "complete"]


@dataclass(frozen=True)
class ProgressEvent:
    phase: ProgressPhase
    message: str
    image_index: int = 0
    image_count: int = 0
    row_index: int = 0
    rows_in_image: int = 0
    completed_rows: int = 0
    total_rows: int = 0
    output_path: Path | None = None


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass(frozen=True)
class _ImageRows:
    path: Path
    rows: list[RowRegion]


def run_pipeline(
    image_paths: list[Path],
    out_dir: Path,
    *,
    write_debug: bool = True,
    ocr: OcrEngine | None = None,
    ocr_engine: OcrEngineName = "tesseract",
    paddle_model: str = "en_PP-OCRv5_mobile_rec",
    progress: ProgressCallback | None = None,
) -> PipelineResult:
    timer = _PipelineTimer()
    _emit_progress(progress, ProgressEvent("prepare", "Preparing output directories", output_path=out_dir))
    with timer.step("prepare output directories"):
        out_dir.mkdir(parents=True, exist_ok=True)
        debug_dir = out_dir / "debug"
        if write_debug:
            debug_dir.mkdir(parents=True, exist_ok=True)

    _emit_progress(progress, ProgressEvent("prepare", "Initializing OCR engine", output_path=out_dir))
    with timer.step("initialize OCR engine"):
        engine = ocr or create_ocr_engine(ocr_engine, paddle_model=paddle_model)
        engine.assert_available()

    image_rows: list[_ImageRows] = []
    image_count = len(image_paths)
    for image_index, image_path in enumerate(image_paths, start=1):
        _emit_progress(
            progress,
            ProgressEvent(
                "detect",
                f"Detecting image {image_index}/{image_count}",
                image_index=image_index,
                image_count=image_count,
                output_path=out_dir,
            ),
        )
        with timer.step("load images"):
            image_rgb = load_rgb(image_path)
        with timer.step("detect rows"):
            rows = detect_rows(image_rgb)
        if write_debug:
            with timer.step("write debug annotations"):
                annotate_rows(image_rgb, rows).save(debug_dir / f"{image_path.stem}_annotated.png")
        image_rows.append(_ImageRows(path=image_path, rows=rows))

    total_rows = sum(len(plan.rows) for plan in image_rows)
    _emit_progress(
        progress,
        ProgressEvent(
            "detect",
            f"Detected {total_rows} row(s) across {image_count} image(s)",
            image_count=image_count,
            total_rows=total_rows,
            output_path=out_dir,
        ),
    )

    candidates: list[TransactionCandidate] = []
    completed_rows = 0
    for image_index, plan in enumerate(image_rows, start=1):
        with timer.step("load images"):
            image_rgb = load_rgb(plan.path)
        rows_in_image = len(plan.rows)
        for row_index, row in enumerate(plan.rows, start=1):
            _emit_progress(
                progress,
                ProgressEvent(
                    "ocr",
                    f"Image {image_index}/{image_count}, row {row_index}/{rows_in_image}, "
                    f"OCR row {completed_rows + 1}/{total_rows}",
                    image_index=image_index,
                    image_count=image_count,
                    row_index=row_index,
                    rows_in_image=rows_in_image,
                    completed_rows=completed_rows,
                    total_rows=total_rows,
                    output_path=out_dir,
                ),
            )
            with timer.step("OCR rows"):
                candidate = _read_row(
                    image_rgb,
                    plan.path,
                    row_index - 1,
                    row,
                    engine,
                    debug_dir if write_debug else None,
                )
            candidates.append(candidate)
            completed_rows += 1
            _emit_progress(
                progress,
                ProgressEvent(
                    "ocr",
                    f"Image {image_index}/{image_count}, row {row_index}/{rows_in_image}, "
                    f"OCR row {completed_rows}/{total_rows}",
                    image_index=image_index,
                    image_count=image_count,
                    row_index=row_index,
                    rows_in_image=rows_in_image,
                    completed_rows=completed_rows,
                    total_rows=total_rows,
                    output_path=out_dir,
                ),
            )

    _emit_progress(progress, ProgressEvent("export", "Writing output files", completed_rows=completed_rows, total_rows=total_rows, output_path=out_dir))
    with timer.step("dedupe transactions"):
        records = dedupe_candidates(candidates)
    result = PipelineResult(
        generated_at=datetime.now(),
        input_images=[str(path) for path in image_paths],
        candidates=candidates,
        transactions=records,
    )
    with timer.step("write review.csv"):
        write_review_csv(records, out_dir / "review.csv")
    with timer.step("write monthly_category_totals.csv"):
        write_monthly_category_csv(records, out_dir / "monthly_category_totals.csv")
    with timer.step("write actual.ofx"):
        write_ofx(records, out_dir / "actual.ofx")
    result.timings = list(timer.timings)
    with timer.step("write transactions.json"):
        write_json(result, out_dir / "transactions.json")
    result.timings = list(timer.timings)
    _emit_progress(
        progress,
        ProgressEvent(
            "complete",
            "Complete",
            image_count=image_count,
            completed_rows=completed_rows,
            total_rows=total_rows,
            output_path=out_dir,
        ),
    )
    return result


def _emit_progress(progress: ProgressCallback | None, event: ProgressEvent) -> None:
    if progress is not None:
        progress(event)


def _read_row(
    image_rgb,
    image_path: Path,
    row_index: int,
    row,
    ocr_engine: OcrEngine,
    debug_dir: Path | None,
) -> TransactionCandidate:
    warnings: list[str] = []
    fields: dict[str, OcrField] = {}
    field_specs = [
        ("payee", row.payee_bbox, "payee"),
        ("datetime", row.datetime_bbox, "datetime"),
        ("amount", row.amount_bbox, "amount"),
    ]
    for name, bbox, _mode in field_specs:
        crop = to_pil_crop(image_rgb, bbox)
        if debug_dir:
            crop.save(debug_dir / f"{image_path.stem}_{row_index:02d}_{name}.png")

    read_row_fields = getattr(ocr_engine, "read_row_fields", None)
    if callable(read_row_fields):
        try:
            row_crop = to_pil_crop(image_rgb, row.row_bbox)
            row_reader = cast(Callable[[Any, BBox, dict[str, BBox]], dict[str, OcrField]], read_row_fields)
            fields = row_reader(
                row_crop,
                row.row_bbox,
                {name: bbox for name, bbox, _mode in field_specs},
            )
        except Exception as exc:  # noqa: BLE001 - preserve partial row evidence for review.
            warnings.append(f"row OCR failed: {exc}")

    for name, bbox, mode in field_specs:
        if name in fields:
            continue
        crop = to_pil_crop(image_rgb, bbox)
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


class _PipelineTimer:
    def __init__(self) -> None:
        self._by_name: dict[str, ProcessTiming] = {}
        self.timings: list[ProcessTiming] = []

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - started
            existing = self._by_name.get(name)
            if existing is None:
                timing = ProcessTiming(name=name, seconds=elapsed)
                self._by_name[name] = timing
                self.timings.append(timing)
            else:
                existing.seconds += elapsed
