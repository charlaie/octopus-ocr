from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from decimal import Decimal
from html import escape
from pathlib import Path
from typing import get_args

from octopus_ocr.models import Category, PipelineResult, TransactionRecord


CSV_FIELDS = [
    "datetime",
    "date",
    "payee",
    "amount",
    "direction",
    "category",
    "dedupe_status",
    "duplicate_group",
    "fitid",
    "source_images",
    "row_refs",
    "confidence",
    "warnings",
]

MONTHLY_CATEGORY_FIELDS = [
    "month",
    "category",
    "transaction_count",
    "total_amount",
]


def write_json(result: PipelineResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def write_review_csv(records: list[TransactionRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.datetime, reverse=True):
            writer.writerow(
                {
                    "datetime": record.datetime.isoformat(timespec="minutes"),
                    "date": record.datetime.date().isoformat(),
                    "payee": record.payee,
                    "amount": _decimal_text(record.amount),
                    "direction": record.direction,
                    "category": record.category,
                    "dedupe_status": record.dedupe_status,
                    "duplicate_group": record.duplicate_group,
                    "fitid": record.fitid,
                    "source_images": ";".join(record.source_images),
                    "row_refs": ";".join(record.row_refs),
                    "confidence": "" if record.confidence is None else record.confidence,
                    "warnings": " | ".join(record.warnings),
                }
            )


def write_monthly_category_csv(records: list[TransactionRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    included = [record for record in records if record.dedupe_status in {"kept", "review"}]
    month_category_totals: dict[str, dict[str, Decimal]] = {}
    month_category_counts: dict[str, dict[str, int]] = {}
    for record in included:
        month = record.datetime.strftime("%Y-%m")
        month_totals = month_category_totals.setdefault(month, {})
        month_counts = month_category_counts.setdefault(month, {})
        month_totals[record.category] = month_totals.get(record.category, Decimal("0")) + record.amount
        month_counts[record.category] = month_counts.get(record.category, 0) + 1

    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=MONTHLY_CATEGORY_FIELDS)
        writer.writeheader()
        category_order = {category: index for index, category in enumerate(get_args(Category))}
        for month in sorted(month_category_totals, reverse=True):
            categories = sorted(
                month_category_totals[month],
                key=lambda category: (category_order.get(category, len(category_order)), category),
            )
            for category in categories:
                writer.writerow(
                    {
                        "month": month,
                        "category": category,
                        "transaction_count": month_category_counts[month][category],
                        "total_amount": _decimal_text(month_category_totals[month][category]),
                    }
                )
            writer.writerow(
                {
                    "month": month,
                    "category": "TOTAL",
                    "transaction_count": sum(month_category_counts[month].values()),
                    "total_amount": _decimal_text(sum(month_category_totals[month].values(), Decimal("0"))),
                }
            )


def write_ofx(records: list[TransactionRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = [record for record in records if record.dedupe_status in {"kept", "review"}]
    start = min((record.datetime for record in kept), default=datetime.now()).strftime("%Y%m%d")
    end = max((record.datetime for record in kept), default=datetime.now()).strftime("%Y%m%d")
    now = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    transactions = "\n".join(_ofx_transaction(record) for record in sorted(kept, key=lambda item: item.datetime))
    content = f"""OFXHEADER:100
DATA:OFXSGML
VERSION:102
SECURITY:NONE
ENCODING:UTF-8
CHARSET:UTF-8
COMPRESSION:NONE
OLDFILEUID:NONE
NEWFILEUID:NONE

<OFX>
  <SIGNONMSGSRSV1>
    <SONRS>
      <STATUS>
        <CODE>0
        <SEVERITY>INFO
      </STATUS>
      <DTSERVER>{now}
      <LANGUAGE>ENG
    </SONRS>
  </SIGNONMSGSRSV1>
  <BANKMSGSRSV1>
    <STMTTRNRS>
      <TRNUID>1
      <STATUS>
        <CODE>0
        <SEVERITY>INFO
      </STATUS>
      <STMTRS>
        <CURDEF>HKD
        <BANKACCTFROM>
          <BANKID>OCTOPUS
          <ACCTID>OCTOPUS
          <ACCTTYPE>CHECKING
        </BANKACCTFROM>
        <BANKTRANLIST>
          <DTSTART>{start}
          <DTEND>{end}
{transactions}
        </BANKTRANLIST>
        <LEDGERBAL>
          <BALAMT>0.00
          <DTASOF>{end}
        </LEDGERBAL>
      </STMTRS>
    </STMTTRNRS>
  </BANKMSGSRSV1>
</OFX>
"""
    path.write_text(content, encoding="utf-8")


def _ofx_transaction(record: TransactionRecord) -> str:
    trntype = "CREDIT" if record.amount >= 0 else "DEBIT"
    memo = record.category
    return f"""          <STMTTRN>
            <TRNTYPE>{trntype}
            <DTPOSTED>{record.datetime.strftime("%Y%m%d%H%M%S")}
            <TRNAMT>{_decimal_text(record.amount)}
            <FITID>{escape(record.fitid)}
            <NAME>{escape(record.payee)}
            <MEMO>{escape(memo)}
          </STMTTRN>"""


def _decimal_text(value: Decimal) -> str:
    return f"{value:.2f}"
