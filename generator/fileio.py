"""Детерминированная (де)сериализация файлов эпизода.

JSON: sort_keys, UTF-8, "\n". JSONL: одна запись на строку, компактные разделители.
CSV: заголовок фиксирован, разделитель ",", перевод строки "\n", null -> пустое поле.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable

LEDGER_COLUMNS = [
    "row_id",
    "timestamp",
    "account",
    "debit_minor",
    "credit_minor",
    "currency",
    "entity_id",
    "invoice_id",
    "payment_id",
]
FX_COLUMNS = ["date", "source_currency", "target_currency", "rate"]
PUBLIC_FILES = ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv")
PRIVATE_FILES = ("truth.json", "provenance.json")
PUBLIC_DIR = "public"
PRIVATE_DIR = "private"


def json_text(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def jsonl_text(records: Iterable[dict]) -> str:
    return "".join(
        json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records
    )


def csv_text(columns: list[str], rows: Iterable[dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["" if row[c] is None else row[c] for c in columns])
    return buf.getvalue()


def parse_jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_ledger(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != LEDGER_COLUMNS:
        raise ValueError(f"ledger.csv header mismatch: {reader.fieldnames}")
    rows = []
    for raw in reader:
        row: dict[str, Any] = {k: (v if v != "" else None) for k, v in raw.items()}
        row["debit_minor"] = int(row["debit_minor"])
        row["credit_minor"] = int(row["credit_minor"])
        rows.append(row)
    return rows


def parse_fx(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != FX_COLUMNS:
        raise ValueError(f"fx.csv header mismatch: {reader.fieldnames}")
    return [dict(r) for r in reader]


def fx_record_id(source: str, target: str, day: str) -> str:
    """ID записи курса для evidence_ids и get_record('fx', id)."""
    return f"FX:{source}:{target}:{day}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def write_episode_dir(out_dir: str | os.PathLike, public: dict[str, str], private: dict[str, str]) -> Path:
    out = Path(out_dir)
    for name, text in public.items():
        write_text(out / PUBLIC_DIR / name, text)
    for name, text in private.items():
        write_text(out / PRIVATE_DIR / name, text)
    return out


def read_texts(directory: str | os.PathLike, names: Iterable[str]) -> dict[str, str]:
    base = Path(directory)
    texts = {}
    for name in names:
        with open(base / name, encoding="utf-8", newline="") as fh:
            texts[name] = fh.read()
    return texts


def read_episode_dir(episode_dir: str | os.PathLike) -> tuple[dict[str, str], dict[str, str]]:
    base = Path(episode_dir)
    return read_texts(base / PUBLIC_DIR, PUBLIC_FILES), read_texts(base / PRIVATE_DIR, PRIVATE_FILES)
