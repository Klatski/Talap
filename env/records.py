"""Открытые записи эпизода в памяти среды (только чтение).

Парсер написан независимо от generator/: среда знает только опубликованный формат
(docs/contract.md, раздел 2). Пустое поле CSV -> JSON null, суммы -> int, курсы — строки.
Каждой записи fx добавляется вычисляемый ключ fx_id = FX:<source>:<target>:<date>.
Закрытых файлов (truth/provenance) этот модуль не знает и не принимает.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field

from .constants import FX_COLUMNS, LEDGER_COLUMNS, PUBLIC_FILES, TABLES


class EpisodeDataError(ValueError):
    """Открытые файлы эпизода не соответствуют контракту (ошибка данных, не агента)."""


def fx_id(source: str, target: str, day: str) -> str:
    return f"FX:{source}:{target}:{day}"


def _parse_ledger(text: str) -> list[dict]:
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, None)
    if tuple(header or ()) != LEDGER_COLUMNS:
        raise EpisodeDataError("ledger.csv header mismatch")
    rows = []
    for raw in reader:
        if not raw:
            continue
        if len(raw) != len(LEDGER_COLUMNS):
            raise EpisodeDataError("ledger.csv row width mismatch")
        row = {k: (v if v != "" else None) for k, v in zip(LEDGER_COLUMNS, raw)}
        try:
            row["debit_minor"] = int(row["debit_minor"])
            row["credit_minor"] = int(row["credit_minor"])
        except (TypeError, ValueError) as exc:
            raise EpisodeDataError("ledger.csv amount is not an integer") from exc
        rows.append(row)
    return rows


def _parse_fx(text: str) -> list[dict]:
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, None)
    if tuple(header or ()) != FX_COLUMNS:
        raise EpisodeDataError("fx.csv header mismatch")
    rows = []
    for raw in reader:
        if not raw:
            continue
        if len(raw) != len(FX_COLUMNS):
            raise EpisodeDataError("fx.csv row width mismatch")
        row = dict(zip(FX_COLUMNS, raw))
        row["fx_id"] = fx_id(row["source_currency"], row["target_currency"], row["date"])
        rows.append(row)
    return rows


def _parse_jsonl(text: str, name: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EpisodeDataError(f"{name} is not valid JSONL") from exc
        if not isinstance(rec, dict):
            raise EpisodeDataError(f"{name} record is not an object")
        out.append(rec)
    return out


@dataclass
class PublicEpisode:
    manifest: dict
    policy: dict
    tables: dict[str, list[dict]]
    by_id: dict[str, dict[str, dict]] = field(default_factory=dict)
    documents: dict[str, tuple[str, dict]] = field(default_factory=dict)

    @property
    def episode_id(self) -> str:
        return self.manifest["episode_id"]

    def counts(self) -> dict[str, int]:
        return {name: len(rows) for name, rows in self.tables.items()}


def load_public(texts: dict[str, str]) -> PublicEpisode:
    """Разобрать шесть открытых файлов. Лишние или закрытые файлы — ошибка."""
    if not isinstance(texts, dict) or set(texts) != set(PUBLIC_FILES):
        raise EpisodeDataError("expected exactly the six public files")
    try:
        manifest = json.loads(texts["manifest.json"])
        policy = json.loads(texts["policy.json"])
    except json.JSONDecodeError as exc:
        raise EpisodeDataError("manifest/policy is not valid JSON") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("episode_id"), str):
        raise EpisodeDataError("manifest.episode_id missing")
    tables = {
        "ledger": _parse_ledger(texts["ledger.csv"]),
        "invoices": _parse_jsonl(texts["invoices.jsonl"], "invoices.jsonl"),
        "payments": _parse_jsonl(texts["payments.jsonl"], "payments.jsonl"),
        "fx": _parse_fx(texts["fx.csv"]),
    }
    ep = PublicEpisode(manifest=manifest, policy=policy, tables=tables)
    for name, rows in tables.items():
        key = TABLES[name]["id_field"]
        index: dict[str, dict] = {}
        for row in rows:
            rid = row.get(key)
            if not isinstance(rid, str) or rid in index:
                raise EpisodeDataError(f"{name}: missing or duplicate {key}")
            index[rid] = row
        ep.by_id[name] = index
        doc_field = TABLES[name]["doc_field"]
        if doc_field:
            kind = "invoice" if name == "invoices" else "payment"
            for row in rows:
                doc = row.get(doc_field)
                if not isinstance(doc, str) or doc in ep.documents:
                    raise EpisodeDataError(f"{name}: missing or duplicate {doc_field}")
                ep.documents[doc] = (kind, row)
    return ep
