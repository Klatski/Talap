"""Открытые записи эпизода в памяти среды (только чтение).

Парсер написан независимо от generator/: среда знает только опубликованный формат
(docs/contract.md, раздел 2). Пустое поле CSV -> JSON null, суммы -> int, курсы — строки.
Каждой записи fx добавляется вычисляемый ключ fx_id = FX:<source>:<target>:<date>; запись того же курса
находится и по ключу в записи жюри <date>:<source>:<target> (синоним, canonical_record_id).
Разбор терпимый к копиям, которые делает жюри: BOM в начале файла снимается, переводы строк LF или CRLF,
колонки CSV читаются по именам заголовка в любом порядке (набор колонок — ровно контрактный), пустой CSV
(0 байт, без заголовка) — пустая таблица; записи отдаются в каноническом порядке полей, поэтому наблюдения
не зависят от порядка колонок в файле.
Закрытых файлов (truth/provenance) этот модуль не знает и не принимает.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field

from .constants import FX_COLUMNS, LEDGER_COLUMNS, PUBLIC_FILES, TABLES


class EpisodeDataError(ValueError):
    """Открытые файлы эпизода не соответствуют контракту (ошибка данных, не агента)."""


def fx_id(source: str, target: str, day: str) -> str:
    return f"FX:{source}:{target}:{day}"


_JURY_FX_KEY = re.compile(r"(\d{4}-\d{2}-\d{2}):([A-Z]{3}):([A-Z]{3})")


def canonical_record_id(rid: object) -> object:
    """Ключ курса в записи жюри <date>:<source>:<target> -> fx_id (FX:<source>:<target>:<date>); прочее как есть."""
    if isinstance(rid, str):
        m = _JURY_FX_KEY.fullmatch(rid)
        if m:
            return fx_id(m.group(2), m.group(3), m.group(1))
    return rid


def _csv_rows(text: str, columns: tuple, name: str):
    """Строки CSV как {колонка: значение} в каноническом порядке колонок; заголовок — те же имена в любом порядке.
    Пустой файл (0 байт или одни пробелы/переводы строк, даже без заголовка) — пустая таблица: так выглядит
    fx.csv компании без валютных операций в фикстуре, собранной вручную."""
    if not text.strip():
        return
    reader = csv.reader(io.StringIO(text, newline=""))
    header = [h.strip() for h in (next(reader, None) or ())]
    if sorted(header) != sorted(columns):
        raise EpisodeDataError(f"{name} header mismatch")
    position = [header.index(c) for c in columns]
    for raw in reader:
        if not raw or (len(raw) == 1 and not raw[0].strip()):
            continue
        if len(raw) != len(header):
            raise EpisodeDataError(f"{name} row width mismatch")
        yield {c: raw[i] for c, i in zip(columns, position)}


def _parse_ledger(text: str) -> list[dict]:
    rows = []
    for raw in _csv_rows(text, LEDGER_COLUMNS, "ledger.csv"):
        row = {k: (v if v != "" else None) for k, v in raw.items()}
        try:
            row["debit_minor"] = int(row["debit_minor"])
            row["credit_minor"] = int(row["credit_minor"])
        except (TypeError, ValueError) as exc:
            raise EpisodeDataError("ledger.csv amount is not an integer") from exc
        rows.append(row)
    return rows


def _parse_fx(text: str) -> list[dict]:
    rows = []
    for row in _csv_rows(text, FX_COLUMNS, "fx.csv"):
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
    if not all(isinstance(t, str) for t in texts.values()):
        raise EpisodeDataError("public files must be texts")
    texts = {k: t[1:] if t.startswith("\ufeff") else t for k, t in texts.items()}  # BOM
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
