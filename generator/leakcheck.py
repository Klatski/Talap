"""Детектор утечек через артефакты: «тупые» признаки документов vs. принадлежность к ошибке.

Для каждого документа считаются признаки, которые НЕ требуют анализа доказательств:
позиция в файле, круглость суммы, смещение записи, близость к границе, выходной день,
задержка проводки, длина/формат ID. Если какой-то признак сам по себе выделяет
ошибочные документы заметно лучше случайного, генератор выдаёт ошибки по артефактам.

Отчёт: для каждого признака — доля ошибочных среди документов с признаком (precision)
и доля ошибок, которые признак покрывает (recall), против базовой частоты ошибок.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta
from typing import Iterable

from .fileio import parse_jsonl, parse_ledger
from .timeutil import parse_iso, parse_offset


def document_features(public: dict[str, str]) -> list[tuple[str, dict[str, bool]]]:
    manifest = json.loads(public["manifest.json"])
    offset = parse_offset(manifest["timezone"])
    start = parse_iso(manifest["period_start"])
    end = parse_iso(manifest["period_end"])
    invoices = parse_jsonl(public["invoices.jsonl"])
    payments = parse_jsonl(public["payments.jsonl"])
    ledger = parse_ledger(public["ledger.csv"])
    first_row_ts: dict[str, object] = {}
    for row in ledger:
        key = row["payment_id"] or row["invoice_id"]
        if key and key not in first_row_ts:
            first_row_ts[key] = parse_iso(row["timestamp"])
    out = []
    for records, ts_field, amount_field, id_field in (
        (invoices, "issued_at", "total_minor", "invoice_id"),
        (payments, "timestamp", "amount_minor", "payment_id"),
    ):
        n = len(records)
        for pos, rec in enumerate(records):
            ts_text = rec[ts_field]
            ts = parse_iso(ts_text)
            amount = abs(rec[amount_field])
            local = ts.astimezone(parse_iso(manifest["period_start"]).tzinfo)
            posted_ts = first_row_ts.get(rec[id_field])
            feats = {
                "last_10pct_of_file": pos >= n * 0.9,
                "round_amount_1000": amount % 1000 == 0,
                "non_company_offset": not ts_text.endswith(manifest["timezone"]),
                "utc_offset": ts_text.endswith("+00:00"),
                "within_6h_of_boundary": min(abs(ts - start), abs(ts - end)) <= timedelta(hours=6),
                "weekend_company_tz": local.weekday() >= 5,
                "posted": posted_ts is not None,
                "posting_differs_from_document_time": posted_ts is not None and posted_ts != ts,
                "foreign_currency": rec["currency"] != manifest["base_currency"],
            }
            out.append((rec[id_field], feats))
    del offset
    return out


def leak_report(episodes: Iterable[tuple[dict[str, str], dict]]) -> dict:
    """episodes: пары (public_texts, truth). Возвращает статистику по признакам."""
    total = errors = 0
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"with": 0, "with_error": 0})
    for public, truth in episodes:
        bad = {i for issue in truth["issues"] for i in issue["affected_ids"]}
        for doc_id, feats in document_features(public):
            total += 1
            is_err = doc_id in bad
            errors += is_err
            for name, value in feats.items():
                if value:
                    stats[name]["with"] += 1
                    stats[name]["with_error"] += is_err
    base_rate = errors / total if total else 0.0
    report = {"documents": total, "error_documents": errors, "base_rate": round(base_rate, 4), "features": {}}
    for name, s in sorted(stats.items()):
        precision = s["with_error"] / s["with"] if s["with"] else 0.0
        recall = s["with_error"] / errors if errors else 0.0
        report["features"][name] = {
            "documents": s["with"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "lift": round(precision / base_rate, 2) if base_rate else 0.0,
        }
    return report
