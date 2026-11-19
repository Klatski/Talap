"""Детектор утечек через артефакты: «тупые» признаки документов vs. принадлежность к ошибке.

Для каждого документа считаются признаки, которые НЕ требуют анализа доказательств:
позиция в файле, круглость суммы, смещение записи, близость к границе, выходной день,
задержка проводки, длина/формат ID. Если какой-то признак сам по себе выделяет
ошибочные документы заметно лучше случайного, генератор выдаёт ошибки по артефактам.

Отчёт: для каждого признака — доля ошибочных среди документов с признаком (precision)
и доля ошибок, которые признак покрывает (recall), против базовой частоты ошибок.
Плюс «тупой классификатор» (наивный Байес, обучение/проверка на разных эпизодах): AUC и
precision@k по всем признакам, по артефактным и по артефактным среди проведённых документов периода.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta
from typing import Iterable

from .fileio import parse_jsonl, parse_ledger
from .timeutil import parse_iso


def document_features(public: dict[str, str]) -> list[tuple[str, dict[str, bool]]]:
    manifest = json.loads(public["manifest.json"])
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
    return out


# Признаки формы записи (позиция, формат, ID, круглость, смещение) — не содержание. Остальные
# (проведён ли документ, валюта, близость к границе, число строк проводки, invoice_refs) —
# содержательные: ошибка конкретного типа по определению ими обладает.
ARTIFACT_FEATURES = (
    "last_10pct_of_file", "round_amount_1000", "non_company_offset", "utc_offset", "weekend_company_tz",
    "posting_differs_from_document_time", "file_position_decile", "ledger_position_decile", "hour_utc",
    "amount_magnitude", "line_items", "id_first_char", "doc_id_first_char",
)


def _bucket(value: float, edges: tuple[float, ...]) -> int:
    return sum(value >= e for e in edges)


def document_categorical(public: dict[str, str]) -> list[tuple[str, dict[str, object]]]:
    """Бинарные признаки + категориальные артефакты (позиция, час, порядок величины суммы,
    число строк документа и проводки, символы ID). Только то, что видно без анализа доказательств."""
    invoices = {r["invoice_id"]: r for r in parse_jsonl(public["invoices.jsonl"])}
    payments = {r["payment_id"]: r for r in parse_jsonl(public["payments.jsonl"])}
    ledger = parse_ledger(public["ledger.csv"])
    rows_per_doc: dict[str, int] = defaultdict(int)
    first_row_pos: dict[str, int] = {}
    for pos, row in enumerate(ledger):
        key = row["payment_id"] or row["invoice_id"]
        if key:
            rows_per_doc[key] += 1
            first_row_pos.setdefault(key, pos)
    manifest = json.loads(public["manifest.json"])
    start, end = parse_iso(manifest["period_start"]), parse_iso(manifest["period_end"])
    order = {k: i / max(1, len(invoices)) for i, k in enumerate(invoices)}
    order.update({k: i / max(1, len(payments)) for i, k in enumerate(payments)})
    out = []
    for doc_key, feats in document_features(public):
        rec = invoices.get(doc_key) or payments[doc_key]
        ts = parse_iso(rec.get("issued_at") or rec["timestamp"])
        amount = abs(rec.get("total_minor", rec.get("amount_minor")))
        cats: dict[str, object] = dict(feats)
        cats.update(
            {
                "kind": "invoice" if doc_key in invoices else "payment",
                "file_position_decile": _bucket(order[doc_key], tuple(i / 10 for i in range(1, 10))),
                "ledger_position_decile": _bucket(first_row_pos.get(doc_key, -1) / max(1, len(ledger)), tuple(i / 10 for i in range(1, 10))),
                "hour_utc": ts.hour,
                "amount_magnitude": len(str(amount)),
                "line_items": len(rec.get("line_items") or []),
                "ledger_rows": min(rows_per_doc.get(doc_key, 0), 6),
                "refs": "null" if rec.get("invoice_refs", "absent") is None else len(rec.get("invoice_refs") or []),
                "id_first_char": doc_key.split("-", 1)[1][0],
                "doc_id_first_char": rec["doc_id"].split("-", 1)[1][0],
                "in_period": start <= ts < end,
            }
        )
        out.append((doc_key, cats))
    return out


def full_report(episodes: Iterable[tuple[dict[str, str], dict]]) -> dict:
    episodes = list(episodes)
    return {
        "single_features": leak_report(episodes),
        "classifier_all_features": classifier_report(episodes),
        "classifier_artifact_features": classifier_report(episodes, ARTIFACT_FEATURES),
        "classifier_artifact_features_posted_in_period": classifier_report(episodes, ARTIFACT_FEATURES, True),
    }


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


def _labelled(episodes, features, posted_in_period_only) -> list[list[tuple[dict[str, object], bool]]]:
    out = []
    for public, truth in episodes:
        bad = {i for issue in truth["issues"] for i in issue["affected_ids"]}
        rows = []
        for key, cats in document_categorical(public):
            if posted_in_period_only and not (cats["posted"] and cats["in_period"]):
                continue
            rows.append(({n: v for n, v in cats.items() if features is None or n in features}, key in bad))
        out.append(rows)
    return out


def _auc(scored: list[tuple[float, bool]]) -> float:
    pos = [s for s, y in scored if y]
    neg = [s for s, y in scored if not y]
    if not pos or not neg:
        return 0.5
    ranked = sorted(scored, key=lambda t: t[0])
    rank_sum, i = 0.0, 0
    while i < len(ranked):  # средние ранги для равных оценок
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for t in ranked[i:j] if t[1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def classifier_report(
    episodes: Iterable[tuple[dict[str, str], dict]],
    features: Iterable[str] | None = None,
    posted_in_period_only: bool = False,
) -> dict:
    """«Тупой классификатор» (наивный Байес по признакам документа).

    Обучается на чётных эпизодах, оценивается на нечётных (и наоборот). Если ошибки
    находятся по артефактам, AUC близок к 1 и precision@k много выше базовой частоты.
    features=None — все признаки; ARTIFACT_FEATURES — только форма записи.
    posted_in_period_only — сравнивать только проведённые документы периода, т.е. убрать
    содержательный сигнал «документ другого периода попал в книгу» (это и есть cut-off).
    """
    import math

    data = _labelled(list(episodes), set(features) if features is not None else None, posted_in_period_only)
    folds = [[d for i, d in enumerate(data) if i % 2 == f] for f in (0, 1)]
    scored: list[tuple[float, bool]] = []
    for train_fold, test_fold in ((folds[0], folds[1]), (folds[1], folds[0])):
        rows = [r for ep in train_fold for r in ep]
        n_pos = sum(y for _, y in rows)
        n_neg = len(rows) - n_pos
        counts: dict[tuple[str, object, bool], int] = defaultdict(int)
        values: dict[str, set] = defaultdict(set)
        for cats, y in rows:
            for name, v in cats.items():
                counts[(name, v, y)] += 1
                values[name].add(v)
        prior = math.log((n_pos + 1) / (n_neg + 1))
        for ep in test_fold:
            for cats, y in ep:
                score = prior
                for name, v in cats.items():
                    k = len(values[name]) + 1
                    score += math.log((counts[(name, v, True)] + 1) / (n_pos + k))
                    score -= math.log((counts[(name, v, False)] + 1) / (n_neg + k))
                scored.append((score, y))
    n_err = sum(y for _, y in scored)
    base = n_err / len(scored) if scored else 0.0
    top = sorted(scored, key=lambda t: -t[0])[:n_err]
    precision_at_k = sum(y for _, y in top) / n_err if n_err else 0.0
    return {
        "documents": len(scored),
        "error_documents": n_err,
        "base_rate": round(base, 4),
        "auc": round(_auc(scored), 4),
        "precision_at_k": round(precision_at_k, 4),
        "lift_at_k": round(precision_at_k / base, 2) if base else 0.0,
    }
