"""Рендер истории в файлы эпизода + эталон (truth) + provenance.

ID назначаются ТОЛЬКО здесь, после внесения ошибок, из отдельного локального RNG;
порядок строк — по времени с случайным тай-брейком. Поэтому ни позиция, ни ID,
ни формат строки не выдают внесённые изменения.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any

from .fileio import FX_COLUMNS, LEDGER_COLUMNS, csv_text, fx_record_id, json_text, jsonl_text
from .history import CHART_OF_ACCOUNTS, History, Invoice, Payment
from .ids import IdFactory
from .inject import IssueSpec
from .money import format_rate
from .timeutil import format_offset, to_iso
from .version import GENERATOR_VERSION, SCHEMA_VERSION

# Допуск суммы τ по типу (truth.amount_tolerance_minor). Точные суммы — 0: дубль и cut-off — total документа,
# неразнесённый платёж — строка 2300 книги. Курсовая разница — 1: пересчёт по курсу с округлением
# (policy.json) — единственное место, где агент может разойтись с эталоном на единицу из-за округления.
AMOUNT_TOLERANCE_MINOR = {"duplicate_invoice": 0, "unapplied_payment": 0, "cutoff_error": 0, "fx_rate_error": 1}


@dataclass
class Rendered:
    episode_id: str
    public: dict[str, str]
    truth: dict
    provenance: dict


class KeyedIds:
    """ID выводится из (материал, внутренний ключ записи): у одной и той же записи один и тот же
    ID и в чистой (контрфактической), и в испорченной версии эпизода. Порядок назначения
    не влияет на результат; коллизии разрешаются детерминированно."""

    def __init__(self, material: str) -> None:
        self.material = material
        self.used: set[str] = set()

    def new(self, prefix: str, length: int, key: str) -> str:
        attempt = 0
        while True:
            rng = random.Random(f"{self.material}|{prefix}|{key}|{attempt}")
            candidate = IdFactory(rng).new(prefix, length)
            if candidate not in self.used:
                self.used.add(candidate)
                return candidate
            attempt += 1


class IdMaps:
    def __init__(self, h: History, material: str) -> None:
        ids = KeyedIds(material)
        self.episode_id = ids.new("ep-", 10, "episode")
        self.entity = {k: ids.new("C-", 6, k) for k in h.customers}
        self.entity.update({k: ids.new("V-", 6, k) for k in h.vendors})
        self.invoice_id: dict[str, str] = {}
        self.payment_id: dict[str, str] = {}
        self.doc_id: dict[str, str] = {}
        for key in h.invoices:
            self.invoice_id[key] = ids.new("INV-", 7, key)
            self.doc_id[key] = ids.new("D-", 8, key)
        for key in h.payments:
            self.payment_id[key] = ids.new("PAY-", 7, key)
            self.doc_id[key] = ids.new("D-", 8, key)
        self.order: dict[str, str] = {}
        for inv in h.invoices.values():
            for line in inv.lines:
                if line.order not in self.order:
                    self.order[line.order] = ids.new("SO-", 7, line.order)
        self.ids = ids

    def business_id(self, key: str) -> str:
        return self.invoice_id.get(key) or self.payment_id[key]


def _policy(h: History) -> dict:
    rule_text = {
        "document_date": "rate of the document's calendar date in company_timezone",
        "previous_day": "rate of the calendar day before the document's date in company_timezone",
    }
    rounding_text = {
        "half_up": "round half away from zero to an integer number of minor units",
        "half_even": "round half to even (banker's rounding) to an integer number of minor units",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "company_timezone": format_offset(h.offset),
        "ledger_currency": h.base,
        "ledger_scope": "Postings of documents that belong to the reporting period only; no opening balances.",
        "document_window": "invoices.jsonl and payments.jsonl also contain documents from about 30 days before and 5 days after the period; those must not be posted in this ledger.",
        "period": {
            "start": to_iso(h.start, h.offset),
            "end": to_iso(h.end, h.offset),
            "start_inclusive": True,
            "end_inclusive": False,
            "attribution": "A document belongs to the period that contains its own timestamp (invoices: issued_at, payments: timestamp) converted to company_timezone.",
        },
        "fx": {
            "rate_date_rule": h.p.rate_rule,
            "rate_date_rule_description": rule_text[h.p.rate_rule],
            "missing_rate": "latest_prior_available",
            "missing_rate_description": "if fx.csv has no rate for the selected date, use the latest earlier date that has one",
            "quote": "rate = units of target_currency for 1 unit of source_currency",
            "formula": "target_minor = round(source_minor * rate * 10^(exponent(target) - exponent(source)))",
            "invoice_posting": "1200 debit and 4000 credit, both converted at the invoice rate",
            "payment_posting": "1000 and 2300 converted at the payment rate; each 1200 credit converted at the rate of the invoice it settles; the difference goes to 7900",
            "record_id": "FX:<source_currency>:<target_currency>:<date>",
        },
        "rounding": {
            "mode": h.p.rounding,
            "description": rounding_text[h.p.rounding],
            "applies_to": "every converted ledger line, separately",
        },
        "payments": {
            "allocation_source": "invoice_refs: list of {invoice_id, amount_minor} in the payment currency (always the invoice currency); null when the payer gave no reference",
            "unallocated_amounts": "payments without invoice_refs and any excess of amount_minor over the sum of invoice_refs are credited to 2300",
            "refunds": "negative amount_minor, no invoice_refs; debit 2300, credit 1000; ledger currency only",
        },
        "billing": {
            "order_lines": "each (order_ref, sku) pair is billed at most once; recurring services get a new order_ref for every billing",
        },
        "accounts": dict(CHART_OF_ACCOUNTS),
    }


def _tiebreak(material: str, key: str) -> float:
    return random.Random(f"{material}|order|{key}").random()


def render(
    h: History,
    issues: list[IssueSpec],
    id_material: str,
    order_material: str,
    meta: dict[str, Any],
) -> Rendered:
    maps = IdMaps(h, id_material)
    entries = h.all_entries()
    tiebreak = {e.key: _tiebreak(order_material, e.key) for e in entries}
    entries_sorted = sorted(entries, key=lambda e: (e.ts, tiebreak[e.key]))

    ledger_rows = []
    row_ids: dict[tuple[str, int], str] = {}
    for entry in entries_sorted:
        ts = to_iso(entry.ts, entry.offset)
        seen_roles: dict[tuple[str, str], int] = {}
        for i, line in enumerate(entry.lines):
            role_key = (line.role, line.invoice or "")
            n = seen_roles.get(role_key, 0)
            seen_roles[role_key] = n + 1
            row_id = maps.ids.new("L-", 8, f"{entry.key}|{line.role}|{line.invoice or ''}|{n}")
            row_ids[(entry.key, i)] = row_id
            ledger_rows.append(
                {
                    "row_id": row_id,
                    "timestamp": ts,
                    "account": line.account,
                    "debit_minor": line.debit,
                    "credit_minor": line.credit,
                    "currency": h.base,
                    "entity_id": maps.entity[line.entity] if line.entity else None,
                    "invoice_id": maps.invoice_id[line.invoice] if line.invoice else None,
                    "payment_id": maps.payment_id[line.payment] if line.payment else None,
                }
            )

    inv_tie = {k: _tiebreak(order_material, k) for k in h.invoices}
    invoices = []
    for inv in sorted(h.invoices.values(), key=lambda i: (i.issued_at, inv_tie[i.key])):
        invoices.append(
            {
                "doc_id": maps.doc_id[inv.key],
                "invoice_id": maps.invoice_id[inv.key],
                "entity_id": maps.entity[inv.customer],
                "issued_at": to_iso(inv.issued_at, inv.offset),
                "due_at": to_iso(inv.due_at, inv.offset),
                "currency": inv.currency,
                "total_minor": inv.total_minor,
                "line_items": [
                    {
                        "line_no": n,
                        "sku": line.sku,
                        "description": line.description,
                        "quantity": line.quantity,
                        "unit_price_minor": line.unit_price_minor,
                        "amount_minor": line.amount_minor,
                        "order_ref": maps.order[line.order],
                    }
                    for n, line in enumerate(inv.lines, start=1)
                ],
            }
        )

    pay_tie = {k: _tiebreak(order_material, k) for k in h.payments}
    payments = []
    for pay in sorted(h.payments.values(), key=lambda p: (p.ts, pay_tie[p.key])):
        payments.append(
            {
                "doc_id": maps.doc_id[pay.key],
                "payment_id": maps.payment_id[pay.key],
                "timestamp": to_iso(pay.ts, pay.offset),
                "currency": pay.currency,
                "amount_minor": pay.amount_minor,
                "invoice_refs": None
                if pay.allocations is None
                else [{"invoice_id": maps.invoice_id[a.invoice], "amount_minor": a.amount_minor} for a in pay.allocations],
            }
        )

    fx_rows = []
    for d in h.fx_dates:
        for cur in h.foreign:
            fx_rows.append(
                {"date": d.isoformat(), "source_currency": cur, "target_currency": h.base, "rate": format_rate(h.fx[cur][d])}
            )

    policy = _policy(h)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "episode_id": maps.episode_id,
        "base_currency": h.base,
        "currency_exponents": dict(h.exponents),
        "period_start": to_iso(h.start, h.offset),
        "period_end": to_iso(h.end, h.offset),
        "timezone": format_offset(h.offset),
        "counts": {
            "ledger_rows": len(ledger_rows),
            "invoices": len(invoices),
            "payments": len(payments),
            "documents": len(invoices) + len(payments),
            "fx_rates": len(fx_rows),
        },
        "rounding": {"mode": h.p.rounding, "unit": "minor"},
        "money_format": "integer minor units; exponent per currency in currency_exponents",
        "fx_rate_format": "decimal string",
        "tables": {
            "ledger": {"file": "ledger.csv", "id_field": "row_id"},
            "invoices": {"file": "invoices.jsonl", "id_field": "invoice_id", "document_id_field": "doc_id"},
            "payments": {"file": "payments.jsonl", "id_field": "payment_id", "document_id_field": "doc_id"},
            "fx": {"file": "fx.csv", "id_format": "FX:<source_currency>:<target_currency>:<date>"},
        },
    }

    public = {
        "manifest.json": json_text(manifest),
        "policy.json": json_text(policy),
        "ledger.csv": csv_text(LEDGER_COLUMNS, ledger_rows),
        "invoices.jsonl": jsonl_text(invoices),
        "payments.jsonl": jsonl_text(payments),
        "fx.csv": csv_text(FX_COLUMNS, fx_rows),
    }

    # ---------------------------------------------------------------- эталон
    entry_by_doc = {e.doc: e for e in h.entries if e.doc}
    group_ids: dict[int, str] = {}

    def resolve(ref: tuple) -> list[str]:
        if ref[0] == "doc":
            return [maps.doc_id[ref[1]]]
        if ref[0] == "row":
            entry = entry_by_doc[ref[1]]
            found = [row_ids[(entry.key, i)] for i, ln in enumerate(entry.lines) if ref[2] in ("*", ln.role)]
            if not found:
                raise AssertionError(f"no evidence row {ref}")
            return found
        if ref[0] == "fx":
            return [fx_record_id(ref[1], h.base, ref[2].isoformat())]
        raise AssertionError(ref)

    truth_issues = []
    provenance_injections = []
    for issue in issues:
        sets = set()
        for template in issue.evidence:
            for combo in itertools.product(*(resolve(r) for r in template)):
                sets.add(tuple(sorted(set(combo))))
        group = None
        if issue.group is not None:
            if issue.group not in group_ids:
                group_ids[issue.group] = maps.ids.new("SCN-", 6, f"group{issue.group}")
            group = group_ids[issue.group]
        record = {
            "issue_id": None,
            "type": issue.type,
            "affected_ids": sorted(maps.business_id(k) for k in issue.affected),
            "expected_amount_minor": issue.amount_minor,
            "currency": issue.currency,
            "amount_tolerance_minor": AMOUNT_TOLERANCE_MINOR[issue.type],
            "evidence_sets": [list(s) for s in sorted(sets)],
            "scenario": issue.scenario,
            "scenario_group": group,
        }
        truth_issues.append((record, issue))

    truth_issues.sort(key=lambda pair: (pair[0]["type"], pair[0]["affected_ids"]))
    for record, issue in truth_issues:
        record["issue_id"] = maps.ids.new("ISS-", 8, f"{issue.type}|{'|'.join(sorted(issue.affected))}")
        provenance_injections.append(
            {
                "issue_id": record["issue_id"],
                "type": issue.type,
                "variant": issue.variant,
                "cause": issue.cause,
                "scenario": issue.scenario,
                "scenario_group": record["scenario_group"],
                "affected_ids": record["affected_ids"],
                "affected_doc_ids": sorted(maps.doc_id[k] for k in issue.affected),
                "before": _resolve_values(issue.before, maps),
                "after": _resolve_values(issue.after, maps),
                "expected_amount_minor": record["expected_amount_minor"],
                "currency": record["currency"],
            }
        )

    truth = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": maps.episode_id,
        "issues": [record for record, _ in truth_issues],
    }

    distractors = []
    for item in h.distractors:
        distractors.append(_resolve_values(item, maps))
    distractors.sort(key=lambda d: (d["kind"], str(sorted(d.items()))))

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "episode_id": maps.episode_id,
        **meta,
        "history": {
            "customers": len(h.customers),
            "vendors": len(h.vendors),
            "products": len(h.products),
            "invoices_by_period": _count_by_period(h, h.invoices.values()),
            "payments_by_period": _count_by_period(h, h.payments.values()),
            "doc_offsets": [format_offset(o) for o in h.doc_palette],
        },
        "injections": provenance_injections,
        "distractors": distractors,
        "transforms": [],
    }
    return Rendered(maps.episode_id, public, truth, provenance)


def _count_by_period(h: History, docs) -> dict[str, int]:
    counts = {"prev": 0, "cur": 0, "next": 0}
    for doc in docs:
        counts[h.period_of(doc.ts)] += 1
    return counts


def _resolve_values(value: Any, maps: IdMaps) -> Any:
    """Заменить внутренние ключи документов на публичные ID в записях provenance."""
    if isinstance(value, dict):
        return {k: _resolve_values(v, maps) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_values(v, maps) for v in value]
    if isinstance(value, str):
        if value in maps.invoice_id:
            return maps.invoice_id[value]
        if value in maps.payment_id:
            return maps.payment_id[value]
    return value
