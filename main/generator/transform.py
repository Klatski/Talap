"""Безвредное преобразование эпизода: перестановка строк + смена всех ID.

transform(эпизод, seed) -> новый эпизод, где
  * строки ledger/invoices/payments/fx перемешаны;
  * все ID (episode, entity, invoice, payment, doc, row, order_ref, issue, scenario)
    заменены на новые случайные того же формата;
  * truth.json и provenance.json переотображены на новые ID.
Бухгалтерское содержание не меняется, поэтому ответ oracle и награда обязаны
сохраниться (с учётом новых ID). ID курсов (FX:<src>:<dst>:<date>) — естественные
ключи, они не меняются.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .fileio import (
    FX_COLUMNS,
    LEDGER_COLUMNS,
    csv_text,
    json_text,
    jsonl_text,
    parse_fx,
    parse_jsonl,
    parse_ledger,
    read_episode_dir,
    write_episode_dir,
)
from .ids import IdFactory


class _Renamer:
    def __init__(self, rng: random.Random, existing: set[str]) -> None:
        self.ids = IdFactory(rng, used=existing)
        self.map: dict[str, str] = {}

    def __call__(self, old: str | None) -> str | None:
        if old is None:
            return None
        if old.startswith("FX:"):
            return old
        if old not in self.map:
            prefix, sep, tail = old.partition("-")
            if not sep or not tail:
                raise ValueError(f"unexpected id format: {old!r}")
            self.map[old] = self.ids.new(prefix + "-", len(tail))
        return self.map[old]


def _collect_ids(ledger, invoices, payments, truth) -> list[str]:
    found: list[str] = [truth["episode_id"]]
    for r in ledger:
        found += [r["row_id"], r["entity_id"], r["invoice_id"], r["payment_id"]]
    for inv in invoices:
        found += [inv["doc_id"], inv["invoice_id"], inv["entity_id"]]
        found += [li["order_ref"] for li in inv["line_items"]]
    for pay in payments:
        found += [pay["doc_id"], pay["payment_id"]]
        found += [ref["invoice_id"] for ref in pay["invoice_refs"] or []]
    for issue in truth["issues"]:
        found += [issue["issue_id"], issue["scenario_group"], *issue["affected_ids"]]
        for ev in issue["evidence_sets"]:
            found += ev
    return [x for x in found if x is not None]


def transform_texts(public: dict[str, str], private: dict[str, str], seed: int | str) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    manifest = json.loads(public["manifest.json"])
    ledger = parse_ledger(public["ledger.csv"])
    invoices = parse_jsonl(public["invoices.jsonl"])
    payments = parse_jsonl(public["payments.jsonl"])
    fx_rows = parse_fx(public["fx.csv"])
    truth = json.loads(private["truth.json"])
    provenance = json.loads(private["provenance.json"])

    rng = random.Random(f"transform|{seed}|{manifest['episode_id']}")
    all_ids = _collect_ids(ledger, invoices, payments, truth)
    rename = _Renamer(rng, set(all_ids))
    for old in all_ids:  # фиксированный порядок обхода -> детерминированные новые ID
        rename(old)

    new_ledger = [
        {
            **r,
            "row_id": rename(r["row_id"]),
            "entity_id": rename(r["entity_id"]),
            "invoice_id": rename(r["invoice_id"]),
            "payment_id": rename(r["payment_id"]),
        }
        for r in ledger
    ]
    new_invoices = []
    for inv in invoices:
        new_invoices.append(
            {
                **inv,
                "doc_id": rename(inv["doc_id"]),
                "invoice_id": rename(inv["invoice_id"]),
                "entity_id": rename(inv["entity_id"]),
                "line_items": [{**li, "order_ref": rename(li["order_ref"])} for li in inv["line_items"]],
            }
        )
    new_payments = []
    for pay in payments:
        refs = pay["invoice_refs"]
        new_payments.append(
            {
                **pay,
                "doc_id": rename(pay["doc_id"]),
                "payment_id": rename(pay["payment_id"]),
                "invoice_refs": None
                if refs is None
                else [{**ref, "invoice_id": rename(ref["invoice_id"])} for ref in refs],
            }
        )
    rng.shuffle(new_ledger)
    rng.shuffle(new_invoices)
    rng.shuffle(new_payments)
    new_fx = list(fx_rows)
    rng.shuffle(new_fx)

    old_episode = manifest["episode_id"]
    new_episode = rename(old_episode)
    manifest = {**manifest, "episode_id": new_episode}

    new_issues = []
    for issue in truth["issues"]:
        new_issues.append(
            {
                **issue,
                "issue_id": rename(issue["issue_id"]),
                "scenario_group": rename(issue["scenario_group"]),
                "affected_ids": sorted(rename(x) for x in issue["affected_ids"]),
                "evidence_sets": sorted(sorted(rename(x) for x in ev) for ev in issue["evidence_sets"]),
            }
        )
    new_issues.sort(key=lambda i: (i["type"], i["affected_ids"]))
    truth = {**truth, "episode_id": new_episode, "issues": new_issues}

    provenance = _replace_strings(provenance, rename.map)
    provenance.setdefault("transforms", []).append(
        {"kind": "permute_and_rename", "seed": str(seed), "source_episode_id": old_episode}
    )

    new_public = {
        "manifest.json": json_text(manifest),
        "policy.json": public["policy.json"],
        "ledger.csv": csv_text(LEDGER_COLUMNS, new_ledger),
        "invoices.jsonl": jsonl_text(new_invoices),
        "payments.jsonl": jsonl_text(new_payments),
        "fx.csv": csv_text(FX_COLUMNS, new_fx),
    }
    new_private = {"truth.json": json_text(truth), "provenance.json": json_text(provenance)}
    return new_public, new_private, dict(rename.map)


def _replace_strings(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {k: _replace_strings(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_strings(v, mapping) for v in value]
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


def transform_episode(src_dir: str | Path, dst_dir: str | Path, seed: int | str) -> dict[str, str]:
    """Прочитать эпизод из src_dir, записать преобразованный в dst_dir. Возвращает карту старый->новый ID."""
    public, private = read_episode_dir(src_dir)
    new_public, new_private, mapping = transform_texts(public, private, seed)
    write_episode_dir(dst_dir, new_public, new_private)
    return mapping


def remap_findings(findings: dict, mapping: dict[str, str]) -> dict:
    """Переотобразить findings (ответ агента/oracle) на новые ID — для парных проверок."""
    return _replace_strings(findings, mapping)
