"""Первое наблюдение эпизода: задача, manifest, политика учёта, справка по действиям и бюджет.

Всё здесь — открытый контракт (docs/contract.md). Никаких сведений об эталоне:
ни числа ошибок, ни их типов в этом эпизоде, ни seed, ни путей к файлам.
"""

from __future__ import annotations

import copy

from .constants import ENV_VERSION, MAX_ACTION_BYTES, MAX_SUBMIT_BYTES, PAGE_LIMIT, TABLES
from .records import PublicEpisode

TASK = (
    "You are auditing the books of one fictional company for one reporting period. "
    "The ledger (ledger.csv) should contain exactly the postings of the documents "
    "(invoices, payments) that belong to the period, made according to the accounting policy. "
    "Find the real discrepancies of four known types and submit them with evidence. "
    "A book may also be completely clean: then submit an empty list. "
    "Plausible but correct cases exist (recurring invoices, split, partial, batch and on-account "
    "payments, refunds, documents near the period boundary, delayed postings) and must not be reported. "
    "Compare money only in one currency (convert with fx.csv and the policy), compare moments in time "
    "only after applying their UTC offsets. One document may carry issues of two different types; "
    "report each type as a separate finding. Every step costs a little, the episode ends on submit."
)

ISSUE_TYPES = {
    "duplicate_invoice": {
        "meaning": "the same invoice (same customer, same issued_at instant, identical line items "
                   "including order_ref) was entered twice and both copies are posted",
        "affected_ids": "both invoice_id values of the pair",
        "amount": "total_minor of one invoice, in the invoice currency",
        "evidence": "both invoice doc_id values",
    },
    "unapplied_payment": {
        "meaning": "invoice_refs of a payment allocate the whole payment to invoice X, but the ledger "
                   "credits 1200 for X with less than allocated; the rest sits on 2300",
        "affected_ids": "payment_id and the invoice_id X",
        "amount": "the 2300 credit of that payment as posted, in the ledger (base) currency",
        "evidence": "payment doc_id and the row_id of that 2300 row",
    },
    "fx_rate_error": {
        "meaning": "a converted ledger amount differs from the conversion at the rate date required by the policy",
        "affected_ids": "the invoice_id or the payment_id",
        "amount": "|posted - correct| on the 1200 line (invoice) or on the 1000 line (payment), in the base currency",
        "evidence": "document doc_id, the converted ledger row (invoice: 1200 or 4000; payment: 1000) "
                    "and the fx_id of the rate record the policy selects",
    },
    "cutoff_error": {
        "meaning": "a document that belongs to another period (by its own time in company_timezone) is posted in this ledger",
        "affected_ids": "the invoice_id or the payment_id",
        "amount": "|total_minor| or |amount_minor| of the document, in the document currency",
        "evidence": "document doc_id and any ledger row of that document",
    },
}

MATCHING = (
    "A finding counts only if its type, the set of affected_ids, the currency (exact string: no case change "
    "or trimming) and the amount (|claimed - expected| <= the published tolerance: 1 minor unit for every type "
    "in generated episodes) are right, every evidence id exists in this episode (a row_id, invoice_id, "
    "payment_id, fx_id or doc_id) and evidence_ids contain at least one complete evidence set (extra existing "
    "ids are allowed). Matching to the reference is one-to-one (maximum matching): repeated findings are false "
    "positives. A well-formed but wrong finding (unknown type, wrong currency, negative amount, unknown id) is "
    "a false positive. A malformed submission (wrong JSON types, missing or extra fields, more than 30 findings, "
    "more than 10 evidence ids in a finding, explanation over 1000 characters, another episode_id) ends the "
    "episode with reward -1. The text of explanation never changes the reward."
)

ACTIONS_HELP = {
    "list_records": {"args": {"table": "ledger | invoices | payments | fx",
                              "cursor": "null for the first page, then next_cursor from the previous page",
                              "limit": f"1..{PAGE_LIMIT} (larger values are reduced to {PAGE_LIMIT})"},
                     "result": "{records: [record], next_cursor: null | string}",
                     "example": {"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": PAGE_LIMIT}}},
    "get_record": {"args": {"table": "ledger | invoices | payments | fx",
                            "id": "row_id | invoice_id | payment_id | fx_id (FX:<source>:<target>:<date>)"},
                   "result": "the record object",
                   "example": {"name": "get_record", "args": {"table": "fx", "id": "FX:USD:KZT:2026-03-02"}}},
    "get_document": {"args": {"doc_id": "doc_id of an invoice or a payment"},
                     "result": "the document object (an invoice or a payment record)",
                     "example": {"name": "get_document", "args": {"doc_id": "D-XXXXXXXX"}}},
    "calculator": {"args": {"expression": "decimal arithmetic: + - * / ( ), abs(x), round_half_up(x[, n]), round_half_even(x[, n])"},
                   "result": "a decimal string, e.g. \"0.3\" for 0.1+0.2",
                   "example": {"name": "calculator", "args": {"expression": "round_half_up(23208 * 516.3571)"}}},
    "submit": {"args": {"findings": "{episode_id, findings: [{type, affected_ids, claimed_amount_minor, currency, evidence_ids, explanation}]}"},
               "result": "{accepted: true}; the episode ends (terminated=true)",
               "example": {"name": "submit", "args": {"findings": {"episode_id": "<manifest.episode_id>", "findings": []}}}},
}
ACTION_FORMAT = ("An action is {\"name\": <action>, \"args\": {...}}; the flat form {\"action\": <action>, ...args} is "
                 "accepted too. Every step() uses one step of the budget, a failing one too.")


def initial_observation(ep: PublicEpisode, max_steps: int, time_limit_s: float) -> dict:
    tables = {}
    for name, rows in ep.tables.items():
        spec = TABLES[name]
        fields = sorted(rows[0].keys()) if rows else []
        tables[name] = {"rows": len(rows), "id_field": spec["id_field"], "fields": fields}
        if spec["doc_field"]:
            tables[name]["document_id_field"] = spec["doc_field"]
    return {
        "task": TASK,
        "episode_id": ep.episode_id,
        "manifest": copy.deepcopy(ep.manifest),
        "policy": copy.deepcopy(ep.policy),
        "tables": tables,
        "issue_types": copy.deepcopy(ISSUE_TYPES),
        "matching": MATCHING,
        "actions": copy.deepcopy(ACTIONS_HELP),
        "action_format": ACTION_FORMAT,
        "limits": {
            "max_steps": max_steps,
            "page_limit": PAGE_LIMIT,
            "time_limit_seconds": time_limit_s,
            "max_findings": 30,
            "max_evidence_ids": 10,
            "max_explanation_chars": 1000,
            "max_action_bytes": MAX_ACTION_BYTES,
            "max_submit_bytes": MAX_SUBMIT_BYTES,
            "inspection_step_cost": 0.002,
        },
        "remaining_steps": max_steps,
        "env_version": ENV_VERSION,
    }
