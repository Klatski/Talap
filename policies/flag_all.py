"""Атака «пометить всё»: каждый документ × каждый подходящий тип, без анализа доказательств.

Читает только счета и платежи (≤ 8 шагов), затем отправляет до 30 выводов по кругу типов:
сумма = сумма документа, валюта = валюта документа, доказательство = doc_id (+ ссылка на счёт
у платежа). Обязана получать плохую награду: так проверяется устойчивость награды.
"""

from __future__ import annotations

from .base import ScriptedPolicy, TableReadError, submit_action

MAX_FINDINGS = 30


class FlagAllPolicy(ScriptedPolicy):
    name = "flag_all"

    def script(self, obs0: dict):
        try:
            invoices = yield from self.read_table("invoices")
            payments = yield from self.read_table("payments")
        except TableReadError:
            invoices, payments = [], []
        candidates: list[list[dict]] = [[], [], [], []]
        for inv in invoices:
            base = {"claimed_amount_minor": abs(inv["total_minor"]), "currency": inv["currency"],
                    "evidence_ids": [inv["doc_id"]], "explanation": "flagged"}
            candidates[0].append({"type": "duplicate_invoice", "affected_ids": [inv["invoice_id"]], **base})
            candidates[1].append({"type": "cutoff_error", "affected_ids": [inv["invoice_id"]], **base})
            candidates[2].append({"type": "fx_rate_error", "affected_ids": [inv["invoice_id"]], **base})
        for pay in payments:
            refs = [r["invoice_id"] for r in pay.get("invoice_refs") or []]
            base = {"claimed_amount_minor": abs(pay["amount_minor"]), "currency": pay["currency"],
                    "evidence_ids": [pay["doc_id"]], "explanation": "flagged"}
            candidates[3].append({"type": "unapplied_payment", "affected_ids": [pay["payment_id"], *refs[:1]], **base})
            candidates[1].append({"type": "cutoff_error", "affected_ids": [pay["payment_id"]], **base})
            candidates[2].append({"type": "fx_rate_error", "affected_ids": [pay["payment_id"]], **base})
        findings: list[dict] = []
        k = 0
        while len(findings) < MAX_FINDINGS and any(candidates):
            bucket = candidates[k % 4]
            if bucket:
                findings.append(bucket.pop(0))
            k += 1
        yield submit_action(self.episode_id, findings)
