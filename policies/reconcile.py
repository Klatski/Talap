"""Простой алгоритм сверки (базовая политика и детектор утечек через артефакты).

Читает все таблицы и сверяет книгу с документами «по поверхности», без анализа политики:
  * даты берутся как записаны в строке времени (первые 10 символов), часовые пояса игнорируются;
  * курс — на записанную дату документа (или ближайший предыдущий), правило previous_day
    и режим округления политики игнорируются (всегда half_up);
  * дубль — любые два счёта одного клиента с одинаковыми валютой и суммой;
  * неразнесённый платёж — любой кредит 2300 у платежа со ссылками на счёт.
Такая сверка должна попадаться на отвлекающие случаи (повторяющиеся счета, переплаты,
документы у границы в «обманчивых» смещениях). Если она находит ошибки почти без промахов,
генератор выдаёт ошибки по артефактам — это и есть смысл детектора.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal, localcontext

from .base import ScriptedPolicy, TableReadError, submit_action


def _conv(amount: int, src: str, base: str, rate: Decimal, exp: dict) -> int:
    with localcontext() as ctx:
        ctx.prec = 60
        v = Decimal(amount) * rate * (Decimal(10) ** (exp[base] - exp[src]))
        return int(v.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def naive_findings(manifest: dict, ledger: list[dict], invoices: list[dict], payments: list[dict], fx: list[dict]) -> list[dict]:
    base, exp = manifest["base_currency"], manifest["currency_exponents"]
    start, end = manifest["period_start"][:10], manifest["period_end"][:10]
    rates: dict[str, list[tuple[str, Decimal, str]]] = defaultdict(list)
    for r in fx:
        rates[r["source_currency"]].append((r["date"], Decimal(r["rate"]), r.get("fx_id") or f"FX:{r['source_currency']}:{r['target_currency']}:{r['date']}"))
    for cur in rates:
        rates[cur].sort()

    def rate(cur: str, ts: str):
        series = rates.get(cur, [])
        pos = bisect.bisect_right([d for d, _, _ in series], ts[:10])
        return series[pos - 1] if pos else None

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in ledger:
        key = row.get("payment_id") or row.get("invoice_id")
        if key:
            by_doc[key].append(row)
    out: list[dict] = []
    for inv in invoices:
        rows = by_doc.get(inv["invoice_id"])
        if not rows:
            continue
        if not start <= inv["issued_at"][:10] < end:
            out.append({"type": "cutoff_error", "affected_ids": [inv["invoice_id"]], "claimed_amount_minor": abs(inv["total_minor"]),
                        "currency": inv["currency"], "evidence_ids": [inv["doc_id"], rows[0]["row_id"]], "explanation": "date outside period"})
        ar = [r for r in rows if r["account"] == "1200" and r["debit_minor"]]
        if inv["currency"] != base and ar and (found := rate(inv["currency"], inv["issued_at"])):
            diff = abs(ar[0]["debit_minor"] - _conv(inv["total_minor"], inv["currency"], base, found[1], exp))
            if diff:
                out.append({"type": "fx_rate_error", "affected_ids": [inv["invoice_id"]], "claimed_amount_minor": diff, "currency": base,
                            "evidence_ids": [inv["doc_id"], ar[0]["row_id"], found[2]], "explanation": "amount differs"})
    for pay in payments:
        rows = by_doc.get(pay["payment_id"])
        if not rows:
            continue
        if not start <= pay["timestamp"][:10] < end:
            out.append({"type": "cutoff_error", "affected_ids": [pay["payment_id"]], "claimed_amount_minor": abs(pay["amount_minor"]),
                        "currency": pay["currency"], "evidence_ids": [pay["doc_id"], rows[0]["row_id"]], "explanation": "date outside period"})
        bank = [r for r in rows if r["account"] == "1000" and r["debit_minor"]]
        if pay["currency"] != base and bank and pay["amount_minor"] > 0 and (found := rate(pay["currency"], pay["timestamp"])):
            diff = abs(bank[0]["debit_minor"] - _conv(pay["amount_minor"], pay["currency"], base, found[1], exp))
            if diff:
                out.append({"type": "fx_rate_error", "affected_ids": [pay["payment_id"]], "claimed_amount_minor": diff, "currency": base,
                            "evidence_ids": [pay["doc_id"], bank[0]["row_id"], found[2]], "explanation": "amount differs"})
        un = [r for r in rows if r["account"] == "2300" and r["credit_minor"]]
        refs = pay.get("invoice_refs") or []
        if un and refs:
            out.append({"type": "unapplied_payment", "affected_ids": [pay["payment_id"], refs[0]["invoice_id"]],
                        "claimed_amount_minor": un[0]["credit_minor"], "currency": base,
                        "evidence_ids": [pay["doc_id"], un[0]["row_id"]], "explanation": "amount left on 2300"})
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for inv in invoices:
        groups[(inv["entity_id"], inv["currency"], inv["total_minor"])].append(inv)
    for members in groups.values():
        members = sorted(members, key=lambda m: (m["issued_at"], m["due_at"]))
        for a, b in zip(members, members[1:]):
            out.append({"type": "duplicate_invoice", "affected_ids": [a["invoice_id"], b["invoice_id"]],
                        "claimed_amount_minor": a["total_minor"], "currency": a["currency"],
                        "evidence_ids": [a["doc_id"], b["doc_id"]], "explanation": "same customer and amount"})
    # Обрезка до 30 по содержимому (тип, сумма, валюта, время документа), а не по ID и порядку строк:
    # ответ не меняется при перестановке строк и смене ID (парные проверки).
    when = {inv["invoice_id"]: inv["issued_at"] for inv in invoices}
    when.update({pay["payment_id"]: pay["timestamp"] for pay in payments})
    out.sort(key=lambda f: (f["type"], f["claimed_amount_minor"], f["currency"],
                            sorted(when.get(a, "") for a in f["affected_ids"])))
    return out[:30]


class ReconcilePolicy(ScriptedPolicy):
    name = "reconcile"

    def script(self, obs0: dict):
        tables = {}
        try:
            for table in ("invoices", "payments", "fx", "ledger"):
                tables[table] = yield from self.read_table(table)
        except TableReadError:
            yield submit_action(self.episode_id, [])
            return
        yield submit_action(self.episode_id, naive_findings(obs0["manifest"], tables["ledger"], tables["invoices"],
                                                            tables["payments"], tables["fx"]))
