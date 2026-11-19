"""Скриптовый аудитор (зона B): демо-политика и независимый валидатор данных A.

Читает ВСЕ открытые таблицы через инструменты среды (list_records с курсорами, ≤ ~41 шага
на самых больших эпизодах), затем по одному опубликованному контракту (docs/contract.md,
разделы 3–4; policy.json и manifest.json эпизода) выводит расхождения четырёх типов с
минимальными наборами доказательств и отправляет их.

Код написан независимо от generator/ (не импортирует его): если аудитор на эпизоде A
получает F1 < 1, значит эталон A расходится с опубликованным контрактом или данными —
это перекрёстная проверка «B проверяет данные A». Функция derive_findings() используется
также тестами напрямую (без среды) на сотнях сгенерированных эпизодов.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext
from typing import Any

from .base import ScriptedPolicy, TableReadError, submit_action

BANK, AR, UNAPPLIED, REVENUE = "1000", "1200", "2300", "4000"
ROUNDING = {"half_up": ROUND_HALF_UP, "half_even": ROUND_HALF_EVEN}


def _offset(text: str) -> timezone:
    sign = -1 if text.startswith("-") else 1
    hh, mm = text[1:].split(":")
    return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))


def _instant(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise ValueError("timestamp without offset")
    return value


class Books:
    """Открытые записи + правила политики эпизода."""

    def __init__(self, manifest: dict, policy: dict, ledger: list[dict], invoices: list[dict],
                 payments: list[dict], fx: list[dict]) -> None:
        self.base = manifest["base_currency"]
        self.exp = manifest["currency_exponents"]
        self.tz = _offset(policy.get("company_timezone") or manifest["timezone"])
        self.start = _instant(manifest["period_start"])
        self.end = _instant(manifest["period_end"])
        self.rule = policy["fx"]["rate_date_rule"]
        self.rounding = ROUNDING[policy["rounding"]["mode"]]
        self.ledger, self.invoices, self.payments = ledger, invoices, payments
        self.inv_by_id = {r["invoice_id"]: r for r in invoices}
        rates: dict[str, list[tuple[date, Decimal, str]]] = defaultdict(list)
        for row in fx:
            if row["target_currency"] != self.base:
                continue
            fid = row.get("fx_id") or f"FX:{row['source_currency']}:{row['target_currency']}:{row['date']}"
            rates[row["source_currency"]].append((date.fromisoformat(row["date"]), Decimal(row["rate"]), fid))
        self.rates = {cur: sorted(vals) for cur, vals in rates.items()}
        self.inv_rows: dict[str, list[dict]] = defaultdict(list)
        self.pay_rows: dict[str, list[dict]] = defaultdict(list)
        for row in ledger:
            if row.get("payment_id"):
                self.pay_rows[row["payment_id"]].append(row)
            elif row.get("invoice_id"):
                self.inv_rows[row["invoice_id"]].append(row)

    def in_period(self, ts: str) -> bool:
        return self.start <= _instant(ts) < self.end

    def rate_for(self, currency: str, ts: str) -> tuple[str, Decimal] | None:
        """(fx_id, курс) по правилу политики; нет курса на дату -> последняя более ранняя дата."""
        day = _instant(ts).astimezone(self.tz).date()
        if self.rule == "previous_day":
            day -= timedelta(days=1)
        series = self.rates.get(currency, [])
        pos = bisect.bisect_right([d for d, _, _ in series], day)
        if pos == 0:
            return None
        _, rate, fid = series[pos - 1]
        return fid, rate

    def convert(self, amount: int, currency: str, rate: Decimal) -> int:
        if currency == self.base:
            return amount
        with localcontext() as ctx:
            ctx.prec = 60
            value = Decimal(amount) * rate * (Decimal(10) ** (self.exp[self.base] - self.exp[currency]))
            return int(value.quantize(Decimal(1), rounding=self.rounding))


def _finding(kind: str, affected: list[str], amount: int, currency: str, evidence: list[str], why: str) -> dict:
    return {"type": kind, "affected_ids": sorted(affected), "claimed_amount_minor": amount, "currency": currency,
            "evidence_ids": evidence, "explanation": why}


def _line_sig(li: dict) -> tuple:
    return (li.get("order_ref"), li.get("sku"), li.get("description"), li.get("quantity"),
            li.get("unit_price_minor"), li.get("amount_minor"))


def derive_findings(manifest: dict, policy: dict, ledger: list[dict], invoices: list[dict],
                    payments: list[dict], fx: list[dict]) -> list[dict]:
    b = Books(manifest, policy, ledger, invoices, payments, fx)
    out: list[dict] = []

    # ------------------------------------------------------------ счета: cut-off и курс
    for inv in invoices:
        iid, cur = inv["invoice_id"], inv["currency"]
        rows = sorted(b.inv_rows.get(iid, []), key=lambda r: r["row_id"])
        if not rows:
            continue
        ar = [r for r in rows if r["account"] == AR and r["debit_minor"] > 0]
        if not b.in_period(inv["issued_at"]):
            out.append(_finding("cutoff_error", [iid], abs(inv["total_minor"]), cur, [inv["doc_id"], (ar or rows)[0]["row_id"]],
                                "Invoice date in company timezone is outside the period, but it is posted here."))
        if cur != b.base and ar:
            found = b.rate_for(cur, inv["issued_at"])
            if found:
                fid, rate = found
                diff = abs(ar[0]["debit_minor"] - b.convert(inv["total_minor"], cur, rate))
                if diff:
                    out.append(_finding("fx_rate_error", [iid], diff, b.base, [inv["doc_id"], ar[0]["row_id"], fid],
                                        "AR/revenue amount differs from conversion at the policy rate date."))

    # ------------------------------------------------------------ платежи: cut-off, курс, разнесение
    for pay in payments:
        pid, cur, amount = pay["payment_id"], pay["currency"], pay["amount_minor"]
        rows = sorted(b.pay_rows.get(pid, []), key=lambda r: r["row_id"])
        if not rows:
            continue
        if not b.in_period(pay["timestamp"]):
            out.append(_finding("cutoff_error", [pid], abs(amount), cur, [pay["doc_id"], rows[0]["row_id"]],
                                "Payment date in company timezone is outside the period, but it is posted here."))
        if amount < 0:
            continue  # возврат: Дт 2300 / Кт 1000 в базовой валюте
        bank = [r for r in rows if r["account"] == BANK and r["debit_minor"] > 0]
        if cur != b.base and bank:
            found = b.rate_for(cur, pay["timestamp"])
            if found:
                fid, rate = found
                diff = abs(bank[0]["debit_minor"] - b.convert(amount, cur, rate))
                if diff:
                    out.append(_finding("fx_rate_error", [pid], diff, b.base, [pay["doc_id"], bank[0]["row_id"], fid],
                                        "Bank amount differs from conversion at the policy rate date."))
        refs = pay.get("invoice_refs")
        if not refs or sum(r["amount_minor"] for r in refs) != amount:
            continue  # без ссылок или с законным остатком — 2300 корректен
        unapplied = [r for r in rows if r["account"] == UNAPPLIED and r["credit_minor"] > 0]
        if not unapplied:
            continue
        credited: dict[str, int] = defaultdict(int)
        for r in rows:
            if r["account"] == AR and r.get("invoice_id"):
                credited[r["invoice_id"]] += r["credit_minor"]
        short = []
        for ref in refs:
            target = b.inv_by_id.get(ref["invoice_id"])
            if target is None:
                continue
            found = ("", Decimal(1)) if cur == b.base else b.rate_for(cur, target["issued_at"])
            if found is None:
                continue
            if credited.get(ref["invoice_id"], 0) < b.convert(ref["amount_minor"], cur, found[1]):
                short.append(ref["invoice_id"])
        if len(short) == 1:
            out.append(_finding("unapplied_payment", [pid, short[0]], sum(r["credit_minor"] for r in unapplied), b.base,
                                [pay["doc_id"], unapplied[0]["row_id"]],
                                "Payment references the invoice in full, but part of it stayed on 2300."))

    # ------------------------------------------------------------ дубли счетов
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for inv in invoices:
        key = (inv["entity_id"], inv["currency"], inv["total_minor"], _instant(inv["issued_at"]),
               tuple(sorted(_line_sig(li) for li in inv["line_items"])))
        groups[key].append(inv)
    for members in groups.values():
        posted = [m for m in members if b.inv_rows.get(m["invoice_id"])]
        if len(members) == 2 and len(posted) == 2:
            a, c = members
            out.append(_finding("duplicate_invoice", [a["invoice_id"], c["invoice_id"]], a["total_minor"], a["currency"],
                                sorted([a["doc_id"], c["doc_id"]]),
                                "Same customer, instant and order lines; both copies are posted."))
    out.sort(key=lambda f: (f["type"], f["affected_ids"]))
    return out[:30]


class AuditorPolicy(ScriptedPolicy):
    name = "auditor"

    def script(self, obs0: dict):
        tables = {}
        try:
            for table in ("invoices", "payments", "fx", "ledger"):
                tables[table] = yield from self.read_table(table)
        except TableReadError:
            yield submit_action(self.episode_id, [])
            return
        findings = derive_findings(obs0["manifest"], obs0["policy"], tables["ledger"], tables["invoices"],
                                   tables["payments"], tables["fx"])
        yield submit_action(self.episode_id, findings)


def derive_from_public_texts(texts: dict[str, Any]) -> list[dict]:
    """То же по шести открытым файлам (для тестов-валидаторов без среды)."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from env.records import load_public

    ep = load_public(texts)
    t = ep.tables
    return derive_findings(ep.manifest, ep.policy, t["ledger"], t["invoices"], t["payments"], t["fx"])
