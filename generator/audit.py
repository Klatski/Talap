"""Самопроверка эталона: заново выводит все расхождения ТОЛЬКО по открытым файлам.

Модуль не импортирует history/inject/render: он знает лишь опубликованный контракт
(manifest, policy, форматы файлов). Используется
  * генератором — эпизод не выпускается, если audit(public) != truth;
  * тестами — на сотнях seed и конфигов;
  * CLI: python -m generator audit <episode_dir>.

Кроме ошибок четырёх типов audit собирает «аномалии» — всё, что не объясняется ни
корректным учётом, ни объявленной ошибкой (несбалансированные строки, висячие ссылки,
непроведённые документы периода и т.п.). У корректного эпизода аномалий нет.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from .fileio import fx_record_id, parse_fx, parse_jsonl, parse_ledger
from .money import convert_minor
from .timeutil import local_date, parse_iso, parse_offset

BANK, AR, UNAPPLIED, REVENUE, FX_GAIN_LOSS = "1000", "1200", "2300", "4000", "7900"


@dataclass
class Public:
    manifest: dict
    policy: dict
    ledger: list[dict]
    invoices: list[dict]
    payments: list[dict]
    fx_rows: list[dict]

    def __post_init__(self) -> None:
        m = self.manifest
        self.base = m["base_currency"]
        self.exponents = m["currency_exponents"]
        self.offset = parse_offset(m["timezone"])
        self.start = parse_iso(m["period_start"])
        self.end = parse_iso(m["period_end"])
        self.rule = self.policy["fx"]["rate_date_rule"]
        self.mode = self.policy["rounding"]["mode"]
        self.fx: dict[tuple[str, date], Decimal] = {}
        self.fx_ids: dict[str, dict] = {}
        for row in self.fx_rows:
            d = date.fromisoformat(row["date"])
            self.fx[(row["source_currency"], d)] = Decimal(row["rate"])
            self.fx_ids[fx_record_id(row["source_currency"], row["target_currency"], row["date"])] = row
        self.inv_by_id = {r["invoice_id"]: r for r in self.invoices}
        self.pay_by_id = {r["payment_id"]: r for r in self.payments}
        self.doc_by_id = {r["doc_id"]: r for r in [*self.invoices, *self.payments]}
        self.row_by_id = {r["row_id"]: r for r in self.ledger}

    def in_period(self, ts: datetime) -> bool:
        return self.start <= ts < self.end

    def rate(self, currency: str, instant: datetime) -> tuple[date | None, Decimal | None]:
        if currency == self.base:
            return None, Decimal(1)
        d = local_date(instant, self.offset)
        if self.rule == "previous_day":
            d -= timedelta(days=1)
        for back in range(15):
            cand = d - timedelta(days=back)
            if (currency, cand) in self.fx:
                return cand, self.fx[(currency, cand)]
        return None, None

    def conv(self, amount: int, currency: str, rate: Decimal) -> int:
        return convert_minor(amount, currency, self.base, rate, self.mode, self.exponents)


def parse_public(texts: dict[str, str]) -> Public:
    return Public(
        manifest=json.loads(texts["manifest.json"]),
        policy=json.loads(texts["policy.json"]),
        ledger=parse_ledger(texts["ledger.csv"]),
        invoices=parse_jsonl(texts["invoices.jsonl"]),
        payments=parse_jsonl(texts["payments.jsonl"]),
        fx_rows=parse_fx(texts["fx.csv"]),
    )


@dataclass
class AuditResult:
    issues: list[dict] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)

    def issue_keys(self) -> list[tuple]:
        return sorted(
            (i["type"], tuple(i["affected_ids"]), i["amount_minor"], i["currency"]) for i in self.issues
        )


def _issue(kind: str, affected: list[str], amount: int, currency: str) -> dict:
    return {"type": kind, "affected_ids": sorted(affected), "amount_minor": amount, "currency": currency}


def _line_signature(line: dict) -> tuple:
    return (
        line["order_ref"],
        line["sku"],
        line["description"],
        line["quantity"],
        line["unit_price_minor"],
        line["amount_minor"],
    )


def audit(pub: Public) -> AuditResult:
    res = AuditResult()
    bad = res.anomalies.append
    accounts = set(pub.policy["accounts"])

    # ---------------------------------------------------------------- книга целиком
    total_debit = total_credit = 0
    inv_rows: dict[str, list[dict]] = defaultdict(list)
    pay_rows: dict[str, list[dict]] = defaultdict(list)
    for row in pub.ledger:
        rid = row["row_id"]
        if row["currency"] != pub.base:
            bad(f"{rid}: ledger currency {row['currency']} != base")
        d, c = row["debit_minor"], row["credit_minor"]
        if d < 0 or c < 0 or (d > 0) == (c > 0):
            bad(f"{rid}: row must have exactly one positive side")
        if row["account"] not in accounts:
            bad(f"{rid}: unknown account {row['account']}")
        total_debit += d
        total_credit += c
        if row["invoice_id"] is not None and row["invoice_id"] not in pub.inv_by_id:
            bad(f"{rid}: dangling invoice_id")
        if row["payment_id"] is not None and row["payment_id"] not in pub.pay_by_id:
            bad(f"{rid}: dangling payment_id")
        if row["payment_id"] is not None:
            pay_rows[row["payment_id"]].append(row)
        elif row["invoice_id"] is not None:
            inv_rows[row["invoice_id"]].append(row)
        else:
            if not pub.in_period(parse_iso(row["timestamp"])):
                bad(f"{rid}: non-document posting outside period")
    if total_debit != total_credit:
        bad(f"ledger not balanced: {total_debit} != {total_credit}")

    # ---------------------------------------------------------------- счета
    posted_invoices: set[str] = set()
    for inv in pub.invoices:
        iid = inv["invoice_id"]
        cur = inv["currency"]
        lines = inv["line_items"]
        if any(li["amount_minor"] != li["quantity"] * li["unit_price_minor"] for li in lines):
            bad(f"{iid}: line amount != quantity * unit_price")
        if inv["total_minor"] != sum(li["amount_minor"] for li in lines):
            bad(f"{iid}: total != sum(line_items)")
        if cur not in pub.exponents:
            bad(f"{iid}: currency without exponent")
        ts = parse_iso(inv["issued_at"])
        rows = inv_rows.get(iid, [])
        if not rows:
            if pub.in_period(ts):
                bad(f"{iid}: in-period invoice is not posted")
            continue
        posted_invoices.add(iid)
        ar = [r for r in rows if r["account"] == AR]
        rev = [r for r in rows if r["account"] == REVENUE]
        if len(rows) != 2 or len(ar) != 1 or len(rev) != 1 or ar[0]["debit_minor"] == 0 or rev[0]["credit_minor"] == 0:
            bad(f"{iid}: unexpected posting structure")
            continue
        posted = ar[0]["debit_minor"]
        if rev[0]["credit_minor"] != posted:
            bad(f"{iid}: AR != revenue")
        if not pub.in_period(ts):
            res.issues.append(_issue("cutoff_error", [iid], abs(inv["total_minor"]), cur))
        elif any(not pub.in_period(parse_iso(r["timestamp"])) for r in rows):
            bad(f"{iid}: in-period invoice posted outside period")
        _, rate = pub.rate(cur, ts)
        if rate is None:
            bad(f"{iid}: no fx rate")
            continue
        expected = pub.conv(inv["total_minor"], cur, rate)
        if posted != expected:
            if cur == pub.base:
                bad(f"{iid}: base-currency invoice posted with different amount")
            else:
                res.issues.append(_issue("fx_rate_error", [iid], abs(posted - expected), pub.base))

    # ---------------------------------------------------------------- платежи
    allocated_total: dict[str, int] = defaultdict(int)
    for pay in pub.payments:
        pid = pay["payment_id"]
        cur = pay["currency"]
        amount = pay["amount_minor"]
        refs = pay["invoice_refs"]
        ts = parse_iso(pay["timestamp"])
        for ref in refs or []:
            target = pub.inv_by_id.get(ref["invoice_id"])
            if target is None:
                bad(f"{pid}: invoice_refs points to unknown invoice")
                continue
            if target["currency"] != cur:
                bad(f"{pid}: allocation currency mismatch")
            if ref["amount_minor"] <= 0:
                bad(f"{pid}: non-positive allocation")
            if parse_iso(target["issued_at"]) >= ts:
                bad(f"{pid}: pays an invoice issued later")
            allocated_total[ref["invoice_id"]] += ref["amount_minor"]
        if refs is not None and sum(r["amount_minor"] for r in refs) > amount:
            bad(f"{pid}: allocations exceed payment amount")
        rows = pay_rows.get(pid, [])
        if not rows:
            if pub.in_period(ts):
                bad(f"{pid}: in-period payment is not posted")
            continue
        if not pub.in_period(ts):
            res.issues.append(_issue("cutoff_error", [pid], abs(amount), cur))
        elif any(not pub.in_period(parse_iso(r["timestamp"])) for r in rows):
            bad(f"{pid}: in-period payment posted outside period")
        if sum(r["debit_minor"] for r in rows) != sum(r["credit_minor"] for r in rows):
            bad(f"{pid}: payment entry not balanced")

        if amount < 0:
            ok = (
                cur == pub.base
                and refs is None
                and len(rows) == 2
                and any(r["account"] == UNAPPLIED and r["debit_minor"] == -amount for r in rows)
                and any(r["account"] == BANK and r["credit_minor"] == -amount for r in rows)
            )
            if not ok:
                bad(f"{pid}: unexpected refund posting")
            continue

        bank = [r for r in rows if r["account"] == BANK]
        if len(bank) != 1 or bank[0]["debit_minor"] <= 0:
            bad(f"{pid}: expected exactly one bank debit")
            continue
        _, rate_p = pub.rate(cur, ts)
        if rate_p is None:
            bad(f"{pid}: no fx rate")
            continue
        bank_value = bank[0]["debit_minor"]
        expected_bank = pub.conv(amount, cur, rate_p)
        fx_flag = False
        if bank_value != expected_bank:
            if cur == pub.base:
                bad(f"{pid}: base-currency payment posted with different amount")
            else:
                res.issues.append(_issue("fx_rate_error", [pid], abs(bank_value - expected_bank), pub.base))
                fx_flag = True

        ar_credit: dict[str, int] = defaultdict(int)
        unapplied = 0
        fx_net = 0
        for r in rows:
            acc = r["account"]
            if acc == BANK:
                continue
            if acc == AR:
                if r["debit_minor"] or r["invoice_id"] is None:
                    bad(f"{pid}: unexpected AR row")
                ar_credit[r["invoice_id"]] += r["credit_minor"]
            elif acc == UNAPPLIED:
                if r["debit_minor"]:
                    bad(f"{pid}: unexpected 2300 debit")
                unapplied += r["credit_minor"]
            elif acc == FX_GAIN_LOSS:
                if cur == pub.base:
                    bad(f"{pid}: FX line on base-currency payment")
                fx_net += r["credit_minor"] - r["debit_minor"]
            else:
                bad(f"{pid}: unexpected account {acc}")
        ref_ids = {r["invoice_id"] for r in refs or []}
        if set(ar_credit) - ref_ids:
            bad(f"{pid}: AR credit for invoice not in invoice_refs")
        shortfalls = []
        for ref in refs or []:
            target = pub.inv_by_id.get(ref["invoice_id"])
            if target is None:
                continue
            _, rate_inv = pub.rate(cur, parse_iso(target["issued_at"]))
            expected_credit = pub.conv(ref["amount_minor"], cur, rate_inv)
            got = ar_credit.get(ref["invoice_id"], 0)
            if got < expected_credit:
                shortfalls.append(ref["invoice_id"])
            elif got > expected_credit:
                bad(f"{pid}: AR credit above allocation")
        remainder = amount - sum(r["amount_minor"] for r in refs or [])
        if shortfalls:
            if len(shortfalls) > 1 or remainder != 0 or unapplied <= 0:
                bad(f"{pid}: ambiguous unapplied amount")
            else:
                res.issues.append(_issue("unapplied_payment", [pid, shortfalls[0]], unapplied, pub.base))
        elif remainder == 0 and unapplied != 0:
            bad(f"{pid}: 2300 credit without reason")
        elif remainder > 0 and not fx_flag and unapplied != pub.conv(remainder, cur, rate_p):
            bad(f"{pid}: 2300 credit != unallocated remainder")
        if bank_value - sum(ar_credit.values()) - unapplied != fx_net:
            bad(f"{pid}: FX gain/loss line does not close the entry")

    for iid, value in allocated_total.items():
        if iid in pub.inv_by_id and value > pub.inv_by_id[iid]["total_minor"]:
            bad(f"{iid}: over-allocated")

    # ---------------------------------------------------------------- дубли
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for inv in pub.invoices:
        key = (
            inv["entity_id"],
            inv["currency"],
            inv["total_minor"],
            parse_iso(inv["issued_at"]),
            tuple(sorted(_line_signature(li) for li in inv["line_items"])),
        )
        groups[key].append(inv)
    dup_members: set[str] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        ids = [m["invoice_id"] for m in members]
        dup_members.update(ids)
        if len(members) == 2 and all(i in posted_invoices for i in ids):
            res.issues.append(_issue("duplicate_invoice", ids, members[0]["total_minor"], members[0]["currency"]))
        else:
            bad(f"{ids}: duplicate group that is not a posted pair")
    seen_lines: dict[tuple[str, str], str] = {}
    for inv in pub.invoices:
        if inv["invoice_id"] in dup_members:
            continue
        for li in inv["line_items"]:
            k = (li["order_ref"], li["sku"])
            if k in seen_lines and seen_lines[k] != inv["invoice_id"]:
                bad(f"{inv['invoice_id']}: order line billed twice without being a duplicate")
            seen_lines[k] = inv["invoice_id"]
    return res


# -------------------------------------------------------------------- доказательства
def check_evidence_set(pub: Public, issue: dict, evidence: list[str]) -> list[str]:
    """Достаточен ли ОДИН набор доказательств сам по себе (плюс manifest и policy).

    Проверка использует только записи из набора: так проверяется «независимая
    достаточность» — агент, видевший эти записи, может обосновать вывод и сумму.
    """
    problems = []
    for rid in evidence:
        if rid not in pub.row_by_id and rid not in pub.doc_by_id and rid not in pub.fx_ids:
            problems.append(f"unknown evidence id {rid}")
    if problems:
        return problems
    rows = [pub.row_by_id[r] for r in evidence if r in pub.row_by_id]
    docs = [pub.doc_by_id[r] for r in evidence if r in pub.doc_by_id]
    fxs = [pub.fx_ids[r] for r in evidence if r in pub.fx_ids]
    kind = issue["type"]
    expected = issue["expected_amount_minor"]
    affected = issue["affected_ids"]

    if kind == "duplicate_invoice":
        invs = [d for d in docs if d.get("invoice_id") in affected]
        if len(invs) != 2:
            return ["duplicate: both invoice documents required"]
        a, b = invs
        same = (
            a["entity_id"] == b["entity_id"]
            and a["currency"] == b["currency"]
            and a["total_minor"] == b["total_minor"]
            and parse_iso(a["issued_at"]) == parse_iso(b["issued_at"])
            and sorted(map(_line_signature, a["line_items"])) == sorted(map(_line_signature, b["line_items"]))
        )
        if not same:
            problems.append("duplicate: documents differ")
        if a["total_minor"] != expected or a["currency"] != issue["currency"]:
            problems.append("duplicate: amount/currency mismatch")
        return problems

    if kind == "cutoff_error":
        target = [d for d in docs if (d.get("invoice_id") or d.get("payment_id")) in affected]
        if len(target) != 1:
            return ["cutoff: document required"]
        doc = target[0]
        doc_key = doc.get("invoice_id") or doc.get("payment_id")
        ts = parse_iso(doc.get("issued_at") or doc.get("timestamp"))
        if pub.in_period(ts):
            problems.append("cutoff: document is inside the period")
        if not any(doc_key in (r["invoice_id"], r["payment_id"]) for r in rows):
            problems.append("cutoff: ledger row of the document required")
        amount = abs(doc.get("total_minor", doc.get("amount_minor")))
        if amount != expected or doc["currency"] != issue["currency"]:
            problems.append("cutoff: amount/currency mismatch")
        return problems

    if kind == "unapplied_payment":
        pays = [d for d in docs if d.get("payment_id") in affected]
        if len(pays) != 1:
            return ["unapplied: payment document required"]
        pay = pays[0]
        pid, cur = pay["payment_id"], pay["currency"]
        refs = pay["invoice_refs"] or []
        if not refs:
            return ["unapplied: payment references no invoice"]
        if sum(r["amount_minor"] for r in refs) != pay["amount_minor"]:
            problems.append("unapplied: payment has a legitimate unallocated remainder")
        un = [r for r in rows if r["account"] == UNAPPLIED and r["payment_id"] == pid]
        if len(un) != 1 or un[0]["credit_minor"] != expected:
            problems.append("unapplied: 2300 row with the amount required")
        if len(refs) == 1:
            short = [refs[0]["invoice_id"]]
        else:
            # Счетов несколько: недоразнесённый счёт выводится только из записей набора — Кт 1200 этого
            # платежа по каждому счёту против суммы ссылки по курсу счёта (дата счёта по политике).
            short = []
            for ref in refs:
                inv = next((d for d in docs if d.get("invoice_id") == ref["invoice_id"]), None)
                if inv is None:
                    return ["unapplied: document of every referenced invoice required"]
                rate_date, rate = pub.rate(cur, parse_iso(inv["issued_at"]))
                if cur != pub.base:
                    fx_row = next((f for f in fxs if f["source_currency"] == cur and rate_date is not None
                                   and f["date"] == rate_date.isoformat()), None)
                    if fx_row is None:
                        return ["unapplied: policy rate of every referenced invoice required"]
                    rate = Decimal(fx_row["rate"])
                expected_credit = pub.conv(ref["amount_minor"], cur, rate)
                got = sum(r["credit_minor"] for r in rows
                          if r["account"] == AR and r["payment_id"] == pid and r["invoice_id"] == ref["invoice_id"])
                if got < expected_credit:
                    short.append(ref["invoice_id"])
                elif got > expected_credit:
                    problems.append("unapplied: AR credit above allocation")
        if len(short) != 1 or sorted(affected) != sorted([pid, short[0]]):
            problems.append("unapplied: the evidence does not determine the under-allocated invoice")
        return problems

    if kind == "fx_rate_error":
        target = [d for d in docs if (d.get("invoice_id") or d.get("payment_id")) in affected]
        if len(target) != 1 or len(fxs) < 1:
            return ["fx: document and fx record required"]
        doc = target[0]
        doc_key = doc.get("invoice_id") or doc.get("payment_id")
        ts = parse_iso(doc.get("issued_at") or doc.get("timestamp"))
        want_date, _ = pub.rate(doc["currency"], ts)
        fx_rows = [f for f in fxs if f["source_currency"] == doc["currency"] and f["date"] == (want_date and want_date.isoformat())]
        if not fx_rows:
            return ["fx: rate record for the policy date required"]
        rate = Decimal(fx_rows[0]["rate"])
        amount = doc.get("total_minor", doc.get("amount_minor"))
        correct = pub.conv(amount, doc["currency"], rate)
        accounts = (AR, REVENUE) if "invoice_id" in doc else (BANK,)
        cand = [r for r in rows if r["account"] in accounts and doc_key in (r["invoice_id"], r["payment_id"])]
        if not cand:
            return ["fx: converted ledger row required"]
        posted = cand[0]["debit_minor"] + cand[0]["credit_minor"]
        if abs(posted - correct) != expected:
            problems.append("fx: amount cannot be reproduced from evidence")
        return problems

    return [f"unknown issue type {kind}"]


def compare_with_truth(pub: Public, truth: dict) -> list[str]:
    """Пустой список = эталон совпадает с выведенным по открытым данным, доказательства достаточны."""
    problems = []
    result = audit(pub)
    problems += [f"anomaly: {a}" for a in result.anomalies]
    truth_keys = sorted(
        (i["type"], tuple(i["affected_ids"]), i["expected_amount_minor"], i["currency"]) for i in truth["issues"]
    )
    derived = result.issue_keys()
    if truth_keys != derived:
        missing = [k for k in truth_keys if k not in derived]
        extra = [k for k in derived if k not in truth_keys]
        problems.append(f"truth mismatch: not derivable={missing} undeclared={extra}")
    for issue in truth["issues"]:
        if not issue["evidence_sets"]:
            problems.append(f"{issue['issue_id']}: no evidence sets")
        for ev in issue["evidence_sets"]:
            for p in check_evidence_set(pub, issue, ev):
                problems.append(f"{issue['issue_id']}: {p}")
    if truth["episode_id"] != pub.manifest["episode_id"]:
        problems.append("episode_id mismatch")
    return problems


def summarize(result: AuditResult) -> dict[str, Any]:
    by_type: dict[str, int] = defaultdict(int)
    for i in result.issues:
        by_type[i["type"]] += 1
    return {"issues": dict(sorted(by_type.items())), "anomalies": len(result.anomalies)}
