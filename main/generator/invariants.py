"""Бухгалтерские инварианты скрытой истории (до и после внесения ошибок).

До внесения ошибок история должна быть безупречной: любой провал здесь — баг
генератора (InvariantError), а не «сложность». После внесения ошибок проверяется,
что книга по-прежнему сбалансирована, а единственные расхождения — объявленные
(это делает audit по открытым файлам).
"""

from __future__ import annotations

from collections import defaultdict

from .history import BANK, History, Invoice


class InvariantError(AssertionError):
    pass


def _entries_balanced(h: History, problems: list[str]) -> int:
    n = 0
    for e in h.all_entries():
        n += 1
        debit = credit = 0
        if len(e.lines) < 2:
            problems.append(f"{e.key}: entry with <2 lines")
        for ln in e.lines:
            if not isinstance(ln.debit, int) or not isinstance(ln.credit, int):
                problems.append(f"{e.key}: non-int amount")
            if ln.debit < 0 or ln.credit < 0 or (ln.debit > 0) == (ln.credit > 0):
                problems.append(f"{e.key}: line must have exactly one positive side")
            debit += ln.debit
            credit += ln.credit
        if debit != credit:
            problems.append(f"{e.key}: unbalanced {debit} != {credit}")
    return n


def check_clean(h: History) -> dict:
    """Инварианты чистой истории. Вызывать после build_doc_entries() и ДО inject_all()."""
    problems: list[str] = []
    order_lines: dict[tuple[str, str], str] = {}
    for inv in h.invoices.values():
        if not inv.lines or inv.total_minor <= 0:
            problems.append(f"{inv.key}: empty or non-positive invoice")
        if h.customers[inv.customer] != inv.currency:
            problems.append(f"{inv.key}: currency differs from customer currency")
        for ln in inv.lines:
            if ln.quantity <= 0 or ln.unit_price_minor <= 0:
                problems.append(f"{inv.key}: bad line")
            k = (ln.order, ln.sku)
            if k in order_lines:
                problems.append(f"{inv.key}: order line {k} billed twice")
            order_lines[k] = inv.key
        if inv.due_at <= inv.issued_at:
            problems.append(f"{inv.key}: due_at not after issued_at")

    allocated: dict[str, int] = defaultdict(int)
    for pay in h.payments.values():
        if pay.amount_minor == 0:
            problems.append(f"{pay.key}: zero payment")
        if pay.amount_minor < 0:
            if pay.allocations is not None or pay.currency != h.base:
                problems.append(f"{pay.key}: refund must be unreferenced and in base currency")
            continue
        if pay.allocations is not None:
            if not pay.allocations:
                problems.append(f"{pay.key}: empty allocations list")
            if sum(a.amount_minor for a in pay.allocations) > pay.amount_minor:
                problems.append(f"{pay.key}: allocations exceed amount")
            for a in pay.allocations:
                inv = h.invoices[a.invoice]
                if a.amount_minor <= 0:
                    problems.append(f"{pay.key}: non-positive allocation")
                if inv.currency != pay.currency or inv.customer != pay.customer:
                    problems.append(f"{pay.key}: allocation to foreign invoice")
                if inv.issued_at >= pay.ts:
                    problems.append(f"{pay.key}: pays a later invoice")
                allocated[a.invoice] += a.amount_minor
    for key, value in allocated.items():
        if value > h.invoices[key].total_minor:
            problems.append(f"{key}: over-allocated")

    docs = [*h.invoices.values(), *h.payments.values()]
    posted = 0
    for doc in docs:
        in_period = h.period_of(doc.ts) == "cur"
        spec = doc.posting
        if spec.posted != in_period:
            problems.append(f"{doc.key}: posted={spec.posted} but in_period={in_period}")
        if spec.posted:
            posted += 1
            if not h.start <= spec.ts < h.end or spec.ts < doc.ts:
                problems.append(f"{doc.key}: posting timestamp outside period or before document")
        if spec.rate_override or spec.shortfall:
            problems.append(f"{doc.key}: clean history has an error modifier")
        h.policy_rate(doc.currency, doc.ts)  # курс обязан существовать

    n_entries = _entries_balanced(h, problems)
    by_doc = {e.doc: e for e in h.entries}
    for doc in docs:
        if not doc.posting.posted:
            continue
        entry = by_doc[doc.key]
        if isinstance(doc, Invoice):
            expected = h.conv(doc.total_minor, doc.currency, h.policy_rate(doc.currency, doc.ts)[1])
            if any(ln.debit + ln.credit != expected for ln in entry.lines):
                problems.append(f"{doc.key}: invoice posting != converted total")
        elif doc.amount_minor > 0:
            expected = h.conv(doc.amount_minor, doc.currency, h.policy_rate(doc.currency, doc.ts)[1])
            bank = [ln for ln in entry.lines if ln.account == BANK]
            if len(bank) != 1 or bank[0].debit != expected:
                problems.append(f"{doc.key}: bank line != converted amount")
    if problems:
        raise InvariantError("; ".join(problems[:20]))
    return {
        "invoices": len(h.invoices),
        "payments": len(h.payments),
        "posted_documents": posted,
        "entries_balanced": n_entries,
        "order_lines_unique": len(order_lines),
        "allocations_within_totals": len(allocated),
        "conversions_match_policy": posted,
        "status": "passed",
    }


def check_after(h: History) -> dict:
    problems: list[str] = []
    n_entries = _entries_balanced(h, problems)
    debit = sum(ln.debit for e in h.all_entries() for ln in e.lines)
    credit = sum(ln.credit for e in h.all_entries() for ln in e.lines)
    if debit != credit:
        problems.append("ledger not balanced")
    for doc in [*h.invoices.values(), *h.payments.values()]:
        if doc.posting.posted and doc.posting.ts is None:
            problems.append(f"{doc.key}: posted without timestamp")
    if problems:
        raise InvariantError("; ".join(problems[:20]))
    rows = sum(len(e.lines) for e in h.all_entries())
    return {
        "entries_balanced": n_entries,
        "ledger_rows": rows,
        "ledger_debit_equals_credit": True,
        "status": "passed",
    }
