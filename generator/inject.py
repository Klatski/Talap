"""Внесение ошибок в историю как записанное причинное изменение.

Каждая ошибка меняет только PostingSpec документа (как он проведён) или добавляет
документ-дубль. Документы-первоисточники (счета, платежи, курсы) не портятся:
по ним всегда можно восстановить верный учёт — это и есть доказательства.

Сумма каждой ошибки вычисляется ПОСЛЕ перестроения проводок (finalize_issues) из
фактических строк книги — один источник правды для эталона.

Проводка ошибки всегда датирована внутри периода (invariants.check_after): документ соседнего
периода «попадает в книгу» текущего, а не в книгу с чужими датами. Поэтому cut-off не виден по одной
книге — нужен сам документ (его дата) — и «проводка раньше документа» встречается и у корректных
документов (history.early_postings).

Вариант ошибки можно заказать (config.variants, Params.variant_for); без заказа он случайный.
Новая случайность (время проводки «entered_in_period») берётся из отдельного потока aux, поэтому
эпизоды без этих вариантов совпадают с прежними побайтово.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from .history import MIN_FX_DIFF, GenerationShortfall, History, Invoice, Payment, PostingSpec
from .money import format_rate
from .timeutil import local_date


@dataclass
class IssueSpec:
    type: str
    affected: list[str]  # внутренние ключи документов
    currency: str
    amount_rule: tuple
    evidence: list[list[tuple]]  # шаблоны наборов доказательств
    variant: str
    cause: str
    scenario: str | None = None
    group: int | None = None
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    amount_minor: int | None = None


def _doc_amount(doc: Invoice | Payment) -> int:
    return doc.total_minor if isinstance(doc, Invoice) else doc.amount_minor


class Injector:
    def __init__(self, h: History, rng: random.Random, aux: random.Random | None = None) -> None:
        self.h = h
        self.rng = rng
        self.aux = aux if aux is not None else random.Random("inject-aux")
        self.used: set[str] = set()
        self.issues: list[IssueSpec] = []
        self._group = 0

    def free(self, *keys: str) -> bool:
        return not any(k in self.used for k in keys)

    def take(self, *keys: str) -> None:
        self.used.update(keys)

    def next_group(self) -> int:
        self._group += 1
        return self._group

    # ------------------------------------------------------------------ курс
    def fx_options(self, doc: Invoice | Payment) -> list[tuple[str, date, Decimal]]:
        """Варианты «не того» курса, дающие расхождение >= MIN_FX_DIFF."""
        h = self.h
        cur = doc.currency
        instant = doc.ts
        company_day = local_date(instant, h.offset)
        correct_date, correct_rate = h.policy_rate(cur, instant)
        amount = abs(_doc_amount(doc))
        correct_value = h.conv(amount, cur, correct_rate)
        # Порядок = приоритет при совпадении дат курса у разных вариантов.
        raw: list[tuple[str, date, Decimal]] = []
        doc_day = local_date(instant, doc.offset)
        if doc_day != company_day:
            raw.append(("document_offset_date", *h.rate_for_date(cur, doc_day)))
        utc_day = local_date(instant, 0)
        if utc_day != company_day and utc_day != doc_day:
            raw.append(("utc_date", *h.rate_for_date(cur, utc_day)))
        other_rule = "previous_day" if h.p.rate_rule == "document_date" else "document_date"
        raw.append(("other_rule", *h.rate_for_date(cur, company_day, other_rule)))
        if isinstance(doc, Invoice):
            try:
                raw.append(("due_date", *h.rate_for_date(cur, local_date(doc.due_at, h.offset))))
            except GenerationShortfall:
                pass
        idx = h.fx_dates.index(correct_date)
        for delta in (-3, -2, -1, 1, 2, 3):
            j = idx + delta
            if 0 <= j < len(h.fx_dates):
                d = h.fx_dates[j]
                raw.append(("adjacent_date", d, h.fx[cur][d]))
        options = []
        seen = set()
        for variant, d, rate in raw:
            if d == correct_date or d in seen:
                continue
            if abs(h.conv(amount, cur, rate) - correct_value) < MIN_FX_DIFF:
                continue
            seen.add(d)
            options.append((variant, d, rate))
        return options

    def _apply_fx(self, doc: Invoice | Payment, option: tuple[str, date, Decimal], scenario=None, group=None) -> None:
        h = self.h
        variant, d, rate = option
        correct_date, correct_rate = h.policy_rate(doc.currency, doc.ts)
        doc.posting.rate_override = (d, rate)
        k = doc.key
        fx_ref = ("fx", doc.currency, correct_date)
        if isinstance(doc, Invoice):
            evidence = [[("doc", k), ("row", k, "ar"), fx_ref], [("doc", k), ("row", k, "revenue"), fx_ref]]
            role = "ar"
        else:
            evidence = [[("doc", k), ("row", k, "bank"), fx_ref]]
            role = "bank"
        causes = {
            "adjacent_date": "курс взят за соседнюю дату вместо даты по политике",
            "other_rule": "применено не то правило выбора даты курса (document_date/previous_day)",
            "due_date": "курс взят по сроку оплаты счёта, а не по дате документа",
            "document_offset_date": "дата курса взята в часовом поясе записи документа, а не компании",
            "utc_date": "дата курса взята по UTC, а не по часовому поясу компании",
        }
        self.issues.append(
            IssueSpec(
                type="fx_rate_error",
                affected=[k],
                currency=h.base,
                amount_rule=("fx_diff", k, role),
                evidence=evidence,
                variant=variant,
                cause=causes[variant],
                scenario=scenario,
                group=group,
                before={"rate_date": correct_date.isoformat(), "rate": format_rate(correct_rate)},
                after={"rate_date": d.isoformat(), "rate": format_rate(rate)},
            )
        )

    def fx(self, variant: str | None = None) -> None:
        h = self.h
        cands: list[Invoice | Payment] = [
            inv
            for inv in h.invoices.values()
            if inv.currency != h.base and inv.posting.posted and h.period_of(inv.issued_at) == "cur"
            and "boundary" not in inv.tags and "duplicate" not in inv.tags and self.free(inv.key)
        ]
        cands += [
            pay
            for pay in h.payments.values()
            if pay.currency != h.base and pay.posting.posted and pay.amount_minor > 0
            and h.period_of(pay.ts) == "cur" and "boundary" not in pay.tags and self.free(pay.key)
        ]
        self.rng.shuffle(cands)
        for doc in cands:
            options = self.fx_options(doc)
            if variant is not None:
                options = [o for o in options if o[0] == variant]
            if options:
                self._apply_fx(doc, self.rng.choice(options))
                self.take(doc.key)
                return
        raise GenerationShortfall(f"no candidate for fx_rate_error{f' ({variant})' if variant else ''}")

    # ------------------------------------------------------------------ разнесение
    def _apply_shortfall(self, pay: Payment, scenario=None, group=None, variant: str | None = None) -> bool:
        """variant: None — случайно (45% полный пропуск), "missing_allocation" — счёт не получил ничего,
        "partial_allocation" — счёт получил часть (15–85%), остаток на 2300. False — у этого платежа
        заказанный вариант невозможен (слишком малая сумма), вызывающий берёт другого кандидата."""
        h = self.h
        rng = self.rng
        k = rng.randrange(len(pay.allocations))
        alloc = pay.allocations[k]
        a = alloc.amount_minor
        if variant is None:
            full = a < 50 or rng.random() < 0.45
        elif variant == "missing_allocation":
            full = True
        else:
            if a < 50:
                return False
            rng.random()  # тот же расход случайности, что у случайного выбора: прежние частичные случаи те же
            full = False
        u = a if full else rng.randint(max(1, a * 15 // 100), max(1, a * 85 // 100))
        rate_p = pay.posting.rate_override[1] if pay.posting.rate_override else h.policy_rate(pay.currency, pay.ts)[1]
        inv_rate = h.invoice_rate(h.invoices[alloc.invoice])
        if h.conv(u, pay.currency, rate_p) < 10 or (
            not full and h.conv(a - u, pay.currency, inv_rate) >= h.conv(a, pay.currency, inv_rate)
        ):
            if variant == "partial_allocation":
                return False
            u, full = a, True
            if h.conv(u, pay.currency, rate_p) < 10:
                return False
        pay.posting.shortfall = (k, u)
        self.take(pay.key, alloc.invoice)
        self.issues.append(
            IssueSpec(
                type="unapplied_payment",
                affected=[pay.key, alloc.invoice],
                currency=h.base,
                amount_rule=("posted_line", pay.key, "unapplied"),
                evidence=[[("doc", pay.key), ("row", pay.key, "unapplied")]],
                variant="missing_allocation" if full else "partial_allocation",
                cause=(
                    "платёж не разнесён на указанный в invoice_refs счёт, сумма осталась на 2300"
                    if full
                    else "платёж разнесён на счёт частично, остаток осел на 2300"
                ),
                scenario=scenario,
                group=group,
                before={"allocated_minor": a, "unapplied_minor": 0},
                after={"allocated_minor": a - u, "unapplied_minor": u},
            )
        )
        return True

    def unapplied(self, variant: str | None = None) -> None:
        h = self.h
        cands = [
            pay
            for pay in h.payments.values()
            if h.is_unapplied_candidate(pay)
            and pay.posting.posted
            and self.free(pay.key, *(a.invoice for a in pay.allocations))
        ]
        self.rng.shuffle(cands)
        for pay in cands:
            if self._apply_shortfall(pay, variant=variant):
                return
        raise GenerationShortfall(f"no candidate for unapplied_payment{f' ({variant})' if variant else ''}")

    # ------------------------------------------------------------------ дубль
    def duplicate(self) -> None:
        h = self.h
        rng = self.rng
        cands = [
            inv
            for inv in h.invoices.values()
            if inv.posting.posted and h.period_of(inv.issued_at) == "cur"
            and "boundary" not in inv.tags and "duplicate" not in inv.tags and self.free(inv.key)
        ]
        if not cands:
            raise GenerationShortfall("no candidate for duplicate_invoice")
        inv = rng.choice(cands)
        dup = Invoice(
            key=h.new_key("inv"),
            customer=inv.customer,
            currency=inv.currency,
            issued_at=inv.issued_at,
            offset=h.doc_offset(rng),
            due_at=inv.due_at,
            lines=copy.deepcopy(inv.lines),
            tags={"duplicate"},
        )
        latest = h.end - timedelta(minutes=1)
        ts = min(inv.posting.ts + timedelta(minutes=rng.randint(0, 72 * 60)), latest)
        ts = max(ts, inv.posting.ts)
        dup.posting = PostingSpec(posted=True, ts=ts, offset=h.ledger_offset(rng))
        h.invoices[dup.key] = dup
        self.take(inv.key, dup.key)
        self.issues.append(
            IssueSpec(
                type="duplicate_invoice",
                affected=sorted([inv.key, dup.key]),
                currency=inv.currency,
                amount_rule=("doc_total", inv.key),
                evidence=[[("doc", inv.key), ("doc", dup.key)]],
                variant="reentered_document",
                cause="счёт повторно введён (те же клиент, время, заказ и строки) и проведён второй раз",
                before={"documents_for_order_lines": 1},
                after={"documents_for_order_lines": 2, "original": inv.key, "duplicate": dup.key},
            )
        )

    # ------------------------------------------------------------------ период
    def _post_out_of_period(self, doc: Invoice | Payment, variant: str, offset: int | None = None) -> None:
        """Провести документ соседнего периода в текущем. Момент проводки всегда внутри периода:
        shifted_into_period — у самой границы (до 6 ч до конца / 12 ч после начала);
        entered_in_period — в любой момент последних 7 дней (документ следующего периода, «задним
        числом») или первых 10 дней (документ прошлого периода, введён с опозданием);
        document_offset_misread — цифры времени документа в ЕГО смещении прочитаны как время компании."""
        h = self.h
        rng = self.rng
        side = h.period_of(doc.ts)
        if variant == "entered_in_period":
            if side == "next":
                lo, hi = max(h.start, h.end - timedelta(days=7)), h.end - timedelta(minutes=1)
            else:
                lo, hi = h.start + timedelta(minutes=1), min(h.end - timedelta(minutes=1), h.start + timedelta(days=10))
            ts = h.random_instant(self.aux, lo, hi)
        elif variant == "document_offset_misread":
            ts = doc.ts + timedelta(minutes=doc.offset - h.offset)
        elif side == "next":
            ts = h.end - timedelta(minutes=rng.randint(1, 360), seconds=rng.randint(0, 59))
        else:
            ts = h.start + timedelta(minutes=rng.randint(1, 720), seconds=rng.randint(0, 59))
        if not h.start <= ts < h.end:
            raise AssertionError(f"cutoff posting of {doc.key} outside the period ({variant})")
        doc.posting = PostingSpec(posted=True, ts=ts, offset=h.ledger_offset(rng) if offset is None else offset)

    def _cutoff_issue(self, doc: Invoice | Payment, variant: str, scenario=None, group=None) -> None:
        side = self.h.period_of(doc.ts)
        self.issues.append(
            IssueSpec(
                type="cutoff_error",
                affected=[doc.key],
                currency=doc.currency,
                amount_rule=("doc_total", doc.key),
                evidence=[[("doc", doc.key), ("row", doc.key, "*")]],
                variant=variant,
                cause=(
                    "документ следующего периода проведён в текущем"
                    if side == "next"
                    else "документ прошлого периода проведён в текущем"
                ),
                scenario=scenario,
                group=group,
                before={"posted_in_period": False, "document_period": side},
                after={"posted_in_period": True, "ledger_timestamp_variant": variant},
            )
        )

    def cutoff(self, variant: str | None = None) -> None:
        h = self.h
        rng = self.rng
        pool: list[Invoice | Payment] = [
            inv
            for inv in h.invoices.values()
            if h.period_of(inv.issued_at) != "cur"
            and not ({"boundary_fx_invoice", "boundary_support", "duplicate"} & inv.tags)
            and self.free(inv.key)
        ]
        pool += [
            pay
            for pay in h.payments.values()
            if h.period_of(pay.ts) == "next" and pay.amount_minor > 0 and pay.allocations
            and all(h.period_of(h.invoices[a.invoice].issued_at) != "next" for a in pay.allocations)
            and "boundary" not in pay.tags and self.free(pay.key)
        ]
        if not pool:
            raise GenerationShortfall("no candidate for cutoff_error")
        doc = rng.choice(pool)
        chosen = rng.choice(["entered_in_period", "shifted_into_period"])
        variant = variant or chosen
        self._post_out_of_period(doc, variant)
        self.take(doc.key)
        self._cutoff_issue(doc, variant)

    # ------------------------------------------------------------------ сценарии
    def scenario_fx_partial(self) -> None:
        """Частичный платёж в иностранной валюте на границе периода."""
        h = self.h
        cands = [
            pay
            for pay in h.payments.values()
            if "boundary_fx_payment" in pay.tags and pay.posting.posted
            and sum(a.amount_minor for a in pay.allocations) == pay.amount_minor
            and self.free(pay.key, *(a.invoice for a in pay.allocations))
        ]
        self.rng.shuffle(cands)
        name = "fx_partial_boundary_payment"
        for pay in cands:
            option = _scenario_option(self.fx_options(pay))
            if option is None:
                continue
            group = self.next_group()
            self._apply_fx(pay, option, name, group)
            # сценарий — именно частичное разнесение (полный пропуск был бы просто unapplied + курс)
            if not self._apply_shortfall(pay, name, group, variant="partial_allocation"):
                pay.posting.rate_override = None
                self.issues.pop()
                continue
            self.take(pay.key)
            return
        raise GenerationShortfall("no candidate for fx_partial_boundary_payment")

    def scenario_cutoff_fx(self) -> None:
        """Счёт соседнего периода в иностранной валюте, записанный «как будто» в текущем."""
        h = self.h
        cands = [inv for inv in h.invoices.values() if "boundary_fx_invoice" in inv.tags and self.free(inv.key)]
        self.rng.shuffle(cands)
        name = "cutoff_fx_boundary_invoice"
        for inv in cands:
            option = _scenario_option(self.fx_options(inv))
            if option is None:
                continue
            group = self.next_group()
            # бухгалтер переписал цифры времени документа (в его смещении) как время компании:
            # проводка внутри периода, в поясе компании, а сам документ — соседнего периода
            self._post_out_of_period(inv, "document_offset_misread", offset=h.offset)
            self._cutoff_issue(inv, "document_offset_misread", name, group)
            self._apply_fx(inv, option, name, group)
            self.take(inv.key)
            return
        raise GenerationShortfall("no candidate for cutoff_fx_boundary_invoice")


def _scenario_option(options: list[tuple[str, date, Decimal]]):
    """Для сценариев на границе предпочтительна ошибка «дата в смещении записи документа».
    Если на выходных обе даты сводятся к одному курсу — берём другой вариант неверного курса."""
    for wanted in ("document_offset_date", "utc_date", "other_rule", "due_date", "adjacent_date"):
        for option in options:
            if option[0] == wanted:
                return option
    return None


def inject_all(h: History, rng: random.Random, aux: random.Random | None = None) -> list[IssueSpec]:
    inj = Injector(h, rng, aux)
    p = h.p
    for _ in range(p.scenarios["fx_partial_boundary_payment"]):
        inj.scenario_fx_partial()
    for _ in range(p.scenarios["cutoff_fx_boundary_invoice"]):
        inj.scenario_cutoff_fx()
    for i in range(p.errors["cutoff_error"]):
        inj.cutoff(p.variant_for("cutoff_error", i))
    for _ in range(p.errors["duplicate_invoice"]):
        inj.duplicate()  # единственный вариант reentered_document
    for i in range(p.errors["unapplied_payment"]):
        inj.unapplied(p.variant_for("unapplied_payment", i))
    for i in range(p.errors["fx_rate_error"]):
        inj.fx(p.variant_for("fx_rate_error", i))
    return inj.issues


def finalize_issues(h: History, issues: list[IssueSpec]) -> None:
    """Сумма каждой ошибки — из фактических строк книги после перестроения проводок."""
    by_doc = {e.doc: e for e in h.entries if e.doc}
    for issue in issues:
        rule = issue.amount_rule
        if rule[0] == "doc_total":
            amount = abs(_doc_amount(h.doc(rule[1])))
        elif rule[0] == "posted_line":
            lines = [ln for ln in by_doc[rule[1]].lines if ln.role == rule[2]]
            if len(lines) != 1:
                raise AssertionError(f"expected one {rule[2]} line for {rule[1]}")
            amount = lines[0].debit + lines[0].credit
        elif rule[0] == "fx_diff":
            doc = h.doc(rule[1])
            lines = [ln for ln in by_doc[rule[1]].lines if ln.role == rule[2]]
            if len(lines) != 1:
                raise AssertionError(f"expected one {rule[2]} line for {rule[1]}")
            posted = lines[0].debit + lines[0].credit
            correct = h.conv(abs(_doc_amount(doc)), doc.currency, h.policy_rate(doc.currency, doc.ts)[1])
            amount = abs(posted - correct)
        else:
            raise AssertionError(rule)
        if amount <= 0:
            raise AssertionError(f"non-positive amount for {issue.type}")
        issue.amount_minor = amount
