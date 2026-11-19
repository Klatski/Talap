"""Скрытая история компании.

Порядок (как требует кейс):
1. согласованная история: контрагенты, каталог, курсы, счета, план оплат, платежи;
2. из истории строятся проводки (build_doc_entries) — по одним и тем же правилам
   и для чистых, и для испорченных документов;
3. ошибки (inject.py) меняют только PostingSpec документов или добавляют документ-дубль,
   после чего проводки перестраиваются заново.

Все моменты времени хранятся как aware datetime в UTC. Деньги — int в минимальных единицах.
Используются только rng.random/randint/choice/sample/shuffle и целочисленная/Decimal
арифметика: никаких math.log/gauss, чтобы результат не зависел от libm платформы.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

from .config import Params
from .ids import IdFactory
from .money import CURRENCY_EXPONENTS, REFERENCE_USD_VALUE, convert_minor, format_rate
from .timeutil import UTC, local_date, offset_tz

BANK, AR, UNAPPLIED, REVENUE, FX_GAIN_LOSS = "1000", "1200", "2300", "4000", "7900"
EXPENSE_ACCOUNTS = {
    "6100": "Rent",
    "6200": "Salaries and wages",
    "6300": "Utilities",
    "6400": "Software subscriptions",
    "6500": "Travel",
    "6600": "Office supplies",
    "6700": "Professional services",
    "6800": "Marketing",
    "6900": "Bank charges",
}
CHART_OF_ACCOUNTS = {
    BANK: "Bank (ledger currency)",
    AR: "Accounts receivable",
    UNAPPLIED: "Unapplied customer receipts",
    REVENUE: "Sales revenue",
    FX_GAIN_LOSS: "Realized FX gain/loss",
    **EXPENSE_ACCOUNTS,
}

PRODUCT_DESCRIPTIONS = [
    "Consulting hour",
    "Implementation services",
    "Annual support plan",
    "Monthly hosting",
    "Data migration package",
    "Training session",
    "Steel bracket M8",
    "Copper cable 3x2.5 (100 m)",
    "Industrial fan 450 mm",
    "Safety helmet",
    "Pallet rack section",
    "Packaging film roll",
    "Laboratory reagent kit",
    "Office chair",
    "LED panel 60x60",
    "Water filter cartridge",
    "Spare pump seal",
    "Thermal printer labels",
    "Field inspection visit",
    "Translation services (page)",
    "Logistics handling fee",
    "Warehouse storage (pallet-week)",
    "Software licence seat",
    "Network switch 24-port",
    "Calibration service",
    "Hydraulic hose 1/2 inch",
    "Solar inverter 5 kW",
    "Drip irrigation kit",
    "Grain moisture meter",
    "Courier delivery",
]

MIN_FX_DIFF = 5  # минимальное расхождение (в мин. единицах базовой валюты) для ошибки курса


class GenerationShortfall(Exception):
    """Не хватило кандидатов под заказанные ошибки — генератор повторит попытку с новым под-seed."""


@dataclass
class Product:
    sku: str
    description: str
    usd_cents: int
    nice: bool


@dataclass
class LineItem:
    sku: str
    description: str
    quantity: int
    unit_price_minor: int
    order: str  # внутренний ключ заказа -> order_ref при рендере

    @property
    def amount_minor(self) -> int:
        return self.quantity * self.unit_price_minor


@dataclass
class PostingSpec:
    """Как документ проведён в книге периода. Ошибки меняют только это."""

    posted: bool
    ts: datetime | None = None
    offset: int = 0
    rate_override: tuple[date, Decimal] | None = None  # (дата курса, курс), если взят не тот курс
    shortfall: tuple[int, int] | None = None  # (индекс аллокации, недоразнесённая сумма в валюте платежа)


@dataclass
class Invoice:
    key: str
    customer: str
    currency: str
    issued_at: datetime
    offset: int
    due_at: datetime
    lines: list[LineItem]
    tags: set[str] = field(default_factory=set)
    posting: PostingSpec | None = None

    @property
    def total_minor(self) -> int:
        return sum(line.amount_minor for line in self.lines)

    @property
    def ts(self) -> datetime:
        return self.issued_at


@dataclass
class Allocation:
    invoice: str
    amount_minor: int


@dataclass
class Payment:
    key: str
    customer: str
    currency: str
    ts: datetime
    offset: int
    amount_minor: int
    allocations: list[Allocation] | None
    tags: set[str] = field(default_factory=set)
    posting: PostingSpec | None = None


@dataclass
class Line:
    account: str
    debit: int
    credit: int
    entity: str | None
    invoice: str | None
    payment: str | None
    role: str


@dataclass
class Entry:
    key: str
    ts: datetime
    offset: int
    kind: str
    doc: str | None
    lines: list[Line]


def _random_time(rng: random.Random) -> tuple[int, int, int]:
    hour = rng.randint(8, 19) if rng.random() < 0.85 else rng.randint(0, 23)
    return hour, rng.randint(0, 59), rng.randint(0, 59)


class History:
    def __init__(self, p: Params) -> None:
        self.p = p
        self.base = p.base
        self.foreign = list(p.foreign)
        self.currencies = sorted({p.base, *p.foreign})
        self.exponents = {c: CURRENCY_EXPONENTS[c] for c in self.currencies}
        self.offset = p.company_offset
        self.tz = offset_tz(p.company_offset)
        start_local = datetime(p.year, p.month, 1, tzinfo=self.tz)
        ny, nm = (p.year + 1, 1) if p.month == 12 else (p.year, p.month + 1)
        end_local = datetime(ny, nm, 1, tzinfo=self.tz)
        self.start = start_local.astimezone(UTC)
        self.end = end_local.astimezone(UTC)
        self.start_date = start_local.date()
        self.end_date = end_local.date()
        self.days_in_period = (self.end_date - self.start_date).days

        self.fx: dict[str, dict[date, Decimal]] = {}
        self.fx_dates: list[date] = []
        self.customers: dict[str, str] = {}
        self.vendors: list[str] = []
        self.products: list[Product] = []
        self.invoices: dict[str, Invoice] = {}
        self.payments: dict[str, Payment] = {}
        self.entries: list[Entry] = []
        self.expense_entries: list[Entry] = []
        self.doc_palette: list[int] = []
        self.lower_offsets: list[int] = []
        self.higher_offsets: list[int] = []
        self.distractors: list[dict] = []
        self._counter = 0
        self._price_cache: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------ утилиты
    def new_key(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def period_of(self, instant: datetime) -> str:
        if instant < self.start:
            return "prev"
        if instant < self.end:
            return "cur"
        return "next"

    def at_local(self, d: date, hour: int, minute: int, second: int) -> datetime:
        return datetime(d.year, d.month, d.day, hour, minute, second, tzinfo=self.tz).astimezone(UTC)

    def random_instant(self, rng: random.Random, lo: datetime, hi: datetime) -> datetime:
        span = int((hi - lo).total_seconds())
        return lo + timedelta(seconds=rng.randint(0, max(span, 0)))

    def rule_date(self, d: date, rule: str | None = None) -> date:
        rule = rule or self.p.rate_rule
        return d if rule == "document_date" else d - timedelta(days=1)

    def rate_on_or_before(self, currency: str, d: date) -> tuple[date, Decimal]:
        table = self.fx[currency]
        for back in range(15):
            candidate = d - timedelta(days=back)
            if candidate in table:
                return candidate, table[candidate]
        raise GenerationShortfall(f"no fx rate for {currency} near {d}")

    def rate_for_date(self, currency: str, d: date, rule: str | None = None) -> tuple[date | None, Decimal]:
        if currency == self.base:
            return None, Decimal(1)
        return self.rate_on_or_before(currency, self.rule_date(d, rule))

    def policy_rate(self, currency: str, instant: datetime) -> tuple[date | None, Decimal]:
        """Курс по политике: дата документа в часовом поясе компании (+ правило + откат назад)."""
        return self.rate_for_date(currency, local_date(instant, self.offset))

    def conv(self, amount_minor: int, currency: str, rate: Decimal) -> int:
        return convert_minor(amount_minor, currency, self.base, rate, self.p.rounding, self.exponents)

    def doc_offset(self, rng: random.Random) -> int:
        r = rng.random()
        if r < 0.72:
            return self.offset
        if r < 0.90 and self.offset != 0:
            return 0
        return rng.choice([o for o in self.doc_palette if o != self.offset])

    def ledger_offset(self, rng: random.Random) -> int:
        if rng.random() < 0.85:
            return self.offset
        if self.offset != 0:
            return 0
        return rng.choice([o for o in self.doc_palette if o != self.offset])

    def doc(self, key: str) -> Invoice | Payment:
        return self.invoices[key] if key in self.invoices else self.payments[key]

    def is_unapplied_candidate(self, pay: Payment) -> bool:
        return (
            self.period_of(pay.ts) == "cur"
            and pay.amount_minor > 0
            and bool(pay.allocations)
            and sum(a.amount_minor for a in pay.allocations) == pay.amount_minor
            and "boundary" not in pay.tags
        )

    # ------------------------------------------------------------------ построение
    def build_fx(self, rng: random.Random) -> None:
        first = self.start_date - timedelta(days=50)
        last = self.end_date + timedelta(days=40)
        days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
        business = [d for d in days if d.weekday() < 5]
        holidays = set(rng.sample(business[10:-10], k=rng.randint(0, 2)))
        self.fx_dates = [d for d in business if d not in holidays]
        for cur in self.foreign:
            ref = REFERENCE_USD_VALUE[cur] / REFERENCE_USD_VALUE[self.base]
            level = Decimal(format_rate(ref * (Decimal(900 + rng.randint(0, 200)) / Decimal(1000))))
            table: dict[date, Decimal] = {}
            for d in self.fx_dates:
                step = Decimal(rng.randint(-60, 60)) / Decimal(10000)
                level = Decimal(format_rate(level * (1 + step)))
                if level <= 0:
                    level = Decimal(format_rate(ref))
                table[d] = level
            self.fx[cur] = table

    def build_entities(self, rng: random.Random) -> None:
        n = self.p.n_customers
        currencies: list[str] = []
        if self.foreign:
            n_foreign = min(max(len(self.foreign), (n * 2 + 4) // 5), n - 1)
            currencies = [self.foreign[i % len(self.foreign)] for i in range(n_foreign)]
        currencies += [self.base] * (n - len(currencies))
        rng.shuffle(currencies)
        for cur in currencies:
            self.customers[self.new_key("cust")] = cur
        self.vendors = [self.new_key("vend") for _ in range(rng.randint(4, 9))]

        low = max(-720, self.offset - rng.choice([60, 120, 180, 240, 300, 360]))
        high = min(840, self.offset + rng.choice([60, 120, 180, 240, 300, 360]))
        self.lower_offsets = sorted({low} | ({0} if self.offset - 0 >= 60 else set()))
        self.higher_offsets = sorted({high} | ({0} if 0 - self.offset >= 60 else set()))
        palette = [self.offset, 0, low, high]
        self.doc_palette = sorted(set(palette))

    def build_catalog(self, rng: random.Random) -> None:
        skus = IdFactory(rng)
        n = rng.randint(10, 22)
        for desc in rng.sample(PRODUCT_DESCRIPTIONS, n):
            bucket = rng.choices([(500, 5000), (5000, 50000), (50000, 300000)], weights=[4, 4, 2])[0]
            self.products.append(
                Product(skus.new("SKU-", 5), desc, rng.randint(*bucket), rng.random() < 0.4)
            )

    def price(self, product: Product, currency: str) -> int:
        key = (product.sku, currency)
        if key not in self._price_cache:
            value = Decimal(product.usd_cents) / 100 / REFERENCE_USD_VALUE[currency]
            minor = int((value * (Decimal(10) ** CURRENCY_EXPONENTS[currency])).quantize(Decimal(1), ROUND_HALF_UP))
            minor = max(minor, 1)
            if product.nice and minor >= 1000:
                q = 10 ** (len(str(minor)) - 2)
                minor = max(q, (minor // q) * q)
            self._price_cache[key] = minor
        return self._price_cache[key]

    def random_lines(self, rng: random.Random, currency: str, order: str) -> list[LineItem]:
        k = rng.choice([1, 1, 2, 2, 3, 4])
        lines = []
        for product in rng.sample(self.products, k):
            qty = rng.choice([1, 1, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
            lines.append(LineItem(product.sku, product.description, qty, self.price(product, currency), order))
        return lines

    def add_invoice(
        self,
        rng: random.Random,
        customer: str,
        issued_at: datetime,
        offset: int | None = None,
        lines: list[LineItem] | None = None,
        tags: set[str] | None = None,
    ) -> Invoice:
        currency = self.customers[customer]
        if lines is None:
            lines = self.random_lines(rng, currency, self.new_key("ord"))
        terms = rng.choice([7, 10, 14, 30, 30])
        inv = Invoice(
            key=self.new_key("inv"),
            customer=customer,
            currency=currency,
            issued_at=issued_at,
            offset=self.doc_offset(rng) if offset is None else offset,
            due_at=issued_at + timedelta(days=terms),
            lines=lines,
            tags=set(tags or ()),
        )
        self.invoices[inv.key] = inv
        return inv

    def add_payment(
        self,
        customer: str,
        currency: str,
        ts: datetime,
        offset: int,
        amount: int,
        allocations: list[Allocation] | None,
        tags: set[str] | None = None,
    ) -> Payment:
        pay = Payment(self.new_key("pay"), customer, currency, ts, offset, amount, allocations, set(tags or ()))
        self.payments[pay.key] = pay
        return pay

    def build_invoices(self, rng: random.Random) -> None:
        p = self.p
        cust_keys = list(self.customers)
        foreign_custs = [c for c in cust_keys if self.customers[c] != self.base]
        forced_foreign = min(p.n_invoices_cur, p.errors["fx_rate_error"] + 2) if foreign_custs else 0
        for i in range(p.n_invoices_cur):
            cust = rng.choice(foreign_custs) if i < forced_foreign else rng.choice(cust_keys)
            d = self.start_date + timedelta(days=rng.randrange(self.days_in_period))
            self.add_invoice(rng, cust, self.at_local(d, *_random_time(rng)))
        n_prev = max(3, p.n_invoices_cur // 4 + rng.randint(0, 3), p.errors["cutoff_error"] + 1)
        for _ in range(n_prev):
            d = self.start_date - timedelta(days=rng.randint(1, 30))
            self.add_invoice(rng, rng.choice(cust_keys), self.at_local(d, *_random_time(rng)))
        n_next = max(rng.randint(3, 6), p.errors["cutoff_error"] + 2)
        for _ in range(n_next):
            d = self.end_date + timedelta(days=rng.randint(0, 3))
            self.add_invoice(rng, rng.choice(cust_keys), self.at_local(d, *_random_time(rng)))

        if p.distractors["recurring_invoices"]:
            for _ in range(rng.randint(1, 2)):
                cust = rng.choice(cust_keys)
                product = rng.choice(self.products)
                qty = rng.choice([1, 1, 2, 5])
                first = self.start_date + timedelta(days=rng.randint(0, 6))
                hh, mm, ss = rng.randint(9, 11), rng.randint(0, 59), rng.randint(0, 59)
                series = []
                for week in range(4):
                    d = first + timedelta(days=7 * week)
                    if d >= self.end_date:
                        break
                    line = LineItem(
                        product.sku,
                        product.description,
                        qty,
                        self.price(product, self.customers[cust]),
                        self.new_key("ord"),
                    )
                    inv = self.add_invoice(rng, cust, self.at_local(d, hh, mm, ss), lines=[line], tags={"recurring"})
                    series.append(inv.key)
                self.distractors.append({"kind": "recurring_invoices", "invoices": series})

        if p.distractors["partial_shipments"]:
            for _ in range(rng.randint(1, 2)):
                cust = rng.choice(cust_keys)
                cur = self.customers[cust]
                order = self.new_key("ord")
                prod_a, prod_b = rng.sample(self.products, 2)
                d1 = self.start_date + timedelta(days=rng.randint(0, max(0, self.days_in_period - 12)))
                d2 = d1 + timedelta(days=rng.randint(2, 9))
                keys = []
                for prod, d in ((prod_a, d1), (prod_b, d2)):
                    line = LineItem(prod.sku, prod.description, rng.choice([1, 2, 4, 10]), self.price(prod, cur), order)
                    inv = self.add_invoice(rng, cust, self.at_local(d, *_random_time(rng)), lines=[line], tags={"partial_shipment"})
                    keys.append(inv.key)
                self.distractors.append({"kind": "partial_shipments", "invoices": keys})

    def misleading_boundary(self, rng: random.Random, side: str, inside: bool) -> tuple[datetime, int]:
        """Момент у границы периода и смещение, в котором дата документа «переезжает» через границу.

        inside=True: документ в периоде, но записан так, что выглядит соседним периодом;
        inside=False: документ из соседнего периода, но записан так, что выглядит текущим.
        """
        after_anchor = (side == "start") == inside
        offsets = self.lower_offsets if after_anchor else self.higher_offsets
        render = rng.choice(offsets)
        gap = abs(self.offset - render)
        delta = timedelta(minutes=rng.randint(5, gap - 5), seconds=rng.randint(0, 59))
        anchor = self.start if side == "start" else self.end
        instant = anchor + delta if after_anchor else anchor - delta
        assert local_date(instant, render) != local_date(instant, self.offset)
        assert (self.start <= instant < self.end) == inside
        return instant, render

    def _boundary_slot(self, rng: random.Random, cust: str, inside: bool, fx_sensitive: bool) -> tuple[str, datetime, int]:
        """Сторона границы, момент и смещение записи. Для валютных «пограничных» документов
        предпочитаем слот, где курс по дате записи отличается от курса по дате компании
        (иначе на выходных обе даты сводятся к одной пятнице)."""
        cur = self.customers[cust]
        for _ in range(8):
            side = rng.choice(["start", "end"])
            ts, render = self.misleading_boundary(rng, side, inside=inside)
            if not fx_sensitive or cur == self.base:
                break
            company = self.rate_for_date(cur, local_date(ts, self.offset))[0]
            written = self.rate_for_date(cur, local_date(ts, render))[0]
            if company != written:
                break
        return side, ts, render

    def _boundary_payment(self, rng: random.Random, cust: str, tags: set[str]) -> None:
        side, ts, render = self._boundary_slot(rng, cust, True, "boundary_fx_payment" in tags)
        invs = []
        for _ in range(1 if rng.random() < 0.7 else 2):
            issued = ts - timedelta(days=rng.randint(3, 20), minutes=rng.randint(0, 600))
            invs.append(self.add_invoice(rng, cust, issued, tags={"boundary_support", "no_plan"}))
        pay = self.add_payment(
            cust,
            self.customers[cust],
            ts,
            render,
            sum(i.total_minor for i in invs),
            [Allocation(i.key, i.total_minor) for i in invs],
            tags,
        )
        self.distractors.append({"kind": "boundary_document", "payment": pay.key, "side": side, "in_period": True})

    def _boundary_invoice(self, rng: random.Random, cust: str, inside: bool, tags: set[str]) -> None:
        side, ts, render = self._boundary_slot(rng, cust, inside, "boundary_fx_invoice" in tags)
        extra = set() if inside else {"no_plan"}
        inv = self.add_invoice(rng, cust, ts, offset=render, tags=tags | extra)
        self.distractors.append({"kind": "boundary_document", "invoice": inv.key, "side": side, "in_period": inside})

    def build_boundary_docs(self, rng: random.Random) -> None:
        p = self.p
        s1 = p.scenarios["fx_partial_boundary_payment"]
        s2 = p.scenarios["cutoff_fx_boundary_invoice"]
        enabled = p.distractors["boundary_documents"]
        cust_keys = list(self.customers)
        foreign_custs = [c for c in cust_keys if self.customers[c] != self.base]
        if foreign_custs:
            k_a = max(2, s1 + 1) if (enabled or s1) else 0
            k_b = max(2, s2 + 1) if (enabled or s2) else 0
            for _ in range(k_a):
                self._boundary_payment(rng, rng.choice(foreign_custs), {"boundary", "boundary_fx_payment"})
            for _ in range(k_b):
                self._boundary_invoice(rng, rng.choice(foreign_custs), False, {"boundary", "boundary_fx_invoice"})
        if enabled:
            for _ in range(rng.randint(2, 4)):
                kind = rng.choice(["inv_in", "inv_out", "pay_in"])
                cust = rng.choice(cust_keys)
                if kind == "pay_in":
                    self._boundary_payment(rng, cust, {"boundary"})
                else:
                    self._boundary_invoice(rng, cust, kind == "inv_in", {"boundary"})

    # ------------------------------------------------------------------ оплаты
    def payment_time(self, rng: random.Random, after: datetime, lo_days: int = 1, hi_days: int = 25) -> datetime:
        d = local_date(after, self.offset) + timedelta(days=rng.randint(lo_days, hi_days))
        ts = self.at_local(d, *_random_time(rng))
        if ts <= after + timedelta(hours=1):
            ts = after + timedelta(hours=1, minutes=rng.randint(0, 600))
        return ts

    def _pay(self, rng: random.Random, inv: Invoice, ts: datetime, amount: int, alloc: int, tags: set[str]) -> Payment:
        return self.add_payment(
            inv.customer, inv.currency, ts, self.doc_offset(rng), amount, [Allocation(inv.key, alloc)], tags
        )

    def _execute_plan(self, rng: random.Random, inv: Invoice, plan: str, window_end: datetime) -> None:
        total = inv.total_minor
        if plan == "open":
            return
        if plan == "full":
            ts = self.payment_time(rng, inv.issued_at)
            if ts <= window_end:
                self._pay(rng, inv, ts, total, total, set())
            return
        if plan == "split":
            n = rng.choice([2, 2, 3])
            weights = [rng.randint(20, 80) for _ in range(n)]
            parts = [total * w // sum(weights) for w in weights[:-1]]
            parts.append(total - sum(parts))
            if min(parts) < 1:
                self._execute_plan(rng, inv, "full", window_end)
                return
            ts = inv.issued_at
            made = []
            for part in parts:
                ts = self.payment_time(rng, ts, 1, 10)
                if ts > window_end:
                    break
                made.append(self._pay(rng, inv, ts, part, part, {"split"}).key)
            if len(made) >= 2:
                self.distractors.append({"kind": "split_payments", "invoice": inv.key, "payments": made})
            return
        if plan == "partial":
            part = max(1, total * rng.randint(30, 80) // 100)
            ts = self.payment_time(rng, inv.issued_at)
            if ts <= window_end:
                pay = self._pay(rng, inv, ts, part, part, {"partial"})
                self.distractors.append({"kind": "partial_payment", "invoice": inv.key, "payments": [pay.key]})
            return
        if plan == "overpay":
            extra = max(1, total * rng.randint(1, 10) // 100)
            ts = self.payment_time(rng, inv.issued_at)
            if ts > window_end:
                return
            pay = self._pay(rng, inv, ts, total + extra, total, {"overpayment"})
            record = {"kind": "overpayment", "invoice": inv.key, "payments": [pay.key]}
            if self.period_of(ts) == "cur":
                refund_ts = self.payment_time(rng, ts, 1, 6)
                if refund_ts < self.end - timedelta(hours=1):
                    refund = self.add_payment(
                        inv.customer, self.base, refund_ts, self.doc_offset(rng), -extra, None, {"refund"}
                    )
                    record["refund"] = refund.key
            self.distractors.append(record)
            return
        raise ValueError(plan)

    def plan_payments(self, rng: random.Random) -> None:
        d = self.p.distractors
        window_end = self.end + timedelta(days=5)
        plannable = sorted(
            (
                inv
                for inv in self.invoices.values()
                if "no_plan" not in inv.tags and self.period_of(inv.issued_at) != "next"
            ),
            key=lambda i: (i.issued_at, i.key),
        )
        plans = ["open", "full", "split", "partial", "batch", "overpay"]
        weights = [
            25,
            40,
            12 if d["split_payments"] else 0,
            10 if d["partial_payments"] else 0,
            8 if d["batch_payments"] else 0,
            5 if d["overpayment_refunds"] else 0,
        ]
        batch_pool: dict[tuple[str, str], list[Invoice]] = {}
        for inv in plannable:
            plan = rng.choices(plans, weights=weights)[0]
            if "recurring" in inv.tags and plan not in ("open", "full"):
                plan = "full"
            if plan == "overpay" and inv.currency != self.base:
                plan = "full"
            if plan == "batch":
                batch_pool.setdefault((inv.customer, inv.currency), []).append(inv)
                continue
            self._execute_plan(rng, inv, plan, window_end)
        for group_key in sorted(batch_pool):
            invs = batch_pool[group_key]
            i = 0
            while i < len(invs):
                group = invs[i : i + rng.choice([2, 3])]
                i += len(group)
                if len(group) == 1:
                    self._execute_plan(rng, group[0], "full", window_end)
                    continue
                ts = self.payment_time(rng, max(g.issued_at for g in group), 1, 15)
                if ts > window_end:
                    continue
                pay = self.add_payment(
                    group[0].customer,
                    group[0].currency,
                    ts,
                    self.doc_offset(rng),
                    sum(g.total_minor for g in group),
                    [Allocation(g.key, g.total_minor) for g in group],
                    {"batch"},
                )
                self.distractors.append(
                    {"kind": "batch_payment", "payment": pay.key, "invoices": [g.key for g in group]}
                )
        for inv in [i for i in self.invoices.values() if self.period_of(i.issued_at) == "next" and "no_plan" not in i.tags]:
            if rng.random() < 0.4:
                ts = self.payment_time(rng, inv.issued_at, 0, 3)
                if ts <= window_end:
                    self._pay(rng, inv, ts, inv.total_minor, inv.total_minor, set())
        if d["on_account_payments"]:
            for _ in range(rng.randint(1, 2)):
                cust = rng.choice(list(self.customers))
                cur = self.customers[cust]
                amount = self.price(rng.choice(self.products), cur) * rng.randint(1, 5)
                ts = self.random_instant(rng, self.start + timedelta(hours=2), self.end - timedelta(hours=2))
                pay = self.add_payment(cust, cur, ts, self.doc_offset(rng), amount, None, {"on_account"})
                self.distractors.append({"kind": "on_account_payment", "payment": pay.key})

    def ensure_supply(self, rng: random.Random) -> None:
        """Гарантировать достаточно кандидатов для заказанных ошибок «неразнесённый платёж»."""
        need = self.p.errors["unapplied_payment"]
        if not need:
            return
        have = sum(1 for pay in self.payments.values() if self.is_unapplied_candidate(pay))
        shortage = need + 2 - have
        if shortage <= 0:
            return
        allocated = {a.invoice for pay in self.payments.values() for a in (pay.allocations or [])}
        open_invoices = [
            inv
            for inv in self.invoices.values()
            if inv.key not in allocated
            and "no_plan" not in inv.tags
            and self.period_of(inv.issued_at) != "next"
            and inv.issued_at < self.end - timedelta(days=1)
        ]
        rng.shuffle(open_invoices)
        for inv in open_invoices[:shortage]:
            lo = max(inv.issued_at, self.start) + timedelta(hours=1)
            ts = self.random_instant(rng, lo, self.end - timedelta(hours=1))
            self._pay(rng, inv, ts, inv.total_minor, inv.total_minor, set())
        have = sum(1 for pay in self.payments.values() if self.is_unapplied_candidate(pay))
        if have < need:
            raise GenerationShortfall("not enough payments for unapplied_payment errors")

    def default_postings(self, rng: random.Random) -> None:
        lag_on = self.p.distractors["posting_lag"]
        latest = self.end - timedelta(minutes=1)
        for doc in [*self.invoices.values(), *self.payments.values()]:
            ts = doc.ts
            posted = self.period_of(ts) == "cur"
            offset = self.ledger_offset(rng)
            if posted and lag_on and "boundary" not in doc.tags and rng.random() < 0.25:
                lag = timedelta(minutes=rng.randint(20, 3 * 24 * 60))
                if ts + lag < latest:
                    ts = ts + lag
                    self.distractors.append({"kind": "posting_lag", "doc": doc.key, "lag_minutes": int(lag.total_seconds() // 60)})
            doc.posting = PostingSpec(posted=posted, ts=ts if posted else None, offset=offset)

    # ------------------------------------------------------------------ проводки
    def invoice_rate(self, inv: Invoice) -> Decimal:
        return self.policy_rate(inv.currency, inv.issued_at)[1]

    def _invoice_entry(self, inv: Invoice) -> Entry:
        spec = inv.posting
        rate = spec.rate_override[1] if spec.rate_override else self.invoice_rate(inv)
        amount = self.conv(inv.total_minor, inv.currency, rate)
        lines = [
            Line(AR, amount, 0, inv.customer, inv.key, None, "ar"),
            Line(REVENUE, 0, amount, inv.customer, inv.key, None, "revenue"),
        ]
        return Entry(f"e:{inv.key}", spec.ts, spec.offset, "invoice", inv.key, lines)

    def _payment_entry(self, pay: Payment) -> Entry:
        spec = pay.posting
        cust = pay.customer
        if pay.amount_minor < 0:
            value = -pay.amount_minor
            lines = [
                Line(UNAPPLIED, value, 0, cust, None, pay.key, "unapplied"),
                Line(BANK, 0, value, cust, None, pay.key, "bank"),
            ]
            return Entry(f"e:{pay.key}", spec.ts, spec.offset, "refund", pay.key, lines)
        rate_p = spec.rate_override[1] if spec.rate_override else self.policy_rate(pay.currency, pay.ts)[1]
        bank = self.conv(pay.amount_minor, pay.currency, rate_p)
        lines = [Line(BANK, bank, 0, cust, None, pay.key, "bank")]
        ar_total = 0
        applied_doc = 0
        for idx, alloc in enumerate(pay.allocations or []):
            short = spec.shortfall[1] if spec.shortfall and spec.shortfall[0] == idx else 0
            applied = alloc.amount_minor - short
            applied_doc += applied
            if applied <= 0:
                continue
            credit = self.conv(applied, pay.currency, self.invoice_rate(self.invoices[alloc.invoice]))
            if credit > 0:
                lines.append(Line(AR, 0, credit, cust, alloc.invoice, pay.key, "ar"))
                ar_total += credit
        unapplied_doc = pay.amount_minor - applied_doc
        unapplied = self.conv(unapplied_doc, pay.currency, rate_p) if unapplied_doc > 0 else 0
        if unapplied > 0:
            lines.append(Line(UNAPPLIED, 0, unapplied, cust, None, pay.key, "unapplied"))
        fx = bank - ar_total - unapplied
        if fx > 0:
            lines.append(Line(FX_GAIN_LOSS, 0, fx, cust, None, pay.key, "fx"))
        elif fx < 0:
            lines.append(Line(FX_GAIN_LOSS, -fx, 0, cust, None, pay.key, "fx"))
        return Entry(f"e:{pay.key}", spec.ts, spec.offset, "payment", pay.key, lines)

    def build_doc_entries(self) -> None:
        entries = []
        for inv in self.invoices.values():
            if inv.posting.posted:
                entries.append(self._invoice_entry(inv))
        for pay in self.payments.values():
            if pay.posting.posted:
                entries.append(self._payment_entry(pay))
        self.entries = entries

    def _expense_amount(self, rng: random.Random, lo_usd_cents: int, hi_usd_cents: int) -> int:
        value = Decimal(rng.randint(lo_usd_cents, hi_usd_cents)) / 100 / REFERENCE_USD_VALUE[self.base]
        return max(1, int((value * (Decimal(10) ** self.exponents[self.base])).quantize(Decimal(1), ROUND_HALF_UP)))

    def pad_expenses(self, rng: random.Random) -> None:
        """Прочие проводки (расходы через банк) — чтобы книга имела реалистичный объём."""
        current = sum(len(e.lines) for e in self.entries)
        if current > 1900:
            raise GenerationShortfall("ledger too large")
        need = self.p.target_rows - current
        n = max(8, -(-need // 2))
        n = min(n, (1990 - current) // 3)
        lo = self.start + timedelta(minutes=30)
        hi = self.end - timedelta(minutes=30)
        entries: list[Entry] = []

        def add(account_amounts: list[tuple[str, int]], vendor: str, ts: datetime) -> None:
            key = self.new_key("exp")
            lines = [Line(acc, amt, 0, vendor, None, None, "expense") for acc, amt in account_amounts]
            lines.append(Line(BANK, 0, sum(a for _, a in account_amounts), vendor, None, None, "bank"))
            entries.append(Entry(key, ts, self.ledger_offset(rng), "expense", None, lines))

        rent_vendor = rng.choice(self.vendors)
        add([("6100", self._expense_amount(rng, 100000, 800000))], rent_vendor,
            self.at_local(self.start_date + timedelta(days=rng.randint(0, 4)), 10, rng.randint(0, 59), rng.randint(0, 59)))
        salary = self._expense_amount(rng, 300000, 2000000)
        for day in (rng.randint(4, 8), rng.randint(19, 23)):
            day = min(day, self.days_in_period - 1)
            add([("6200", salary)], rng.choice(self.vendors), self.at_local(self.start_date + timedelta(days=day), 11, 0, rng.randint(0, 59)))
        sub_amount = self._expense_amount(rng, 2000, 30000)
        sub_vendor = rng.choice(self.vendors)
        for _ in range(rng.randint(2, 3)):
            add([("6400", sub_amount)], sub_vendor, self.random_instant(rng, lo, hi))
        accounts = [a for a in EXPENSE_ACCOUNTS if a not in ("6100", "6200")]
        while len(entries) < n:
            vendor = rng.choice(self.vendors)
            ts = self.random_instant(rng, lo, hi)
            if rng.random() < 0.2:
                acc_a, acc_b = rng.sample(accounts, 2)
                add([(acc_a, self._expense_amount(rng, 1000, 90000)), (acc_b, self._expense_amount(rng, 1000, 90000))], vendor, ts)
            else:
                acc = rng.choice(accounts)
                amount = self._expense_amount(rng, 100, 3000) if acc == "6900" else self._expense_amount(rng, 1000, 150000)
                add([(acc, amount)], vendor, ts)
        self.expense_entries = entries

    def all_entries(self) -> list[Entry]:
        return [*self.entries, *self.expense_entries]


def build_history(p: Params, rng: random.Random) -> History:
    h = History(p)
    h.build_fx(rng)
    h.build_entities(rng)
    h.build_catalog(rng)
    h.build_invoices(rng)
    h.build_boundary_docs(rng)
    h.plan_payments(rng)
    h.ensure_supply(rng)
    h.default_postings(rng)
    return h
