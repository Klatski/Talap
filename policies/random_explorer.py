"""Случайный исследователь с фиксированным seed — проверяет контракт, ошибки, лимиты и воспроизводимость.

Поток случайности: random.Random("random-explorer|<seed>|<episode_id>") — одинаковые seed,
эпизод и наблюдения дают одинаковые действия в любом процессе.
Поведение: случайное число шагов исследования (иногда больше лимита -> усечение), смесь
корректных и заведомо неверных действий (неизвестные таблицы/ID/действия, испорченные
курсоры, неверные limit, опасные выражения калькулятора), затем случайный submit
(иногда с неверной структурой -> награда -1).
"""

from __future__ import annotations

import random
from typing import Any

TABLES = ("ledger", "invoices", "payments", "fx")
ID_KEYS = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": "fx_id"}
TYPES = ("duplicate_invoice", "unapplied_payment", "fx_rate_error", "cutoff_error")
FAKE_IDS = ("L-ZZZZZZZZ", "INV-ZZZZZZZ", "PAY-ZZZZZZZ", "D-ZZZZZZZZ", "FX:XXX:YYY:2026-01-01",
            "../../etc/passwd", "' OR 1=1 --", "")
CALC = ("{a} * {r}", "round_half_up({a} * {r})", "round_half_even({a} / 3, 2)", "{a} - {b}", "abs({b} - {a})",
        "({a} + {b}) / 2", "1 / 0", "2 ** 64", "__import__('os').system('ls')", "open('secret.txt')", "1e309",
        "(" * 40 + "1" + ")" * 40, "9" * 41)


class RandomExplorer:
    name = "random"

    def __init__(self, seed: int = 0, submit_malformed_p: float = 0.15) -> None:
        self.seed = seed
        self.submit_malformed_p = submit_malformed_p

    def reset(self, observation: dict, info: dict | None = None) -> None:
        self.episode_id = observation.get("episode_id", "")
        self.rng = random.Random(f"random-explorer|{self.seed}|{self.episode_id}")
        self.currencies = sorted((observation.get("manifest") or {}).get("currency_exponents", {"XXX": 2}))
        self.budget = self.rng.randint(0, 115)  # >= лимита шагов -> эпизод усечётся
        self.ids: dict[str, list[str]] = {t: [] for t in TABLES}
        self.docs: list[str] = []
        self.amounts: list[int] = [0]
        self.cursors: dict[str, list[Any]] = {t: [None] for t in TABLES}
        self.taken = 0

    # ------------------------------------------------------------------ наблюдения
    def _absorb(self, obs: dict) -> None:
        result = obs.get("result") if isinstance(obs, dict) else None
        if not isinstance(result, dict):
            return
        records = list(result.get("records") or [])
        if isinstance(result.get("record"), dict):
            records.append(result["record"])
        if isinstance(result.get("document"), dict):
            records.append(result["document"])
        table = result.get("table")
        if table in TABLES and result.get("next_cursor"):
            if result["next_cursor"] not in self.cursors[table]:
                self.cursors[table].append(result["next_cursor"])
        for rec in records:
            if "account" in rec:
                t = "ledger"
            elif "line_items" in rec:
                t = "invoices"
            elif "invoice_refs" in rec:
                t = "payments"
            elif "fx_id" in rec:
                t = "fx"
            else:
                continue
            rid = rec.get(ID_KEYS[t])
            if isinstance(rid, str) and rid not in self.ids[t] and len(self.ids[t]) < 500:
                self.ids[t].append(rid)
            if isinstance(rec.get("doc_id"), str) and rec["doc_id"] not in self.docs:
                self.docs.append(rec["doc_id"])
            for key in ("debit_minor", "credit_minor", "total_minor", "amount_minor"):
                if isinstance(rec.get(key), int) and rec[key] and len(self.amounts) < 500:
                    self.amounts.append(abs(rec[key]))

    # ------------------------------------------------------------------ действия
    def _pick(self, pool: list, fallback: Any = None) -> Any:
        return self.rng.choice(pool) if pool else fallback

    def _seen_id(self, table: str) -> str:
        roll = self.rng.random()
        if roll < 0.7 and self.ids[table]:
            return self.rng.choice(self.ids[table])
        if roll < 0.85:
            other = self.rng.choice(TABLES)
            if self.ids[other]:
                return self.rng.choice(self.ids[other])
        return self.rng.choice(FAKE_IDS)

    def _random_action(self) -> Any:
        rng = self.rng
        kind = rng.choices(
            ["list", "get_record", "get_document", "calculator", "unknown", "malformed"],
            weights=[40, 18, 12, 15, 5, 10],
        )[0]
        if kind == "list":
            table = rng.choice(TABLES) if rng.random() > 0.05 else "accounts"
            cursor = self._pick(self.cursors.get(table, [None]))
            if rng.random() < 0.05:
                cursor = rng.choice(["ledger.0.ffffffffffff", "invoices.10", 42, "../.."])
            limit = rng.choice([50, 50, 50, 20, 5, 1, 0, 75, -3, "5"])
            return {"action": "list_records", "table": table, "cursor": cursor, "limit": limit}
        if kind == "get_record":
            table = rng.choice(TABLES) if rng.random() > 0.05 else "accounts"
            return {"action": "get_record", "table": table, "id": self._seen_id(table if table in TABLES else "ledger")}
        if kind == "get_document":
            doc = self._pick(self.docs, "D-ZZZZZZZZ") if rng.random() < 0.75 else self._seen_id("invoices")
            return {"action": "get_document", "doc_id": doc}
        if kind == "calculator":
            a, b = self._pick(self.amounts, 1), self._pick(self.amounts, 2)
            r = f"{rng.randint(1, 600)}.{rng.randint(0, 9999):04d}"
            return {"action": "calculator", "expression": rng.choice(CALC).format(a=a, b=b, r=r)}
        if kind == "unknown":
            return {"action": rng.choice(["delete_records", "sql", "shell", "read_file", "dump_all", "SUBMIT"])}
        return rng.choice([
            "list_records",
            ["list_records", "ledger"],
            {"action": 5},
            {"table": "ledger"},
            {"action": "get_record", "table": "ledger"},
            {"action": "get_document", "doc_id": "D-1", "extra": True},
            {"action": "calculator", "expression": None},
            {"action": "list_records", "table": None},
        ])

    def _random_submit(self) -> dict:
        rng = self.rng
        findings = []
        for _ in range(rng.randint(0, 4)):
            pool = self.ids["invoices"] + self.ids["payments"] or ["INV-ZZZZZZZ"]
            findings.append({
                "type": rng.choice(TYPES),
                "affected_ids": rng.sample(pool, min(len(pool), rng.choice([1, 1, 2]))),
                "claimed_amount_minor": self._pick(self.amounts, 0),
                "currency": rng.choice(self.currencies),
                "evidence_ids": rng.sample(self.docs + self.ids["ledger"] + self.ids["fx"] or ["L-ZZZZZZZZ"],
                                           min(3, len(self.docs + self.ids["ledger"] + self.ids["fx"]) or 1)),
                "explanation": "random guess",
            })
        payload: Any = {"episode_id": self.episode_id, "findings": findings}
        if rng.random() < self.submit_malformed_p:
            payload = rng.choice([
                {"episode_id": self.episode_id, "findings": [{"type": "fraud"}]},
                {"episode_id": "ep-WRONG00000", "findings": []},
                {"episode_id": self.episode_id, "findings": findings, "confidence": 0.99},
                "all of them",
                None,
            ])
        return {"action": "submit", "findings": payload}

    def act(self, observation: dict) -> Any:
        self._absorb(observation)
        self.taken += 1
        if self.taken > self.budget:
            return self._random_submit()
        return self._random_action()
