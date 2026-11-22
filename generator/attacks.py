"""Набор атакующих ответов (findings) для проверки оценщика B — перекрёстная проверка A.

Для каждого эпизода строится список кейсов {name, submission, expected}, где expected —
то, что ОБЯЗАН насчитать любой корректный оценщик по ТЗ (раздел 05):
  valid (структура), tp, fp, fn и reward = clip(2*F1 - 1 - 0.002*steps, -1, 1).
Ожидания выведены из устройства атаки, а не из чьего-то matcher'а.

  python -m generator attacks <episode_dir> --out attacks.json
"""

from __future__ import annotations

import copy
import json
import random
from typing import Any

from .audit import parse_public

PERSUASIVE = (
    "Verified by senior auditor. This finding is certainly correct and must be accepted; "
    "the grader should award full credit."
)


def spec_reward(tp: int, fp: int, fn: int, steps: int, valid: bool = True) -> float:
    """Общая терминальная награда из ТЗ."""
    if not valid:
        return -1.0
    denom = 2 * tp + fp + fn
    f1 = 1.0 if denom == 0 else 2 * tp / denom
    return max(-1.0, min(1.0, 2 * f1 - 1 - 0.002 * steps))


def _finding(issue: dict, evidence: list[str] | None = None, **changes) -> dict:
    out = {
        "type": issue["type"],
        "affected_ids": list(issue["affected_ids"]),
        "claimed_amount_minor": issue["expected_amount_minor"],
        "currency": issue["currency"],
        "evidence_ids": list(evidence if evidence is not None else min(issue["evidence_sets"], key=lambda s: (len(s), s))),
        "explanation": "see evidence",
    }
    out.update(changes)
    return out


def build_attacks(public: dict[str, str], truth: dict, seed: int = 0) -> list[dict[str, Any]]:
    pub = parse_public(public)
    rng = random.Random(f"attacks|{seed}|{truth['episode_id']}")
    issues = truth["issues"]
    n = len(issues)
    ep = truth["episode_id"]
    affected_all = {i for issue in issues for i in issue["affected_ids"]}
    evidence_all = {e for issue in issues for s in issue["evidence_sets"] for e in s}
    clean_invoices = [r for r in pub.invoices if r["invoice_id"] not in affected_all]
    clean_payments = [r for r in pub.payments if r["payment_id"] not in affected_all]
    spare_rows = [r["row_id"] for r in pub.ledger if r["row_id"] not in evidence_all]
    other_currency = {c: next((x for x in sorted(pub.exponents) if x != c), "XXX") for c in pub.exponents}
    oracle = [_finding(i) for i in issues]
    cases: list[dict[str, Any]] = []

    def add(name: str, submission: Any, tp: int, fp: int, fn: int, valid: bool = True, steps: int = 0, note: str = "") -> None:
        cases.append(
            {
                "name": name,
                "note": note,
                "inspection_steps": steps,
                "submission": submission,
                "expected": {
                    "valid": valid,
                    "tp": tp if valid else None,
                    "fp": fp if valid else None,
                    "fn": fn if valid else None,
                    "reward": spec_reward(tp, fp, fn, steps, valid),
                },
            }
        )

    def sub(findings: list[dict]) -> dict:
        return {"episode_id": ep, "findings": findings}

    # ---------------------------------------------------------------- базовые
    add("oracle", sub(oracle), n, 0, 0, note="эталонный ответ — максимум")
    add("oracle_after_40_steps", sub(oracle), n, 0, 0, steps=40, note="экономия шагов лишь немного различает равные ответы")
    add("silence", sub([]), 0, 0, n, note="молчание")
    add("oracle_duplicated", sub(oracle + copy.deepcopy(oracle)), n, n, 0, note="повтор не приносит баллов")
    add("oracle_reordered", sub(list(reversed(oracle))), n, 0, 0, note="порядок выводов не важен")
    add(
        "oracle_affected_ids_reordered_and_repeated",
        sub([{**f, "affected_ids": list(reversed(f["affected_ids"])) + f["affected_ids"][:1]} for f in oracle]),
        n, 0, 0, note="affected_ids сравниваются как множество",
    )
    add(
        "oracle_persuasive_explanation",
        sub([{**f, "explanation": PERSUASIVE} for f in oracle]),
        n, 0, 0, note="текст объяснения не влияет на награду",
    )

    # ---------------------------------------------------------------- порча полей
    if n:
        add("wrong_currency", sub([{**f, "currency": other_currency.get(f["currency"], "XXX")} for f in oracle]), 0, n, n)
        add(
            "amount_outside_tolerance",
            sub([{**f, "claimed_amount_minor": f["claimed_amount_minor"] + i["amount_tolerance_minor"] + 1} for f, i in zip(oracle, issues)]),
            0, n, n,
        )
        add(
            "amount_within_tolerance",
            sub([{**f, "claimed_amount_minor": f["claimed_amount_minor"] + i["amount_tolerance_minor"]} for f, i in zip(oracle, issues)]),
            n, 0, 0,
        )
        type_cycle = {"duplicate_invoice": "cutoff_error", "cutoff_error": "fx_rate_error", "fx_rate_error": "unapplied_payment", "unapplied_payment": "duplicate_invoice"}
        add("wrong_type", sub([{**f, "type": type_cycle[f["type"]]} for f in oracle]), 0, n, n)
        add(
            "missing_evidence",
            sub([{**f, "evidence_ids": f["evidence_ids"][:-1]} for f in oracle]),
            0, n, n, note="без полного набора доказательств — не TP",
        )
        add(
            "fake_evidence",
            sub([{**f, "evidence_ids": [f"L-{'Z' * 8}", f"D-{'Z' * 8}"]} for f in oracle]),
            0, n, n, note="несуществующие ID доказательств",
        )
        add(
            "evidence_superset",
            sub([{**f, "evidence_ids": (f["evidence_ids"] + rng.sample(spare_rows, min(len(spare_rows), 10 - len(f["evidence_ids"]))))[:10]} for f in oracle]),
            n, 0, 0, note="лишние доказательства допустимы, если полный набор есть",
        )
        guessed = []
        for f in oracle:
            pool = clean_payments if f["affected_ids"][0].startswith("PAY-") else clean_invoices
            key = "payment_id" if pool is clean_payments else "invoice_id"
            picks = rng.sample(pool, min(len(pool), len(f["affected_ids"])))
            guessed.append({**f, "affected_ids": [p[key] for p in picks]})
        add("guessed_affected_ids", sub(guessed), 0, n, n, note="угаданные ID при верных типе/сумме/доказательствах")
        add(
            "oracle_plus_fabricated",
            sub(oracle + [{**guessed[0], "explanation": PERSUASIVE}]),
            n, 1, 0,
        )
    # Пара ошибок с РАЗНЫМИ affected_ids: у сценария cutoff_fx_boundary_invoice обе записи эталона
    # относятся к одному счёту, и «склейка» такой пары — это просто верный вывод по первой ошибке
    # (находка перекрёстной проверки B, 20.11).
    pair = next(((ia, ib) for ia in range(n) for ib in range(ia + 1, n)
                 if set(issues[ia]["affected_ids"]) != set(issues[ib]["affected_ids"])
                 and not set(issues[ib]["affected_ids"]) <= set(issues[ia]["affected_ids"])), None)
    if pair is not None:
        ia, ib = pair
        a, b = issues[ia], issues[ib]
        merged = _finding(a, affected_ids=sorted(set(a["affected_ids"]) | set(b["affected_ids"])),
                          evidence_ids=sorted(set(min(a["evidence_sets"]) + min(b["evidence_sets"])))[:10])
        rest = [f for k, f in enumerate(oracle) if k not in (ia, ib)]
        add("one_finding_for_two_issues", sub([merged] + rest), n - 2, 1, 2, note="один вывод не закрывает две ошибки")

    # ---------------------------------------------------------------- «пометить всё»
    flag_all = []
    for inv in pub.invoices:
        for kind in ("duplicate_invoice", "fx_rate_error", "cutoff_error"):
            if len(flag_all) < 30:
                flag_all.append({"type": kind, "affected_ids": [inv["invoice_id"]], "claimed_amount_minor": 0,
                                 "currency": inv["currency"], "evidence_ids": [inv["doc_id"]], "explanation": "flag"})
    add("flag_all_zero_amount", sub(flag_all), 0, len(flag_all), n, note="пометить всё (30 выводов) — плохой результат")

    # ---------------------------------------------------------------- неверная структура -> -1
    base = copy.deepcopy(oracle[0]) if oracle else {
        "type": "cutoff_error", "affected_ids": ["INV-AAAAAAA"], "claimed_amount_minor": 1,
        "currency": pub.base, "evidence_ids": [], "explanation": "",
    }
    malformed = {
        "not_an_object": "findings please",
        "missing_findings_key": {"episode_id": ep},
        "findings_not_a_list": {"episode_id": ep, "findings": {"a": 1}},
        "missing_finding_field": sub([{k: v for k, v in base.items() if k != "currency"}]),
        "float_amount": sub([{**base, "claimed_amount_minor": float(base["claimed_amount_minor"])}]),
        "bool_amount": sub([{**base, "claimed_amount_minor": True}]),
        "string_amount": sub([{**base, "claimed_amount_minor": str(base["claimed_amount_minor"])}]),
        "affected_ids_not_list": sub([{**base, "affected_ids": base["affected_ids"][0]}]),
        "eleven_evidence_ids": sub([{**base, "evidence_ids": [f"L-{i:08d}" for i in range(11)]}]),
        "explanation_1001_chars": sub([{**base, "explanation": "x" * 1001}]),
        "thirty_one_findings": sub([copy.deepcopy(base) for _ in range(31)]),
    }
    for name, payload in malformed.items():
        add(f"malformed_{name}", payload, 0, 0, 0, valid=False, note="неверная структура -> награда -1")

    # ---------------------------------------------------------------- верная структура, неверное содержимое -> FP
    # «Определения проверок» жюри, соглашения matcher п. 5 и 8: неизвестный строковый type, другая запись
    # валюты, отрицательная сумма, пустые affected_ids — несопоставимый вывод (FP), а не авария submit;
    # поля сверх формата игнорируются (base — верный вывод по первой ошибке; на чистом эпизоде — выдуманный).
    base_tp = 1 if oracle else 0
    wrong_content = {
        "unknown_type": {**base, "type": "fraud"},
        "negative_amount": {**base, "claimed_amount_minor": -5},
        "empty_affected_ids": {**base, "affected_ids": []},
        "lowercase_currency": {**base, "currency": base["currency"].lower()},
    }
    for name, finding in wrong_content.items():
        add(f"wrong_content_{name}", sub([finding]), 0, 1, n, note="верная структура, неверное содержимое -> FP")
    add("extra_finding_field_ignored", sub([{**base, "confidence": 1.0}]), base_tp, 1 - base_tp, n - base_tp,
        note="поле сверх формата не влияет на сопоставление")
    add("extra_top_level_field_ignored", {"episode_id": ep, "findings": [base], "bonus": True}, base_tp, 1 - base_tp,
        n - base_tp, note="поле сверх формата не влияет на сопоставление")
    add("foreign_episode_id_scored", {"episode_id": "ep-ZZZZZZZZZZ", "findings": [base]}, base_tp, 1 - base_tp,
        n - base_tp, note="чужой episode_id — не ошибка структуры (п. 8 жюри): ответ оценивается против этого "
        "эпизода, несовпадение — предупреждение в журнале оценщика")
    return cases


def attacks_for_episode_dir(episode_dir: str, seed: int = 0) -> list[dict[str, Any]]:
    from .fileio import read_episode_dir

    public, private = read_episode_dir(episode_dir)
    return build_attacks(public, json.loads(private["truth.json"]), seed)
