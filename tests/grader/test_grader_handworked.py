"""Ручные примеры matcher/награды: ответ посчитан вручную по формулам ТЗ (раздел 05), не кодом.

Зачем: тесты свойств и атак писались вместе с оценщиком и могут разделять его слепые зоны. Здесь
каждое ожидаемое число выведено руками в комментарии (дробью) из текста ТЗ:
  TP = верный type + те же affected_ids (как множество) + сумма в допуске + валюта + хотя бы один
       полный допустимый набор доказательств; повтор и несопоставленные — FP; пропущенные — FN;
  P = TP/(TP+FP), R = TP/(TP+FN) (нулевой знаменатель -> 0), F1 = 2TP/(2TP+FP+FN);
  пустой эталон и пустой ответ -> F1 = 1; reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1).

Посмотреть глазами (таблица «вручную | программа»):
    python tests/grader/test_grader_handworked.py --show
Свой ответ против своего эталона: python -m grader --truth T.json --submission S.json --steps N
"""

import json
import sys
import tempfile
import unittest
from fractions import Fraction as Fr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grader import score_submission  # noqa: E402


def issue(iid, kind, ids, amount, cur, tol, evidence_sets):
    return {"issue_id": iid, "type": kind, "affected_ids": ids, "expected_amount_minor": amount, "currency": cur,
            "amount_tolerance_minor": tol, "evidence_sets": evidence_sets, "scenario": None, "scenario_group": None}


def F(kind, ids, amount, cur, evidence, explanation="hand"):
    return {"type": kind, "affected_ids": ids, "claimed_amount_minor": amount, "currency": cur,
            "evidence_ids": evidence, "explanation": explanation}


FX = "FX:USD:KZT:2026-03-02"
# Эталон из трёх ошибок:
#   I1 дубль счёта INV-A/INV-B, 50000 KZT, допуск 0, доказательство {D-A, D-B}
#   I2 курс по платежу PAY-C, 1200 KZT, допуск 1, доказательство {D-C, FX, L-7}
#   I3 cut-off по счёту INV-E, 9900 EUR, допуск 0, доказательство {D-E, L-1} ИЛИ {D-E, L-2}
T = {"episode_id": "ep-HAND000001", "issues": [
    issue("ISS-HAND1", "duplicate_invoice", ["INV-A", "INV-B"], 50000, "KZT", 0, [["D-A", "D-B"]]),
    issue("ISS-HAND2", "fx_rate_error", ["PAY-C"], 1200, "KZT", 1, [["D-C", FX, "L-7"]]),
    issue("ISS-HAND3", "cutoff_error", ["INV-E"], 9900, "EUR", 0, [["D-E", "L-1"], ["D-E", "L-2"]]),
]}
OK1 = F("duplicate_invoice", ["INV-A", "INV-B"], 50000, "KZT", ["D-A", "D-B"])
OK2 = F("fx_rate_error", ["PAY-C"], 1200, "KZT", ["D-C", FX, "L-7"])
OK3 = F("cutoff_error", ["INV-E"], 9900, "EUR", ["D-E", "L-1"])
# Два одинаковых по типу и ID расхождения с разными суммами — ловушка для жадного сопоставления.
T2 = {"episode_id": "ep-HAND000002", "issues": [
    issue("ISS-J1", "fx_rate_error", ["PAY-X"], 1000, "KZT", 5, [["D-X"]]),
    issue("ISS-J2", "fx_rate_error", ["PAY-X"], 1008, "KZT", 5, [["D-X"]]),
]}
T0 = {"episode_id": "ep-HAND000000", "issues": []}  # чистая книга

# (название, эталон, выводы, inspection_steps, ожидание вручную: tp, fp, fn, P, R, F1, reward)
EXAMPLES = [
    # Всё верно; порядок ID переставлен; сумма I2 на 1 выше (= допуск); у I3 второй набор + лишний ID.
    # TP 3, FP 0, FN 0. P = R = F1 = 1. reward = 2*1 - 1 - 0.002*12 = 1 - 0.024 = 0.976 = 122/125.
    ("1. perfect answer, 12 steps", T,
     [F("duplicate_invoice", ["INV-B", "INV-A"], 50000, "KZT", ["D-B", "D-A"]),
      F("fx_rate_error", ["PAY-C"], 1201, "KZT", ["D-C", FX, "L-7"]),
      F("cutoff_error", ["INV-E"], 9900, "EUR", ["L-1000", "L-2", "D-E"])],
     12, (3, 0, 0, Fr(1), Fr(1), Fr(1), Fr(122, 125))),
    # f1 = I1 верно (TP); f2 = копия f1 (повтор -> FP); f3 = I2 с суммой 1202 (|2| > допуска 1 -> FP);
    # f4 = I3 в KZT вместо EUR (FP). I2 и I3 не закрыты -> FN 2.
    # P = 1/4, R = 1/3, F1 = 2/(2+3+2) = 2/7. reward = 4/7 - 1 - 0.06 = -3/7 - 3/50 = -171/350 ≈ -0.4886.
    ("2. duplicate + amount off + wrong currency, 30 steps", T,
     [OK1, dict(OK1), {**OK2, "claimed_amount_minor": 1202}, {**OK3, "currency": "KZT"}],
     30, (1, 3, 2, Fr(1, 4), Fr(1, 3), Fr(2, 7), Fr(-171, 350))),
    # f1 = I1 без D-B (набор неполный -> FP); f2 = I2 верно (TP); f3 = I3 только с D-E (неполный -> FP).
    # TP 1, FP 2, FN 2. P = 1/3, R = 1/3, F1 = 2/(2+2+2) = 1/3. reward = 2/3 - 1 = -1/3.
    ("3. incomplete evidence, 0 steps", T,
     [{**OK1, "evidence_ids": ["D-A", "L-99"]}, OK2, {**OK3, "evidence_ids": ["D-E"]}],
     0, (1, 2, 2, Fr(1, 3), Fr(1, 3), Fr(1, 3), Fr(-1, 3))),
    # f1 = I1 только с INV-A (подмножество -> FP); f2 = I1 с лишним INV-Z (надмножество -> FP);
    # f3 = I2 под типом unapplied_payment (FP); f4 = I3 верно (TP). FN = I1, I2.
    # P = 1/4, R = 1/3, F1 = 2/7. reward = 4/7 - 1 - 0.01 = -3/7 - 1/100 = -307/700 ≈ -0.4386.
    ("4. subset/superset IDs + wrong type, 5 steps", T,
     [{**OK1, "affected_ids": ["INV-A"]}, {**OK1, "affected_ids": ["INV-A", "INV-B", "INV-Z"]},
      {**OK2, "type": "unapplied_payment"}, OK3],
     5, (1, 3, 2, Fr(1, 4), Fr(1, 3), Fr(2, 7), Fr(-307, 700))),
    # g1 = 1004 подходит к обеим (|4| <= 5), g2 = 997 только к J1 (|3| <= 5, |11| > 5).
    # Жадно (g1 -> J1) получится 1 TP; правильно g1 -> J2, g2 -> J1: TP 2, FP 0, FN 0, reward 1.
    ("5. non-greedy matching (order g1, g2)", T2,
     [F("fx_rate_error", ["PAY-X"], 1004, "KZT", ["D-X"]), F("fx_rate_error", ["PAY-X"], 997, "KZT", ["D-X"])],
     0, (2, 0, 0, Fr(1), Fr(1), Fr(1), Fr(1))),
    ("5b. the same, order g2, g1", T2,
     [F("fx_rate_error", ["PAY-X"], 997, "KZT", ["D-X"]), F("fx_rate_error", ["PAY-X"], 1004, "KZT", ["D-X"])],
     0, (2, 0, 0, Fr(1), Fr(1), Fr(1), Fr(1))),
    # Чистая книга, молчание: TP=FP=FN=0 -> P = R = 0 (нулевые знаменатели), F1 = 1 по особому правилу.
    # reward = 1 - 0.002*7 = 0.986 = 493/500.
    ("6. clean books, silence, 7 steps", T0, [], 7, (0, 0, 0, Fr(0), Fr(0), Fr(1), Fr(493, 500))),
    # Чистая книга, одно ложное обвинение: FP 1 -> P = 0/1 = 0, R = 0 (знаменатель 0), F1 = 0/1 = 0.
    # reward = clip(-1 - 0.014) = -1.
    ("7. clean books, one false accusation", T0, [OK3], 7, (0, 1, 0, Fr(0), Fr(0), Fr(0), Fr(-1))),
    # Книга с ошибками, молчание 99 шагов: FN 3 -> P = 0, R = 0/3 = 0, F1 = 0; reward = clip(-1.198) = -1.
    ("8. books with errors, silence, 99 steps", T, [], 99, (0, 0, 3, Fr(0), Fr(0), Fr(0), Fr(-1))),
    # Тот же полный ответ за 99 шагов: 1 - 0.198 = 0.802 = 401/500 > -1 (полный ответ лучше молчания).
    ("9. full answer, 99 steps", T, [OK1, OK2, OK3], 99, (3, 0, 0, Fr(1), Fr(1), Fr(1), Fr(401, 500))),
]

# Неверная структура -> reward -1 (даже если внутри верные выводы). Вручную: valid = False, reward = -1.
INVALID = [
    ("extra field in a finding", {"episode_id": T["episode_id"], "findings": [OK1, OK2, {**OK3, "confidence": 0.9}]}),
    ("other episode_id", {"episode_id": "ep-HAND999999", "findings": [OK1, OK2, OK3]}),
    ("31 findings", {"episode_id": T["episode_id"], "findings": [OK1] * 31}),
    ("explanation of 1001 characters", {"episode_id": T["episode_id"], "findings": [{**OK1, "explanation": "x" * 1001}]}),
    ("11 evidence ids", {"episode_id": T["episode_id"], "findings": [{**OK1, "evidence_ids": [f"L-{k}" for k in range(11)]}]}),
    ("amount as a float", {"episode_id": T["episode_id"], "findings": [{**OK1, "claimed_amount_minor": 50000.0}]}),
    ("explanation missing", {"episode_id": T["episode_id"], "findings": [{k: v for k, v in OK1.items() if k != "explanation"}]}),
]


def greedy_tp(findings, issues):
    """Наивное жадное сопоставление (для доказательства, что пример 5 — настоящая ловушка)."""
    used, tp = set(), 0
    for f in findings:
        for k, i in enumerate(issues):
            if k not in used and f["type"] == i["type"] and set(f["affected_ids"]) == set(i["affected_ids"]) \
                    and abs(f["claimed_amount_minor"] - i["expected_amount_minor"]) <= i["amount_tolerance_minor"] \
                    and f["currency"] == i["currency"] and any(set(s) <= set(f["evidence_ids"]) for s in i["evidence_sets"]):
                used.add(k)
                tp += 1
                break
    return tp


def run(truth, findings, steps):
    return score_submission(truth, {"episode_id": truth["episode_id"], "findings": findings}, steps)


class HandWorked(unittest.TestCase):
    def test_examples(self):
        for name, truth, findings, steps, (tp, fp, fn, p, r, f1, reward) in EXAMPLES:
            with self.subTest(name):
                got = run(truth, findings, steps)
                self.assertTrue(got["valid"], got["errors"])
                self.assertEqual((got["tp"], got["fp"], got["fn"]), (tp, fp, fn))
                self.assertEqual((got["precision"], got["recall"], got["f1"]), (float(p), float(r), float(f1)))
                self.assertEqual(got["reward"], float(reward))

    def test_example_5_is_a_real_greedy_trap(self):
        self.assertEqual(greedy_tp(EXAMPLES[4][2], T2["issues"]), 1)

    def test_explanation_of_1000_characters_is_fine(self):
        got = run(T, [{**OK1, "explanation": "я" * 1000}], 0)
        self.assertTrue(got["valid"])
        self.assertEqual(got["f1"], float(Fr(2, 4)))  # TP 1, FN 2 -> 2/(2+0+2) = 1/2

    def test_invalid_structure(self):
        for name, submission in INVALID:
            with self.subTest(name):
                got = score_submission(T, submission, 0)
                self.assertFalse(got["valid"])
                self.assertEqual(got["reward"], -1.0)

    def test_end_to_end_through_the_isolated_env(self):
        """Агрегированный путь: процессы среды и оценщика, стаб-эпизод, ответ составлен по его truth.json руками.

        Выводы: cut-off INV-JM96D6D верно (второй набор доказательств) -> TP; дубль INV-PK57H2Q/INV-VANY3VP
        верно -> TP; курс PAY-5ZX46M2 на 119358 вместо 119356 (|2| > допуска 1) -> FP; остальные 4 -> FN.
        3 шага исследования + submit: inspection_steps = 3.
        F1 = 2*2/(4+1+4) = 4/9; reward = 8/9 - 1 - 0.006 = -1/9 - 3/500 = -527/4500 ≈ -0.11711.
        """
        from env import AuditEnv

        findings = [
            F("cutoff_error", ["INV-JM96D6D"], 868788, "EUR", ["D-PCS65XK3", "L-T7TNRSHX"]),
            F("duplicate_invoice", ["INV-VANY3VP", "INV-PK57H2Q"], 261798, "EUR", ["D-YVEFVQS7", "D-YJ5KT9VK"]),
            F("fx_rate_error", ["PAY-5ZX46M2"], 119358, "KZT", ["D-Z3TU57FE", "FX:USD:KZT:2026-03-06", "L-2UKNUG5J"]),
        ]
        expected = float(Fr(-527, 4500))
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "grader.jsonl"
            with AuditEnv(grader_log=log) as env:
                for mode in ("train", "eval"):
                    obs, _ = env.reset(seed=0, config={"split": "stub", "mode": mode})
                    self.assertEqual(obs["episode_id"], "ep-SUW6BEVZNR")
                    env.step({"action": "list_records", "table": "ledger", "limit": 5})
                    env.step({"action": "get_document", "doc_id": "D-PCS65XK3"})
                    env.step({"action": "calculator", "expression": "868788 / 100"})
                    _, reward, term, trunc, info = env.step(
                        {"action": "submit", "findings": {"episode_id": obs["episode_id"], "findings": findings}})
                    self.assertTrue(term)
                    if mode == "train":
                        self.assertEqual(reward, expected)
                        self.assertEqual((info["tp"], info["fp"], info["fn"], info["inspection_steps"]), (2, 1, 4, 3))
                    else:
                        self.assertEqual((reward, info), (0.0, {}))  # в eval политика ничего не видит
            records = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(r["mode"], r["reward"], r["tp"], r["fp"], r["fn"]) for r in records],
                         [("train", expected, 2, 1, 4), ("eval", expected, 2, 1, 4)])


def show() -> None:
    def fr(x: Fr) -> str:
        return str(x) if x.denominator != 1 else str(x.numerator)

    print("| example | by hand: TP/FP/FN | program | by hand: P, R, F1 | program | by hand: reward | program | |")
    print("|---|---|---|---|---|---|---|---|")
    for name, truth, findings, steps, (tp, fp, fn, p, r, f1, reward) in EXAMPLES:
        got = run(truth, findings, steps)
        same = (got["tp"], got["fp"], got["fn"]) == (tp, fp, fn) and got["f1"] == float(f1) and got["reward"] == float(reward)
        print(f"| {name} | {tp}/{fp}/{fn} | {got['tp']}/{got['fp']}/{got['fn']} | {fr(p)}, {fr(r)}, {fr(f1)} | "
              f"{got['precision']:.4f}, {got['recall']:.4f}, {got['f1']:.4f} | {fr(reward)} = {float(reward):.6f} | "
              f"{got['reward']:.6f} | {'OK' if same else 'MISMATCH'} |")
    for name, submission in INVALID:
        got = score_submission(T, submission, 0)
        print(f"| invalid: {name} | -1 | {got['reward']} ({got['errors'][0] if got['errors'] else 'valid?!'}) | | | | | "
              f"{'OK' if got['reward'] == -1.0 and not got['valid'] else 'MISMATCH'} |")


if __name__ == "__main__":
    if "--show" in sys.argv:
        show()
    else:
        unittest.main()
