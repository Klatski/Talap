"""Валидация submit по правилам жюри: структура -> -1, неверное содержимое -> FP; согласие со schemas/findings.schema.json."""

import copy
import random
import unittest

from grader_helpers import finding_for, issue, sub, truth_of
from generator.schema_check import load_schema, validate
from grader import score_submission, validate_submission

EP = "ep-TESTTEST00"
GOOD = finding_for(issue())

# Структурно правильные выводы с неверным содержимым: не ошибка структуры, а FP (п. 8 соглашений matcher).
WRONG_CONTENT = {
    "unknown type": {"type": "fraud"},
    "negative amount": {"claimed_amount_minor": -1},
    "currency lower (M06)": {"currency": "kzt"},
    "currency with space (M06)": {"currency": "KZT "},
    "currency 4 letters": {"currency": "KZTT"},
    "currency newline": {"currency": "KZT\n"},
    "affected empty": {"affected_ids": []},
    "affected 11 ids": {"affected_ids": [f"X{i}" for i in range(11)]},
    "affected long id": {"affected_ids": ["X" * 500]},
    "affected empty string": {"affected_ids": [""]},
    "evidence unknown id (M04)": {"evidence_ids": GOOD["evidence_ids"] + ["U-NOT-IN-EPISODE"]},
}


class ValidateTests(unittest.TestCase):
    def test_valid_shapes(self):
        payloads = [sub([]), sub([GOOD]), sub([GOOD] * 30), sub([{**GOOD, "evidence_ids": [], "explanation": ""}]),
                    sub([{**GOOD, "affected_ids": ["X"] * 10, "evidence_ids": ["Y"] * 10, "explanation": "z" * 1000}]),
                    sub([{**GOOD, "claimed_amount_minor": 0}]), sub([{**GOOD, "claimed_amount_minor": 10**30}]),
                    sub([{**GOOD, "explanation": "\U0001F600" * 1000}]),   # 1000 code points, не байтов
                    {**sub([GOOD]), "bonus": 1}, sub([{**GOOD, "confidence": 0.9}])]  # лишние поля игнорируются
        payloads += [sub([{**GOOD, **change}]) for change in WRONG_CONTENT.values()]
        for k, payload in enumerate(payloads):
            with self.subTest(k=k):
                self.assertEqual(validate_submission(payload, EP), [])

    def test_foreign_episode_id_is_scored_with_a_warning(self):
        """п. 8 соглашений жюри: структурная ошибка — тип значения, нет поля, лимит. Строка episode_id с другим
        значением — не она: ответ оценивается против текущего эпизода, в журнале — предупреждение."""
        truth = truth_of(issue())
        known = frozenset(GOOD["evidence_ids"] + GOOD["affected_ids"])
        right = score_submission(truth, sub([GOOD]), 7, public_ids=known)
        self.assertEqual(right["warnings"], [])
        keys = ("valid", "tp", "fp", "fn", "precision", "recall", "f1", "reward")
        for other in ("ep-OTHER00000", "", "EP-TESTTEST00", EP + " "):
            with self.subTest(episode_id=other):
                self.assertEqual(validate_submission(sub([GOOD], episode_id=other), EP), [])
                result = score_submission(truth, sub([GOOD], episode_id=other), 7, public_ids=known)
                self.assertEqual({k: result[k] for k in keys}, {k: right[k] for k in keys})
                self.assertEqual(len(result["warnings"]), 1)
                if other:
                    self.assertNotIn(other, result["warnings"][0])  # без эха значения

    def test_wrong_content_is_a_false_positive_not_a_structure_error(self):
        truth = truth_of(issue())
        known = frozenset(GOOD["evidence_ids"] + GOOD["affected_ids"])
        for name, change in WRONG_CONTENT.items():
            with self.subTest(case=name):
                result = score_submission(truth, sub([{**GOOD, **change}]), 0, public_ids=known)
                self.assertTrue(result["valid"])
                self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))
                self.assertEqual(result["errors"], [])
        ok = score_submission(truth, sub([GOOD]), 0, public_ids=known)
        self.assertEqual((ok["tp"], ok["fp"], ok["fn"], ok["reward"]), (1, 0, 0, 1.0))

    def test_invalid_shapes(self):
        cases = {
            "not object": "findings", "list": [GOOD], "null": None, "number": 3,
            "missing findings": {"episode_id": EP}, "missing episode": {"findings": []},
            "findings not list": {"episode_id": EP, "findings": {}},
            "episode not str": {"episode_id": 5, "findings": []}, "episode null": {"episode_id": None, "findings": []},
            "31 findings": sub([GOOD] * 31),
            "finding not object": sub(["x"]),
            "missing field": sub([{k: v for k, v in GOOD.items() if k != "evidence_ids"}]),
            "type not str": sub([{**GOOD, "type": 1}]), "type null": sub([{**GOOD, "type": None}]),
            "float (M09)": sub([{**GOOD, "claimed_amount_minor": 100.0}]),
            "bool (M09)": sub([{**GOOD, "claimed_amount_minor": True}]),
            "str amount (M09)": sub([{**GOOD, "claimed_amount_minor": "100"}]),
            "currency null": sub([{**GOOD, "currency": None}]), "currency int": sub([{**GOOD, "currency": 398}]),
            "affected str": sub([{**GOOD, "affected_ids": "INV-1"}]), "affected int": sub([{**GOOD, "affected_ids": [1]}]),
            "evidence 11 distinct": sub([{**GOOD, "evidence_ids": [f"E{i}" for i in range(11)]}]),
            "evidence 11 with repeats (M05)": sub([{**GOOD, "evidence_ids": ["E"] * 11}]),
            "evidence null": sub([{**GOOD, "evidence_ids": None}]), "evidence int": sub([{**GOOD, "evidence_ids": [5]}]),
            "explanation 1001": sub([{**GOOD, "explanation": "x" * 1001}]),
            "explanation 1001 code points": sub([{**GOOD, "explanation": "\U0001F600" * 1001}]),
            "explanation null": sub([{**GOOD, "explanation": None}]),
        }
        for name, payload in cases.items():
            with self.subTest(case=name):
                errors = validate_submission(payload, EP)
                self.assertTrue(errors)
                self.assertLessEqual(len(errors), 20)
                result = score_submission(truth_of(issue()), payload, 0)
                self.assertFalse(result["valid"])
                self.assertEqual(result["reward"], -1.0)
                self.assertEqual((result["tp"], result["fp"], result["fn"]), (None, None, None))

    def test_errors_do_not_echo_values(self):
        secret = "ZZ-ECHO-PROBE"
        errors = validate_submission({"episode_id": secret, "findings": [{**GOOD, "type": 7, "bonus": secret}]}, EP)
        self.assertTrue(errors)
        self.assertNotIn(secret, " ".join(errors))

    def test_agrees_with_json_schema_on_fuzzed_submissions(self):
        """Схема и валидатор описывают одну и ту же структуру (episode_id сверяет только оценщик)."""
        schema = load_schema("findings")
        rng = random.Random(5)
        mutations = [
            lambda f: f.pop(rng.choice(sorted(f))),
            lambda f: f.update({rng.choice(sorted(f)): rng.choice([None, 1, -1, 1.5, True, "", "x", [], {}, ["a"], [1]])}),
            lambda f: f.update({"extra": 1}),
            lambda f: f.update({"affected_ids": ["A"] * rng.randint(0, 12)}),
            lambda f: f.update({"evidence_ids": ["E" * rng.randint(0, 70)] * rng.randint(0, 12)}),
            lambda f: f.update({"currency": rng.choice(["USD", "usd", "US", "USDX", "ÜSD", "123", " USD", 840])}),
            lambda f: f.update({"claimed_amount_minor": rng.choice([0, 5, -5, 2**70, 5.0, "5", False])}),
            lambda f: f.update({"explanation": "e" * rng.choice([0, 999, 1000, 1001])}),
            lambda f: None,
        ]
        for _ in range(3000):
            findings = []
            for _ in range(rng.choice([0, 1, 2, 30, 31])):
                f = copy.deepcopy(GOOD)
                for _ in range(rng.randint(0, 2)):
                    rng.choice(mutations)(f)
                findings.append(f)
            payload = sub(findings)
            if rng.random() < 0.05:
                payload["extra"] = 1
            ours = not validate_submission(payload, EP)
            theirs = not validate(payload, schema)
            self.assertEqual(ours, theirs, payload)


if __name__ == "__main__":
    unittest.main()
