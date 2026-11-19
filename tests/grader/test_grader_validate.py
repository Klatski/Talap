"""Строгая валидация submit: неверная структура -> -1; согласие с schemas/findings.schema.json."""

import copy
import random
import unittest

from grader_helpers import finding_for, issue, sub, truth_of
from generator.schema_check import load_schema, validate
from grader import score_submission, validate_submission

EP = "ep-TESTTEST00"
GOOD = finding_for(issue())


class ValidateTests(unittest.TestCase):
    def test_valid_shapes(self):
        for payload in (sub([]), sub([GOOD]), sub([GOOD] * 30), sub([{**GOOD, "evidence_ids": [], "explanation": ""}]),
                        sub([{**GOOD, "affected_ids": ["X"] * 10, "evidence_ids": ["Y"] * 10, "explanation": "z" * 1000}]),
                        sub([{**GOOD, "claimed_amount_minor": 0}]), sub([{**GOOD, "claimed_amount_minor": 10**30}])):
            with self.subTest(n=len(payload["findings"])):
                self.assertEqual(validate_submission(payload, EP), [])

    def test_invalid_shapes(self):
        cases = {
            "not object": "findings", "list": [GOOD], "null": None, "number": 3,
            "missing findings": {"episode_id": EP}, "missing episode": {"findings": []},
            "extra top": {**sub([]), "bonus": 1}, "findings not list": {"episode_id": EP, "findings": {}},
            "wrong episode": sub([], episode_id="ep-OTHER00000"), "episode not str": {"episode_id": 5, "findings": []},
            "empty episode": {"episode_id": "", "findings": []}, "31 findings": sub([GOOD] * 31),
            "finding not object": sub(["x"]), "extra field": sub([{**GOOD, "confidence": 1}]),
            "missing field": sub([{k: v for k, v in GOOD.items() if k != "evidence_ids"}]),
            "unknown type": sub([{**GOOD, "type": "fraud"}]), "type not str": sub([{**GOOD, "type": 1}]),
            "negative": sub([{**GOOD, "claimed_amount_minor": -1}]), "float": sub([{**GOOD, "claimed_amount_minor": 100.0}]),
            "bool": sub([{**GOOD, "claimed_amount_minor": True}]), "str amount": sub([{**GOOD, "claimed_amount_minor": "100"}]),
            "currency lower": sub([{**GOOD, "currency": "kzt"}]), "currency 4": sub([{**GOOD, "currency": "KZTT"}]),
            "currency newline": sub([{**GOOD, "currency": "KZT\n"}]), "currency null": sub([{**GOOD, "currency": None}]),
            "affected empty": sub([{**GOOD, "affected_ids": []}]), "affected str": sub([{**GOOD, "affected_ids": "INV-1"}]),
            "affected 11": sub([{**GOOD, "affected_ids": ["X"] * 11}]), "affected int": sub([{**GOOD, "affected_ids": [1]}]),
            "affected long": sub([{**GOOD, "affected_ids": ["X" * 65]}]), "affected empty str": sub([{**GOOD, "affected_ids": [""]}]),
            "evidence 11": sub([{**GOOD, "evidence_ids": ["E"] * 11}]), "evidence null": sub([{**GOOD, "evidence_ids": None}]),
            "evidence int": sub([{**GOOD, "evidence_ids": [5]}]), "explanation 1001": sub([{**GOOD, "explanation": "x" * 1001}]),
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

    def test_errors_do_not_echo_values(self):
        secret = "ZZ-ECHO-PROBE"
        errors = validate_submission({"episode_id": secret, "findings": [{**GOOD, "type": secret, "bonus": secret}]}, EP)
        self.assertTrue(errors)
        self.assertNotIn(secret, " ".join(errors))

    def test_agrees_with_json_schema_on_fuzzed_submissions(self):
        """Наш валидатор строже схемы только в одном: '$' в схеме пропускает 'KZT\\n' (re.search)."""
        schema = load_schema("findings")
        rng = random.Random(5)
        mutations = [
            lambda f: f.pop(rng.choice(sorted(f))),
            lambda f: f.update({rng.choice(sorted(f)): rng.choice([None, 1, -1, 1.5, True, "", "x", [], {}, ["a"], [1]])}),
            lambda f: f.update({"extra": 1}),
            lambda f: f.update({"affected_ids": ["A"] * rng.randint(0, 12)}),
            lambda f: f.update({"evidence_ids": ["E" * rng.randint(0, 70)] * rng.randint(0, 12)}),
            lambda f: f.update({"currency": rng.choice(["USD", "usd", "US", "USDX", "ÜSD", "123"])}),
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
