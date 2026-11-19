"""Враждебный ввод: ни одно действие агента не роняет step, а итог одинаков в процессе и через изоляцию.

Правило (docs/contract.md, раздел 8): объект с action == "submit" (или JSON-строка такого объекта)
всегда завершает эпизод; если он не JSON (NaN, целое длиннее 4300 цифр) или больше любого валидного
submit (> MAX_ACTION_BYTES), это неверная структура: награда -1 и причина в журнале оценщика.
Прочие нечитаемые действия дают ограниченную ошибку и расходуют шаг.
"""

import json
import unittest

from env_helpers import STUB, local_env
from env import AuditEnv
from env.constants import ERRORS, MAX_ACTION_BYTES, MAX_PARSE_BYTES
from grader.validate import MALFORMED_REASONS

BIG = "9" * 5000  # больше sys.get_int_max_str_digits() по умолчанию
BIG_INT = 10 ** 5000  # такой int есть в Python, но str()/json.dumps на нём падают


def finding(**kw):
    f = {"type": "fx_rate_error", "affected_ids": ["PAY-1"], "claimed_amount_minor": 1, "currency": "KZT",
         "evidence_ids": [], "explanation": ""}
    f.update(kw)
    return f


TOOL_CASES = [
    ("string with 5000-digit integer", '{"action": "calculator", "expression": "1", "x": ' + BIG + "}", "invalid_action"),
    ("string with NaN", '{"action": "calculator", "expression": "1", "x": NaN}', "invalid_action"),
    ("string with Infinity", '{"action": "get_document", "doc_id": Infinity}', "invalid_action"),
    ("deeply nested string", "[" * 100_000 + "]" * 100_000, "invalid_action"),
    ("deeply nested object", None, "invalid_action"),  # строится в тесте
    ("python int above parse limit", {"action": "calculator", "expression": "1", "x": BIG_INT}, "invalid_action"),
    ("float nan argument", {"action": "list_records", "table": "ledger", "limit": float("nan")}, "invalid_action"),
    ("oversize tool action", {"action": "calculator", "expression": "1" * (MAX_ACTION_BYTES + 10)}, "action_too_large"),
    ("string above parse limit", "x" * (MAX_PARSE_BYTES + 1), "action_too_large"),
    ("bytes", b'{"action": "calculator", "expression": "1"}', "invalid_action"),
    ("set", {"action", "calculator"}, "invalid_action"),
    ("mixed key types (sort_keys fails)", {1: "a", "action": "calculator", "expression": "1"}, "invalid_action"),
    ("int keys only in nested object", {"action": "calculator", "expression": "1", "x": {1: "a", 2: "b"}}, "unexpected_argument"),
]


class HostileInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = {"local": local_env(), "isolated": AuditEnv()}

    @classmethod
    def tearDownClass(cls):
        for env in cls.envs.values():
            env.close()

    def run_both(self, action, mode="train"):
        out = {}
        for name, env in self.envs.items():
            obs, info = env.reset(seed=0, config={**STUB, "mode": mode})
            out[name] = (obs["episode_id"], env.step(action))
        return out

    def test_tool_actions_give_bounded_errors_and_consume_a_step(self):
        deep = {"action": "calculator", "expression": "1"}
        for _ in range(50_000):
            deep = {"a": deep}
        for label, action, code in TOOL_CASES:
            with self.subTest(case=label):
                results = self.run_both(deep if action is None else action)
                for name, (_, (obs, reward, term, trunc, info)) in results.items():
                    self.assertEqual(obs["error"]["code"], code, name)
                    self.assertEqual(obs["error"]["message"], ERRORS[code])
                    self.assertEqual((obs["result"], reward, term, trunc, info), (None, 0.0, False, False, {}))
                    self.assertEqual(obs["remaining_steps"], 99)

    def test_unreadable_submit_always_terminates_with_minus_one(self):
        cases = {
            "nan amount": ({"action": "submit", "findings": {"findings": [finding(claimed_amount_minor=float("nan"))]}}, "not_json"),
            "5000-digit amount": ({"action": "submit", "findings": {"findings": [finding(claimed_amount_minor=BIG_INT)]}}, "not_json"),
            "huge explanation": ({"action": "submit", "findings": {"findings": [finding(explanation="x" * 300_000)]}}, "too_large"),
            "huge explanation as JSON string": (None, "too_large"),
            "5000-digit amount as JSON string": (None, "not_json"),
            "extra wrapper field": ({"action": "submit", "findings": {"findings": []}, "confidence": 1}, "wrapper"),
            "missing findings": ({"action": "submit"}, "wrapper"),
        }
        for label, (action, reason) in cases.items():
            with self.subTest(case=label):
                for name, env in self.envs.items():
                    obs, _ = env.reset(seed=0, config={**STUB, "mode": "train"})
                    ep = obs["episode_id"]
                    if label == "huge explanation as JSON string":
                        action = json.dumps({"action": "submit", "findings": {"episode_id": ep, "findings": [finding(explanation="x" * 300_000)]}})
                    if label == "5000-digit amount as JSON string":
                        action = '{"action": "submit", "findings": {"episode_id": "%s", "findings": [{"type": "fx_rate_error", ' \
                                 '"affected_ids": ["PAY-1"], "claimed_amount_minor": %s, "currency": "KZT", "evidence_ids": [], ' \
                                 '"explanation": ""}]}}' % (ep, BIG)
                    obs, reward, term, trunc, info = env.step(action)
                    self.assertEqual((reward, term, trunc), (-1.0, True, False), name)
                    self.assertIsNone(obs["error"])
                    self.assertFalse(info["valid"])
                    self.assertEqual(info["errors"], [f"submit action: {MALFORMED_REASONS[reason]}"], name)
                    self.assertEqual(info["inspection_steps"], 0)
                    after = env.step({"action": "calculator", "expression": "1"})
                    self.assertEqual(after[0]["error"]["code"], "episode_over")

    def test_eval_mode_hides_even_malformed_submit(self):
        results = self.run_both({"action": "submit", "findings": {"findings": [finding(explanation="x" * 300_000)]}}, mode="eval")
        for name, (_, (obs, reward, term, trunc, info)) in results.items():
            self.assertEqual((reward, term, trunc, info), (0.0, True, False, {}), name)

    def test_local_and_isolated_outcomes_are_identical(self):
        actions = [a for _, a, _ in TOOL_CASES if a is not None and not isinstance(a, (bytes, set))]
        for action in actions:
            results = self.run_both(action)
            self.assertEqual(json.dumps(results["local"], sort_keys=True), json.dumps(results["isolated"], sort_keys=True))


if __name__ == "__main__":
    unittest.main()
