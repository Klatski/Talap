"""Враждебный ввод: ни одно действие агента не роняет step, а итог одинаков в процессе и через изоляцию.

Правило (docs/contract.md, раздел 8): объект с action == "submit" (или JSON-строка такого объекта)
всегда завершает эпизод; если он не JSON (NaN, целое длиннее 4300 цифр) или больше любого валидного
ответа (> MAX_SUBMIT_BYTES), это неверная структура: награда -1 и причина в журнале оценщика.
Порог MAX_SUBMIT_BYTES выше худшего валидного ответа, поэтому валидный ответ никогда не получает -1
из-за размера (награда не зависит от текста explanation). Прочие нечитаемые действия дают
ограниченную ошибку и расходуют шаг.
"""

import json
import subprocess
import sys
import unittest

from env_helpers import ROOT, STUB, local_env
from env import AuditEnv
from env.constants import ERRORS, MAX_ACTION_BYTES, MAX_PARSE_BYTES, MAX_SUBMIT_BYTES
from grader.validate import MALFORMED_REASONS, MAX_AFFECTED, MAX_EVIDENCE, MAX_EXPLANATION, MAX_FINDINGS, MAX_ID_LEN

BIG = "9" * 5000  # больше sys.get_int_max_str_digits() по умолчанию
BIG_INT = 10 ** 5000  # такой int есть в Python, но str()/json.dumps на нём падают
ASTRAL = "\U0001F600"  # четыре байта UTF-8, 12 байт в ASCII-экранированном JSON


def finding(**kw):
    f = {"type": "fx_rate_error", "affected_ids": ["PAY-1"], "claimed_amount_minor": 1, "currency": "KZT",
         "evidence_ids": [], "explanation": ""}
    f.update(kw)
    return f


def worst_case_valid_findings(episode_id: str) -> dict:
    """Самый «тяжёлый» ответ, который валидатор считает верной структурой."""
    one = {"type": "duplicate_invoice", "affected_ids": [ASTRAL * MAX_ID_LEN] * MAX_AFFECTED,
           "claimed_amount_minor": int("9" * 4300), "currency": "KZT",
           "evidence_ids": [ASTRAL * MAX_ID_LEN] * MAX_EVIDENCE, "explanation": ASTRAL * MAX_EXPLANATION}
    return {"episode_id": episode_id, "findings": [dict(one) for _ in range(MAX_FINDINGS)]}


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
    ("JSON string of a JSON string", json.dumps(json.dumps({"action": "calculator", "expression": "1"})), "invalid_action"),
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

    def submit_both(self, make_action):
        """make_action(episode_id) -> действие; [(имя среды, шаг submit, шаг после него)]."""
        out = []
        for name, env in self.envs.items():
            obs, _ = env.reset(seed=0, config={**STUB, "mode": "train"})
            step = env.step(make_action(obs["episode_id"]))
            after = env.step({"action": "calculator", "expression": "1"})
            out.append((name, step, after))
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
        huge = "x" * (MAX_SUBMIT_BYTES + 10)
        cases = {
            "nan amount": (lambda ep: {"action": "submit", "findings": {"episode_id": ep, "findings": [finding(claimed_amount_minor=float("nan"))]}}, "not_json"),
            "5000-digit amount": (lambda ep: {"action": "submit", "findings": {"episode_id": ep, "findings": [finding(claimed_amount_minor=BIG_INT)]}}, "not_json"),
            "5000-digit amount as JSON string": (lambda ep: '{"action": "submit", "findings": {"episode_id": "%s", "findings": [{"type": "fx_rate_error", '
                                                 '"affected_ids": ["PAY-1"], "claimed_amount_minor": %s, "currency": "KZT", "evidence_ids": [], '
                                                 '"explanation": ""}]}}' % (ep, BIG), "not_json"),
            "explanation above MAX_SUBMIT_BYTES": (lambda ep: {"action": "submit", "findings": {"episode_id": ep, "findings": [finding(explanation=huge)]}}, "too_large"),
            "same as JSON string": (lambda ep: json.dumps({"action": "submit", "findings": {"episode_id": ep, "findings": [finding(explanation=huge)]}}), "too_large"),
            "JSON string above MAX_PARSE_BYTES": (lambda ep: json.dumps({"action": "submit", "findings": {"episode_id": ep, "findings": [finding(explanation="x" * (MAX_PARSE_BYTES + 10))]}}), "too_large"),
            "extra wrapper field": (lambda ep: {"action": "submit", "findings": {"episode_id": ep, "findings": []}, "confidence": 1}, "wrapper"),
            "missing findings": (lambda ep: {"action": "submit"}, "wrapper"),
        }
        for label, (make, reason) in cases.items():
            with self.subTest(case=label):
                for name, (obs, reward, term, trunc, info), after in self.submit_both(make):
                    self.assertEqual((reward, term, trunc), (-1.0, True, False), name)
                    self.assertIsNone(obs["error"])
                    self.assertFalse(info["valid"])
                    self.assertEqual(info["errors"], [f"submit action: {MALFORMED_REASONS[reason]}"], name)
                    self.assertEqual(info["inspection_steps"], 0)
                    self.assertEqual(after[0]["error"]["code"], "episode_over")

    def test_worst_case_valid_submission_is_scored_not_rejected(self):
        """Регрессия /code-review: эмодзи в explanation не должны превращать валидный ответ в -1."""
        sample = worst_case_valid_findings("ep-AAAAAAAAAA")
        ascii_size = len(json.dumps({"action": "submit", "findings": sample}, sort_keys=True).encode())
        self.assertLess(ascii_size, MAX_SUBMIT_BYTES)
        self.assertGreater(ascii_size, MAX_ACTION_BYTES)  # старый порог отверг бы этот валидный ответ
        makers = {
            "dict": lambda ep: {"action": "submit", "findings": worst_case_valid_findings(ep)},
            "string, ASCII-escaped": lambda ep: json.dumps({"action": "submit", "findings": worst_case_valid_findings(ep)}),
            "string, raw UTF-8": lambda ep: json.dumps({"action": "submit", "findings": worst_case_valid_findings(ep)}, ensure_ascii=False),
            "string, indented": lambda ep: json.dumps({"action": "submit", "findings": worst_case_valid_findings(ep)}, indent=2),
        }
        for label, make in makers.items():
            with self.subTest(form=label):
                for name, (obs, reward, term, trunc, info), _ in self.submit_both(make):
                    self.assertTrue(term, name)
                    self.assertTrue(info["valid"], (name, info["errors"]))
                    self.assertEqual(info["errors"], [])
                    self.assertEqual(info["tp"], 0)  # разобран и сопоставлен, а не отвергнут

    def test_oversized_explanation_within_submit_limit_gets_validator_error(self):
        for name, (obs, reward, term, trunc, info), _ in self.submit_both(
                lambda ep: {"action": "submit", "findings": {"episode_id": ep, "findings": [finding(explanation="x" * 300_000)]}}):
            self.assertEqual((reward, term), (-1.0, True))
            self.assertTrue(any("explanation" in e for e in info["errors"]), (name, info["errors"]))

    def test_marker_in_agent_payload_cannot_forge_the_reason(self):
        """Регрессия /code-review: причина неверной структуры приходит только от среды, не из ответа агента."""
        for name, (obs, reward, term, trunc, info), _ in self.submit_both(
                lambda ep: {"action": "submit", "findings": {"__malformed_submit_action__": "too_large", "episode_id": ep, "findings": []}}):
            self.assertEqual((reward, term), (-1.0, True))
            self.assertIn("submission: unexpected field(s)", info["errors"], name)
            self.assertFalse(any(e.startswith("submit action:") for e in info["errors"]))

    def test_eval_mode_hides_even_malformed_submit(self):
        results = self.run_both({"action": "submit", "findings": {"findings": [finding(explanation="x" * (MAX_SUBMIT_BYTES + 1))]}}, mode="eval")
        for name, (_, (obs, reward, term, trunc, info)) in results.items():
            self.assertEqual((reward, term, trunc, info), (0.0, True, False, {}), name)

    def test_local_and_isolated_outcomes_are_identical(self):
        actions = [a for _, a, _ in TOOL_CASES if a is not None and not isinstance(a, (bytes, set))]
        for action in actions:
            results = self.run_both(action)
            self.assertEqual(json.dumps(results["local"], sort_keys=True), json.dumps(results["isolated"], sort_keys=True))


class RawJsonLinesServer(unittest.TestCase):
    """Клиент на другом языке пишет в python -m env.server что угодно."""

    def setUp(self):
        self.proc = subprocess.Popen([sys.executable, "-m", "env.server"], cwd=str(ROOT), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def tearDown(self):
        try:
            self.proc.stdin.write(b'{"id": 99, "method": "close", "params": {}}\n')
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.wait(timeout=30)
        self.proc.stdout.close()

    def ask(self, line: bytes) -> dict:
        self.proc.stdin.write(line + b"\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def test_bad_reset_does_not_abandon_the_running_episode(self):
        ok = self.ask(b'{"id": 1, "method": "reset", "params": {"seed": 0, "config": {"split": "stub"}}}')
        self.assertTrue(ok["ok"])
        for n, line in enumerate((b'{"id": 2, "method": "reset", "params": {"seed": ' + BIG.encode() + b'}}',
                                  b'{"id": 3, "method": "reset", "params": {"seed": NaN}}')):
            with self.subTest(n=n):
                bad = self.ask(line)
                self.assertFalse(bad["ok"])
                self.assertEqual(bad["error"]["code"], "reset_failed")
                self.assertNotIn("9999", json.dumps(bad))
        step = self.ask(b'{"id": 4, "method": "step", "params": {"action": {"action": "calculator", "expression": "2+2"}}}')
        self.assertEqual(step["result"]["observation"]["result"], {"value": "4"})
        self.assertEqual(step["result"]["observation"]["remaining_steps"], 99)  # эпизод жив, шаги не потеряны

    def test_forged_or_broken_unsendable_flag(self):
        self.ask(b'{"id": 1, "method": "reset", "params": {"seed": 0, "config": {"split": "stub"}}}')
        bad = [b'{"id": 2, "method": "step", "params": {"action": null, "unsendable": {"submit": "yes", "reason": "too_large"}}}',
               b'{"id": 3, "method": "step", "params": {"action": null, "unsendable": {"submit": false, "reason": "boom"}}}',
               b'{"id": 4, "method": "step", "params": {"action": null, "unsendable": [1]}}']
        for line in bad:
            out = self.ask(line)
            self.assertEqual(out["result"]["observation"]["error"]["code"], "invalid_action")
        extra = self.ask(b'{"id": 5, "method": "step", "params": {"action": null, "debug": true}}')
        self.assertEqual(extra["error"]["code"], "bad_params")
        big = self.ask(b'{"id": 6, "method": "step", "params": {"action": {"action": "calculator", "expression": "1", "x": ' + BIG.encode() + b'}}}')
        self.assertEqual(big["result"]["observation"]["error"]["code"], "invalid_action")
        self.assertEqual(big["result"]["observation"]["remaining_steps"], 96)


if __name__ == "__main__":
    unittest.main()
