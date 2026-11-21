"""Враждебный ввод: ни одно действие агента не роняет step, а итог одинаков в процессе и через изоляцию.

Правило (docs/contract.md, раздел 8): submit в любом формате (или JSON-строка такого объекта) всегда
завершает эпизод; если он не JSON (NaN, целое длиннее 4300 цифр) или больше MAX_SUBMIT_BYTES, это неверная
структура: награда -1 и причина в журнале оценщика. Размер считается без текста валидных explanation, поэтому
текст объяснения никогда не решает судьбу ответа и не меняет награду. Прочие нечитаемые действия дают
ограниченную ошибку и расходуют шаг.
"""

import json
import subprocess
import sys
import unittest

from env_helpers import ROOT, STUB, local_env
from env import AuditEnv
from env.constants import ERRORS, MAX_ACTION_BYTES, MAX_EXPLANATION_CHARS, MAX_PARSE_BYTES, MAX_SUBMIT_BYTES
from grader.validate import MALFORMED_REASONS, MAX_EVIDENCE, MAX_EXPLANATION, MAX_FINDINGS

# «Тяжёлый» реалистичный вывод: 10 affected_ids по 64 символа. Структура не ограничивает списки и длину ID
# (правила жюри: такой вывод — просто FP), их ограничивает только транспортный предел submit.
MAX_AFFECTED, MAX_ID_LEN = 10, 64

BIG = "9" * 5000  # больше sys.get_int_max_str_digits() по умолчанию
BIG_INT = 10 ** 5000  # такой int есть в Python, но str()/json.dumps на нём падают
ASTRAL = "\U0001F600"  # четыре байта UTF-8, 12 байт в ASCII-экранированном JSON


def finding(**kw):
    f = {"type": "fx_rate_error", "affected_ids": ["PAY-1"], "claimed_amount_minor": 1, "currency": "KZT",
         "evidence_ids": [], "explanation": ""}
    f.update(kw)
    return f


def heavy_valid_findings(episode_id: str) -> dict:
    """Тяжёлый валидный ответ: 30 выводов, ID по 64 эмодзи, суммы по 4300 цифр, объяснения по 1000 эмодзи."""
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
    ("string above parse limit, submit name only nested",
     json.dumps({"name": "calculator", "args": {"pad": "x" * (MAX_PARSE_BYTES + 10), "name": "submit"}}), "action_too_large"),
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
            "missing findings": (lambda ep: {"action": "submit"}, "wrapper"),
            "jury envelope without args": (lambda ep: {"name": "submit"}, "wrapper"),
            "jury envelope, args not an object": (lambda ep: {"name": "submit", "args": "findings"}, "wrapper"),
            "jury envelope, NaN inside": (lambda ep: {"name": "submit", "args": {"findings": {"episode_id": ep, "findings": [finding(claimed_amount_minor=float("nan"))]}}}, "not_json"),
            "jury envelope above MAX_SUBMIT_BYTES": (lambda ep: json.dumps({"name": "submit", "args": {"findings": {"episode_id": ep, "findings": [finding(explanation="x" * (MAX_PARSE_BYTES + 10))]}}}), "too_large"),
            # регрессия /code-review: json.dumps(sort_keys=True) ставит "args" раньше "name"
            "jury envelope above MAX_PARSE_BYTES, args first": (lambda ep: json.dumps({"name": "submit", "args": {"findings": {"episode_id": ep, "findings": [finding(explanation="x" * (MAX_PARSE_BYTES + 10))]}}}, sort_keys=True), "too_large"),
            "flat form above MAX_PARSE_BYTES, action last": (lambda ep: json.dumps({"findings": {"episode_id": ep, "findings": [finding(explanation="x" * (MAX_PARSE_BYTES + 10))]}, "action": "submit"}), "too_large"),
        }
        for label, (make, reason) in cases.items():
            with self.subTest(case=label):
                for name, (obs, reward, term, trunc, info), after in self.submit_both(make):
                    self.assertEqual((reward, term, trunc), (-1.0, True, False), name)
                    self.assertIsNone(obs["result"])
                    self.assertEqual(obs["error"]["code"], "invalid_submission")
                    self.assertEqual(obs["error"]["detail"], f"submit action: {MALFORMED_REASONS[reason]}")
                    self.assertFalse(info["valid"])
                    self.assertEqual(info["errors"], [f"submit action: {MALFORMED_REASONS[reason]}"], name)
                    self.assertEqual(info["inspection_steps"], 0)
                    self.assertEqual(after[0]["error"]["code"], "episode_over")

    def test_heavy_valid_submission_is_scored_not_rejected(self):
        """Регрессия /code-review: эмодзи в explanation и ID не должны превращать валидный ответ в -1."""
        sample = heavy_valid_findings("ep-AAAAAAAAAA")
        ascii_size = len(json.dumps({"action": "submit", "findings": sample}, sort_keys=True).encode())
        self.assertGreater(ascii_size, MAX_ACTION_BYTES)  # порог обычного действия отверг бы этот валидный ответ
        makers = {
            "dict": lambda ep: {"action": "submit", "findings": heavy_valid_findings(ep)},
            "string, ASCII-escaped": lambda ep: json.dumps({"action": "submit", "findings": heavy_valid_findings(ep)}),
            "string, raw UTF-8": lambda ep: json.dumps({"action": "submit", "findings": heavy_valid_findings(ep)}, ensure_ascii=False),
            "string, indented": lambda ep: json.dumps({"action": "submit", "findings": heavy_valid_findings(ep)}, indent=2),
            "jury envelope": lambda ep: {"name": "submit", "args": {"findings": heavy_valid_findings(ep)}},
        }
        for label, make in makers.items():
            with self.subTest(form=label):
                for name, (obs, reward, term, trunc, info), _ in self.submit_both(make):
                    self.assertTrue(term, name)
                    self.assertTrue(info["valid"], (name, info["errors"]))
                    self.assertEqual(info["errors"], [])
                    self.assertEqual(info["tp"], 0)  # разобран и сопоставлен, а не отвергнут

    def test_explanation_text_never_decides_the_size_limit(self):
        """Регрессия /code-review: размер submit меряется без текста валидных explanation. Ответ, который с
        объяснениями по 1000 эмодзи больше порога, а без них меньше, оценивается так же, как с пустыми."""
        ids = [f"{'X' * 25}{i:04d}" for i in range(900)]  # 30 выводов x 900 ID ~ 0.9 МБ без объяснений

        def make(explanation):
            return lambda ep: {"name": "submit", "args": {"findings": {"episode_id": ep, "findings": [
                finding(affected_ids=ids, explanation=explanation) for _ in range(MAX_FINDINGS)]}}}

        plain, rich = make(""), make(ASTRAL * MAX_EXPLANATION)
        sizes = [len(json.dumps(m("ep-AAAAAAAAAA"), sort_keys=True).encode()) for m in (plain, rich)]
        self.assertLess(sizes[0], MAX_SUBMIT_BYTES)
        self.assertGreater(sizes[1], MAX_SUBMIT_BYTES)
        outcomes = set()
        for make_action in (plain, rich):
            for name, (obs, reward, term, trunc, info), _ in self.submit_both(make_action):
                self.assertTrue(info["valid"], (name, info["errors"]))
                self.assertEqual(info["fp"], MAX_FINDINGS)
                outcomes.add(json.dumps([reward, term, trunc, info], sort_keys=True))
        self.assertEqual(len(outcomes), 1)

    def test_explanation_limit_mirrors_the_validator(self):
        self.assertEqual(MAX_EXPLANATION_CHARS, MAX_EXPLANATION)

    def test_oversized_explanation_within_submit_limit_gets_validator_error(self):
        for name, (obs, reward, term, trunc, info), _ in self.submit_both(
                lambda ep: {"action": "submit", "findings": {"episode_id": ep, "findings": [finding(explanation="x" * 300_000)]}}):
            self.assertEqual((reward, term), (-1.0, True))
            self.assertTrue(any("explanation" in e for e in info["errors"]), (name, info["errors"]))

    def test_marker_in_agent_payload_cannot_forge_the_reason(self):
        """Регрессия /code-review: причина неверной структуры приходит только от среды, не из ответа агента.
        Поле-«маркер» в ответе — просто поле сверх формата: оно игнорируется, ответ оценивается как есть."""
        for name, (obs, reward, term, trunc, info), _ in self.submit_both(
                lambda ep: {"action": "submit", "findings": {"__malformed_submit_action__": "too_large", "episode_id": ep, "findings": []}}):
            self.assertTrue(term)
            self.assertTrue(info["valid"], name)
            self.assertEqual(info["errors"], [])
            self.assertEqual((info["tp"], info["fp"]), (0, 0))

    def test_fields_beyond_the_format_are_ignored(self):
        def make(ep):
            return {"name": "submit", "args": {"findings": {"episode_id": ep, "findings": [finding(confidence=0.99)], "note": "x"},
                                               "extra_arg": True}}
        for name, (obs, reward, term, trunc, info), _ in self.submit_both(make):
            self.assertTrue(term)
            self.assertTrue(info["valid"], (name, info["errors"]))
            self.assertEqual((info["tp"], info["fp"]), (0, 1))
        obs, _ = self.envs["local"].reset(seed=0, config={**STUB, "mode": "train"})
        self.assertIn("Fields beyond the format are ignored", obs["matching"])  # текст правил = поведение оценщика
        self.assertNotIn("extra fields", obs["matching"])

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
        self.assertEqual(step["result"]["observation"]["result"], "4")
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
