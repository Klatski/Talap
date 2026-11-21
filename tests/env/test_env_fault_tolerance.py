"""Жёсткая обёртка step: никакое действие и никакой дефект среды/оценщика не роняют step исключением.

* дефект внутри разбора или выполнения действия -> ограниченная ошибка internal_error, шаг расходуется,
  текст исключения наружу не попадает, следующий шаг работает;
* оценщик упал на submit -> эпизод завершён, ошибка grader_unavailable, исключения нет;
* фаззинг: тысячи случайных JSON-подобных действий подряд (с reset между эпизодами) — ни одного
  исключения, форма наблюдения всегда {result, error, remaining_steps}, коды только из каталога.
"""

import random
import unittest
from unittest import mock

from env_helpers import STUB, local_env
from env.constants import ACTIONS, ERRORS, TABLES
from env.core import EnvCore
from env.service import EnvService, InProcessGrader

SECRET = "boom-internal-detail-7f3a"


class InternalDefects(unittest.TestCase):
    def setUp(self):
        self.env = local_env()
        self.env.reset(seed=0, config={**STUB, "mode": "train"})

    def tearDown(self):
        self.env.close()

    def check_internal_error(self, obs, remaining):
        self.assertEqual(obs["error"]["code"], "internal_error")
        self.assertEqual(obs["error"]["message"], ERRORS["internal_error"])
        self.assertIsNone(obs["result"])
        self.assertEqual(obs["remaining_steps"], remaining)
        self.assertNotIn(SECRET, repr(obs))

    def test_defect_in_dispatch(self):
        with mock.patch.object(EnvCore, "_dispatch", side_effect=RuntimeError(SECRET)):
            obs, reward, term, trunc, info = self.env.step({"action": "list_records", "table": "ledger"})
        self.check_internal_error(obs, 99)
        self.assertEqual((reward, term, trunc, info), (0.0, False, False, {}))
        obs = self.env.step({"action": "calculator", "expression": "2+2"})[0]
        self.assertEqual(obs["result"], "4")  # среда жива

    def test_defect_in_parsing(self):
        with mock.patch.object(EnvCore, "_parse", side_effect=KeyError(SECRET)):
            obs = self.env.step({"action": "calculator", "expression": "1"})[0]
        self.check_internal_error(obs, 99)

    def test_non_json_result_from_a_tool(self):
        with mock.patch.object(EnvCore, "_dispatch", return_value={"value": object()}):
            obs = self.env.step({"action": "calculator", "expression": "1"})[0]
        self.check_internal_error(obs, 99)

    def test_defect_on_the_last_step_still_truncates(self):
        self.env.reset(seed=0, config={**STUB, "max_steps": 2})
        self.env.step({"action": "calculator", "expression": "1"})
        with mock.patch.object(EnvCore, "_dispatch", side_effect=RuntimeError(SECRET)):
            obs, reward, term, trunc, info = self.env.step({"action": "calculator", "expression": "1"})
        self.assertTrue(trunc)
        self.assertEqual(obs["error"]["code"], "internal_error")


class FailingGrader(InProcessGrader):
    def finish(self, *args, **kwargs):
        raise ConnectionError(SECRET)


class GraderFailure(unittest.TestCase):
    def test_submit_when_grader_is_down(self):
        service = EnvService(FailingGrader())
        obs, _ = service.reset(0, {**STUB, "mode": "train"})
        out = service.step({"action": "submit", "findings": {"episode_id": obs["episode_id"], "findings": []}})
        obs, reward, term, trunc, info = out
        self.assertEqual(obs["error"]["code"], "grader_unavailable")
        self.assertEqual((reward, term, trunc, info), (0.0, True, False, {}))
        self.assertNotIn(SECRET, repr(out))
        self.assertEqual(service.step({"action": "calculator", "expression": "1"})[0]["error"]["code"], "episode_over")
        obs, _ = service.reset(0, STUB)  # после сбоя можно начать новый эпизод
        self.assertEqual(obs["remaining_steps"], 100)
        service.reset(0, {**STUB, "max_steps": 1})
        out = service.step({"action": "calculator", "expression": "1"})  # step_limit тоже идёт через finish
        self.assertTrue(out[3])
        self.assertEqual(out[0]["error"]["code"], "grader_unavailable")


def _random_value(rng: random.Random, depth: int = 0):
    kinds = ["str", "int", "float", "bool", "none", "list", "dict", "id", "big"]
    kind = rng.choice(kinds if depth < 3 else kinds[:5])
    if kind == "str":
        return rng.choice(["", "ledger", "x" * rng.randint(0, 300), "../../etc", "\u0000", "\U0001F600", "1+1", "submit"])
    if kind == "int":
        return rng.choice([0, -1, 1, 50, 51, 10**18, -(10**30)])
    if kind == "float":
        return rng.choice([0.5, -0.0, 1e308, float("nan"), float("inf")])
    if kind == "bool":
        return rng.choice([True, False])
    if kind == "none":
        return None
    if kind == "id":
        return rng.choice(["L-00000000", "D-00000000", "INV-0000000", "PAY-0000000", "FX:USD:KZT:2026-03-02"])
    if kind == "big":
        return 10 ** rng.choice([100, 4400, 5000])
    if kind == "list":
        return [_random_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {rng.choice(["action", "table", "id", "doc_id", "cursor", "limit", "expression", "findings", "x"]):
            _random_value(rng, depth + 1) for _ in range(rng.randint(0, 4))}


def random_action(rng: random.Random):
    roll = rng.random()
    if roll < 0.15:
        return _random_value(rng)
    action = {"action": rng.choice(sorted(ACTIONS) + ["shell", "", "SUBMIT"])}
    for key in rng.sample(["table", "id", "doc_id", "cursor", "limit", "expression", "findings", "extra"], rng.randint(0, 4)):
        action[key] = rng.choice([rng.choice(sorted(TABLES)), _random_value(rng)])
    if roll > 0.9:
        import json
        try:
            return json.dumps(action)
        except (TypeError, ValueError):
            return action
    return action


class Fuzz(unittest.TestCase):
    def test_thousands_of_random_actions_never_raise(self):
        rng = random.Random(20261120)
        env = local_env()
        codes = set()
        try:
            env.reset(seed=0, config={**STUB, "mode": "train"})
            for i in range(3000):
                obs, reward, term, trunc, info = env.step(random_action(rng))
                self.assertEqual(set(obs), {"result", "error", "remaining_steps"})
                self.assertIsInstance(reward, float)
                if obs["error"] is not None:
                    self.assertIn(obs["error"]["code"], ERRORS)
                    self.assertEqual(obs["error"]["message"], ERRORS[obs["error"]["code"]])
                    codes.add(obs["error"]["code"])
                if term or trunc or obs["error"] and obs["error"]["code"] == "episode_over":
                    env.reset(seed=i, config={"split": rng.choice(["stub", "train", "dev"]), "mode": rng.choice(["train", "eval"])})
        finally:
            env.close()
        self.assertNotIn("internal_error", codes)  # фаззинг не нашёл дефектов, на которые сработала бы страховка
        self.assertGreaterEqual(len(codes), 8)


if __name__ == "__main__":
    unittest.main()
