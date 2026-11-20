"""Чистота reset(): ничто не переносится из эпизода в эпизод.

* последовательность A -> B -> A с ОДНИМ экземпляром политики даёт для A побайтово одинаковые
  траектории (в процессе и через изолированную среду) — для всех агентских политик;
* две загрузки одного эпизода не делят ни одного изменяемого объекта (списки/словари записей) ни в
  среде, ни у оценщика; порча внутреннего состояния первого эпизода не видна во втором;
* изменение возвращённых наблюдений/info политикой не меняет следующий reset;
* курсоры прошлого эпизода недействительны, бюджет шагов и время начинаются заново.
"""

import json
import unittest

from env_helpers import STUB, local_env
from env import AuditEnv
from policies import make_policy

A = {"split": "train", "index": 2, "mode": "train"}
B = {"split": "dev", "index": 5, "mode": "train"}


def play(env, policy, config, seed=0):
    obs, info = env.reset(seed=seed, config=config)
    policy.reset(obs, info)
    trace = [{"obs": obs, "info": info}]
    done = False
    while not done:
        action = policy.act(obs)
        obs, reward, term, trunc, info = env.step(action)
        trace.append({"action": action, "obs": obs, "reward": reward, "terminated": term, "truncated": trunc, "info": info})
        done = term or trunc
    return json.dumps(trace, sort_keys=True, default=repr)


def mutable_ids(obj, seen=None):
    """id всех списков и словарей, достижимых из obj."""
    seen = set() if seen is None else seen
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, (dict, list)) and id(x) not in seen:
            seen.add(id(x))
            stack.extend(x.values() if isinstance(x, dict) else x)
    return seen


class ResetPurity(unittest.TestCase):
    def test_a_b_a_with_one_policy_instance(self):
        for env_name, factory in (("local", local_env), ("isolated", AuditEnv)):
            env = factory()
            try:
                for name in ("random", "reconcile", "flag_all", "auditor"):
                    with self.subTest(env=env_name, policy=name):
                        policy = make_policy(name, seed=3)
                        first = play(env, policy, A, seed=11)
                        play(env, policy, B, seed=12)
                        again = play(env, policy, A, seed=11)
                        self.assertEqual(first, again)
            finally:
                env.close()

    def test_two_loads_share_no_mutable_objects(self):
        env = local_env()
        try:
            env.reset(seed=0, config=A)
            core = env._service.core
            first_data = core._ep.data
            first_truth = env._grader.session._current.truth
            ids_env = mutable_ids([first_data.tables, first_data.by_id, first_data.manifest, first_data.policy,
                                   list(first_data.documents.values())])
            ids_truth = mutable_ids(first_truth)
            # «баг»: что-то испортило состояние первого эпизода
            first_data.tables["ledger"][0]["debit_minor"] = -999
            first_data.manifest["base_currency"] = "XXX"
            first_truth["issues"].clear()
            env.reset(seed=0, config=B)
            obs, _ = env.reset(seed=0, config=A)
            second_data = core._ep.data
            second_truth = env._grader.session._current.truth
            self.assertFalse(ids_env & mutable_ids([second_data.tables, second_data.by_id, second_data.manifest,
                                                    second_data.policy, list(second_data.documents.values())]))
            self.assertFalse(ids_truth & mutable_ids(second_truth))
            self.assertNotEqual(obs["manifest"]["base_currency"], "XXX")
            self.assertNotEqual(second_data.tables["ledger"][0]["debit_minor"], -999)
            self.assertTrue(second_truth["issues"])  # эталон перечитан, а не взят из прошлого эпизода
            self.assertIsNot(first_data, second_data)
        finally:
            env.close()

    def test_policy_side_mutation_does_not_leak_into_the_env(self):
        env = local_env()
        try:
            obs, info = env.reset(seed=0, config=A)
            pristine = json.dumps(obs, sort_keys=True)
            obs["manifest"]["counts"]["ledger_rows"] = 0
            obs["policy"].clear()
            info["mode"] = "eval"
            page = env.step({"action": "list_records", "table": "invoices", "limit": 2})[0]
            page["result"]["records"][0]["total_minor"] = -1
            again, info2 = env.reset(seed=0, config=A)
            self.assertEqual(json.dumps(again, sort_keys=True), pristine)
            self.assertEqual(info2["mode"], "train")
            page2 = env.step({"action": "list_records", "table": "invoices", "limit": 2})[0]
            self.assertNotEqual(page2["result"]["records"][0]["total_minor"], -1)
        finally:
            env.close()

    def test_budget_clock_and_cursors_restart(self):
        env = local_env()
        try:
            env.reset(seed=0, config={**A, "max_steps": 5})
            cursor = env.step({"action": "list_records", "table": "ledger", "limit": 10})[0]["result"]["next_cursor"]
            for _ in range(3):
                env.step({"action": "calculator", "expression": "1"})
            obs, _ = env.reset(seed=0, config=A)  # тот же эпизод заново
            self.assertEqual(obs["remaining_steps"], 100)
            step = env.step({"action": "list_records", "table": "ledger", "cursor": cursor, "limit": 10})[0]
            self.assertEqual(step["remaining_steps"], 99)
            # курсор того же эпизода по смыслу остаётся валидным (он детерминирован), но шаги не унаследованы
            self.assertIsNone(step["error"])
            env.reset(seed=0, config=B)
            stale = env.step({"action": "list_records", "table": "ledger", "cursor": cursor, "limit": 10})[0]
            self.assertEqual(stale["error"]["code"], "invalid_cursor")  # курсор другого эпизода не принимается
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
