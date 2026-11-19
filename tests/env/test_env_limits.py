"""Лимиты и завершение: 100 шагов вкл. submit, тайм-аут, сторожевой таймер, шаг после конца, повторный submit."""

import time
import unittest

from env_helpers import STUB, clean_episode, local_env, stub_truth
from policies.oracle import oracle_findings

LIST = {"action": "list_records", "table": "ledger"}


class StepLimitTests(unittest.TestCase):
    def test_hundred_steps_truncate_with_empty_answer(self):
        env = local_env()
        env.reset(seed=0, config=STUB)
        for i in range(1, 100):
            obs, r, term, trunc, info = env.step(LIST)
            self.assertFalse(term or trunc, i)
            self.assertEqual(obs["remaining_steps"], 100 - i)
        obs, r, term, trunc, info = env.step(LIST)
        self.assertTrue(trunc)
        self.assertFalse(term)
        self.assertEqual(obs["remaining_steps"], 0)
        self.assertIsNotNone(obs["result"])  # сотое действие выполнено
        rec = env.grader_records[-1]
        self.assertEqual((rec["end_reason"], rec["inspection_steps"], rec["steps_used"]), ("step_limit", 100, 100))
        self.assertEqual(rec["tp"], 0)
        self.assertEqual(r, -1.0)  # пустой ответ на книге с ошибками
        self.assertEqual(info["truncated_reason"], "step_limit")
        env.close()

    def test_truncation_on_clean_book_scores_empty_answer(self):
        split, idx = clean_episode()
        env = local_env()
        env.reset(seed=idx, config={"split": split})
        for _ in range(100):
            obs, r, term, trunc, info = env.step(LIST)
        self.assertTrue(trunc)
        self.assertAlmostEqual(r, 1 - 0.002 * 100)  # общая награда: F1 = 1 для пустого эталона
        self.assertEqual(info["reward_experimental"], -1.0)  # экспериментальная закрывает лазейку
        env.close()

    def test_submit_on_last_step_terminates(self):
        truth = stub_truth()
        env = local_env()
        env.reset(seed=0, config=STUB)
        for _ in range(99):
            env.step(LIST)
        obs, r, term, trunc, info = env.step({"action": "submit", "findings": oracle_findings(truth)})
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(info["inspection_steps"], 99)
        self.assertAlmostEqual(r, 1 - 0.002 * 99)
        env.close()

    def test_after_end_and_resubmission(self):
        truth = stub_truth()
        env = local_env()
        env.reset(seed=0, config=STUB)
        env.step({"action": "submit", "findings": {"episode_id": truth["episode_id"], "findings": []}})
        for action in ({"action": "submit", "findings": oracle_findings(truth)}, LIST, "garbage"):
            obs, r, term, trunc, info = env.step(action)
            self.assertEqual(obs["error"]["code"], "episode_over")
            self.assertEqual((r, term, trunc, info), (0.0, True, False, {}))
        self.assertEqual(len(env.grader_records), 1)
        self.assertEqual(env.grader_records[0]["tp"], 0)  # первый ответ остаётся окончательным
        env.close()

    def test_custom_max_steps_and_bounds(self):
        env = local_env()
        env.reset(seed=0, config={**STUB, "max_steps": 3})
        for _ in range(2):
            self.assertFalse(env.step(LIST)[3])
        self.assertTrue(env.step(LIST)[3])
        for bad in (0, 101, "5", True, 2.5, None):
            with self.subTest(max_steps=bad):
                with self.assertRaises(ValueError):
                    env.reset(seed=0, config={**STUB, "max_steps": bad})
        env.close()

    def test_reset_mid_episode_records_abandoned(self):
        env = local_env()
        env.reset(seed=0, config=STUB)
        env.step(LIST)
        env.reset(seed=0, config=STUB)
        rec = env.grader_records[-1]
        self.assertEqual((rec["end_reason"], rec["truncated"], rec["inspection_steps"]), ("abandoned", True, 1))
        env.close()
        self.assertEqual(env.grader_records[-1]["end_reason"], "abandoned")  # close тоже закрывает эпизод


class TimeLimitTests(unittest.TestCase):
    def test_late_action_is_not_executed(self):
        env = local_env()
        env.reset(seed=0, config={**STUB, "time_limit_s": 1.0})
        env.step(LIST)
        time.sleep(1.3)
        obs, r, term, trunc, info = env.step(LIST)
        self.assertTrue(trunc)
        self.assertFalse(term)
        self.assertEqual(obs["error"]["code"], "time_limit")
        self.assertIsNone(obs["result"])
        rec = env.grader_records[-1]
        self.assertEqual((rec["end_reason"], rec["inspection_steps"]), ("time_limit", 1))
        self.assertEqual(env.step(LIST)[0]["error"]["code"], "episode_over")
        env.close()

    def test_watchdog_finishes_without_steps(self):
        env = local_env()
        env.reset(seed=0, config={**STUB, "time_limit_s": 0.3, "mode": "eval"})
        time.sleep(1.2)
        self.assertEqual(len(env.grader_records), 1)  # оценщик уже получил пустой ответ
        self.assertEqual(env.grader_records[0]["end_reason"], "time_limit")
        obs, r, term, trunc, info = env.step(LIST)
        self.assertEqual((obs["error"]["code"], r, term, trunc, info), ("time_limit", 0.0, False, True, {}))
        env.close()

    def test_time_limit_bounds(self):
        env = local_env()
        for bad in (0, -1, 61, "60", True, None):
            with self.subTest(limit=bad):
                with self.assertRaises(ValueError):
                    env.reset(seed=0, config={**STUB, "time_limit_s": bad})
        env.close()


if __name__ == "__main__":
    unittest.main()
