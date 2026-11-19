"""Режимы train/eval: награда по ходу 0; в eval политика не видит ни награды, ни диагностики."""

import unittest

from env_helpers import STUB, local_env, stub_truth
from policies.oracle import oracle_findings

LIST = {"action": "list_records", "table": "invoices"}
INFO_KEYS = {"valid", "errors", "tp", "fp", "fn", "precision", "recall", "f1", "inspection_steps",
             "truncated_reason", "reward_experimental"}


class ModeTests(unittest.TestCase):
    def setUp(self):
        self.truth = stub_truth()

    def _episode(self, mode, findings):
        env = local_env()
        env.reset(seed=0, config={**STUB, "mode": mode})
        seen = []
        for _ in range(3):
            seen.append(env.step(LIST))
        seen.append(env.step({"action": "submit", "findings": findings}))
        rec = env.grader_records[-1]
        env.close()
        return seen, rec

    def test_train_returns_terminal_reward_and_metrics(self):
        seen, rec = self._episode("train", oracle_findings(self.truth))
        for obs, r, term, trunc, info in seen[:-1]:
            self.assertEqual((r, term, trunc, info), (0.0, False, False, {}))
        obs, r, term, trunc, info = seen[-1]
        self.assertTrue(term)
        self.assertAlmostEqual(r, 1 - 0.002 * 3)
        self.assertEqual(r, rec["reward"])
        self.assertEqual(set(info), INFO_KEYS)
        self.assertEqual((info["tp"], info["fp"], info["fn"], info["f1"]), (len(self.truth["issues"]), 0, 0, 1.0))
        self.assertEqual(obs["result"], {"status": "submitted"})

    def test_eval_hides_reward_and_diagnostics(self):
        seen, rec = self._episode("eval", oracle_findings(self.truth))
        for obs, r, term, trunc, info in seen:
            self.assertEqual(r, 0.0)
            self.assertEqual(info, {})
        self.assertEqual(seen[-1][0], {"result": {"status": "submitted"}, "error": None, "remaining_steps": 96})
        self.assertAlmostEqual(rec["reward"], 1 - 0.002 * 3)  # у оценщика награда есть
        self.assertEqual(rec["mode"], "eval")

    def test_eval_invalid_submission_is_still_hidden(self):
        seen, rec = self._episode("eval", {"episode_id": self.truth["episode_id"], "findings": [{"type": "fraud"}]})
        self.assertEqual(seen[-1][1:], (0.0, True, False, {}))
        self.assertFalse(rec["valid"])
        self.assertEqual(rec["reward"], -1.0)
        self.assertTrue(rec["errors"])

    def test_train_invalid_submission(self):
        for payload in (None, "all", {"episode_id": "ep-OTHER00000", "findings": []}, {"findings": []}):
            with self.subTest(payload=payload):
                seen, rec = self._episode("train", payload)
                obs, r, term, trunc, info = seen[-1]
                self.assertEqual((r, term, trunc), (-1.0, True, False))
                self.assertFalse(info["valid"])
                self.assertEqual(info["tp"], None)

    def test_malformed_submit_wrapper(self):
        env = local_env()
        env.reset(seed=0, config=STUB)
        obs, r, term, trunc, info = env.step({"action": "submit", "findings": oracle_findings(self.truth), "bonus": 1})
        self.assertEqual((r, term), (-1.0, True))
        env.reset(seed=0, config=STUB)
        obs, r, term, trunc, info = env.step({"action": "submit"})
        self.assertEqual((r, term), (-1.0, True))
        env.close()

    def test_bad_mode(self):
        env = local_env()
        for bad in ("test", None, 1, "EVAL"):
            with self.assertRaises(ValueError):
                env.reset(seed=0, config={**STUB, "mode": bad})
        env.close()


if __name__ == "__main__":
    unittest.main()
