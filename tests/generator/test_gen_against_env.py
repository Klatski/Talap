"""Перекрёстная проверка «A проверяет среду B» (зона A): сквозной путь агент -> среда -> оценщик.

* oracle через ИЗОЛИРОВАННУЮ среду (env.AuditEnv, отдельные процессы) на каждом открытом эпизоде:
  train -> награда 1.0 и F1 = 1; eval -> политика видит 0 и пустой info;
* все атаки generator.attacks подаются через step(submit) среды (а не прямо в оценщик):
  награда совпадает с ожидаемой по ТЗ, неверная структура -> -1;
* сданный trajectories/trajectories.jsonl проходит сканер утечек A.
Тесты пропускаются, пока среды B нет (env/ не импортируется).
"""

import importlib
import json
import unittest
from pathlib import Path

from gen_helpers import ROOT, episode
from generator.attacks import build_attacks
from generator.leakscan import scan_trajectories

EPISODES = ROOT / "episodes"


def _env_module():
    try:
        return importlib.import_module("env")
    except ImportError:
        return None


class OracleThroughIsolatedEnv(unittest.TestCase):
    def setUp(self):
        self.env_mod = _env_module()
        if self.env_mod is None:
            self.skipTest("env (zone B) is not available")

    def test_oracle_is_perfect_in_train_and_hidden_in_eval(self):
        from policies.oracle import OraclePolicy

        dirs = sorted(EPISODES.glob("*/ep-*"))
        with self.env_mod.AuditEnv() as env:
            for d in dirs:
                split = d.parent.name
                for mode in ("train", "eval"):
                    with self.subTest(episode=d.name, mode=mode):
                        obs, info = env.reset(seed=0, config={"split": split, "episode_id": d.name, "mode": mode})
                        self.assertEqual(obs["episode_id"], d.name)
                        pol = OraclePolicy.from_episode_dir(d)
                        pol.reset(obs, info)
                        obs, reward, term, trunc, info = env.step(pol.act(obs))
                        self.assertTrue(term and not trunc)
                        if mode == "train":
                            self.assertEqual(reward, 1.0)
                            self.assertEqual(info["f1"], 1.0)
                        else:
                            self.assertEqual((reward, info), (0.0, {}))


class AttacksThroughEnvStep(unittest.TestCase):
    def setUp(self):
        self.env_mod = _env_module()
        if self.env_mod is None:
            self.skipTest("env (zone B) is not available")

    def test_attack_rewards_via_submit_action(self):
        stub = next((EPISODES / "stub").glob("ep-*"))
        from generator.fileio import read_episode_dir

        public, private = read_episode_dir(stub)
        cases = build_attacks(public, json.loads(private["truth.json"]), seed=2)
        env = self.env_mod.LocalAuditEnv()
        try:
            for case in cases:
                with self.subTest(case=case["name"]):
                    env.reset(seed=0, config={"split": "stub", "mode": "train"})
                    for _ in range(case["inspection_steps"]):
                        env.step({"action": "calculator", "expression": "1+1"})
                    obs, reward, term, trunc, info = env.step({"action": "submit", "findings": case["submission"]})
                    self.assertTrue(term)
                    self.assertAlmostEqual(reward, case["expected"]["reward"], places=12)
                    self.assertEqual(info["valid"], case["expected"]["valid"])
                    self.assertNotIn("issue_id", json.dumps(info))
        finally:
            env.close()

    def test_generated_episode_matches_generator_truth(self):
        """reset(seed, {"generator": ...}) в среде даёт тот же эпизод, что generate_episode(seed, config)."""
        ep = episode("all_types", 3)
        env = self.env_mod.LocalAuditEnv()
        try:
            obs, _ = env.reset(seed=3, config={"generator": {"name": "t_all", "size": "small",
                                                             "errors": {"duplicate_invoice": 1, "unapplied_payment": 1,
                                                                        "fx_rate_error": 1, "cutoff_error": 1}}})
            self.assertEqual(obs["episode_id"], ep.episode_id)
            self.assertEqual(obs["manifest"], json.loads(ep.public["manifest.json"]))
        finally:
            env.close()


class SubmittedTrajectories(unittest.TestCase):
    def test_trajectories_file_has_no_leaks(self):
        path = ROOT / "trajectories" / "trajectories.jsonl"
        if not path.is_file():
            self.skipTest("trajectories.jsonl not produced yet")
        self.assertEqual(scan_trajectories(path, EPISODES), [])
        with open(path, encoding="utf-8") as fh:
            first = json.loads(fh.readline())
        self.assertLessEqual({"episode_id", "step", "action", "observation", "reward", "terminated", "truncated"}, set(first))


if __name__ == "__main__":
    unittest.main()
