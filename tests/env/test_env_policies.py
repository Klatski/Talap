"""Базовые политики-адаптеры: работают через контракт среды и дают ожидаемый по смыслу результат."""

import json
import unittest

from env_helpers import EPISODES, clean_episode, local_env, stub_truth
from grader.episodes import EpisodeStore, read_episode
from policies import ALL_POLICIES, BASELINES, make_policy


def _run(env, name, seed, config, truth=None, policy_seed=0):
    obs, info = env.reset(seed=seed, config=config)
    pol = make_policy(name, seed=policy_seed, truth=truth)
    pol.reset(obs, info)
    actions, done, reward, info = [], False, 0.0, {}
    while not done:
        action = pol.act(obs)
        actions.append(action)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
    return actions, reward, info, (term, trunc)


class PolicyTests(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(BASELINES, ("random", "reconcile", "flag_all", "oracle"))
        self.assertIn("auditor", ALL_POLICIES)
        with self.assertRaises(ValueError):
            make_policy("oracle")  # эталон нужен явно — oracle живёт на стороне оценщика
        with self.assertRaises(ValueError):
            make_policy("cheater")

    def test_oracle_and_auditor_reach_f1_one_on_every_open_episode(self):
        store = EpisodeStore()
        env = local_env()
        for split in ("train", "dev", "stub"):
            for i, name in enumerate(store.split_episodes(split)):
                truth = read_episode(EPISODES / split / name)[1]
                for policy in ("oracle", "auditor"):
                    with self.subTest(split=split, episode=name, policy=policy):
                        actions, reward, info, (term, trunc) = _run(env, policy, i, {"split": split}, truth=truth)
                        self.assertTrue(term and not trunc)
                        self.assertEqual(info["f1"], 1.0)
                        self.assertLessEqual(len(actions), 45)
                        if policy == "oracle":
                            self.assertEqual(reward, 1.0)
        env.close()

    def test_flag_all_is_punished(self):
        env = local_env()
        for split, idx in (("stub", 0), clean_episode(), ("dev", 3)):
            actions, reward, info, _ = _run(env, "flag_all", idx, {"split": split})
            self.assertLessEqual(len(actions[-1]["args"]["findings"]["findings"]), 30)
            self.assertEqual(info["tp"], 0)
            self.assertEqual(reward, -1.0)
        env.close()

    def test_reconcile_runs_and_is_far_from_oracle(self):
        env = local_env()
        actions, reward, info, (term, _) = _run(env, "reconcile", 0, {"split": "stub"})
        self.assertTrue(term)
        self.assertLess(info["f1"], 0.8)
        env.close()

    def test_random_explorer_exercises_errors_and_limits(self):
        env = local_env()
        codes, truncated, invalid = set(), 0, 0
        for seed in range(12):
            obs, info = env.reset(seed=seed % 10, config={"split": "train"})
            pol = make_policy("random", seed=seed)
            pol.reset(obs, info)
            done = False
            while not done:
                obs, r, term, trunc, info = env.step(pol.act(obs))
                if obs["error"]:
                    codes.add(obs["error"]["code"])
                done = term or trunc
            truncated += trunc
            invalid += bool(term and info.get("valid") is False)
        env.close()
        for code in ("unknown_action", "invalid_action", "not_found", "calculator_error", "unknown_table"):
            self.assertIn(code, codes)
        self.assertGreater(truncated, 0)

    def test_policies_never_see_reference_through_observations(self):
        truth = stub_truth()
        env = local_env()
        obs, info = env.reset(seed=0, config={"split": "stub", "mode": "eval"})
        blob = json.dumps(obs) + json.dumps(info)
        for issue in truth["issues"]:
            self.assertNotIn(issue["issue_id"], blob)
        self.assertNotIn("expected_amount_minor", blob)
        env.close()


if __name__ == "__main__":
    unittest.main()
