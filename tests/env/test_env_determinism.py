"""Воспроизводимость и обобщение: одинаковые seed/config/действия -> те же записи, переходы и награды;
перестановка строк + смена ID сохраняют ответ oracle и награду (с учётом новых ID)."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from env_helpers import EPISODES, ROOT, TempRoot, dumps, local_env
from generator.fileio import read_episode_dir
from generator.transform import remap_findings, transform_texts
from policies import make_policy
from policies.auditor import derive_from_public_texts


def _play(env, policy_name, seed, config, policy_seed=0, truth=None):
    obs, info = env.reset(seed=seed, config=config)
    pol = make_policy(policy_name, seed=policy_seed, truth=truth)
    pol.reset(obs, info)
    trace = [dumps(obs), dumps(info)]
    done, total = False, 0.0
    last = None
    while not done:
        action = pol.act(obs)
        obs, r, term, trunc, info = env.step(action)
        trace.append(dumps([action, obs, r, term, trunc, info]))
        done = term or trunc
        total += r
        last = action
    return trace, total, last


class SameProcess(unittest.TestCase):
    def test_random_explorer_replays_exactly(self):
        for seed in (0, 1, 2):
            a = _play(local_env(), "random", seed, {"split": "train"}, policy_seed=11)[0]
            b = _play(local_env(), "random", seed, {"split": "train"}, policy_seed=11)[0]
            self.assertEqual(a, b)
        c = _play(local_env(), "random", 0, {"split": "train"}, policy_seed=12)[0]
        self.assertNotEqual(a, c)

    def test_generated_episodes_are_reproducible(self):
        env = local_env()
        o1, _ = env.reset(seed=7, config={"generator": "all_types"})
        o2, _ = env.reset(seed=7, config={"generator": "all_types"})
        o3, _ = env.reset(seed=8, config={"generator": "all_types"})
        self.assertEqual(dumps(o1), dumps(o2))
        self.assertNotEqual(o1["episode_id"], o3["episode_id"])
        env.close()
        salted = local_env(salt="secret")
        o4, _ = salted.reset(seed=7, config={"generator": "all_types"})
        self.assertNotEqual(o1["episode_id"], o4["episode_id"])
        trace, total, _ = _play(salted, "auditor", 7, {"generator": "all_types"})
        self.assertGreater(total, 0.9)
        salted.close()


class CrossProcess(unittest.TestCase):
    def _run(self, hashseed: str, out: Path, local: bool = False) -> str:
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        cmd = [sys.executable, "-m", "env.runner", "--policies", "random,auditor,reconcile", "--splits", "stub",
               "--mode", "train", "--trajectories", str(out), "--report-json", str(out) + ".report.json"]
        if local:
            cmd.append("--local")
        subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True)
        return hashlib.sha256(out.read_bytes()).hexdigest()

    def test_trajectories_identical_across_processes_and_hash_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            h1 = self._run("0", Path(tmp) / "a.jsonl")
            h2 = self._run("12345", Path(tmp) / "b.jsonl")
            h3 = self._run("777", Path(tmp) / "c.jsonl", local=True)
            self.assertEqual(h1, h2)
            self.assertEqual(h1, h3)  # изолированный и внутрипроцессный режимы дают одно и то же
            r1 = json.loads((Path(tmp) / "a.jsonl.report.json").read_text(encoding="utf-8"))
            r2 = json.loads((Path(tmp) / "b.jsonl.report.json").read_text(encoding="utf-8"))
            self.assertEqual([e["grader"]["reward"] for e in r1["episodes"]], [e["grader"]["reward"] for e in r2["episodes"]])


class PairedTransform(unittest.TestCase):
    """Как парные проверки жюри: тот же эпизод после перестановки и смены всех ID."""

    @classmethod
    def setUpClass(cls):
        cls.root = TempRoot()
        cls.pairs = []
        sources = [EPISODES / "stub" / p.name for p in (EPISODES / "stub").glob("ep-*")]
        sources += [EPISODES / "train" / n for n in ("ep-5C8KTAPXJA", "ep-D2Z88UACCQ", "ep-775BGFMEPG")]
        for i, src in enumerate(sources):
            public, private = read_episode_dir(src)
            core = {k: v for k, v in private.items() if "/" not in k}
            cls.root.write("orig", public, core)
            new_public, new_private, mapping = transform_texts(public, core, seed=f"pair-{i}")
            cls.root.write("pair", new_public, new_private)
            cls.pairs.append((json.loads(public["manifest.json"])["episode_id"],
                              json.loads(new_public["manifest.json"])["episode_id"], mapping, public, new_public))

    @classmethod
    def tearDownClass(cls):
        cls.root.cleanup()

    def test_rewards_survive_permutation_and_renaming(self):
        from grader.episodes import EpisodeStore, read_episode

        store = EpisodeStore(self.root.path)
        for old_id, new_id, mapping, _, _ in self.pairs:
            truths = {s: read_episode(store.episode_dir(s, e))[1] for s, e in (("orig", old_id), ("pair", new_id))}
            for policy in ("oracle", "auditor", "reconcile", "flag_all"):
                with self.subTest(episode=old_id, policy=policy):
                    rewards = []
                    for split, ep in (("orig", old_id), ("pair", new_id)):
                        env = local_env(episodes_root=self.root.path)
                        _, total, _ = _play(env, policy, 0, {"split": split, "episode_id": ep}, truth=truths[split])
                        env.close()
                        rewards.append(total)
                    self.assertEqual(rewards[0], rewards[1])
                    if policy in ("oracle",):
                        self.assertEqual(rewards[0], 1.0)

    def test_auditor_answer_maps_to_the_new_ids(self):
        for old_id, new_id, mapping, public, new_public in self.pairs:
            with self.subTest(episode=old_id):
                before = remap_findings({"episode_id": old_id, "findings": derive_from_public_texts(public)}, mapping)
                after = {"episode_id": new_id, "findings": derive_from_public_texts(new_public)}
                key = lambda f: (f["type"], sorted(f["affected_ids"]), f["claimed_amount_minor"], f["currency"], sorted(f["evidence_ids"]))  # noqa: E731
                self.assertEqual(before["episode_id"], after["episode_id"])
                self.assertEqual(sorted(map(key, before["findings"])), sorted(map(key, after["findings"])))


if __name__ == "__main__":
    unittest.main()
