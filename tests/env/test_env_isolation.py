"""Изоляция и отсутствие утечек: эталон только в процессе оценщика, наблюдения чистые."""

import ast
import json
import subprocess
import sys
import textwrap
import unittest

from env_helpers import EPISODES, ROOT, STUB, TempRoot, local_env, stub_truth
from env.protocol import Channel, RemoteError
from generator.leakscan import scan_records
from grader.episodes import PUBLIC_FILES
from grader.session import GraderSession
from policies import make_policy

AGENT_SIDE = ("calculator", "client", "constants", "core", "cursor", "observation", "protocol", "records", "server")
PRIVATE_WORDS = ("truth", "provenance", "private", "issue_id", "evidence_sets", "expected_amount")


def _string_constants(tree: ast.AST) -> list[str]:
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


class StaticIsolation(unittest.TestCase):
    def test_agent_side_modules_never_touch_the_reference(self):
        for name in AGENT_SIDE:
            tree = ast.parse((ROOT / "env" / f"{name}.py").read_text(encoding="utf-8"))
            with self.subTest(module=name):
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse((node.module or "").startswith(("grader", "generator", "policies")), node.module)
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertFalse(alias.name.startswith(("grader", "generator", "policies")))
                for text in _string_constants(tree):
                    for word in PRIVATE_WORDS:
                        self.assertNotIn(word, text.lower(), f"{name}: {text[:60]!r}")

    def test_service_imports_grader_only_for_in_process_mode(self):
        tree = ast.parse((ROOT / "env" / "service.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.assertFalse(getattr(node, "module", "") and node.module.startswith("grader"))

    def test_env_server_process_does_not_load_grader_code(self):
        code = "import sys, env.server; print([m for m in sys.modules if m.split('.')[0] in ('grader', 'generator', 'policies')])"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "[]")


class GraderBoundary(unittest.TestCase):
    def test_load_sends_only_public_files(self):
        session = GraderSession()
        out = session.load(0, STUB, "train")
        self.assertEqual(set(out), {"public", "mode"})
        self.assertEqual(set(out["public"]), set(PUBLIC_FILES))
        blob = json.dumps(out)
        for issue in stub_truth()["issues"]:
            self.assertNotIn(issue["issue_id"], blob)
        for word in ("expected_amount_minor", "evidence_sets", "provenance", "injections", "\"seed\""):
            self.assertNotIn(word, blob)

    def test_reset_cannot_point_the_grader_elsewhere(self):
        env = local_env()
        bad = [
            {"episode_dir": str(EPISODES / "stub")}, {"split": "../episodes"}, {"split": "stub/.."},
            {"split": "stub", "episode_id": "ep-NOTEXIST00"}, {"split": "nonexistent"}, {"split": "stub", "index": 5},
            {"split": "stub", "truth": {"issues": []}}, {"generator": "../../etc/passwd"}, {"split": "stub", "generator": "stub"},
            "split=stub", {"split": "stub", "index": True},
        ]
        for config in bad:
            with self.subTest(config=config):
                with self.assertRaises(ValueError) as ctx:
                    env.reset(seed=0, config=config)
                msg = str(ctx.exception)
                self.assertNotIn(str(ROOT), msg)
                self.assertNotIn("private", msg)
        env.close()

    def test_isolated_env_rejects_grader_methods(self):
        ch = Channel([sys.executable, "-m", "env.server"], cwd=str(ROOT))
        try:
            for method in ("load", "finish", "truth", "close_grader"):
                with self.subTest(method=method):
                    with self.assertRaises(RemoteError) as ctx:
                        ch.request(method, {"seed": 0, "config": STUB, "mode": "train"})
                    self.assertEqual(ctx.exception.code, "unknown_method")
        finally:
            ch.request("close", {})
            ch.close()


class AgentProcessHasNoReference(unittest.TestCase):
    def test_full_isolated_episode(self):
        script = textwrap.dedent(f"""
            import gc, json, re, sys
            sys.path.insert(0, {str(ROOT)!r})
            opened = []
            sys.addaudithook(lambda ev, args: opened.append(str(args[0])) if ev == "open" and args else None)
            from env import AuditEnv
            from policies import make_policy
            env = AuditEnv()
            obs, info = env.reset(seed=0, config={{"split": "stub", "mode": "train"}})
            pol = make_policy("auditor"); pol.reset(obs, info)
            done, steps, reward = False, 0, None
            while not done:
                obs, reward, term, trunc, info = env.step(pol.act(obs)); steps += 1; done = term or trunc
            env.close()
            keys = {{"expected_amount_minor", "evidence_sets", "amount_tolerance_minor", "scenario_group", "injections"}}
            def scan():
                bad = 0
                for o in gc.get_objects():
                    if isinstance(o, dict):
                        if keys & {{k for k in o if isinstance(k, str)}} or any(isinstance(v, str) and re.fullmatch(r"ISS-[A-Z0-9]{{8}}", v) for v in o.values()):
                            bad += 1
                    elif isinstance(o, (list, tuple)) and any(isinstance(v, str) and re.fullmatch(r"ISS-[A-Z0-9]{{8}}", v) for v in o):
                        bad += 1
                return bad
            private_opens = [p for p in opened if "private" in p or "truth" in p]
            clean = scan()
            from grader.episodes import EpisodeStore, read_episode  # отрицательный контроль: эталон загружен нарочно
            store = EpisodeStore()
            control = read_episode(store.episode_dir("stub", store.split_episodes("stub")[0]))[1]
            print(json.dumps({{"reward": reward, "steps": steps, "f1": info.get("f1"), "private_opens": private_opens,
                               "private_objects": clean, "control_objects": scan()}}))
        """)
        out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        res = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(res["f1"], 1.0)
        self.assertAlmostEqual(res["reward"], 1 - 0.002 * (res["steps"] - 1))
        self.assertEqual(res["private_opens"], [])
        self.assertEqual(res["private_objects"], 0)
        self.assertGreater(res["control_objects"], 0)  # детектор работает


class ObservationLeakScan(unittest.TestCase):
    def test_trajectories_of_all_agent_policies_are_clean(self):
        records = []
        env = local_env()
        for name in ("random", "reconcile", "flag_all", "auditor"):
            for split, idx in (("stub", 0), ("train", 0), ("train", 5), ("dev", 2)):
                for mode in ("train", "eval"):
                    obs, info = env.reset(seed=idx, config={"split": split, "mode": mode})
                    ep = obs["episode_id"]
                    records.append({"episode_id": ep, "step": 0, "action": None, "observation": obs, "info": info})
                    pol = make_policy(name, seed=3)
                    pol.reset(obs, info)
                    done, k = False, 0
                    while not done:
                        action = pol.act(obs)
                        obs, r, term, trunc, info = env.step(action)
                        k += 1
                        done = term or trunc
                        records.append({"episode_id": ep, "step": k, "action": action, "observation": obs, "reward": r,
                                        "terminated": term, "truncated": trunc, "info": info})
        env.close()
        self.assertEqual(scan_records(records, EPISODES), [])
        issue_ids = {i["issue_id"] for d in EPISODES.glob("*/ep-*/private/truth.json")
                     for i in json.loads(d.read_text(encoding="utf-8"))["issues"]}
        blob = json.dumps(records)
        self.assertFalse([i for i in issue_ids if i in blob])

    def test_reset_isolates_episode_state(self):
        env = local_env()
        env.reset(seed=0, config=STUB)
        for _ in range(10):
            env.step({"action": "list_records", "table": "ledger"})
        obs, info = env.reset(seed=1, config={"split": "train"})
        self.assertEqual(obs["remaining_steps"], 100)
        step = env.step({"action": "calculator", "expression": "1"})[0]
        self.assertEqual(step["remaining_steps"], 99)
        env.close()


class TempRootSplits(unittest.TestCase):
    def test_custom_root_without_index(self):
        root = TempRoot()
        try:
            src = next((EPISODES / "stub").glob("ep-*"))
            root.copy("holdout", src)
            env = local_env(episodes_root=root.path)
            obs, _ = env.reset(seed=7, config={"split": "holdout"})
            self.assertEqual(obs["episode_id"], src.name)
            with self.assertRaises(ValueError):
                env.reset(seed=0, config={"split": "train"})  # корень обвязки заменён: открытых сплитов здесь нет
            env.close()
        finally:
            root.cleanup()


if __name__ == "__main__":
    unittest.main()
