"""Граница доступа политики («Граница доступа» правил жюри): политика — в отдельном ограниченном процессе.

Негативный контроль: враждебная политика tests/env/attack_policies/read_truth.py В ПРОЦЕССЕ обвязки читает
truth.json по известному пути и получает 12 из 12 и награду 1.0 — атака настоящая. В песочнице env/sandbox.py тот же
код не добывает эталон ни одним способом (файлы, процессы, импорт оценщика, gc, ctypes, сеть, /proc, сигналы),
процесс-жертва жив, а ответ — один ложный вывод-отчёт. Политика, которая зависла или упала, не роняет прогон:
её процесс убивается, эпизод завершает среда (truncated / abandoned, пустые findings), следующий эпизод идёт
в новом процессе.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from env_helpers import EPISODES, ROOT
from env import AuditEnv
from env.runner import POLICY_GRACE_S, load_policy_file, run_episode
from env.sandbox import PolicyError, SandboxedPolicy

ATTACK = ROOT / "tests" / "env" / "attack_policies" / "read_truth.py"
VICTIM = EPISODES / "train" / "ep-5C8KTAPXJA"  # 12 ошибок
READ_ROUTES = {"open", "io.FileIO", "os.open", "io.open_code", "pathlib", "subprocess", "os.system", "import grader",
               "gc scan", "import policies.oracle", "os.listdir", "os.scandir", "ctypes", "socket", "code.replace",
               "/proc/self/environ", "/proc/<parent>/mem", "_winapi.OpenProcess", "os.kill"}
SECCOMP_EXPECTED = sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64", "aarch64", "arm64")

FLAKY_POLICY = '''
import time


class Flaky:
    def __init__(self, hang_on=None, crash_on=None):
        self.hang_on, self.crash_on = hang_on, crash_on

    def reset(self, observation, info=None):
        self.episode_id = observation["episode_id"]

    def act(self, observation):
        if self.episode_id == self.hang_on:
            while True:
                time.sleep(0.05)
        if self.episode_id == self.crash_on:
            raise RuntimeError("policy bug")
        return {"name": "submit", "args": {"findings": {"episode_id": self.episode_id, "findings": []}}}


def make_policy(hang_on=None, crash_on=None):
    return Flaky(hang_on, crash_on)
'''


def read_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class Sandbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="sandbox-test-")
        cls.root = Path(cls.tmp.name)
        cls.copy = cls.root / "каталог жюри (копия) 12"
        shutil.copytree(VICTIM, cls.copy)
        cls.truth_paths = [str(cls.copy / "private" / "truth.json"), str(VICTIM / "private" / "truth.json")]
        cls.truth = json.loads((VICTIM / "private" / "truth.json").read_text(encoding="utf-8"))
        assert len(cls.truth["issues"]) == 12
        cls.flaky = cls.root / "flaky_policy.py"
        cls.flaky.write_bytes(FLAKY_POLICY.encode("utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def closed(self, d: Path, **extra) -> dict:
        return {"mode": "closed", "episode_dir": str(d), **extra}

    def test_isolation_is_reported(self):
        with SandboxedPolicy.builtin("reconcile") as policy:
            iso = policy.isolation
        self.assertEqual((iso["process"], iso["audit_hook"], iso["files_open"]), ("separate", True, False))
        if SECCOMP_EXPECTED:
            self.assertEqual(iso["seccomp"], "active")  # на Linux — фильтр ядра, а не только аудит-хук
        else:
            self.assertNotEqual(iso["seccomp"], "active")

    def test_oracle_never_enters_the_sandbox(self):
        with self.assertRaises(ValueError):
            SandboxedPolicy.builtin("oracle")

    def test_negative_control_in_process_attack_reads_the_truth(self):
        """Без песочницы (политика в процессе обвязки) атака работает — значит тест ниже проверяет настоящую границу."""
        log = self.root / "negative.jsonl"
        policy = load_policy_file(ATTACK, {"roots": [str(ROOT)], "paths": self.truth_paths})
        with AuditEnv(grader_log=log) as env:
            run = run_episode(env, policy, 0, self.closed(self.copy))
        rec = read_log(log)[-1]
        self.assertEqual(run["return_seen_by_policy"], 0.0)  # closed: политика награду не видит
        self.assertEqual((rec["tp"], rec["fp"], rec["fn"], rec["reward"]), (12, 0, 0, 1.0))

    def test_attack_from_the_sandbox_gets_nothing(self):
        victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        log = self.root / "attack.jsonl"
        try:
            with AuditEnv(grader_log=log) as env, SandboxedPolicy.from_file(
                    ATTACK, kwargs={"roots": [str(ROOT)], "paths": self.truth_paths, "victim_pid": victim.pid}) as policy:
                iso = policy.isolation
                obs, info = env.reset(seed=0, config=self.closed(self.copy))
                policy.reset(obs, info)
                action = policy.act(obs)
                out = env.step(action)
            time.sleep(0.2)
            self.assertIsNone(victim.poll(), "the sandboxed policy killed another process")
        finally:
            victim.kill()
            victim.wait()
        findings = action["args"]["findings"]["findings"]
        self.assertEqual(len(findings), 1)
        report = json.loads(findings[0]["explanation"])
        self.assertIs(report["stolen"], False)
        self.assertEqual(set(report["ok"]) & READ_ROUTES, set(), report)
        allowed = set() if iso["seccomp"] == "active" else {"os.stat"}  # Windows: только метаданные, без содержимого
        self.assertLessEqual(set(report["ok"]), allowed, report)
        self.assertGreaterEqual(sum(report["tried"].values()), 20)  # попыток было много — все отбиты
        self.assertEqual((out[1], out[2], out[4]), (0.0, True, {}))
        rec = read_log(log)[-1]
        self.assertEqual((rec["tp"], rec["fp"], rec["fn"], rec["reward"]), (0, 1, 12, -1.0))

    def test_hung_policy_is_stopped_and_the_environment_ends_the_episode(self):
        stub = EPISODES / "stub" / "ep-SUW6BEVZNR"
        log = self.root / "hung.jsonl"
        with AuditEnv(grader_log=log) as env, SandboxedPolicy.from_file(
                self.flaky, kwargs={"hang_on": "ep-5C8KTAPXJA"}) as policy:
            t0 = time.monotonic()
            run = run_episode(env, policy, 0, self.closed(self.copy, time_limit_s=1.0))
            elapsed = time.monotonic() - t0
            self.assertIn("policy_error", run)
            self.assertLess(elapsed, 1.0 + POLICY_GRACE_S + 5.0)
            run2 = run_episode(env, policy, 0, self.closed(stub))  # новый процесс политики
            self.assertNotIn("policy_error", run2)
            self.assertTrue(run2["terminated"])
            self.assertEqual(policy.restarts, 1)
        recs = read_log(log)
        self.assertEqual([(r["episode_id"], r["end_reason"], r["truncated"], r["tp"], r["fp"]) for r in recs],
                         [("ep-5C8KTAPXJA", "time_limit", True, 0, 0), ("ep-SUW6BEVZNR", "submit", False, 0, 0)])

    def test_crashing_policy_is_reported_and_the_episode_is_abandoned(self):
        stub = EPISODES / "stub" / "ep-SUW6BEVZNR"
        log = self.root / "crash.jsonl"
        with AuditEnv(grader_log=log) as env, SandboxedPolicy.from_file(
                self.flaky, kwargs={"crash_on": "ep-5C8KTAPXJA"}) as policy:
            run = run_episode(env, policy, 0, self.closed(self.copy))
            self.assertIn("policy_error", run)
            run2 = run_episode(env, policy, 0, self.closed(stub))
            self.assertTrue(run2["terminated"])
            self.assertEqual(policy.restarts, 0)  # исключение в коде политики процесс не ломает
            with self.assertRaises(PolicyError):
                policy.reset({"episode_id": "ep-5C8KTAPXJA"}, {})
                policy.act({})
        recs = read_log(log)
        self.assertEqual([(r["end_reason"], r["truncated"], r["tp"], r["fp"]) for r in recs],
                         [("abandoned", True, 0, 0), ("submit", False, 0, 0)])


if __name__ == "__main__":
    unittest.main()
