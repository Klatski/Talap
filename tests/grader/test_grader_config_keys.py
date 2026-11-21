"""Ключи config, похожие на известные (опечатки), версия кода в журнале и запись журнала из нескольких процессов.

* Опечатка в ключе верхнего уровня — ошибка, а не молчаливо другой эпизод: {"error": …}, {"Errors": …},
  {"errors ": …} у генератора (и внутри team_config), "Mode", "episode-dir", "team_confg" у reset.
  Явно чужие ключи (kind, label, note, case, seed, episode, ...) по-прежнему игнорируются и пишутся в журнал.
* code_version (grader/codeversion.py): sha256 исходников команды, без CRLF-зависимости, файлы жюри рядом его
  не меняют; пишется в каждую строку журнала оценщика.
* Несколько процессов оценщика дописывают один журнал без потери и перемешивания строк (Windows: O_APPEND
  там не атомарен между процессами, см. grader/session.py::_open_append).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grader import codeversion  # noqa: E402
from grader.episodes import (GRADER_KEYS, RESET_ENV_KEYS, EpisodeLoadError, EpisodeStore, generator_keys,  # noqa: E402
                             misspelled_key, resolve_generator_config)
from grader.judge import PROTOCOL_KEYS  # noqa: E402
from grader.session import GraderSession  # noqa: E402

ONE_DUPLICATE = {"duplicate_invoice": 1}
FOREIGN = ("kind", "label", "note", "notes", "case", "case_id", "seed", "tag", "id", "type", "title", "jury_label",
           "jury_case", "description", "comment", "difficulty", "check", "fixture", "level", "salt", "group", "test",
           "D06", "truth", "episode", "episodes", "config")


class MisspelledKeys(unittest.TestCase):
    def test_generator_key_typos_are_recognised(self):
        known = generator_keys() | PROTOCOL_KEYS
        for key, expected in (("error", "errors"), ("Errors", "errors"), ("errors ", "errors"), ("ERRORS", "errors"),
                              ("erors", "errors"), ("scenario", "scenarios"), ("Scenarois", "scenarios"),
                              ("distracters", "distractors"), ("Size", "size"), ("sizes", "size"),
                              ("presets", "preset"), ("base_currencies", "base_currency"), ("time zone", "timezone"),
                              ("rounding-mode", "rounding_mode"), ("team_confg", "team_config"),
                              ("TeamConfig", "team_config"), ("Mode", "mode")):
            with self.subTest(key=key):
                self.assertEqual(misspelled_key(key, known), expected)

    def test_reset_key_typos_are_recognised(self):
        known = GRADER_KEYS | RESET_ENV_KEYS
        for key, expected in (("Mode", "mode"), ("mode ", "mode"), ("episode-dir", "episode_dir"),
                              ("episode_dri", "episode_dir"), ("EpisodeDir", "episode_dir"),
                              ("team_confg", "team_config"), ("Split", "split"), ("max_step", "max_steps"),
                              ("time_limit", "time_limit_s")):
            with self.subTest(key=key):
                self.assertEqual(misspelled_key(key, known), expected)

    def test_foreign_keys_and_known_keys_are_not_typos(self):
        for known in (generator_keys() | PROTOCOL_KEYS, GRADER_KEYS | RESET_ENV_KEYS):
            for key in FOREIGN + tuple(known):
                with self.subTest(key=key):
                    self.assertIsNone(misspelled_key(key, known))

    def test_generator_config_typo_is_an_error_and_foreign_keys_pass(self):
        for config in ({"error": ONE_DUPLICATE}, {"Errors": ONE_DUPLICATE}, {"errors ": ONE_DUPLICATE},
                       {"size": "small", "scenario": {"fx_partial_boundary_payment": 1}}):
            with self.subTest(config=sorted(config)), self.assertRaises(EpisodeLoadError) as ctx:
                resolve_generator_config(config, [], prefix="team_config.")
            self.assertIn("misspelling", str(ctx.exception))
            self.assertIn("team_config.", str(ctx.exception))
        ignored: list = []
        cfg = resolve_generator_config({"kind": "D06", "label": "x", "errors": ONE_DUPLICATE}, ignored)
        self.assertEqual(cfg["errors"]["duplicate_invoice"], 1)
        self.assertEqual(ignored, ["kind", "label"])

    def test_reset_typo_is_an_error_not_another_episode(self):
        store = EpisodeStore()
        for config in ({"Mode": "closed", "split": "stub"}, {"episode-dir": "x"}, {"team_confg": {}},
                       {"Split": "dev"}, {"team_config": {"Errors": ONE_DUPLICATE}}):
            with self.subTest(config=sorted(config)), self.assertRaises(EpisodeLoadError) as ctx:
                store.load(0, config)
            self.assertIn("misspelling", str(ctx.exception))
        loaded = store.load(0, {"split": "stub", "kind": "C01", "note": 1})
        self.assertEqual(loaded.source["ignored_config_keys"], ["kind", "note"])

    def test_error_message_is_bounded(self):
        with self.assertRaises(EpisodeLoadError) as ctx:
            resolve_generator_config({"Errors" + " " * 500: ONE_DUPLICATE}, [])
        self.assertLess(len(str(ctx.exception)), 200)


class CodeVersion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="codeversion-")
        self.root = Path(self.tmp.name)
        for rel in ("env/a.py", "grader/b.py", "generator/configs/p.json", "schemas/s.schema.json", "check.py"):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_bytes(f"# {rel}\nx = 1\n".encode())

    def tearDown(self):
        self.tmp.cleanup()

    def test_digest_covers_team_sources_only_and_ignores_crlf(self):
        base, count = codeversion.code_digest(self.root)
        self.assertEqual(count, 5)
        (self.root / "my_jury_policy.py").write_text("x = 2\n")  # файл жюри рядом — не код команды
        (self.root / "env" / "__pycache__").mkdir()
        (self.root / "env" / "__pycache__" / "a.cpython-311.py").write_text("junk")
        (self.root / "env" / "notes.txt").write_text("not code")
        self.assertEqual(codeversion.code_digest(self.root), (base, 5))
        (self.root / "env" / "a.py").write_bytes(b"# env/a.py\r\nx = 1\r\n")  # checkout с autocrlf
        self.assertEqual(codeversion.code_digest(self.root)[0], base)
        (self.root / "env" / "a.py").write_bytes(b"# env/a.py\nx = 2\n")
        self.assertNotEqual(codeversion.code_digest(self.root)[0], base)
        (self.root / "env" / "a.py").write_bytes(b"# env/a.py\nx = 1\n")
        (self.root / "generator" / "configs" / "p.json").write_bytes(b"{}")
        self.assertNotEqual(codeversion.code_digest(self.root)[0], base)

    def test_git_commit_from_loose_and_packed_refs(self):
        git = self.root / ".git"
        self.assertIsNone(codeversion.git_commit(self.root))
        (git / "refs" / "heads").mkdir(parents=True)
        (git / "HEAD").write_text("ref: refs/heads/main\n")
        (git / "packed-refs").write_text("# pack-refs with: peeled\n" + "a" * 40 + " refs/heads/main\n")
        self.assertEqual(codeversion.git_commit(self.root), "a" * 40)
        (git / "refs" / "heads" / "main").write_text("b" * 40 + "\n")
        self.assertEqual(codeversion.git_commit(self.root), "b" * 40)
        (git / "HEAD").write_text("c" * 40)  # detached HEAD
        self.assertEqual(codeversion.git_commit(self.root), "c" * 40)

    def test_every_episode_record_carries_the_code_version(self):
        expected = json.loads(subprocess.run([sys.executable, "-m", "grader.codeversion"], cwd=ROOT,
                                             capture_output=True, text=True, check=True).stdout)
        self.assertRegex(expected["sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(expected["files"], 50)
        log = self.root / "log.jsonl"
        session = GraderSession(log_path=log)
        session.load(0, {"split": "stub"}, "eval")
        session.finish(None, 0, 0, "abandoned")
        record = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(record["code_version"], expected)


WRITER = r"""
import sys
sys.path.insert(0, sys.argv[1])
from grader.session import append_private_log
path, tag, n = sys.argv[2], sys.argv[3], int(sys.argv[4])
for i in range(n):
    append_private_log(path, '{"writer": "%s", "i": %d, "pad": "%s"}\n' % (tag, i, tag * 20000))
"""


class ConcurrentLog(unittest.TestCase):
    def test_parallel_graders_append_whole_lines(self):
        tmp = Path(tempfile.mkdtemp(prefix="grader-log-race-"))
        try:
            log = tmp / "shared.jsonl"
            procs = [subprocess.Popen([sys.executable, "-c", WRITER, str(ROOT), str(log), tag, "40"])
                     for tag in "abcd"]
            for p in procs:
                self.assertEqual(p.wait(timeout=120), 0)
            lines = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 160)
            records = [json.loads(line) for line in lines]  # ни одной склеенной или оборванной строки
            for tag in "abcd":
                mine = [r["i"] for r in records if r["writer"] == tag]
                self.assertEqual(mine, list(range(40)))
                self.assertTrue(all(r["pad"] == r["writer"] * 20000 for r in records))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
