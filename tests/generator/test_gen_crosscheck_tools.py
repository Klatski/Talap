"""Инструменты перекрёстной проверки A: атаки на оценщик и сканер утечек в траекториях.

Тест «против оценщика B» автоматически включится, когда в grader/ появится функция
score_submission(truth, submission, inspection_steps) -> dict с ключами valid, tp, fp, fn, reward
(предложение из docs/contract.md, раздел 7). До этого он пропускается.
"""

import importlib
import json
import tempfile
import unittest
from pathlib import Path

from gen_helpers import CONFIGS, episode
from generator.attacks import build_attacks, spec_reward
from generator.fileio import write_episode_dir
from generator.leakscan import scan_records
from generator.schema_check import load_schema, validate

SEMANTIC_ONLY_INVALID: set[str] = set()  # схема валидна, а ответ неверен; чужой episode_id сюда больше не входит


def _cases(key, seed):
    ep = episode(key, seed)
    return ep, {c["name"]: c for c in build_attacks(ep.public, ep.truth, seed)}


class AttackSuiteTests(unittest.TestCase):
    def test_schema_agrees_with_expected_validity(self):
        schema = load_schema("findings")
        for key in ("clean", "all_types", "scenarios"):
            _, cases = _cases(key, 1)
            for name, case in cases.items():
                ok = not validate(case["submission"], schema)
                with self.subTest(config=key, case=name):
                    if name in SEMANTIC_ONLY_INVALID:
                        self.assertTrue(ok)
                    else:
                        self.assertEqual(ok, case["expected"]["valid"])

    def test_spec_relations_hold(self):
        for key in ("all_types", "scenarios", "heavy"):
            for seed in range(3):
                _, c = _cases(key, seed)
                r = {name: case["expected"]["reward"] for name, case in c.items()}
                with self.subTest(config=key, seed=seed):
                    self.assertEqual(r["oracle"], 1.0)
                    self.assertGreater(r["oracle"], r["silence"])  # полный ответ лучше молчания
                    self.assertLess(r["oracle_duplicated"], r["oracle"])  # дубли не выгодны
                    self.assertLess(r["flag_all_zero_amount"], 0)  # «пометить всё» плохо
                    self.assertLess(r["oracle"] - r["oracle_after_40_steps"], 0.1)  # шаги — лишь немного
                    self.assertGreater(r["oracle_after_40_steps"], r["oracle_duplicated"])
                    for name in ("wrong_currency", "guessed_affected_ids", "missing_evidence", "fake_evidence"):
                        self.assertEqual(r[name], -1.0)
                    for name, value in r.items():
                        if name.startswith("malformed_"):
                            self.assertEqual(value, -1.0)

    def test_clean_company_silence_beats_false_accusations(self):
        for seed in range(3):
            _, c = _cases("clean", seed)
            self.assertEqual(c["silence"]["expected"]["reward"], 1.0)
            self.assertEqual(c["flag_all_zero_amount"]["expected"]["reward"], -1.0)

    def test_spec_reward_edge_cases(self):
        self.assertEqual(spec_reward(0, 0, 0, 0), 1.0)  # пустой эталон и пустой ответ
        self.assertEqual(spec_reward(0, 3, 0, 0), -1.0)  # пустой эталон, есть выводы
        self.assertEqual(spec_reward(0, 0, 4, 0), -1.0)
        self.assertAlmostEqual(spec_reward(2, 0, 0, 10), 0.98)
        self.assertEqual(spec_reward(5, 0, 0, 0, valid=False), -1.0)
        self.assertEqual(spec_reward(0, 0, 5, 1000), -1.0)  # clip


class AgainstGraderTests(unittest.TestCase):
    """Прогон атак через оценщик B. Пропускается, пока score_submission не реализован."""

    def setUp(self):
        try:
            module = importlib.import_module("grader")
        except ImportError:
            self.skipTest("grader package not available yet")
        self.score = getattr(module, "score_submission", None)
        if self.score is None:
            self.skipTest("grader.score_submission not implemented yet")

    def test_attacks(self):
        for key in ("clean", "all_types", "scenarios", "heavy"):
            ep, cases = _cases(key, 0)
            for name, case in cases.items():
                result = self.score(ep.truth, case["submission"], case["inspection_steps"])
                exp = case["expected"]
                with self.subTest(config=key, case=name, note=case["note"]):
                    self.assertEqual(bool(result["valid"]), exp["valid"])
                    if exp["valid"]:
                        self.assertEqual((result["tp"], result["fp"], result["fn"]), (exp["tp"], exp["fp"], exp["fn"]))
                    self.assertAlmostEqual(result["reward"], exp["reward"], places=9)


class LeakScanTests(unittest.TestCase):
    def setUp(self):
        self.ep = episode("scenarios", 2)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_episode_dir(self.root / "train" / self.ep.episode_id, self.ep.public, self.ep.private)

    def tearDown(self):
        self.tmp.cleanup()

    def _record(self, observation, info=None):
        return {"episode_id": self.ep.episode_id, "step": 1, "action": {"action": "list_records", "table": "ledger"},
                "observation": observation, "reward": 0.0, "terminated": False, "truncated": False, "info": info or {}}

    def test_public_records_are_clean(self):
        ledger_head = self.ep.public["ledger.csv"].splitlines()[:20]
        invoices = [json.loads(line) for line in self.ep.public["invoices.jsonl"].splitlines()[:10]]
        recs = [self._record({"rows": ledger_head, "remaining_steps": 99}), self._record({"records": invoices})]
        self.assertEqual(scan_records(recs, self.root), [])

    def test_leaks_are_detected(self):
        issue = self.ep.truth["issues"][0]
        leaks = [
            self._record({"rows": []}, {"issue_id": issue["issue_id"]}),
            self._record({"debug": {"expected_amount_minor": 5}}),
            self._record({"error": "file not found: episodes/train/x/private/truth.json"}),
            self._record({"hint": issue["scenario"] or "fx_partial_boundary_payment"}),
            self._record({}, {"seed": self.ep.provenance["seed"]}),
        ]
        for rec in leaks:
            with self.subTest(rec=rec["observation"], info=rec["info"]):
                self.assertTrue(scan_records([rec], self.root))


if __name__ == "__main__":
    unittest.main()
