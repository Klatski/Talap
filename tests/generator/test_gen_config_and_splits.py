"""Валидация конфигов, открытые сплиты в репозитории и oracle."""

import json
import unittest
from collections import Counter

import gen_helpers  # noqa: F401
from generator.audit import compare_with_truth, parse_public
from generator.config import ConfigError, config_digest, normalize_config
from generator.fileio import read_episode_dir
from generator.schema_check import load_schema, validate, validate_episode_texts
from generator.splits import DEFAULT_OUT, check_splits, iter_episode_dirs, load_splits
from policies.oracle import OraclePolicy, oracle_findings


class ConfigTests(unittest.TestCase):
    def test_defaults_and_digest_are_stable(self):
        a = normalize_config({"name": "x"})
        b = normalize_config({"name": "x", "errors": {"duplicate_invoice": 0}})
        self.assertEqual(a, b)
        self.assertEqual(config_digest(a), config_digest(b))

    def test_invalid_configs_rejected(self):
        bad = [
            {"unknown": 1},
            {"errors": {"nope": 1}},
            {"errors": {"duplicate_invoice": -1}},
            {"errors": {"duplicate_invoice": 13}},
            {"errors": {"duplicate_invoice": [3, 1]}},
            {"errors": {"duplicate_invoice": 1.5}},
            {"errors": {"duplicate_invoice": 6, "cutoff_error": 6}, "scenarios": {"fx_partial_boundary_payment": 1}},
            {"size": "huge"},
            {"base_currency": "XXX"},
            {"foreign_currencies": ["KZT"], "base_currency": "KZT"},
            {"foreign_currencies": [], "errors": {"fx_rate_error": 1}},
            {"timezone": "+5"},
            {"timezone": "+14:00"},
            {"period": "2026-13"},
            {"distractors": {"posting_lag": "yes"}},
            {"config_version": "2.0"},
        ]
        for raw in bad:
            with self.subTest(raw=raw):
                with self.assertRaises(ConfigError):
                    normalize_config(raw)

    def test_preset_configs_match_schema(self):
        from generator.splits import CONFIG_DIR

        schema = load_schema("generator_config")
        for path in sorted(CONFIG_DIR.glob("*.json")):
            if path.name == "splits.json":
                continue
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            with self.subTest(config=path.name):
                self.assertEqual(validate(raw, schema), [])
                normalize_config(raw)


class CommittedSplitsTests(unittest.TestCase):
    def test_split_sizes(self):
        splits = load_splits()
        self.assertEqual(len(splits["train"]), 10)
        self.assertEqual(len(splits["dev"]), 10)
        train_seeds = {i["seed"] for i in splits["train"]}
        dev_seeds = {i["seed"] for i in splits["dev"]}
        self.assertFalse(train_seeds & dev_seeds)

    def test_committed_episodes_regenerate_byte_for_byte(self):
        self.assertEqual(check_splits(), [])

    def test_committed_episodes_are_valid(self):
        for split, path in iter_episode_dirs(DEFAULT_OUT):
            public, private = read_episode_dir(path)
            truth = json.loads(private["truth.json"])
            with self.subTest(split=split, episode=path.name):
                self.assertEqual(validate_episode_texts(public, private), [])
                self.assertEqual(compare_with_truth(parse_public(public), truth), [])
                findings = oracle_findings(truth)
                self.assertEqual(validate(findings, load_schema("findings")), [])

    def test_split_coverage(self):
        """В каждом сплите: >=2 чистых, >=2 с взаимодействующими ошибками, все 4 типа."""
        for split in ("train", "dev"):
            clean = interacting = 0
            types = Counter()
            for name, path in iter_episode_dirs(DEFAULT_OUT, (split,)):
                _, private = read_episode_dir(path)
                truth = json.loads(private["truth.json"])
                clean += not truth["issues"]
                interacting += any(i["scenario"] for i in truth["issues"])
                types.update(i["type"] for i in truth["issues"])
            with self.subTest(split=split):
                self.assertGreaterEqual(clean, 2)
                self.assertGreaterEqual(interacting, 2)
                self.assertEqual(set(types), {"duplicate_invoice", "unapplied_payment", "fx_rate_error", "cutoff_error"})


class OracleTests(unittest.TestCase):
    def test_oracle_submits_once_with_full_evidence(self):
        for split, path in iter_episode_dirs(DEFAULT_OUT, ("stub",)):
            policy = OraclePolicy.from_episode_dir(path)
            policy.reset({}, {})
            action = policy.act({})
            self.assertEqual(action["action"], "submit")
            _, private = read_episode_dir(path)
            truth = json.loads(private["truth.json"])
            self.assertEqual(len(action["findings"]["findings"]), len(truth["issues"]))
            for finding, issue in zip(action["findings"]["findings"], truth["issues"]):
                self.assertEqual(finding["type"], issue["type"])
                self.assertIn(sorted(finding["evidence_ids"]), [sorted(s) for s in issue["evidence_sets"]])
                self.assertLessEqual(len(finding["evidence_ids"]), 10)


if __name__ == "__main__":
    unittest.main()
