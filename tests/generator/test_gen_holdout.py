"""Имитация закрытой проверки: новые сочетания параметров, соль, 5 чистых / 5 взаимодействующих, пары."""

import json
import unittest
from collections import Counter

import gen_helpers  # noqa: F401
from generator.holdout import build_holdout, holdout_composition, stress
from generator.splits import DEFAULT_OUT, iter_episode_dirs, load_splits


class HoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_holdout("unit-test-salt", 20)

    def test_composition_like_jury(self):
        self.assertEqual(Counter(holdout_composition(20)), {"clean": 5, "interacting": 5, "mixed": 10})
        kinds = Counter(e["kind"] for e in self.report["episodes"])
        self.assertEqual(kinds, {"clean": 5, "interacting": 5, "mixed": 10})

    def test_every_episode_and_pair_passes(self):
        for e in self.report["episodes"]:
            with self.subTest(episode=e["episode_id"], kind=e["kind"]):
                self.assertEqual(e["problems"], [])
                self.assertLess(e["generation_seconds"], 5)

    def test_new_parameter_combinations_and_seeds(self):
        open_seeds = {i["seed"] for items in load_splits().values() for i in items}
        open_ids = set()
        for _, path in iter_episode_dirs(DEFAULT_OUT):
            with open(path / "public" / "manifest.json", encoding="utf-8") as fh:
                open_ids.add(json.load(fh)["episode_id"])
        configs = {json.dumps(e["config"], sort_keys=True) for e in self.report["episodes"]}
        self.assertEqual(len(configs), 20)
        for e in self.report["episodes"]:
            self.assertNotIn(e["seed"], open_seeds)
            self.assertNotIn(e["episode_id"], open_ids)
        self.assertGreater(len({e["config"]["timezone"] for e in self.report["episodes"]}), 5)
        self.assertGreater(len({e["config"]["base_currency"] for e in self.report["episodes"]}), 2)

    def test_salt_is_required_and_changes_everything(self):
        with self.assertRaises(ValueError):
            build_holdout("", 2)
        other = build_holdout("another-salt", 2)
        self.assertNotEqual(
            [e["episode_id"] for e in other["episodes"]],
            [e["episode_id"] for e in self.report["episodes"][:2]],
        )

    def test_stress_small(self):
        self.assertEqual(stress(24, "unit")["failures"], 0)


if __name__ == "__main__":
    unittest.main()
