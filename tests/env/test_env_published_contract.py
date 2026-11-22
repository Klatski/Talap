"""Опубликованный агенту контракт совпадает с тем, что делают генератор и оценщик.

* τ (допуск суммы) по типу: текст задачи, константа среды, константа генератора и amount_tolerance_minor
  каждой ошибки в открытых эпизодах — одни и те же числа; имени поля truth в первом наблюдении нет.
* Чужой episode_id не назван ошибкой структуры (он ею и не является, grader/validate.py).
* Набор доказательств неразнесённого платежа на несколько счетов описан полностью (строки 1200, счета, курсы).
"""

import json
import unittest

from env_helpers import ROOT
from env.constants import AMOUNT_TOLERANCE_MINOR
from env.observation import ISSUE_TYPES, MATCHING
from generator.render import AMOUNT_TOLERANCE_MINOR as GENERATOR_TOLERANCE
from grader.validate import ISSUE_TYPES as GRADER_TYPES


class PublishedToleranceTests(unittest.TestCase):
    def test_one_tolerance_everywhere(self):
        self.assertEqual(set(AMOUNT_TOLERANCE_MINOR), set(GRADER_TYPES))
        self.assertEqual(AMOUNT_TOLERANCE_MINOR, GENERATOR_TOLERANCE)
        for kind, tolerance in AMOUNT_TOLERANCE_MINOR.items():
            with self.subTest(type=kind):
                self.assertIn(f"{kind} {tolerance}", MATCHING)
        self.assertNotIn("every type", MATCHING)
        published = json.dumps({"matching": MATCHING, "issue_types": ISSUE_TYPES})
        self.assertNotIn("amount_tolerance_minor", published)  # имя закрытого поля truth — для сканеров утечка

    def test_open_episodes_use_the_published_tolerance(self):
        seen = set()
        for split in ("train", "dev", "stub"):
            for truth_path in sorted((ROOT / "episodes" / split).glob("*/private/truth.json")):
                for issue in json.loads(truth_path.read_text(encoding="utf-8"))["issues"]:
                    with self.subTest(episode=truth_path.parent.parent.name, issue=issue["issue_id"]):
                        self.assertEqual(issue["amount_tolerance_minor"], AMOUNT_TOLERANCE_MINOR[issue["type"]])
                    seen.add(issue["type"])
        self.assertEqual(seen, set(AMOUNT_TOLERANCE_MINOR))


class PublishedRulesTests(unittest.TestCase):
    def test_foreign_episode_id_is_not_called_malformed(self):
        self.assertNotIn("another episode_id", MATCHING)
        self.assertIn("manifest.episode_id", MATCHING)

    def test_multi_invoice_unapplied_evidence_is_described(self):
        text = ISSUE_TYPES["unapplied_payment"]["evidence"]
        for word in ("2300", "1200", "every referenced invoice", "fx_id"):
            self.assertIn(word, text)


if __name__ == "__main__":
    unittest.main()
