"""Обязательные свойства награды (ТЗ, раздел 05) на всех открытых эпизодах + атаки A (generator.attacks).

* полный верный ответ с доказательствами лучше молчания на ошибочных книгах;
* молчание лучше ложных обвинений на чистых;
* дубли не дают выгоды; пометка всех строк/типов даёт плохой результат;
* экономия шагов лишь немного различает равные ответы;
* угаданные ID, неподтверждённые доказательства, чужая валюта, неверный ввод -> нет выгоды.
"""

import copy
import unittest

from grader_helpers import open_episodes
from generator.attacks import build_attacks
from grader import score_submission
from policies.oracle import oracle_findings

EPISODES = open_episodes()


def reward(truth, findings, steps=0):
    return score_submission(truth, {"episode_id": truth["episode_id"], "findings": findings}, steps)["reward"]


class RewardProperties(unittest.TestCase):
    def test_attack_suite_of_A_on_every_open_episode(self):
        total = 0
        for split, name, public, truth, _ in EPISODES:
            for case in build_attacks(public, truth, seed=1):
                r = score_submission(truth, case["submission"], case["inspection_steps"])
                exp = case["expected"]
                with self.subTest(episode=name, case=case["name"]):
                    self.assertEqual(r["valid"], exp["valid"])
                    if exp["valid"]:
                        self.assertEqual((r["tp"], r["fp"], r["fn"]), (exp["tp"], exp["fp"], exp["fn"]))
                    self.assertAlmostEqual(r["reward"], exp["reward"], places=12)
                total += 1
        self.assertGreater(total, 600)

    def test_relations(self):
        for split, name, public, truth, _ in EPISODES:
            oracle = oracle_findings(truth)["findings"]
            with self.subTest(episode=name):
                if truth["issues"]:
                    self.assertGreater(reward(truth, oracle), reward(truth, []))                 # полный ответ > молчание
                    self.assertLess(reward(truth, oracle + copy.deepcopy(oracle)), reward(truth, oracle))  # дубли
                    partial = oracle[: max(1, len(oracle) // 2)]
                    self.assertLess(reward(truth, partial + copy.deepcopy(partial)), reward(truth, partial) + 1e-12)
                else:
                    fake = [{"type": t, "affected_ids": ["INV-ZZZZZZZ"], "claimed_amount_minor": 1, "currency": "USD",
                             "evidence_ids": [], "explanation": ""} for t in ("cutoff_error", "fx_rate_error")]
                    self.assertGreater(reward(truth, []), reward(truth, fake))                    # молчание > ложь
                    self.assertEqual(reward(truth, []), 1.0)
                    self.assertEqual(reward(truth, fake[:1]), -1.0)
                self.assertLessEqual(reward(truth, oracle, 0) - reward(truth, oracle, 99), 0.2 + 1e-12)  # шаги — немного
                self.assertEqual(reward(truth, oracle), 1.0)

    def test_flag_everything_is_bad(self):
        for split, name, public, truth, _ in EPISODES:
            cases = {c["name"]: c for c in build_attacks(public, truth)}
            with self.subTest(episode=name):
                self.assertLessEqual(score_submission(truth, cases["flag_all_zero_amount"]["submission"], 0)["reward"], -0.9)

    def test_cross_episode_answers_do_not_score(self):
        """Ответ oracle одного эпизода, поданный в другой (с любым episode_id), не даёт TP: ID записей разные.
        Чужой episode_id — не ошибка структуры (п. 8 жюри), а предупреждение в журнале оценщика."""
        for (_, _, _, a, _), (_, _, _, b, _) in zip(EPISODES, EPISODES[1:]):
            foreign = oracle_findings(a)["findings"]
            r = score_submission(b, {"episode_id": b["episode_id"], "findings": foreign}, 0)
            self.assertEqual(r["tp"], 0)
            r2 = score_submission(b, {"episode_id": a["episode_id"], "findings": foreign}, 0)
            self.assertTrue(r2["valid"])
            self.assertEqual((r2["tp"], r2["reward"]), (0, r["reward"]))
            self.assertTrue(r2["warnings"])


if __name__ == "__main__":
    unittest.main()
