"""Метрики и награда — формулы ТЗ, граничные случаи, клиппинг, стоимость шагов."""

import unittest
from fractions import Fraction

import grader_helpers  # noqa: F401
from grader import experimental_reward, prf, score_submission, shared_reward
from grader_helpers import finding_for, issue, sub, truth_of


class MetricTests(unittest.TestCase):
    def test_prf(self):
        self.assertEqual(prf(3, 1, 2), (Fraction(3, 4), Fraction(3, 5), Fraction(6, 9)))
        self.assertEqual(prf(0, 0, 0), (0, 0, 1))      # пустой эталон и пустой ответ
        self.assertEqual(prf(0, 5, 0), (0, 0, 0))      # пустой эталон, есть выводы
        self.assertEqual(prf(0, 0, 4), (0, 0, 0))
        self.assertEqual(prf(2, 0, 0), (1, 1, 1))

    def test_shared_reward_formula(self):
        self.assertEqual(shared_reward(Fraction(1), 0), 1.0)
        self.assertAlmostEqual(shared_reward(Fraction(1), 10), 0.98)
        self.assertAlmostEqual(shared_reward(Fraction(1, 2), 0), 0.0)
        self.assertEqual(shared_reward(Fraction(0), 0), -1.0)
        self.assertEqual(shared_reward(Fraction(0), 500), -1.0)  # клиппинг снизу
        self.assertEqual(shared_reward(Fraction(1), 0, valid=False), -1.0)
        for steps in range(0, 101):
            self.assertAlmostEqual(shared_reward(Fraction(3, 4), steps), 0.5 - 0.002 * steps)

    def test_experimental_reward_is_separate(self):
        self.assertEqual(experimental_reward(Fraction(1), 10, True, truncated=True), -1.0)
        self.assertEqual(experimental_reward(Fraction(1), 10, True, truncated=False), shared_reward(Fraction(1), 10))
        r = score_submission(truth_of(), sub([]), 100, truncated=True)
        self.assertAlmostEqual(r["reward"], 0.8)
        self.assertEqual(r["reward_experimental"], -1.0)

    def test_step_economy_only_slightly_separates_equal_answers(self):
        t = truth_of(issue())
        best = score_submission(t, sub([finding_for(issue())]), 0)["reward"]
        slow = score_submission(t, sub([finding_for(issue())]), 99)["reward"]
        self.assertLessEqual(best - slow, 0.2 + 1e-12)
        self.assertGreater(slow, score_submission(t, sub([]), 0)["reward"])

    def test_inspection_steps_validation(self):
        for bad in (-1, True, 1.5, "3", None):
            with self.assertRaises(ValueError):
                score_submission(truth_of(), sub([]), bad)

    def test_reward_is_deterministic_float(self):
        t = truth_of(issue(), issue("ISS-AAAAAAA9", affected=("INV-B",)))
        a = score_submission(t, sub([finding_for(issue())]), 37)["reward"]
        b = score_submission(t, sub([finding_for(issue())]), 37)["reward"]
        self.assertEqual(a, b)
        self.assertEqual(a, float(2 * Fraction(2, 3) - 1 - Fraction(74, 1000)))


if __name__ == "__main__":
    unittest.main()
