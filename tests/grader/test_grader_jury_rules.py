"""Блок 3 жюри (20 тестов matcher и награды) и открытые примеры M01–M12 — дословно по «Определениям проверок».

Всё идёт через закрытый вызов жюри judge_adapter.score_episode(episode_dir, findings, inspection_steps) на
фикстуре-каталоге с произвольным именем: открытые файлы — заглушка stub, эталон — синтетический, как в PDF:
одна ошибка A типа T с канонической записью R1, суммой 10000 KZT, допустимым набором {D1, D2} и τ = 0.
inspection_steps по умолчанию 10. Вещественные результаты сравниваются с абсолютной погрешностью 1e-9.
"""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from fractions import Fraction as Fr
from pathlib import Path

from grader_helpers import EPISODES, ROOT  # noqa: F401 — ROOT кладёт корень репозитория в sys.path
import judge_adapter
from grader.episodes import EpisodeLoadError, load_episode_dir
from grader.matcher import match
from grader.session import GraderSession

STUB = EPISODES / "stub" / "ep-SUW6BEVZNR"
EPS = 1e-9
T, T_OTHER = "duplicate_invoice", "cutoff_error"
TYPES = ("duplicate_invoice", "unapplied_payment", "fx_rate_error", "cutoff_error")


def _ids():
    with open(STUB / "public" / "invoices.jsonl", encoding="utf-8") as fh:
        invoices = [json.loads(line) for line in fh if line.strip()]
    kzt = [i for i in invoices if i["currency"] == "KZT"]
    return [i["invoice_id"] for i in kzt[:5]], [i["doc_id"] for i in invoices[:3]]


(R1, R2, R3, R4, R5), (D1, D2, D3) = _ids()
U = "U-ABSENT-0000"
ISSUE_A = {"issue_id": "ISS-A", "type": T, "affected_ids": [R1], "expected_amount_minor": 10000, "currency": "KZT",
           "amount_tolerance_minor": 0, "evidence_sets": [[D1, D2]]}
ISSUE_B = {"issue_id": "ISS-B", "type": "unapplied_payment", "affected_ids": [R2], "expected_amount_minor": 5000,
           "currency": "KZT", "amount_tolerance_minor": 0, "evidence_sets": [[D3]]}


def F(**changes):
    """Правильный вывод F для A (с изменениями теста)."""
    f = {"type": T, "affected_ids": [R1], "claimed_amount_minor": 10000, "currency": "KZT", "evidence_ids": [D1, D2],
         "explanation": "Invoice R1 is posted twice."}
    f.update(changes)
    return f


def F_B():
    return {"type": "unapplied_payment", "affected_ids": [R2], "claimed_amount_minor": 5000, "currency": "KZT",
            "evidence_ids": [D3], "explanation": "Payment not applied."}


def reward(tp, fp, fn, steps=10):
    p = Fr(tp, tp + fp) if tp + fp else Fr(0)
    r = Fr(tp, tp + fn) if tp + fn else Fr(0)
    f1 = Fr(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else Fr(1)
    return max(Fr(-1), min(Fr(1), 2 * f1 - 1 - Fr(2, 1000) * steps)), p, r, f1


class JuryFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="jury-block3-")
        root = Path(cls.tmp.name)
        cls.log = root / "журнал оценщика.jsonl"
        cls._old_log = os.environ.get("AUDIT_GRADER_LOG")
        os.environ["AUDIT_GRADER_LOG"] = str(cls.log)
        cls.dirs = {}
        for key, issues, layout in (("A", [ISSUE_A], "split"), ("empty", [], "split"), ("AB", [ISSUE_A, ISSUE_B], "flat")):
            d = root / f"фикстура {key} (копия жюри) №1"
            truth = {"schema_version": "1.0", "episode_id": "ep-SUW6BEVZNR", "issues": issues}
            if layout == "split":
                shutil.copytree(STUB / "public", d / "public")
                (d / "private").mkdir()
                (d / "private" / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
            else:  # плоская раскладка: открытые файлы и truth.json в одном каталоге
                shutil.copytree(STUB / "public", d)
                (d / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
            cls.dirs[key] = str(d)
        known = load_episode_dir(cls.dirs["A"]).public_ids
        assert {R1, R2, R3, R4, R5, D1, D2, D3} <= known and U not in known  # U выбран по данным фикстуры

    @classmethod
    def tearDownClass(cls):
        if cls._old_log is None:
            os.environ.pop("AUDIT_GRADER_LOG", None)
        else:
            os.environ["AUDIT_GRADER_LOG"] = cls._old_log
        cls.tmp.cleanup()

    def score(self, key, findings, steps=10):
        return judge_adapter.score_episode(self.dirs[key], {"episode_id": "ep-SUW6BEVZNR", "findings": findings}, steps)

    def assertScore(self, got, tp, fp, fn, steps=10):
        exp_reward, p, r, f1 = reward(tp, fp, fn, steps)
        self.assertEqual(set(got), {"tp", "fp", "fn", "precision", "recall", "f1", "return"})
        self.assertEqual((got["tp"], got["fp"], got["fn"]), (tp, fp, fn))
        for name, exp in (("precision", p), ("recall", r), ("f1", f1), ("return", exp_reward)):
            self.assertAlmostEqual(got[name], float(exp), delta=EPS, msg=name)
            self.assertIsInstance(got[name], float)

    def assertStructureError(self, got):
        self.assertEqual(got, {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None,
                               "return": -1.0})
        with open(self.log, encoding="utf-8") as fh:
            last = json.loads(fh.read().splitlines()[-1])
        self.assertFalse(last["valid"])
        self.assertTrue(last["errors"])  # ошибка записана в журнал оценщика


class Block3(JuryFixtures):
    def test_01_one_correct_finding(self):
        got = self.score("A", [F()])
        self.assertScore(got, 1, 0, 0)
        self.assertAlmostEqual(got["return"], 0.98, delta=EPS)

    def test_02_empty_findings(self):
        got = self.score("A", [])
        self.assertScore(got, 0, 0, 1)
        self.assertEqual(got["return"], -1.0)

    def test_03_empty_truth_empty_findings(self):
        got = self.score("empty", [])
        self.assertScore(got, 0, 0, 0)
        self.assertEqual((got["precision"], got["recall"], got["f1"]), (0.0, 0.0, 1.0))
        self.assertAlmostEqual(got["return"], 0.98, delta=EPS)

    def test_04_empty_truth_one_false_finding(self):
        got = self.score("empty", [F()])
        self.assertScore(got, 0, 1, 0)
        self.assertEqual(got["return"], -1.0)

    def test_05_two_identical_correct_findings(self):
        got = self.score("A", [F(), F()])
        self.assertScore(got, 1, 1, 0)
        self.assertAlmostEqual(got["return"], 47 / 150, delta=EPS)
        self.assertLess(got["return"], self.score("A", [F()])["return"])

    def test_06_to_11_one_wrong_field_is_a_false_positive(self):
        cases = {
            "6 wrong type": F(type=T_OTHER),
            "7 other existing record": F(affected_ids=[R2]),
            "8 amount 10001 with tolerance 0": F(claimed_amount_minor=10001),
            "9 other currency": F(currency="USD"),
            "10 only D1": F(evidence_ids=[D1]),
            "11 absent evidence id instead of D2": F(evidence_ids=[D1, U]),
        }
        for name, finding in cases.items():
            with self.subTest(name):
                got = self.score("A", [finding])
                self.assertScore(got, 0, 1, 1)
                self.assertEqual(got["return"], -1.0)

    def test_12_flag_five_records_with_all_four_types(self):
        findings = [F(type=t, affected_ids=[r]) for r in (R1, R2, R3, R4, R5) for t in TYPES]
        self.assertEqual(len(findings), 20)
        got = self.score("A", findings)
        self.assertScore(got, 1, 19, 0)
        self.assertAlmostEqual(got["return"], -871 / 1050, delta=EPS)
        self.assertAlmostEqual(got["f1"], 2 / 21, delta=EPS)

    def test_13_two_issues_only_a_found(self):
        got = self.score("AB", [F()])
        self.assertScore(got, 1, 0, 1)
        self.assertAlmostEqual(got["return"], 47 / 150, delta=EPS)

    def test_14_two_issues_a_plus_one_false(self):
        got = self.score("AB", [F(), F(affected_ids=[R3])])
        self.assertScore(got, 1, 1, 1)
        self.assertAlmostEqual(got["return"], -0.02, delta=EPS)

    def test_15_to_17_step_cost(self):
        for steps, expected in ((0, 1.0), (50, 0.90), (100, 0.80)):
            with self.subTest(steps=steps):
                got = self.score("A", [F()], steps)
                self.assertScore(got, 1, 0, 0, steps)
                self.assertAlmostEqual(got["return"], expected, delta=EPS)
        self.assertAlmostEqual(self.score("A", [F()], 10)["return"] - self.score("A", [F()], 50)["return"], 0.08, delta=EPS)

    def test_18_thirty_one_findings(self):
        self.assertStructureError(self.score("A", [F()] * 31))

    def test_19_eleven_distinct_evidence_ids(self):
        self.assertStructureError(self.score("A", [F(evidence_ids=[D1, D2, D3, R1, R2, R3, R4, R5, U, "X1", "X2"])]))

    def test_20_explanation_of_1001_ascii_characters(self):
        self.assertStructureError(self.score("A", [F(explanation="a" * 1001)]))
        self.assertScore(self.score("A", [F(explanation="a" * 1000)]), 1, 0, 0)  # 1000 — ещё можно


class OpenExamples(JuryFixtures):
    """M01–M12: одна ошибка A с affected_ids {R1, R2}, 10000 KZT, набор {D1, D2}, τ = 0; D3 есть, U нет."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        root = Path(cls.tmp.name)
        a2 = {**ISSUE_A, "affected_ids": [R1, R2]}
        for key, issues in (("M", [a2]), ("M08", [{**a2, "amount_tolerance_minor": 1}])):
            d = root / f"пример {key} — любое имя"
            shutil.copytree(STUB / "public", d / "public")
            (d / "private").mkdir()
            (d / "private" / "truth.json").write_text(json.dumps({"episode_id": "ep-SUW6BEVZNR", "issues": issues}),
                                                      encoding="utf-8")
            cls.dirs[key] = str(d)

    def G(self, **changes):
        return F(**{"affected_ids": [R1, R2], **changes})

    def test_m01_affected_ids_are_a_set(self):
        self.assertScore(self.score("M", [self.G(affected_ids=[R2, R1, R1])]), 1, 0, 0)

    def test_m02_extra_existing_affected_id(self):
        self.assertScore(self.score("M", [self.G(affected_ids=[R1, R2, R3])]), 0, 1, 1)

    def test_m03_extra_existing_evidence_is_allowed(self):
        self.assertScore(self.score("M", [self.G(evidence_ids=[D2, D1, D3])]), 1, 0, 0)

    def test_m04_unknown_evidence_id_is_a_false_positive(self):
        self.assertScore(self.score("M", [self.G(evidence_ids=[D1, D2, U])]), 0, 1, 1)

    def test_m05_eleven_evidence_ids_with_repeats(self):
        self.assertStructureError(self.score("M", [self.G(evidence_ids=[D1] * 10 + [D2])]))

    def test_m06_currency_case_or_whitespace(self):
        for cur in ("kzt", "Kzt", "KZT ", " KZT", "KZT\n"):
            with self.subTest(currency=repr(cur)):
                self.assertScore(self.score("M", [self.G(currency=cur)]), 0, 1, 1)

    def test_m07_amount_off_by_one_with_zero_tolerance(self):
        self.assertScore(self.score("M", [self.G(claimed_amount_minor=10001)]), 0, 1, 1)

    def test_m08_published_tolerance_one(self):
        self.assertScore(self.score("M08", [self.G(claimed_amount_minor=10001)]), 1, 0, 0)
        self.assertScore(self.score("M08", [self.G(claimed_amount_minor=9999)]), 1, 0, 0)
        self.assertScore(self.score("M08", [self.G(claimed_amount_minor=10002)]), 0, 1, 1)

    def test_m09_amount_as_string_or_bool(self):
        for amount in ("10000", True, 10000.0):
            with self.subTest(amount=repr(amount)):
                self.assertStructureError(self.score("M", [self.G(claimed_amount_minor=amount)]))

    def test_unknown_type_negative_amount_empty_affected_are_false_positives(self):
        for change in ({"type": "fraud"}, {"type": ""}, {"claimed_amount_minor": -10000}, {"affected_ids": []}):
            with self.subTest(change=change):
                self.assertScore(self.score("M", [self.G(**change)]), 0, 1, 1)

    def _two(self, ev_a, ev_b):
        a = {**ISSUE_A, "issue_id": "ISS-A", "affected_ids": [R1, R2], "evidence_sets": [ev_a]}
        b = {**ISSUE_A, "issue_id": "ISS-B", "affected_ids": [R1, R2], "evidence_sets": [ev_b]}
        return a, b

    def test_m10_maximum_matching_not_greedy(self):
        a, b = self._two([D1, D2], [D3])
        f0 = self.G(evidence_ids=[D1, D2, D3])  # подходит A и B
        f1 = self.G(evidence_ids=[D1, D2])      # подходит только A
        for issues in ([a, b], [b, a]):
            res = match([f0, f1], issues)
            by_issue = {issues[i]["issue_id"]: f for f, i in res.pairs}
            self.assertEqual((res.tp, res.fp, res.fn), (2, 0, 0))
            self.assertEqual(by_issue, {"ISS-A": 1, "ISS-B": 0})  # F1 -> A, F0 -> B

    def test_m11_deterministic_tie_break(self):
        a, b = self._two([D1, D2], [D1, D2])
        f0, f1 = self.G(), self.G(explanation="same finding, other text")
        for issues in ([a, b], [b, a]):
            res = match([f0, f1], issues)
            by_issue = {issues[i]["issue_id"]: f for f, i in res.pairs}
            self.assertEqual(res.tp, 2)
            self.assertEqual(by_issue, {"ISS-A": 0, "ISS-B": 1})  # A < B: A -> F0, B -> F1
        self.assertEqual(match([f1, f0], [a, b]).tp, 2)  # перестановка выводов не меняет TP

    def test_m12_two_identical_correct_findings(self):
        got = self.score("M", [self.G(), self.G()])
        self.assertScore(got, 1, 1, 0)
        self.assertEqual(got, self.score("A", [F(), F()]))  # совпадает с вектором теста 5


class FixtureFormat(JuryFixtures):
    """Фикстура в формате README («Интерфейс жюри»): issue_id и amount_tolerance_minor необязательны (τ = 0, null — то же),
    episode_id в truth можно не повторять. Без обязательного поля эталон отвергается при загрузке понятной ошибкой,
    а не аварией посреди оценки (регрессия /code-review: KeyError 'issue_id' в разборе пропусков)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        root = Path(cls.tmp.name)
        sparse_a = {k: v for k, v in ISSUE_A.items() if k not in ("issue_id", "amount_tolerance_minor")}
        sparse_b = {**ISSUE_B, "issue_id": None, "amount_tolerance_minor": None}
        for key, issues in (("sparse", [sparse_a, sparse_b]),
                            ("no type", [{k: v for k, v in ISSUE_A.items() if k != "type"}]),
                            ("string amount", [{**ISSUE_A, "expected_amount_minor": "10000"}]),
                            ("negative tolerance", [{**ISSUE_A, "amount_tolerance_minor": -1}]),
                            ("flat evidence", [{**ISSUE_A, "evidence_sets": [D1, D2]}])):
            d = root / f"фикстура {key}"
            shutil.copytree(STUB / "public", d)
            (d / "truth.json").write_text(json.dumps({"issues": issues}), encoding="utf-8")
            cls.dirs[key] = str(d)

    def test_issue_id_and_tolerance_are_optional(self):
        self.assertScore(self.score("sparse", [F(), F_B()]), 2, 0, 0)
        self.assertScore(self.score("sparse", [F(claimed_amount_minor=10001)]), 0, 1, 2)  # τ по умолчанию 0
        self.assertScore(self.score("sparse", []), 0, 0, 2)  # разбор пропусков без issue_id

    def test_reference_without_a_required_field_is_rejected_at_load(self):
        for key in ("no type", "string amount", "negative tolerance", "flat evidence"):
            with self.subTest(fixture=key), self.assertRaises(EpisodeLoadError):
                self.score(key, [])


class GraderLog(unittest.TestCase):
    def test_unwritable_log_does_not_break_the_episode(self):
        """Регрессия /code-review: журнал пишется сразу по завершении эпизода. Если писать нельзя (каталог только
        для чтения), submit не падает, запись остаётся в памяти и дописывается при close()."""
        with tempfile.TemporaryDirectory(prefix="grader-log-") as tmp:
            blocker = Path(tmp) / "not a directory"
            blocker.write_text("x", encoding="utf-8")
            log = blocker / "log.jsonl"
            session = GraderSession(log_path=log)
            session.load(None, {"episode_dir": str(STUB)}, "closed")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                out = session.finish({"episode_id": "ep-SUW6BEVZNR", "findings": []}, 0, 1, "submit")
            self.assertEqual(out, {"reward": 0.0, "info": {}})
            self.assertEqual(len(session.records), 1)
            self.assertIn("grader log is not writable", err.getvalue())
            self.assertNotIn(tmp, err.getvalue())  # без путей
            blocker.unlink()  # журнал снова доступен
            session.close()
            with open(log, encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual([(r["episode_id"], r["end_reason"]) for r in lines], [("ep-SUW6BEVZNR", "submit")])


if __name__ == "__main__":
    unittest.main()
