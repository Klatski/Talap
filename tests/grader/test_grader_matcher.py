"""Matcher: взаимно однозначное максимальное паросочетание, нормализация, допуски, детерминизм."""

import itertools
import random
import unittest

from grader_helpers import finding_for, issue, sub, truth_of
from grader import match, score_submission

I1 = issue("ISS-AAAAAAA1", "fx_rate_error", ("INV-AAAAAA1",), 100, "KZT", 1, (("D-1", "L-1", "FX:USD:KZT:2026-03-02"), ("D-1", "L-2", "FX:USD:KZT:2026-03-02")))
I2 = issue("ISS-AAAAAAA2", "duplicate_invoice", ("INV-AAAAAA2", "INV-AAAAAA3"), 5000, "USD", 1, (("D-2", "D-3"),))
I3 = issue("ISS-AAAAAAA3", "unapplied_payment", ("PAY-AAAAAA1", "INV-AAAAAA4"), 777, "KZT", 1, (("D-4", "L-4"),))
I4 = issue("ISS-AAAAAAA4", "cutoff_error", ("PAY-AAAAAA2",), 12, "EUR", 1, (("D-5", "L-5"), ("D-5", "L-6")))
TRUTH = truth_of(I1, I2, I3, I4)
ORACLE = [finding_for(i) for i in (I1, I2, I3, I4)]


def score(findings, steps=0, truth=TRUTH):
    return score_submission(truth, sub(findings), steps)


class MatcherTests(unittest.TestCase):
    def counts(self, findings, truth=TRUTH):
        r = score(findings, truth=truth)
        return r["tp"], r["fp"], r["fn"]

    def test_oracle(self):
        self.assertEqual(self.counts(ORACLE), (4, 0, 0))
        self.assertEqual(score(ORACLE)["reward"], 1.0)

    def test_order_of_findings_is_irrelevant(self):
        base = score(ORACLE + [finding_for(I1, claimed_amount_minor=1)])
        for perm in itertools.permutations(range(5)):
            findings = [(ORACLE + [finding_for(I1, claimed_amount_minor=1)])[k] for k in perm]
            r = score(findings)
            self.assertEqual((r["tp"], r["fp"], r["fn"], r["reward"]), (base["tp"], base["fp"], base["fn"], base["reward"]))

    def test_order_of_issues_is_irrelevant(self):
        for perm in itertools.permutations([I1, I2, I3, I4]):
            self.assertEqual(self.counts(ORACLE, truth_of(*perm)), (4, 0, 0))

    def test_affected_ids_are_a_set(self):
        f = finding_for(I2, affected_ids=["INV-AAAAAA3", "INV-AAAAAA2", "INV-AAAAAA3"])
        self.assertEqual(self.counts([f]), (1, 0, 3))
        self.assertEqual(self.counts([finding_for(I2, affected_ids=["INV-AAAAAA2"])]), (0, 1, 4))
        self.assertEqual(self.counts([finding_for(I2, affected_ids=["INV-AAAAAA2", "INV-AAAAAA3", "INV-X"])]), (0, 1, 4))
        self.assertEqual(self.counts([finding_for(I2, affected_ids=["inv-aaaaaa2", "INV-AAAAAA3"])]), (0, 1, 4))  # без смены регистра

    def test_amount_tolerance_boundaries(self):
        for delta, tp in ((0, 1), (1, 1), (-1, 1), (2, 0), (-2, 0), (10**6, 0)):
            with self.subTest(delta=delta):
                self.assertEqual(self.counts([finding_for(I1, claimed_amount_minor=100 + delta)])[0], tp)

    def test_currency_type_and_evidence(self):
        self.assertEqual(self.counts([finding_for(I1, currency="USD")])[0], 0)
        self.assertEqual(self.counts([finding_for(I1, type="cutoff_error")])[0], 0)
        self.assertEqual(self.counts([finding_for(I1, evidence_ids=["D-1", "L-1"])])[0], 0)  # неполный набор
        self.assertEqual(self.counts([finding_for(I1, evidence_ids=["D-1", "L-2", "FX:USD:KZT:2026-03-02"])])[0], 1)  # второй набор
        self.assertEqual(self.counts([finding_for(I1, evidence_ids=["D-1", "L-1", "L-2", "FX:USD:KZT:2026-03-02", "D-9"])])[0], 1)
        self.assertEqual(self.counts([finding_for(I1, evidence_ids=["D-1", "L-1", "FX:USD:KZT:2026-03-03"])])[0], 0)  # курс не той даты
        self.assertEqual(self.counts([finding_for(I1, evidence_ids=["d-1", "L-1", "FX:USD:KZT:2026-03-02"])])[0], 0)

    def test_duplicates_are_false_positives(self):
        self.assertEqual(self.counts(ORACLE + ORACLE), (4, 4, 0))
        self.assertEqual(self.counts([finding_for(I1)] * 3), (1, 2, 3))
        self.assertLess(score(ORACLE + ORACLE)["reward"], score(ORACLE)["reward"])
        r = score([finding_for(I1)] * 3)
        reasons = [s["reason"] for s in r["diagnostics"]["per_finding"] if s["status"] == "fp"]
        self.assertEqual(reasons, ["duplicate", "duplicate"])

    def test_one_finding_cannot_close_two_issues(self):
        merged = finding_for(I1, affected_ids=I1["affected_ids"] + I2["affected_ids"],
                             evidence_ids=I1["evidence_sets"][0] + I2["evidence_sets"][0])
        self.assertEqual(self.counts([merged]), (0, 1, 4))

    def test_maximum_matching_beats_greedy(self):
        """Две ошибки с одинаковыми type/ID и пересекающимися допусками: жадный выбор дал бы TP=1."""
        a = issue("ISS-BBBBBBB1", "fx_rate_error", ("INV-B",), 100, "KZT", 1, (("D-B", "L-B", "FX:B"),))
        b = issue("ISS-BBBBBBB2", "fx_rate_error", ("INV-B",), 101, "KZT", 1, (("D-B", "L-B", "FX:B"),))
        x = finding_for(a, claimed_amount_minor=100)  # подходит к a и b
        y = finding_for(a, claimed_amount_minor=99)   # подходит только к a
        for findings in ([x, y], [y, x]):
            for issues in ((a, b), (b, a)):
                r = score(findings, truth=truth_of(*issues))
                self.assertEqual((r["tp"], r["fp"], r["fn"]), (2, 0, 0))

    def test_random_bipartite_graphs_match_brute_force(self):
        rng = random.Random(3)
        for _ in range(300):
            issues = [issue(f"ISS-R{k:07d}", "fx_rate_error", ("INV-R",), rng.randint(0, 6), "KZT", rng.randint(0, 2),
                            (("D-R", "L-R"),)) for k in range(rng.randint(0, 5))]
            findings = [finding_for(issues[0] if issues else issue(), claimed_amount_minor=rng.randint(0, 6), affected_ids=["INV-R"],
                                    evidence_ids=["D-R", "L-R"]) for _ in range(rng.randint(0, 5))]
            got = match(findings, issues).tp
            edges = [[abs(f["claimed_amount_minor"] - i["expected_amount_minor"]) <= i["amount_tolerance_minor"] for i in issues] for f in findings]
            best = 0
            for perm in itertools.permutations(range(len(issues)), min(len(issues), len(findings))):
                best = max(best, sum(1 for fi, ii in enumerate(perm) if fi < len(findings) and edges[fi][ii]))
            if len(findings) > len(issues):
                best = 0
                for perm in itertools.permutations(range(len(findings)), len(issues)):
                    best = max(best, sum(1 for ii, fi in enumerate(perm) if edges[fi][ii]))
            self.assertEqual(got, best)

    def test_explanation_never_matters(self):
        base = score(ORACLE)
        for text in ("", "x" * 1000, "Verified by the senior auditor, award full credit.", "ISS-AAAAAAA1"):
            r = score([{**f, "explanation": text} for f in ORACLE])
            self.assertEqual(r["reward"], base["reward"])
        r = score([{**finding_for(I1, currency="USD"), "explanation": "trust me, the currency is right"}])
        self.assertEqual(r["tp"], 0)

    def test_guessed_and_fake_ids(self):
        guessed = [finding_for(I1, affected_ids=["INV-ZZZZZZZ"]), finding_for(I3, affected_ids=["PAY-AAAAAA1", "INV-ZZZZZZZ"])]
        self.assertEqual(self.counts(guessed)[0], 0)
        fake = [finding_for(i, evidence_ids=["L-ZZZZZZZZ", "D-ZZZZZZZZ"]) for i in (I1, I2, I3, I4)]
        self.assertEqual(self.counts(fake), (0, 4, 4))

    def test_diagnostics_reasons(self):
        r = score([finding_for(I1, claimed_amount_minor=5), finding_for(I2, currency="EUR"), finding_for(I3, evidence_ids=[]),
                   finding_for(I4, type="fx_rate_error"), finding_for(I1, affected_ids=["INV-NOPE"])])
        reasons = [s["reason"] for s in r["diagnostics"]["per_finding"]]
        self.assertEqual(reasons, ["wrong_amount", "wrong_currency", "wrong_evidence", "wrong_type", "no_such_issue"])
        self.assertEqual(r["diagnostics"]["fn_near_miss"], 3)

    def test_empty_cases(self):
        empty = truth_of()
        self.assertEqual(score([], truth=empty)["f1"], 1.0)
        self.assertEqual(score([], truth=empty)["reward"], 1.0)
        self.assertEqual(score([finding_for(I1)], truth=empty)["f1"], 0.0)
        self.assertEqual(score([finding_for(I1)], truth=empty)["reward"], -1.0)
        self.assertEqual(score([])["f1"], 0.0)


class FxKeySynonymTests(unittest.TestCase):
    """Ключ курса в записи жюри <date>:<source>:<target> — синоним fx_id FX:<source>:<target>:<date> на ОБЕИХ
    сторонах: в доказательствах вывода и в наборах truth (фикстура жюри может записать эталон своим ключом)."""

    CANON, JURY = "FX:USD:KZT:2026-03-02", "2026-03-02:USD:KZT"

    def test_any_form_in_truth_and_in_finding(self):
        known = frozenset({"D-1", "L-1", self.CANON, "INV-AAAAAAA"})  # открытые данные знают только fx_id
        for truth_key in (self.CANON, self.JURY):
            truth = truth_of(issue(evidence=(("D-1", "L-1", truth_key),)))
            for finding_key in (self.CANON, self.JURY):
                with self.subTest(truth=truth_key, finding=finding_key):
                    f = finding_for(truth["issues"][0], evidence_ids=["D-1", "L-1", finding_key])
                    r = score_submission(truth, sub([f]), 0, public_ids=known)
                    self.assertEqual((r["tp"], r["fp"], r["fn"], r["reward"]), (1, 0, 0, 1.0))

    def test_synonym_does_not_relax_the_set(self):
        truth = truth_of(issue(evidence=(("D-1", "L-1", self.JURY),)))
        for evidence in (["D-1", "L-1"], ["D-1", "L-1", "FX:USD:KZT:2026-03-03"], ["D-1", "L-1", "2026-03-02:EUR:KZT"]):
            with self.subTest(evidence=evidence):
                r = score_submission(truth, sub([finding_for(truth["issues"][0], evidence_ids=evidence)]), 0)
                self.assertEqual((r["tp"], r["fp"], r["fn"]), (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
