from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from grader import TruthIssue, evaluate_findings, validate_findings_payload


class TestGraderMatchingAndF1(unittest.TestCase):
    def test_one_to_one_matching_prevents_double_counting_tp(self) -> None:
        truth = [
            TruthIssue(
                issue_id="iss_dup_1",
                issue_type="duplicate_invoice_posting",
                affected_ids=("r1", "r2"),
                expected_amount_minor=5000,
                currency="USD",
                tolerance_minor=0,
                valid_evidence_sets=(frozenset({"r1", "r2", "inv_10"}),),
            )
        ]
        f1 = {
            "type": "duplicate_invoice_posting",
            "affected_ids": ["r1", "r2"],
            "claimed_amount_minor": 5000,
            "currency": "USD",
            "evidence_ids": ["r1", "r2", "inv_10"],
            "explanation": "First candidate for duplicate posting",
        }
        f2 = {
            "type": "duplicate_invoice_posting",
            "affected_ids": ["r2", "r1"],
            "claimed_amount_minor": 5000,
            "currency": "USD",
            "evidence_ids": ["r1", "r2", "inv_10"],
            "explanation": "Second candidate matching the same truth issue",
        }
        parsed = validate_findings_payload(
            {"episode_id": "ep_match_1", "findings": [f1, f2]},
            expected_episode_id="ep_match_1",
        )
        result = evaluate_findings(parsed, truth, inspection_steps=4)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 0)
        self.assertAlmostEqual(float(result["f1"]), 2.0 / 3.0, places=6)

    def test_one_to_one_matching_across_multiple_issues(self) -> None:
        truth = [
            TruthIssue(
                issue_id="iss_a",
                issue_type="duplicate_invoice_posting",
                affected_ids=("r1", "r2"),
                expected_amount_minor=1000,
                currency="USD",
                tolerance_minor=0,
                valid_evidence_sets=(frozenset({"r1", "r2"}),),
            ),
            TruthIssue(
                issue_id="iss_b",
                issue_type="payment_invoice_mismatch",
                affected_ids=("inv_2", "pay_2"),
                expected_amount_minor=250,
                currency="USD",
                tolerance_minor=5,
                valid_evidence_sets=(frozenset({"inv_2", "pay_2"}),),
            ),
        ]
        parsed = validate_findings_payload(
            {
                "episode_id": "ep_match_2",
                "findings": [
                    {
                        "type": "duplicate_invoice_posting",
                        "affected_ids": ["r1", "r2"],
                        "claimed_amount_minor": 1000,
                        "currency": "USD",
                        "evidence_ids": ["r1", "r2"],
                        "explanation": "Match A",
                    },
                    {
                        "type": "payment_invoice_mismatch",
                        "affected_ids": ["inv_2", "pay_2"],
                        "claimed_amount_minor": 253,
                        "currency": "USD",
                        "evidence_ids": ["inv_2", "pay_2"],
                        "explanation": "Match B within tolerance",
                    },
                ],
            },
            expected_episode_id="ep_match_2",
        )
        result = evaluate_findings(parsed, truth, inspection_steps=5)
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["fn"], 0)
        self.assertAlmostEqual(float(result["f1"]), 1.0, places=6)

    def test_zero_denominator_f1_is_zero(self) -> None:
        result = evaluate_findings([], [], inspection_steps=1)
        # Contract: empty truth + empty findings -> F1=1.0
        self.assertAlmostEqual(float(result["f1"]), 1.0, places=6)

        # Zero-denominator case when tp=0, fp=0, fn=0 is modeled directly if denominator is 0:
        # verify helper/metric behavior when tp + fp + fn == 0 vs non-empty truth with empty findings.
        with_truth_empty_pred = evaluate_findings(
            [],
            [
                TruthIssue(
                    issue_id="iss_1",
                    issue_type="duplicate_invoice_posting",
                    affected_ids=("r1",),
                    expected_amount_minor=100,
                    currency="USD",
                    tolerance_minor=0,
                    valid_evidence_sets=(frozenset({"r1"}),),
                )
            ],
            inspection_steps=1,
        )
        self.assertEqual(with_truth_empty_pred["tp"], 0)
        self.assertEqual(with_truth_empty_pred["fp"], 0)
        self.assertEqual(with_truth_empty_pred["fn"], 1)
        self.assertAlmostEqual(float(with_truth_empty_pred["f1"]), 0.0, places=6)

    def test_empty_truth_f1_and_fp_penalty(self) -> None:
        empty_f1 = evaluate_findings([], [], inspection_steps=2)
        self.assertAlmostEqual(float(empty_f1["f1"]), 1.0, places=6)

        fp_parsed = validate_findings_payload(
            {
                "episode_id": "ep_empty_truth",
                "findings": [
                    {
                        "type": "duplicate_invoice_posting",
                        "affected_ids": ["r9"],
                        "claimed_amount_minor": 900,
                        "currency": "USD",
                        "evidence_ids": ["r9"],
                        "explanation": "Spurious finding on clean company",
                    }
                ],
            },
            expected_episode_id="ep_empty_truth",
        )
        fp_res = evaluate_findings(fp_parsed, [], inspection_steps=2)
        self.assertEqual(fp_res["tp"], 0)
        self.assertEqual(fp_res["fp"], 1)
        self.assertEqual(fp_res["fn"], 0)
        self.assertAlmostEqual(float(fp_res["f1"]), 0.0, places=6)

    def test_reward_does_not_depend_on_explanation_text(self) -> None:
        truth = [
            TruthIssue(
                issue_id="iss_exp",
                issue_type="duplicate_invoice_posting",
                affected_ids=("r1", "r2"),
                expected_amount_minor=2000,
                currency="USD",
                tolerance_minor=0,
                valid_evidence_sets=(frozenset({"r1", "r2"}),),
            )
        ]
        p_short = validate_findings_payload(
            {
                "episode_id": "ep_exp",
                "findings": [
                    {
                        "type": "duplicate_invoice_posting",
                        "affected_ids": ["r1", "r2"],
                        "claimed_amount_minor": 2000,
                        "currency": "USD",
                        "evidence_ids": ["r1", "r2"],
                        "explanation": "x",
                    }
                ],
            },
            expected_episode_id="ep_exp",
        )
        p_long = validate_findings_payload(
            {
                "episode_id": "ep_exp",
                "findings": [
                    {
                        "type": "duplicate_invoice_posting",
                        "affected_ids": ["r1", "r2"],
                        "claimed_amount_minor": 2000,
                        "currency": "USD",
                        "evidence_ids": ["r1", "r2"],
                        "explanation": "Detailed audit note " * 40,
                    }
                ],
            },
            expected_episode_id="ep_exp",
        )
        r_short = evaluate_findings(p_short, truth, inspection_steps=6)
        r_long = evaluate_findings(p_long, truth, inspection_steps=6)
        self.assertEqual(r_short["reward"], r_long["reward"])
        self.assertEqual(r_short["tp"], r_long["tp"])
        self.assertEqual(r_short["fp"], r_long["fp"])
        self.assertEqual(r_short["fn"], r_long["fn"])


if __name__ == "__main__":
    unittest.main()