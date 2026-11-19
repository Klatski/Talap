from __future__ import annotations

from pathlib import Path
import sys
import unittest
from decimal import Decimal

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from grader import (
    SubmissionSchemaError,
    TruthIssue,
    compute_terminal_reward,
    evaluate_findings,
    validate_findings_payload,
)


class TestGrader(unittest.TestCase):
    def setUp(self) -> None:
        self.truth = [
            TruthIssue(
                issue_id="iss_1",
                issue_type="duplicate_invoice_posting",
                affected_ids=("r1", "r2"),
                expected_amount_minor=10000,
                currency="USD",
                tolerance_minor=0,
                valid_evidence_sets=(frozenset({"r1", "r2", "doc_1"}),),
            )
        ]

    def test_valid_finding_beats_silence_and_step_penalty_breaks_ties(self) -> None:
        valid = validate_findings_payload(
            {
                "episode_id": "ep_1",
                "findings": [
                    {
                        "type": "duplicate_invoice_posting",
                        "affected_ids": ["r2", "r1"],
                        "claimed_amount_minor": 10000,
                        "currency": "USD",
                        "evidence_ids": ["r1", "r2", "doc_1"],
                        "explanation": "Duplicate credit entry",
                    }
                ],
            },
            expected_episode_id="ep_1",
        )
        res_fast = evaluate_findings(valid, self.truth, inspection_steps=2)
        res_slow = evaluate_findings(valid, self.truth, inspection_steps=30)
        res_silent = evaluate_findings([], self.truth, inspection_steps=2)

        self.assertEqual(res_fast["tp"], 1)
        self.assertEqual(res_fast["fp"], 0)
        self.assertEqual(res_fast["fn"], 0)
        self.assertGreater(res_fast["reward"], res_silent["reward"])
        self.assertGreater(res_fast["reward"], res_slow["reward"])

    def test_clean_episode_silence_beats_false_positive(self) -> None:
        fp_item = validate_findings_payload(
            {
                "episode_id": "ep_clean",
                "findings": [
                    {
                        "type": "duplicate_invoice_posting",
                        "affected_ids": ["r1"],
                        "claimed_amount_minor": 10000,
                        "currency": "USD",
                        "evidence_ids": ["r1"],
                        "explanation": "False alarm on clean book",
                    }
                ],
            },
            expected_episode_id="ep_clean",
        )
        clean_silent = evaluate_findings([], [], inspection_steps=3)
        clean_fp = evaluate_findings(fp_item, [], inspection_steps=3)

        self.assertEqual(clean_silent["f1"], 1.0)
        self.assertEqual(clean_fp["f1"], 0.0)
        self.assertGreater(clean_silent["reward"], clean_fp["reward"])

    def test_duplicates_give_no_extra_tp_and_bad_evidence_or_currency_fails(self) -> None:
        one = {
            "type": "duplicate_invoice_posting",
            "affected_ids": ["r1", "r2"],
            "claimed_amount_minor": 10000,
            "currency": "USD",
            "evidence_ids": ["r1", "r2", "doc_1"],
            "explanation": "Valid finding",
        }
        dup = validate_findings_payload(
            {"episode_id": "ep_1", "findings": [one, dict(one)]},
            expected_episode_id="ep_1",
        )
        dup_eval = evaluate_findings(dup, self.truth, inspection_steps=2)
        self.assertEqual(dup_eval["tp"], 1)
        self.assertEqual(dup_eval["fp"], 1)

        wrong_curr = validate_findings_payload(
            {
                "episode_id": "ep_1",
                "findings": [{**one, "currency": "EUR"}],
            },
            expected_episode_id="ep_1",
        )
        self.assertEqual(evaluate_findings(wrong_curr, self.truth, inspection_steps=2)["tp"], 0)

        missing_ev = validate_findings_payload(
            {
                "episode_id": "ep_1",
                "findings": [{**one, "evidence_ids": ["r1"]}],
            },
            expected_episode_id="ep_1",
        )
        self.assertEqual(evaluate_findings(missing_ev, self.truth, inspection_steps=2)["tp"], 0)

    def test_invalid_schema_raises_and_reward_clamped(self) -> None:
        with self.assertRaises(SubmissionSchemaError):
            validate_findings_payload({"episode_id": "wrong", "findings": []}, "ep_1")

        self.assertEqual(compute_terminal_reward(Decimal("0"), 500), Decimal("-1.000000"))
        self.assertEqual(compute_terminal_reward(Decimal("1"), 0), Decimal("1.000000"))


if __name__ == "__main__":
    unittest.main()