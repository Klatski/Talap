from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import AuditEnv, EpisodeBundle
from policies import build_policy, run_suite


class TestPoliciesAndTrajectories(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = EpisodeBundle(
            episode_id="ep_pol_1",
            manifest={
                "schema_version": "1.0",
                "episode_id": "ep_pol_1",
                "base_currency": "USD",
                "period_start": "2026-01-01T00:00:00+00:00",
                "period_end": "2026-01-31T23:59:59+00:00",
                "timezone": "UTC",
                "rounding": "HALF_UP",
            },
            policy={
                "fx_date_rule": "invoice_or_payment_date",
                "period_boundaries": "inclusive",
                "rounding": "HALF_UP",
            },
            ledger=[
                {
                    "row_id": "r1",
                    "timestamp": "2026-01-10T00:00:00+00:00",
                    "account": "1000",
                    "debit_minor": "1000",
                    "credit_minor": "0",
                    "currency": "USD",
                    "entity_id": "e1",
                    "invoice_id": "inv_1",
                    "payment_id": None,
                },
                {
                    "row_id": "r2",
                    "timestamp": "2026-01-11T00:00:00+00:00",
                    "account": "1000",
                    "debit_minor": "1000",
                    "credit_minor": "0",
                    "currency": "USD",
                    "entity_id": "e1",
                    "invoice_id": "inv_1",
                    "payment_id": None,
                },
            ],
            invoices=[{"invoice_id": "inv_1", "amount_minor": "1000", "currency": "USD"}],
            payments=[],
            fx=[],
        )
        self.env = AuditEnv(
            episode_loader=lambda seed, config, include_truth: self.bundle,
            submit_evaluator=lambda **kwargs: {"reward": 0.5, "tp": 1, "fp": 0, "fn": 0, "f1": 1.0},
            max_steps=100,
            timeout_sec=60.0,
        )

    def test_baselines_finish_within_budget_and_log_clean_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_path = Path(tmpdir) / "trajectories.jsonl"
            summaries = run_suite(
                self.env,
                policy_names=["random", "reconcile", "all"],
                seeds=[7],
                mode="eval",
                trajectory_path=traj_path,
            )
            self.assertEqual(len(summaries), 3)
            for s in summaries:
                self.assertTrue(s["terminated"])
                self.assertFalse(s["truncated"])
                self.assertLessEqual(s["steps_used"], 100)

            rows = [
                json.loads(line)
                for line in traj_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreater(len(rows), 0)
            expected_keys = {
                "episode_id",
                "step",
                "action",
                "observation",
                "reward",
                "terminated",
                "truncated",
            }
            for row in rows:
                self.assertEqual(set(row.keys()), expected_keys)
                self.assertNotIn("seed", row)
                self.assertNotIn("truth", row)
                self.assertNotIn("provenance", row)

    def test_build_policy_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            build_policy("unknown_policy", seed=0)


if __name__ == "__main__":
    unittest.main()