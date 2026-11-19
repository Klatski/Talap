from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import AuditEnv, EpisodeBundle, safe_calculate


def make_bundle(episode_id: str = "ep_env_1") -> EpisodeBundle:
    manifest = {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "base_currency": "USD",
        "period_start": "2026-01-01T00:00:00+00:00",
        "period_end": "2026-01-31T23:59:59+00:00",
        "timezone": "UTC",
        "rounding": "HALF_UP",
    }
    policy = {
        "fx_date_rule": "invoice_or_payment_date",
        "period_boundaries": "inclusive",
        "rounding": "HALF_UP",
    }
    ledger = [
        {
            "row_id": f"row_{i}",
            "timestamp": "2026-01-15T00:00:00+00:00",
            "account": "1000_CASH",
            "debit_minor": "1000",
            "credit_minor": "0",
            "currency": "USD",
            "entity_id": "ent_a",
            "invoice_id": None,
            "payment_id": None,
        }
        for i in range(55)
    ]
    return EpisodeBundle(
        episode_id=episode_id,
        manifest=manifest,
        policy=policy,
        ledger=ledger,
        invoices=[],
        payments=[],
        fx=[],
    )


class TestAuditEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = make_bundle()

        def fake_evaluator(**kwargs):
            return {"reward": 0.5, "tp": 1, "fp": 0, "fn": 0}

        self.env = AuditEnv(
            episode_loader=lambda seed, config, include_truth: self.bundle,
            submit_evaluator=fake_evaluator,
            max_steps=100,
            timeout_sec=60.0,
        )

    def test_reset_observation_and_pagination_cap(self) -> None:
        obs0, info0 = self.env.reset(seed=10, config={"mode": "eval"})
        self.assertEqual(info0["mode"], "eval")
        self.assertEqual(obs0["steps_remaining"], 100)
        self.assertIn("manifest", obs0)
        self.assertIn("policy_summary", obs0)

        obs1, reward, term, trunc, _ = self.env.step(
            {"name": "list_records", "args": {"table": "ledger", "cursor": 0, "limit": 80}}
        )
        self.assertEqual(reward, 0.0)
        self.assertFalse(term)
        self.assertFalse(trunc)
        self.assertEqual(obs1["returned"], 50)
        self.assertEqual(obs1["next_cursor"], 50)

    def test_calculator_blocks_code_and_invalid_submit_returns_minus_one(self) -> None:
        self.env.reset(seed=10, config={"mode": "train"})
        obs_calc, r_calc, term_calc, _, _ = self.env.step(
            {"name": "calculator", "args": {"expression": "__import__('os').listdir()"}}
        )
        self.assertIn("error", obs_calc)
        self.assertEqual(r_calc, 0.0)
        self.assertFalse(term_calc)

        _, r_bad, term_bad, _, _ = self.env.step(
            {"name": "submit", "args": {"episode_id": "wrong_ep", "findings": []}}
        )
        self.assertTrue(term_bad)
        self.assertEqual(r_bad, -1.0)
        self.assertEqual(safe_calculate("10.5 * 2"), "21")


if __name__ == "__main__":
    unittest.main()