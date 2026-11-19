"""Данные и эталон: инварианты, эталон == вывод по открытым данным, достаточность доказательств."""

import json
import random
import unittest
from collections import Counter

from gen_helpers import CONFIGS, episode
from generator.audit import audit, compare_with_truth, parse_public
from generator.config import DISTRACTOR_TYPES, ERROR_TYPES, SCENARIO_TYPES, SIZE_PROFILES
from generator.fileio import parse_jsonl, parse_ledger
from generator.generate import generate_episode
from generator.schema_check import load_schema, validate, validate_episode_texts
from policies.oracle import oracle_findings

SEEDS = range(6)


class EpisodeDataTests(unittest.TestCase):
    def test_truth_matches_independent_audit(self):
        for key in CONFIGS:
            for seed in SEEDS:
                ep = episode(key, seed)
                with self.subTest(config=key, seed=seed):
                    self.assertEqual(compare_with_truth(parse_public(ep.public), ep.truth), [])

    def test_schemas(self):
        findings_schema = load_schema("findings")
        for key in CONFIGS:
            for seed in SEEDS[:3]:
                ep = episode(key, seed)
                with self.subTest(config=key, seed=seed):
                    self.assertEqual(validate_episode_texts(ep.public, ep.private), [])
                    self.assertEqual(validate(oracle_findings(ep.truth), findings_schema), [])

    def test_sizes_within_case_limits(self):
        for key in CONFIGS:
            for seed in SEEDS:
                counts = json.loads(episode(key, seed).public["manifest.json"])["counts"]
                with self.subTest(config=key, seed=seed):
                    self.assertTrue(200 <= counts["ledger_rows"] <= 2000, counts)
                    self.assertTrue(20 <= counts["documents"] <= 200, counts)
                    self.assertLessEqual(len(episode(key, seed).truth["issues"]), 12)

    def test_requested_error_counts_are_produced(self):
        for key in CONFIGS:
            cfg = CONFIGS[key]
            expected = Counter({t: n for t, n in cfg.get("errors", {}).items() if n})
            for name, n in cfg.get("scenarios", {}).items():
                for t in SCENARIO_TYPES[name]:
                    expected[t] += n
            for seed in SEEDS[:3]:
                truth = episode(key, seed).truth
                with self.subTest(config=key, seed=seed):
                    self.assertEqual(Counter(i["type"] for i in truth["issues"]), expected)
                    scen = Counter(i["scenario"] for i in truth["issues"] if i["scenario"])
                    for name, n in cfg.get("scenarios", {}).items():
                        self.assertEqual(scen[name], 2 * n)

    def test_clean_company_has_no_issues_but_has_distractors(self):
        for seed in SEEDS:
            ep = episode("clean", seed)
            self.assertEqual(ep.truth["issues"], [])
            kinds = {d["kind"] for d in ep.provenance["distractors"]}
            self.assertGreaterEqual(len(kinds), 4, kinds)
            self.assertEqual(audit(parse_public(ep.public)).issues, [])

    def test_ledger_balanced_and_in_base_currency(self):
        for key in CONFIGS:
            ep = episode(key, 1)
            rows = parse_ledger(ep.public["ledger.csv"])
            base = json.loads(ep.public["manifest.json"])["base_currency"]
            self.assertEqual(sum(r["debit_minor"] for r in rows), sum(r["credit_minor"] for r in rows))
            self.assertTrue(all(r["currency"] == base for r in rows))
            self.assertTrue(all((r["debit_minor"] > 0) != (r["credit_minor"] > 0) for r in rows))

    def test_documents_internally_consistent(self):
        for key in CONFIGS:
            ep = episode(key, 2)
            for inv in parse_jsonl(ep.public["invoices.jsonl"]):
                self.assertEqual(inv["total_minor"], sum(li["amount_minor"] for li in inv["line_items"]))
                for li in inv["line_items"]:
                    self.assertEqual(li["amount_minor"], li["quantity"] * li["unit_price_minor"])
            for pay in parse_jsonl(ep.public["payments.jsonl"]):
                refs = pay["invoice_refs"]
                if refs is not None:
                    self.assertLessEqual(sum(r["amount_minor"] for r in refs), pay["amount_minor"])
                if pay["amount_minor"] < 0:
                    self.assertIsNone(refs)

    def test_no_float_money_in_files(self):
        def walk(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    if k.endswith("_minor"):
                        self.assertIs(type(v), int, k)
                    walk(v)
            elif isinstance(value, list):
                for v in value:
                    walk(v)

        ep = episode("heavy", 0)
        for name in ("invoices.jsonl", "payments.jsonl"):
            for rec in parse_jsonl(ep.public[name]):
                walk(rec)
        walk(ep.truth)

    def test_checks_before_and_after_injection_recorded(self):
        prov = episode("heavy", 3).provenance
        self.assertEqual(prov["checks_before_injection"]["status"], "passed")
        self.assertEqual(prov["checks_after_injection"]["status"], "passed")
        self.assertEqual(len(prov["injections"]), 12)
        for inj in prov["injections"]:
            self.assertTrue(inj["cause"])
            self.assertIn("before", inj)
            self.assertIn("after", inj)

    def test_random_configs_fuzz(self):
        """Новые сочетания параметров (как у жюри) не ломают генератор и эталон."""
        rng = random.Random(2026)
        for i in range(60):
            budget = 12
            scenarios = {}
            for s in SCENARIO_TYPES:
                k = min(rng.choice([0, 0, 1, 2]), budget // 2)
                scenarios[s] = k
                budget -= 2 * k
            errors = {}
            for t in ERROR_TYPES:
                k = min(rng.choice([0, 1, 2, 3]), budget)
                errors[t] = k
                budget -= k
            cfg = {
                "name": f"fuzz{i}",
                "size": rng.choice(sorted(SIZE_PROFILES)),
                "errors": errors,
                "scenarios": scenarios,
                "distractors": {d: rng.random() < 0.7 for d in DISTRACTOR_TYPES},
                "timezone": rng.choice([None, "+05:00", "-05:00", "+00:00", "+05:30", "-11:00", "+13:00"]),
            }
            seed = rng.randint(0, 2**63)
            with self.subTest(i=i, cfg=cfg, seed=seed):
                ep = generate_episode(seed, cfg)
                self.assertEqual(compare_with_truth(parse_public(ep.public), ep.truth), [])


class AuditDetectsTamperingTests(unittest.TestCase):
    """Самопроверка сама проверена: порча данных или эталона обязана ловиться."""

    def setUp(self):
        self.ep = episode("all_types", 4)

    def test_dropping_truth_issue_is_detected(self):
        truth = json.loads(json.dumps(self.ep.truth))
        truth["issues"].pop()
        self.assertTrue(compare_with_truth(parse_public(self.ep.public), truth))

    def test_wrong_expected_amount_is_detected(self):
        truth = json.loads(json.dumps(self.ep.truth))
        truth["issues"][0]["expected_amount_minor"] += 7
        self.assertTrue(compare_with_truth(parse_public(self.ep.public), truth))

    def test_unknown_evidence_is_detected(self):
        truth = json.loads(json.dumps(self.ep.truth))
        truth["issues"][0]["evidence_sets"][0] = ["L-ZZZZZZZZ"]
        self.assertTrue(compare_with_truth(parse_public(self.ep.public), truth))

    def test_unbalanced_ledger_is_detected(self):
        public = dict(self.ep.public)
        lines = public["ledger.csv"].splitlines(keepends=True)
        cells = lines[5].rstrip("\n").split(",")
        if int(cells[3]):
            cells[3] = str(int(cells[3]) + 1)
        else:
            cells[4] = str(int(cells[4]) + 1)
        lines[5] = ",".join(cells) + "\n"
        public["ledger.csv"] = "".join(lines)
        result = audit(parse_public(public))
        self.assertTrue(result.anomalies)


if __name__ == "__main__":
    unittest.main()
