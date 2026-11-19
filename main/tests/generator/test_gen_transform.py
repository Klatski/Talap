"""Перестановка строк и смена ID сохраняют эталон, ответ oracle и бухгалтерское содержание."""

import json
import tempfile
import unittest
from pathlib import Path

from gen_helpers import CONFIGS, episode
from generator.audit import compare_with_truth, parse_public
from generator.fileio import parse_jsonl, parse_ledger, read_episode_dir, write_episode_dir
from generator.schema_check import validate_episode_texts
from generator.transform import remap_findings, transform_episode, transform_texts
from policies.oracle import oracle_findings


def canonical_findings(findings):
    return sorted(
        (f["type"], tuple(sorted(f["affected_ids"])), f["claimed_amount_minor"], f["currency"])
        for f in findings["findings"]
    )


class TransformTests(unittest.TestCase):
    def test_truth_still_derivable_after_transform(self):
        for key in CONFIGS:
            for seed in range(3):
                ep = episode(key, seed)
                pub, priv, _ = transform_texts(ep.public, ep.private, seed + 100)
                with self.subTest(config=key, seed=seed):
                    self.assertEqual(compare_with_truth(parse_public(pub), json.loads(priv["truth.json"])), [])
                    self.assertEqual(validate_episode_texts(pub, priv), [])

    def test_oracle_answer_is_remapped_exactly(self):
        for key in ("all_types", "scenarios", "heavy"):
            ep = episode(key, 1)
            pub, priv, mapping = transform_texts(ep.public, ep.private, 7)
            new_truth = json.loads(priv["truth.json"])
            remapped = remap_findings(oracle_findings(ep.truth), mapping)
            self.assertEqual(canonical_findings(remapped), canonical_findings(oracle_findings(new_truth)))
            self.assertEqual(remapped["episode_id"], new_truth["episode_id"])
            # переотображённые доказательства — один из допустимых наборов нового эталона
            by_key = {(i["type"], tuple(i["affected_ids"])): i for i in new_truth["issues"]}
            for f in remapped["findings"]:
                issue = by_key[(f["type"], tuple(sorted(f["affected_ids"])))]
                self.assertIn(sorted(f["evidence_ids"]), [sorted(s) for s in issue["evidence_sets"]])

    def test_all_ids_change_and_rows_are_permuted(self):
        ep = episode("heavy", 2)
        pub, priv, mapping = transform_texts(ep.public, ep.private, 3)
        old_rows = parse_ledger(ep.public["ledger.csv"])
        new_rows = parse_ledger(pub["ledger.csv"])
        old_ids = {r["row_id"] for r in old_rows}
        self.assertFalse(old_ids & {r["row_id"] for r in new_rows})
        self.assertEqual(len(old_rows), len(new_rows))
        self.assertNotEqual([mapping[r["row_id"]] for r in old_rows], [r["row_id"] for r in new_rows])
        old_inv = {i["invoice_id"] for i in parse_jsonl(ep.public["invoices.jsonl"])}
        new_inv = {i["invoice_id"] for i in parse_jsonl(pub["invoices.jsonl"])}
        self.assertFalse(old_inv & new_inv)
        for old in list(old_ids)[:20]:
            self.assertNotIn(old, pub["ledger.csv"] + priv["truth.json"])
        self.assertNotEqual(json.loads(pub["manifest.json"])["episode_id"], ep.episode_id)

    def test_content_invariants_preserved(self):
        ep = episode("scenarios", 0)
        pub, _, _ = transform_texts(ep.public, ep.private, 11)
        for name in ("invoices.jsonl", "payments.jsonl"):
            amounts_old = sorted(r.get("total_minor", r.get("amount_minor")) for r in parse_jsonl(ep.public[name]))
            amounts_new = sorted(r.get("total_minor", r.get("amount_minor")) for r in parse_jsonl(pub[name]))
            self.assertEqual(amounts_old, amounts_new)
        self.assertEqual(pub["policy.json"], ep.public["policy.json"])
        self.assertEqual(sorted(pub["fx.csv"].splitlines()), sorted(ep.public["fx.csv"].splitlines()))

    def test_transform_is_deterministic_and_seed_dependent(self):
        ep = episode("all_types", 3)
        a = transform_texts(ep.public, ep.private, 5)
        b = transform_texts(ep.public, ep.private, 5)
        c = transform_texts(ep.public, ep.private, 6)
        self.assertEqual(a[0], b[0])
        self.assertNotEqual(a[0]["ledger.csv"], c[0]["ledger.csv"])

    def test_transform_episode_on_disk(self):
        ep = episode("all_types", 5)
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "src", Path(tmp) / "dst"
            write_episode_dir(src, ep.public, ep.private)
            mapping = transform_episode(src, dst, 9)
            pub, priv = read_episode_dir(dst)
            self.assertEqual(compare_with_truth(parse_public(pub), json.loads(priv["truth.json"])), [])
            prov = json.loads(priv["provenance.json"])
            self.assertEqual(prov["transforms"][-1]["source_episode_id"], ep.episode_id)
            self.assertTrue(mapping)


if __name__ == "__main__":
    unittest.main()
