"""Контрфактическая чистая версия: равенства до внесения ошибок + diff ровно по объявленным ошибкам."""

import json
import unittest

from gen_helpers import CONFIGS, episode
from generator.audit import audit, parse_public
from generator.fileio import parse_jsonl, parse_ledger
from generator.generate import COUNTERFACTUAL_DIR


def clean_public(ep):
    prefix = COUNTERFACTUAL_DIR + "/"
    return {k[len(prefix):]: v for k, v in ep.private.items() if k.startswith(prefix)}


class CounterfactualTests(unittest.TestCase):
    def test_clean_version_has_no_discrepancies(self):
        for key in CONFIGS:
            for seed in range(4):
                result = audit(parse_public(clean_public(episode(key, seed))))
                with self.subTest(config=key, seed=seed):
                    self.assertEqual(result.issues, [])
                    self.assertEqual(result.anomalies, [])

    def test_diff_touches_only_declared_documents(self):
        for key in ("all_types", "scenarios", "heavy"):
            for seed in range(4):
                ep = episode(key, seed)
                clean = clean_public(ep)
                dirty_rows = {r["row_id"]: r for r in parse_ledger(ep.public["ledger.csv"])}
                clean_rows = {r["row_id"]: r for r in parse_ledger(clean["ledger.csv"])}
                affected = {i for issue in ep.truth["issues"] for i in issue["affected_ids"]}
                changed = [
                    dirty_rows.get(rid) or clean_rows[rid]
                    for rid in set(dirty_rows) | set(clean_rows)
                    if dirty_rows.get(rid) != clean_rows.get(rid)
                ]
                with self.subTest(config=key, seed=seed):
                    self.assertTrue(changed)
                    for row in changed:
                        self.assertTrue({row["invoice_id"], row["payment_id"]} & affected, row)
                    clean_docs = {d["doc_id"]: d for d in parse_jsonl(clean["invoices.jsonl"]) + parse_jsonl(clean["payments.jsonl"])}
                    dirty_docs = {d["doc_id"]: d for d in parse_jsonl(ep.public["invoices.jsonl"]) + parse_jsonl(ep.public["payments.jsonl"])}
                    for doc_id, doc in clean_docs.items():
                        self.assertEqual(dirty_docs[doc_id], doc)  # первоисточники не меняются
                    extra = set(dirty_docs) - set(clean_docs)
                    n_dups = sum(1 for i in ep.truth["issues"] if i["type"] == "duplicate_invoice")
                    self.assertEqual(len(extra), n_dups)
                    self.assertEqual(clean["fx.csv"], ep.public["fx.csv"])
                    self.assertEqual(clean["policy.json"], ep.public["policy.json"])

    def test_clean_company_counterfactual_is_identical(self):
        ep = episode("clean", 2)
        self.assertEqual(clean_public(ep), ep.public)


if __name__ == "__main__":
    unittest.main()
