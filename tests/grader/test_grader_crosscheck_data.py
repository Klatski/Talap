"""Перекрёстная проверка «B проверяет данные A» (с 21.11).

1. Независимый валидатор: аудитор B (policies/auditor.py, не импортирует generator/) по одним
   открытым записям заново выводит все расхождения с доказательствами — на всех открытых
   эпизодах, их контрфактах (должно быть 0 находок), на новых сочетаниях параметров как у жюри
   и на парах «перестановка + смена ID».
2. Бухгалтерские равенства по открытым данным (своей реализацией B).
3. Простая сверка как детектор утечек: поверхностные признаки не должны находить ошибки.
Находки по данным A оформляются как падение этого теста, данные A здесь не правятся.
"""

import json
import random
import unittest
from collections import defaultdict

from grader_helpers import EPISODES, open_episodes
from env.records import load_public
from generator.fileio import read_texts
from generator.generate import generate_episode
from generator.holdout import sample_config
from generator.transform import transform_texts
from grader import score_submission
from policies.auditor import derive_from_public_texts
from policies.reconcile import naive_findings

OPEN = open_episodes()


def _score(truth, findings):
    return score_submission(truth, {"episode_id": truth["episode_id"], "findings": findings}, 0)


class IndependentValidator(unittest.TestCase):
    def test_open_episodes(self):
        for split, name, public, truth, _ in OPEN:
            with self.subTest(split=split, episode=name):
                r = _score(truth, derive_from_public_texts(public))
                self.assertEqual((r["fp"], r["fn"]), (0, 0), r["diagnostics"])

    def test_counterfactual_versions_are_clean(self):
        for split, name, _, truth, _ in OPEN:
            cf = read_texts(EPISODES / split / name / "private" / "counterfactual",
                            ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv"))
            with self.subTest(episode=name):
                self.assertEqual(derive_from_public_texts(cf), [])

    def test_new_parameter_combinations_and_pairs(self):
        rng = random.Random("b-checks-a")
        kinds = ["clean"] * 10 + ["interacting"] * 15 + ["mixed"] * 25
        for i, kind in enumerate(kinds):
            cfg = sample_config(rng, kind, f"b_check_{i}")
            ep = generate_episode(10_000 + i, cfg, salt="b-crosscheck")
            core = {k: v for k, v in ep.private.items() if "/" not in k}
            with self.subTest(i=i, kind=kind, config=cfg["name"]):
                r = _score(ep.truth, derive_from_public_texts(ep.public))
                self.assertEqual((r["fp"], r["fn"]), (0, 0), r["diagnostics"])
                if kind == "clean":
                    self.assertEqual(ep.truth["issues"], [])
                new_public, new_private, _ = transform_texts(ep.public, core, seed=i)
                r2 = _score(json.loads(new_private["truth.json"]), derive_from_public_texts(new_public))
                self.assertEqual((r2["fp"], r2["fn"]), (0, 0))


class AccountingEqualities(unittest.TestCase):
    def test_open_data_invariants(self):
        for split, name, public, truth, _ in OPEN:
            ep = load_public(public)
            m, t = ep.manifest, ep.tables
            with self.subTest(episode=name):
                ledger = t["ledger"]
                self.assertEqual(sum(r["debit_minor"] for r in ledger), sum(r["credit_minor"] for r in ledger))
                self.assertTrue(all(r["currency"] == m["base_currency"] for r in ledger))
                self.assertTrue(all((r["debit_minor"] > 0) != (r["credit_minor"] > 0) for r in ledger))
                by_doc = defaultdict(lambda: [0, 0])
                for r in ledger:
                    key = r["payment_id"] or r["invoice_id"]
                    if key:
                        by_doc[key][0] += r["debit_minor"]
                        by_doc[key][1] += r["credit_minor"]
                self.assertTrue(all(d == c for d, c in by_doc.values()), "per-document postings balance")
                inv_ids = {i["invoice_id"] for i in t["invoices"]}
                pay_ids = {p["payment_id"] for p in t["payments"]}
                self.assertTrue(all(r["invoice_id"] in inv_ids for r in ledger if r["invoice_id"]))
                self.assertTrue(all(r["payment_id"] in pay_ids for r in ledger if r["payment_id"]))
                for inv in t["invoices"]:
                    self.assertEqual(inv["total_minor"], sum(li["amount_minor"] for li in inv["line_items"]))
                    for li in inv["line_items"]:
                        self.assertEqual(li["amount_minor"], li["quantity"] * li["unit_price_minor"])
                for pay in t["payments"]:
                    refs = pay["invoice_refs"]
                    if refs is not None:
                        self.assertLessEqual(sum(r["amount_minor"] for r in refs), pay["amount_minor"])
                        self.assertTrue(all(r["invoice_id"] in inv_ids for r in refs))
                exps = m["currency_exponents"]
                self.assertTrue(all(d["currency"] in exps for d in t["invoices"] + t["payments"]))
                self.assertEqual(m["counts"]["ledger_rows"], len(ledger))
                self.assertEqual(m["counts"]["documents"], len(t["invoices"]) + len(t["payments"]))
                self.assertTrue(200 <= len(ledger) <= 2000)
                self.assertTrue(20 <= m["counts"]["documents"] <= 200)
                self.assertLessEqual(len(truth["issues"]), 12)
                for issue in truth["issues"]:
                    for ev in issue["evidence_sets"]:
                        for eid in ev:  # каждое доказательство — существующая открытая запись
                            self.assertTrue(eid in ep.documents or eid in ep.by_id["ledger"] or eid in ep.by_id["fx"], eid)


class NaiveReconcileLeakDetector(unittest.TestCase):
    def test_surface_heuristics_do_not_find_errors(self):
        f1s, tp, fp = [], 0, 0
        for split, name, public, truth, _ in OPEN:
            ep = load_public(public)
            t = ep.tables
            r = _score(truth, naive_findings(ep.manifest, t["ledger"], t["invoices"], t["payments"], t["fx"]))
            f1s.append(r["f1"])
            tp += r["tp"]
            fp += r["fp"]
        mean_f1 = sum(f1s) / len(f1s)
        self.assertLess(mean_f1, 0.5, f"naive reconcile mean F1 {mean_f1:.3f}: errors visible without evidence analysis")
        self.assertLess(tp / max(1, tp + fp), 0.3)


if __name__ == "__main__":
    unittest.main()
