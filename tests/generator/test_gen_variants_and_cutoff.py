"""Заказ варианта ошибки (config.variants), частичный сценарий fx_partial, cut-off без артефактов во времени, τ по типу.

Сверка внешнего разбора:
* D09–D11: вариант неразнесённого платежа (полный пропуск / частичное разнесение) заказывается в конфиге;
* fx_partial_boundary_payment — всегда частичное разнесение, а не полный пропуск;
* ни одна строка книги не датирована вне периода, а «проводка раньше документа» бывает и у корректных
  документов — одни временные метки cut-off не выдают (нужна дата самого документа);
* допуск суммы 0 для точных сумм и 1 только для курсовой разницы.
"""

import json
import unittest

from gen_helpers import episode
from generator.config import ConfigError, VARIANTS, config_digest, normalize_config
from generator.fileio import parse_jsonl, parse_ledger
from generator.generate import generate_episode
from generator.timeutil import parse_iso


def injections(ep, error_type, scenario=None):
    return [i for i in ep.provenance["injections"] if i["type"] == error_type and i.get("scenario") == scenario]


def period(ep):
    m = json.loads(ep.public["manifest.json"])
    return parse_iso(m["period_start"]), parse_iso(m["period_end"]), m["timezone"]


def doc_times(ep):
    out = {r["invoice_id"]: parse_iso(r["issued_at"]) for r in parse_jsonl(ep.public["invoices.jsonl"])}
    out.update({r["payment_id"]: parse_iso(r["timestamp"]) for r in parse_jsonl(ep.public["payments.jsonl"])})
    return out


class Variants(unittest.TestCase):
    def test_unapplied_variant_can_be_ordered(self):
        for variant in VARIANTS["unapplied_payment"]:
            cfg = {"size": "small", "errors": {"unapplied_payment": 2}, "variants": {"unapplied_payment": variant}}
            for seed in range(6):
                with self.subTest(variant=variant, seed=seed):
                    ep = generate_episode(seed, cfg)
                    got = injections(ep, "unapplied_payment")
                    self.assertEqual([i["variant"] for i in got], [variant, variant])
                    for inj in got:
                        allocated = inj["after"]["allocated_minor"]
                        self.assertEqual(allocated == 0, variant == "missing_allocation", inj)

    def test_variant_list_gives_each_variant(self):
        """D09–D11: «хотя бы один пропуск и один частичный случай» — в одном эпизоде."""
        cfg = {"size": "small", "errors": {"unapplied_payment": 2},
               "variants": {"unapplied_payment": ["missing_allocation", "partial_allocation"]}}
        for seed in range(8):
            with self.subTest(seed=seed):
                got = sorted(i["variant"] for i in injections(generate_episode(seed, cfg), "unapplied_payment"))
                self.assertEqual(got, ["missing_allocation", "partial_allocation"])

    def test_other_types_take_ordered_variants(self):
        for error_type, variant in (("cutoff_error", "shifted_into_period"), ("cutoff_error", "entered_in_period"),
                                    ("fx_rate_error", "adjacent_date"), ("fx_rate_error", "other_rule"),
                                    ("duplicate_invoice", "reentered_document")):
            cfg = {"size": "medium", "errors": {error_type: 1}, "variants": {error_type: variant}}
            for seed in range(3):
                with self.subTest(type=error_type, variant=variant, seed=seed):
                    got = injections(generate_episode(seed, cfg), error_type)
                    self.assertEqual([i["variant"] for i in got], [variant])

    def test_bad_variants_are_config_errors(self):
        for variants in ({"unapplied": "partial_allocation"}, {"unapplied_payment": "partial"},
                         {"unapplied_payment": []}, {"unapplied_payment": ["partial_allocation"] * 2},
                         {"unapplied_payment": 1}, ["partial_allocation"], {"cutoff_error": "same_instant"}):
            with self.subTest(variants=variants), self.assertRaises(ConfigError):
                normalize_config({"variants": variants})

    def test_configs_without_variants_keep_their_digest(self):
        """Ключа нет в DEFAULT_CONFIG: без него (или с пустым) дайджест и, значит, эпизоды прежние."""
        base = config_digest(normalize_config({"size": "small", "errors": {"unapplied_payment": 1}}))
        for variants in ({}, None, {"unapplied_payment": None}):
            with self.subTest(variants=variants):
                cfg = normalize_config({"size": "small", "errors": {"unapplied_payment": 1}, "variants": variants})
                self.assertNotIn("variants", cfg)
                self.assertEqual(config_digest(cfg), base)
        ordered = normalize_config({"size": "small", "errors": {"unapplied_payment": 1},
                                    "variants": {"unapplied_payment": "partial_allocation"}})
        self.assertNotEqual(config_digest(ordered), base)


class Scenarios(unittest.TestCase):
    def test_fx_partial_boundary_payment_is_always_partial(self):
        cfg = {"size": "small", "scenarios": {"fx_partial_boundary_payment": 1}}
        for seed in range(30):
            with self.subTest(seed=seed):
                (inj,) = injections(generate_episode(seed, cfg), "unapplied_payment", "fx_partial_boundary_payment")
                self.assertEqual(inj["variant"], "partial_allocation")
                self.assertGreater(inj["after"]["allocated_minor"], 0)
                self.assertGreater(inj["after"]["unapplied_minor"], 0)

    def test_misread_invoice_is_posted_in_company_time_inside_the_period(self):
        """cutoff_fx_boundary_invoice: цифры времени документа (в его смещении) прочитаны как время компании."""
        cfg = {"size": "small", "scenarios": {"cutoff_fx_boundary_invoice": 1}}
        for seed in range(12):
            ep = generate_episode(seed, cfg)
            start, end, tz = period(ep)
            (issue,) = [i for i in ep.truth["issues"] if i["type"] == "cutoff_error"]
            inv = next(r for r in parse_jsonl(ep.public["invoices.jsonl"]) if r["invoice_id"] == issue["affected_ids"][0])
            rows = [r for r in parse_ledger(ep.public["ledger.csv"]) if r["invoice_id"] == inv["invoice_id"]]
            with self.subTest(seed=seed):
                self.assertFalse(start <= parse_iso(inv["issued_at"]) < end)  # документ соседнего периода
                for row in rows:
                    self.assertTrue(start <= parse_iso(row["timestamp"]) < end)
                    self.assertTrue(row["timestamp"].endswith(tz))
                    self.assertEqual(row["timestamp"][:19], inv["issued_at"][:19])


class CutoffTimestamps(unittest.TestCase):
    EPISODES = [("heavy", s) for s in range(6)] + [("all_types", s) for s in range(10)] + [("scenarios", s) for s in range(6)]

    def test_no_ledger_row_is_dated_outside_the_period(self):
        for key, seed in self.EPISODES:
            ep = episode(key, seed)
            start, end, _ = period(ep)
            outside = [r["row_id"] for r in parse_ledger(ep.public["ledger.csv"])
                       if not start <= parse_iso(r["timestamp"]) < end]
            with self.subTest(config=key, seed=seed):
                self.assertEqual(outside, [])

    def test_posted_before_its_document_is_not_an_error_signal(self):
        """Правило «проводка раньше документа» ловит часть cut-off, но в основном — корректные ранние проводки."""
        flagged = flagged_errors = 0
        for key, seed in self.EPISODES:
            ep = episode(key, seed)
            times = doc_times(ep)
            bad = {a for i in ep.truth["issues"] if i["type"] == "cutoff_error" for a in i["affected_ids"]}
            first: dict[str, object] = {}
            for row in parse_ledger(ep.public["ledger.csv"]):
                doc = row["payment_id"] or row["invoice_id"]
                ts = parse_iso(row["timestamp"])
                if doc and (doc not in first or ts < first[doc]):
                    first[doc] = ts
            for doc, ts in first.items():
                if doc in times and ts < times[doc]:
                    flagged += 1
                    flagged_errors += doc in bad
        self.assertGreater(flagged, 20)
        self.assertLess(flagged_errors / flagged, 0.35, (flagged_errors, flagged))

    def test_early_postings_are_recorded_and_stay_in_period(self):
        ep = episode("heavy", 0)
        early = [d for d in ep.provenance["distractors"] if d["kind"] == "posting_lag" and d["lag_minutes"] < 0]
        self.assertTrue(early)
        self.assertTrue(all(-3 * 24 * 60 - 1 <= d["lag_minutes"] <= -20 for d in early))


class Tolerance(unittest.TestCase):
    def test_tolerance_is_zero_for_exact_amounts(self):
        expected = {"duplicate_invoice": 0, "unapplied_payment": 0, "cutoff_error": 0, "fx_rate_error": 1}
        seen = set()
        for key, seed in (("heavy", 0), ("heavy", 1), ("all_types", 0)):
            for issue in episode(key, seed).truth["issues"]:
                with self.subTest(type=issue["type"]):
                    self.assertEqual(issue["amount_tolerance_minor"], expected[issue["type"]])
                seen.add(issue["type"])
        self.assertEqual(seen, set(expected))


if __name__ == "__main__":
    unittest.main()
