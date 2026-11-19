"""Нет утечек эталона в открытые файлы и нет признаков-артефактов, выдающих ошибки."""

import json
import re
import unittest

from gen_helpers import CONFIGS, episode
from generator.fileio import parse_fx, parse_jsonl, parse_ledger
from generator.leakcheck import leak_report

ID_LIKE = re.compile(r"(?:L|D|INV|PAY|C|V|SO|SKU|ep)-[0-9A-Z]+")


def _strings(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, str):
        yield value


def _data_strings(public):
    for name in ("invoices.jsonl", "payments.jsonl"):
        for rec in parse_jsonl(public[name]):
            yield from _strings(rec)
    for row in parse_ledger(public["ledger.csv"]) + parse_fx(public["fx.csv"]):
        yield from _strings(row)

FORBIDDEN_WORDS = re.compile(
    r"dup|duplicate|err|error|inject|truth|oracle|provenance|issue|scenario|seed|wrong|fake|bad", re.I
)


class LeakTests(unittest.TestCase):
    def test_private_values_not_in_public_files(self):
        for key in CONFIGS:
            for seed in range(4):
                ep = episode(key, seed)
                public_blob = "".join(ep.public.values())
                with self.subTest(config=key, seed=seed):
                    for issue in ep.truth["issues"]:
                        self.assertNotIn(issue["issue_id"], public_blob)
                        if issue["scenario_group"]:
                            self.assertNotIn(issue["scenario_group"], public_blob)
                    for inj in ep.provenance["injections"]:
                        self.assertNotIn(inj["cause"], public_blob)
                    self.assertNotIn(ep.provenance["config_digest"], public_blob)
                    self.assertNotIn(ep.provenance["config_name"], public_blob)

    def test_no_telltale_words_in_data_files(self):
        """В значениях данных (не в policy) нет слов, намекающих на ошибки или генерацию.
        ID пропускаются: их случайные хвосты могут случайно содержать «ERR»."""
        for key in CONFIGS:
            ep = episode(key, 0)
            for value in _data_strings(ep.public):
                if ID_LIKE.fullmatch(value):
                    continue
                with self.subTest(config=key, value=value):
                    self.assertIsNone(FORBIDDEN_WORDS.search(value))

    def test_manifest_does_not_reveal_config(self):
        for key in CONFIGS:
            manifest = json.loads(episode(key, 0).public["manifest.json"])
            self.assertEqual(
                set(manifest),
                {
                    "schema_version", "generator_version", "episode_id", "base_currency", "currency_exponents",
                    "period_start", "period_end", "timezone", "counts", "rounding", "money_format",
                    "fx_rate_format", "tables",
                },
            )

    def test_ids_have_uniform_format(self):
        """Все ID одного типа имеют одинаковую длину и алфавит — у внесённых записей нет особого формата."""
        pattern = re.compile(r"^(L|D|INV|PAY|C|V|SO|ep)-[23456789ABCDEFGHJKMNPQRSTUVWXYZ]+$")
        ep = episode("heavy", 1)
        ids = re.findall(r"\b(?:L|D|INV|PAY|C|V|SO|ep)-[0-9A-Z]+\b", "".join(ep.public.values()))
        lengths = {}
        for i in ids:
            self.assertRegex(i, pattern)
            lengths.setdefault(i.split("-")[0], set()).add(len(i))
        for prefix, values in lengths.items():
            self.assertEqual(len(values), 1, (prefix, values))

    def test_artifact_features_do_not_reveal_errors(self):
        """Ни один «тупой» признак не даёт точности выше 35% (базовая частота ~6-8%)."""
        episodes = []
        for key in ("all_types", "scenarios", "heavy"):
            for seed in range(12):
                ep = episode(key, seed)
                episodes.append((ep.public, ep.truth))
        report = leak_report(episodes)
        for name, stats in report["features"].items():
            with self.subTest(feature=name, stats=stats):
                self.assertLess(stats["precision"], 0.35)

    def test_injected_rows_are_not_appended_at_the_end(self):
        positions = []
        for seed in range(10):
            ep = episode("heavy", seed)
            lines = ep.public["ledger.csv"].splitlines()[1:]
            ids = {e for issue in ep.truth["issues"] for s in issue["evidence_sets"] for e in s if e.startswith("L-")}
            for pos, line in enumerate(lines):
                if line.split(",", 1)[0] in ids:
                    positions.append(pos / len(lines))
        self.assertTrue(positions)
        mean = sum(positions) / len(positions)
        self.assertGreater(mean, 0.25)
        self.assertLess(mean, 0.75)


if __name__ == "__main__":
    unittest.main()
