"""list_records: ≤ 50 строк, стабильные курсоры, полный обход без пропусков и повторов."""

import csv
import io
import json
import unittest

from env_helpers import EPISODES, STUB, assert_obs_shape, local_env
from env.cursor import encode_cursor
from env.records import load_public
from grader.episodes import read_episode

TABLES = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": "fx_id"}


def _walk(env, table, limit):
    rows, cursor, steps = [], None, 0
    while True:
        obs, _, term, trunc, _ = env.step({"action": "list_records", "table": table, "cursor": cursor, "limit": limit})
        steps += 1
        assert obs["error"] is None, obs["error"]
        res = obs["result"]
        assert len(res["records"]) <= 50
        assert res["offset"] == len(rows)
        rows += res["records"]
        cursor = res["next_cursor"]
        if cursor is None:
            return rows, steps


class PaginationTests(unittest.TestCase):
    def setUp(self):
        self.env = local_env()
        self.obs0, _ = self.env.reset(seed=0, config={**STUB, "max_steps": 100})
        self.counts = self.obs0["manifest"]["counts"]
        self.ep_id = self.obs0["episode_id"]

    def tearDown(self):
        self.env.close()

    def test_full_traversal_matches_files(self):
        name = self.ep_id
        public, _, _ = read_episode(EPISODES / "stub" / name)
        expect = load_public(public).tables
        for table, key in TABLES.items():
            with self.subTest(table=table):
                rows, steps = _walk(self.env, table, 50)
                self.assertEqual(rows, json.loads(json.dumps(expect[table])))
                self.assertEqual(len({r[key] for r in rows}), len(rows))
                self.assertEqual(steps, -(-len(rows) // 50))
        self.assertEqual(len(expect["ledger"]), self.counts["ledger_rows"])
        self.assertEqual(len(expect["fx"]), self.counts["fx_rates"])

    def test_ledger_values_are_typed_like_the_contract(self):
        rows, _ = _walk(self.env, "ledger", 50)
        text = (EPISODES / "stub" / self.ep_id / "public" / "ledger.csv").read_text(encoding="utf-8")
        raw = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(len(raw), len(rows))
        for got, src in zip(rows, raw):
            self.assertIsInstance(got["debit_minor"], int)
            self.assertIsInstance(got["credit_minor"], int)
            for k in ("invoice_id", "payment_id", "entity_id"):
                self.assertEqual(got[k], src[k] or None)
        self.assertTrue(any(r["invoice_id"] is None for r in rows))
        self.assertTrue(any(r["payment_id"] is None for r in rows))

    def test_page_size_does_not_change_content(self):
        a, _ = _walk(self.env, "invoices", 50)
        b, _ = _walk(self.env, "invoices", 7)
        self.assertEqual(a, b)

    def test_cursor_is_stable(self):
        first = self.env.step({"action": "list_records", "table": "ledger"})[0]
        cur = first["result"]["next_cursor"]
        p1 = self.env.step({"action": "list_records", "table": "ledger", "cursor": cur, "limit": 50})[0]["result"]
        p2 = self.env.step({"action": "list_records", "table": "ledger", "cursor": cur, "limit": 50})[0]["result"]
        self.assertEqual(p1, p2)
        self.assertEqual(cur, encode_cursor(self.ep_id, "ledger", 50))

    def test_limit_handling(self):
        obs = self.env.step({"action": "list_records", "table": "ledger", "limit": 500})[0]
        self.assertEqual(len(obs["result"]["records"]), 50)
        obs = self.env.step({"action": "list_records", "table": "fx", "cursor": None, "limit": 3})[0]
        self.assertEqual(len(obs["result"]["records"]), 3)
        for bad in (0, -1, "5", True, 1.5, None, [5]):
            with self.subTest(limit=bad):
                obs = self.env.step({"action": "list_records", "table": "ledger", "limit": bad})[0]
                assert_obs_shape(self, obs)
                self.assertEqual(obs["error"]["code"], "invalid_limit")

    def test_bad_cursors(self):
        good = self.env.step({"action": "list_records", "table": "ledger"})[0]["result"]["next_cursor"]
        table, offset, tag = good.split(".")
        bad = [
            f"invoices.{offset}.{tag}",                 # курсор другой таблицы
            f"{table}.{offset}.{'0' * 12}",             # испорченная подпись
            f"{table}.{int(offset) + 1}.{tag}",         # сдвинутое смещение
            encode_cursor(self.ep_id, "ledger", 10**6),  # за концом таблицы
            encode_cursor("ep-OTHER00000", "ledger", 50),  # курсор другого эпизода
            "ledger.50", "", 50, ["ledger.50"], "../" * 10, "ledger.-1." + tag,
        ]
        for cursor in bad:
            with self.subTest(cursor=cursor):
                obs = self.env.step({"action": "list_records", "table": "ledger", "cursor": cursor})[0]
                self.assertEqual(obs["error"]["code"], "invalid_cursor")
                self.assertIsNone(obs["result"])

    def test_cursor_does_not_survive_reset(self):
        cur = self.env.step({"action": "list_records", "table": "ledger"})[0]["result"]["next_cursor"]
        self.env.reset(seed=0, config={"split": "train"})
        obs = self.env.step({"action": "list_records", "table": "ledger", "cursor": cur})[0]
        self.assertEqual(obs["error"]["code"], "invalid_cursor")

    def test_fx_records_have_ids(self):
        rows, _ = _walk(self.env, "fx", 50)
        for r in rows[:20]:
            self.assertEqual(r["fx_id"], f"FX:{r['source_currency']}:{r['target_currency']}:{r['date']}")
            self.assertIsInstance(r["rate"], str)
        obs = self.env.step({"action": "get_record", "table": "fx", "id": rows[0]["fx_id"]})[0]
        self.assertEqual(obs["result"]["record"], rows[0])


if __name__ == "__main__":
    unittest.main()
