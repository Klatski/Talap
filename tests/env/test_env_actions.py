"""Действия и ограниченные ошибки: каждое неверное действие расходует шаг, текст ошибки фиксирован."""

import json
import unittest

from env_helpers import STUB, assert_obs_shape, local_env
from env.constants import ERRORS, MAX_ACTION_BYTES


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.env = local_env()
        self.obs0, self.info0 = self.env.reset(seed=0, config=STUB)
        first = {}
        for table in ("ledger", "invoices", "payments", "fx"):
            first[table] = self.env.step({"action": "list_records", "table": table, "limit": 2})[0]["result"]["records"][0]
        self.first = first
        self.remaining = 96

    def tearDown(self):
        self.env.close()

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.remaining -= 1
        assert_obs_shape(self, obs)
        self.assertEqual(obs["remaining_steps"], self.remaining)
        self.assertEqual(reward, 0.0)
        self.assertFalse(term or trunc)
        self.assertEqual(info, {})
        return obs

    def test_initial_observation(self):
        o = self.obs0
        for key in ("task", "episode_id", "manifest", "policy", "tables", "issue_types", "actions", "limits", "remaining_steps"):
            self.assertIn(key, o)
        self.assertEqual(o["remaining_steps"], 100)
        self.assertEqual(o["limits"]["max_steps"], 100)
        self.assertEqual(o["limits"]["page_limit"], 50)
        self.assertEqual(set(o["issue_types"]), {"duplicate_invoice", "unapplied_payment", "fx_rate_error", "cutoff_error"})
        self.assertEqual(o["tables"]["ledger"]["rows"], o["manifest"]["counts"]["ledger_rows"])
        self.assertEqual(set(self.info0), {"env_version", "mode"})

    def test_get_record_each_table(self):
        keys = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": "fx_id"}
        for table, key in keys.items():
            with self.subTest(table=table):
                obs = self.step({"action": "get_record", "table": table, "id": self.first[table][key]})
                self.assertEqual(obs["result"], {"table": table, "record": self.first[table]})

    def test_get_document(self):
        inv, pay = self.first["invoices"], self.first["payments"]
        self.assertEqual(self.step({"action": "get_document", "doc_id": inv["doc_id"]})["result"],
                         {"doc_type": "invoice", "document": inv})
        self.assertEqual(self.step({"action": "get_document", "doc_id": pay["doc_id"]})["result"],
                         {"doc_type": "payment", "document": pay})
        # бизнес-ID — не doc_id; doc_id — не ключ таблицы
        self.assertEqual(self.step({"action": "get_document", "doc_id": inv["invoice_id"]})["error"]["code"], "document_not_found")
        self.assertEqual(self.step({"action": "get_record", "table": "invoices", "id": inv["doc_id"]})["error"]["code"], "not_found")

    def test_unknown_ids_do_not_echo_input(self):
        probes = ["L-PROBE1234", "../private/truth.json", "' OR 1=1 --", "INV-" + "Z" * 60, "ISS-AAAAAAAA"]
        for pid in probes:
            for table in ("ledger", "invoices", "payments", "fx"):
                obs = self.step({"action": "get_record", "table": table, "id": pid})
                self.assertEqual(obs["error"]["code"], "not_found")
                self.assertNotIn(pid, json.dumps(obs))
            obs = self.step({"action": "get_document", "doc_id": pid})
            self.assertEqual(obs["error"]["code"], "document_not_found")

    def test_malformed_actions(self):
        cases = [
            ("not json at all", "invalid_action"), (["list_records"], "invalid_action"), (None, "invalid_action"),
            (42, "invalid_action"), ({}, "invalid_action"), ({"action": 5}, "invalid_action"),
            ({"action": None}, "invalid_action"), ({"table": "ledger"}, "invalid_action"),
            ({"action": "drop_table"}, "unknown_action"), ({"action": "LIST_RECORDS"}, "unknown_action"),
            ({"action": "shell", "cmd": "ls"}, "unknown_action"),
            ({"action": "get_record", "table": "ledger"}, "missing_argument"),
            ({"action": "get_document"}, "missing_argument"), ({"action": "calculator"}, "missing_argument"),
            ({"action": "list_records"}, "missing_argument"),
            ({"action": "get_document", "doc_id": "D-1", "path": "/etc"}, "unexpected_argument"),
            ({"action": "list_records", "table": "ledger", "where": "1=1"}, "unexpected_argument"),
            ({"action": "list_records", "table": "accounts"}, "unknown_table"),
            ({"action": "list_records", "table": "truth"}, "unknown_table"),
            ({"action": "list_records", "table": None}, "invalid_argument"),
            ({"action": "get_record", "table": "ledger", "id": 5}, "invalid_argument"),
            ({"action": "get_record", "table": "ledger", "id": ""}, "invalid_argument"),
            ({"action": "get_record", "table": "ledger", "id": "x" * 500}, "invalid_argument"),
            ({"action": "get_document", "doc_id": ["D-1"]}, "invalid_argument"),
            ({"action": "calculator", "expression": "__import__('os')"}, "calculator_error"),
            ({"action": "calculator", "expression": None}, "calculator_error"),
            ({"action": "calculator", "expression": "1" * (MAX_ACTION_BYTES + 10)}, "action_too_large"),
            ({"action": "list_records", "table": "ledger", "limit": float("nan")}, "invalid_action"),
        ]
        for action, code in cases:
            with self.subTest(action=str(action)[:60]):
                obs = self.step(action)
                self.assertEqual(obs["error"]["code"], code)
                self.assertEqual(obs["error"]["message"], ERRORS[code])
                self.assertIsNone(obs["result"])

    def test_json_string_action_is_accepted(self):
        obs = self.step('{"action": "calculator", "expression": "2+2"}')
        self.assertEqual(obs["result"], {"value": "4"})

    def test_returned_records_are_copies(self):
        obs = self.step({"action": "get_record", "table": "invoices", "id": self.first["invoices"]["invoice_id"]})
        obs["result"]["record"]["total_minor"] = -1
        again = self.step({"action": "get_record", "table": "invoices", "id": self.first["invoices"]["invoice_id"]})
        self.assertEqual(again["result"]["record"], self.first["invoices"])

    def test_step_before_reset(self):
        env = local_env()
        obs, reward, term, trunc, info = env.step({"action": "list_records", "table": "ledger"})
        self.assertEqual(obs["error"]["code"], "not_reset")
        self.assertEqual((reward, term, trunc, info), (0.0, False, False, {}))
        env.close()


if __name__ == "__main__":
    unittest.main()
