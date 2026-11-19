"""Протокол процесса среды: мусорный ввод не роняет сервер и не расходует шаги; EOF закрывает всё."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from env_helpers import ROOT, STUB
from env.protocol import MAX_LINE_BYTES, dumps, loads


class RawServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "log.jsonl"
        self.proc = subprocess.Popen([sys.executable, "-m", "env.server", "--grader-log", str(self.log)], cwd=ROOT,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)

    def tearDown(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.tmp.cleanup()

    def send(self, raw: bytes) -> dict:
        self.proc.stdin.write(raw)
        self.proc.stdin.flush()
        return loads(self.proc.stdout.readline())

    def call(self, method, params, rid=1):
        return self.send(dumps({"id": rid, "method": method, "params": params}))

    def test_garbage_and_oversize_lines(self):
        self.assertTrue(self.call("reset", {"seed": 0, "config": STUB})["ok"])
        for raw in (b"not json\n", b"[1,2,3]\n", b'{"method": 5}\n', b'{"id":1,"method":"step","params":{"action":NaN}}\n',
                    b'{"id":1,"method":"step","params":[1]}\n', b"\xff\xfe\n"):
            with self.subTest(raw=raw):
                reply = self.send(raw)
                self.assertFalse(reply["ok"])
                self.assertEqual(reply["error"]["code"], "bad_request")
        reply = self.send(b"x" * (MAX_LINE_BYTES + 100) + b"\n")
        self.assertEqual(reply["error"]["code"], "request_too_large")
        reply = self.call("step", {"action": {"action": "calculator", "expression": "1+1"}})
        self.assertEqual(reply["result"]["observation"]["remaining_steps"], 99)  # мусор шагов не съел
        self.assertEqual(self.call("step", {})["error"]["code"], "bad_params")
        self.assertEqual(self.call("teleport", {})["error"]["code"], "unknown_method")

    def test_eof_closes_env_and_grader(self):
        self.assertTrue(self.call("reset", {"seed": 0, "config": STUB})["ok"])
        self.call("step", {"action": {"action": "list_records", "table": "fx"}})
        self.proc.stdin.close()
        self.assertEqual(self.proc.wait(timeout=30), 0)
        records = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["end_reason"], "abandoned")

    def test_reset_errors_are_bounded(self):
        reply = self.call("reset", {"seed": 0, "config": {"split": "../../"}})
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["error"]["code"], "reset_failed")
        self.assertNotIn(str(ROOT), json.dumps(reply))
        reply = self.call("reset", {"seed": 0, "config": {"split": "stub"}})
        self.assertTrue(reply["ok"])


if __name__ == "__main__":
    unittest.main()
