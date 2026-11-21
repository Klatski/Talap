"""Устойчивость интерфейса жюри к тому, как жюри может его вызвать (сверка внешнего разбора needed_fixes.md).

* Адаптер самодостаточен: у процесса жюри свои пакеты env/, grader/, generator/, policies/ (и PYTHONPATH на них) —
  все три точки входа работают, а пакеты жюри остаются их собственными, sys.path не меняется.
* Один ответ — один результат: findings с кортежами в score_episode дают то же, что через step (JSON).
* Лишние ключи config не мешают (reset и generate_episode) и записываются в журнал оценщика.
* Каталог closed без truth.json проводится (метрики — у жюри); episode_id — из manifest, даже если truth другой.
* Копия жюри с BOM, CRLF, переставленными колонками CSV и ключами JSON даёт те же записи и тот же счёт.
* Неверный submit через step — ограниченная ошибка invalid_submission (result null), а не {"accepted": true}.
* Ключ курса в записи жюри <date>:<source>:<target> находит ту же запись fx.
* Брошенные без close() среды и песочница останавливаются быстро и не оставляют процессов.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from env_helpers import ROOT
import judge_adapter
from test_env_jury_adapter import C_CONFIG, J, JuryAdapter, oracle_findings, read_log, tree_digest


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True, text=True)
        return f'"{pid}"' in out.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover
        return True
    return True


def all_records(env, table: str) -> list[dict]:
    out, cursor = [], None
    while True:
        obs = env.step(J("list_records", table=table, cursor=cursor, limit=50))[0]
        out += obs["result"]["records"]
        cursor = obs["result"]["next_cursor"]
        if cursor is None:
            return out


def jury_style_copy(src: Path, dst: Path) -> None:
    """Копия, как её может записать инструмент жюри: UTF-8 с BOM, CRLF, колонки CSV в обратном порядке,
    ключи объектов JSONL в обратном порядке. Значения и закрытая часть не меняются."""
    shutil.copytree(src, dst)
    pub = dst / "public"
    for name in ("ledger.csv", "fx.csv"):
        rows = list(csv.reader(io.StringIO((pub / name).read_text(encoding="utf-8"), newline="")))
        buf = io.StringIO(newline="")
        csv.writer(buf, lineterminator="\r\n").writerows([list(reversed(r)) for r in rows])
        (pub / name).write_bytes(("﻿" + buf.getvalue()).encode("utf-8"))
    for name in ("invoices.jsonl", "payments.jsonl"):
        lines = [json.loads(line) for line in (pub / name).read_text(encoding="utf-8").splitlines() if line.strip()]
        text = "".join(json.dumps(dict(reversed(list(r.items()))), ensure_ascii=False) + "\r\n" for r in lines)
        (pub / name).write_bytes(("﻿" + text).encode("utf-8"))
    for name in ("manifest.json", "policy.json"):
        text = (pub / name).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\n", "\r\n")
        (pub / name).write_bytes(("﻿" + text).encode("utf-8"))


FAKE_JURY_SCRIPT = r'''
import importlib.util, json, os, sys, time
from pathlib import Path

root, fake, work = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(fake))
import env, grader, generator, policies  # пакеты самого жюри с теми же именами
path_before = list(sys.path)
spec = importlib.util.spec_from_file_location("judge_adapter", root / "judge_adapter.py")
ja = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ja)
ep = work / "эпизод жюри"
ja.generate_episode(11, {"size": "small", "errors": {"duplicate_invoice": 1}}, str(ep))
truth = json.loads((ep / "private" / "truth.json").read_text(encoding="utf-8"))
findings = {"episode_id": truth["episode_id"], "findings": [
    {"type": i["type"], "affected_ids": i["affected_ids"], "claimed_amount_minor": i["expected_amount_minor"],
     "currency": i["currency"], "evidence_ids": i["evidence_sets"][0], "explanation": ""} for i in truth["issues"]]}
e = ja.make_env()
obs, info = e.reset(0, {"mode": "closed", "episode_dir": str(ep), "team_config": {}})
page = e.step({"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 2}})[0]
done = e.step({"name": "submit", "args": {"findings": findings}})[0]
e.close()
score = ja.score_episode(str(ep), findings, 1)
print(json.dumps({"own": [m.JURY_OWN for m in (env, grader, generator, policies)],
                  "same_modules": all(sys.modules[n].JURY_OWN == n for n in ("env", "grader", "generator", "policies")),
                  "sys_path_same": sys.path == path_before, "records": len(page["result"]["records"]),
                  "submit": done["result"], "score": score, "issues": len(truth["issues"])}))
'''

ABANDON_SCRIPT = r'''
import importlib.util, json, sys, time
from pathlib import Path

root, ep, policy_file = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("judge_adapter", root / "judge_adapter.py")
ja = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ja)
envs = [ja.make_env() for _ in range(3)]
for e in envs:
    e.reset(0, {"mode": "closed", "episode_dir": ep, "team_config": {}})
    e.step({"name": "calculator", "args": {"expression": "1+1"}})
policy = ja.load_policy(policy_file)
print(json.dumps({"pids": [e._channel.proc.pid for e in envs] + [policy._channel.proc.pid],
                  "workdir": policy._workdir, "t": time.time()}), flush=True)
# выход без close(): ни среды, ни песочница не закрыты явно
'''

TRIVIAL_POLICY = '''
class P:
    def reset(self, observation, info=None):
        self.episode_id = observation["episode_id"]

    def act(self, observation):
        return {"name": "submit", "args": {"findings": {"episode_id": self.episode_id, "findings": []}}}


def make_policy():
    return P()
'''


class JuryRobustness(JuryAdapter):
    # ------------------------------------------------------------------ адаптер не трогает пакеты жюри
    def test_adapter_is_self_contained_next_to_the_jurys_own_packages(self):
        fake = self.root / "пакеты жюри"
        for pkg in ("env", "grader", "generator", "policies"):
            (fake / pkg).mkdir(parents=True, exist_ok=True)
            (fake / pkg / "__init__.py").write_text(f"JURY_OWN = {pkg!r}\n", encoding="utf-8")
        work = self.root / "рабочий каталог жюри"
        work.mkdir(exist_ok=True)
        env = {**os.environ, "PYTHONPATH": str(fake), "AUDIT_GRADER_LOG": str(self.root / "fake-jury-log.jsonl")}
        out = subprocess.run([sys.executable, "-c", FAKE_JURY_SCRIPT, str(ROOT), str(fake), str(work)], cwd=work,
                             env=env, capture_output=True, text=True, encoding="utf-8", timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        got = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(got["own"], ["env", "grader", "generator", "policies"])
        self.assertTrue(got["same_modules"])
        self.assertTrue(got["sys_path_same"])
        self.assertEqual(got["records"], 2)
        self.assertEqual(got["submit"], {"accepted": True})
        self.assertEqual((got["score"]["tp"], got["score"]["fp"], got["score"]["fn"]), (got["issues"], 0, 0))
        self.assertAlmostEqual(got["score"]["return"], 1.0 - 0.002, places=9)

    # ------------------------------------------------------------------ один ответ — один результат
    def test_findings_with_tuples_score_like_json(self):
        good = oracle_findings(self.e_truth)
        as_tuples = {"episode_id": good["episode_id"], "findings": tuple(
            {**f, "affected_ids": tuple(f["affected_ids"]), "evidence_ids": tuple(f["evidence_ids"])}
            for f in good["findings"])}
        a = judge_adapter.score_episode(str(self.E), good, 3)
        b = judge_adapter.score_episode(str(self.E), as_tuples, 3)
        self.assertEqual(a, b)
        self.assertEqual(b["tp"], 1)
        self.reset(self.E)  # тот же ответ через step
        self.step(J("calculator", expression="1"))
        self.step(J("calculator", expression="1"))
        self.step(J("calculator", expression="1"))
        self.step(J("submit", findings=as_tuples))
        rec = self.records()[-1]
        self.assertEqual((rec["tp"], rec["fp"], rec["fn"], rec["reward"]), (b["tp"], b["fp"], b["fn"], b["return"]))
        bad = judge_adapter.score_episode(str(self.E), {"episode_id": good["episode_id"], "findings": {1, 2}}, 0)
        self.assertEqual(bad["return"], -1.0)  # не-JSON (set) — неверная структура, как в среде
        self.assertIsNone(bad["tp"])

    # ------------------------------------------------------------------ лишние ключи config
    def test_extra_reset_config_keys_are_ignored_and_logged(self):
        obs, _ = self.raw_reset(0, {"mode": "closed", "episode_dir": str(self.E), "team_config": {},
                                    "split": "train", "jury_case": "C19"})
        self.assertEqual(obs["manifest"]["episode_id"], self.e_truth["episode_id"])  # ровно episode_dir
        self.step(J("submit", findings=oracle_findings(self.e_truth)))
        rec = self.records()[-1]
        self.assertEqual(rec["source"]["ignored_config_keys"], ["jury_case", "split"])
        self.assertEqual(rec["tp"], 1)

    def test_extra_generator_config_keys_are_ignored_and_logged(self):
        plain, extra, wrapped = (self.root / n for n in ("gen plain", "gen extra", "gen wrapped"))
        judge_adapter.generate_episode(5, dict(C_CONFIG), str(plain))
        judge_adapter.generate_episode(5, {**C_CONFIG, "jury_label": "D01"}, str(extra))
        judge_adapter.generate_episode(5, {"team_config": {**C_CONFIG, "note": 1}, "mode": "closed", "case": 2},
                                       str(wrapped))
        self.assertEqual(tree_digest(plain), tree_digest(extra))
        self.assertEqual(tree_digest(plain), tree_digest(wrapped))
        logged = [r for r in read_log(self.log) if r.get("kind") == "generate_episode"][-2:]
        self.assertEqual([r["ignored_config_keys"] for r in logged], [["jury_label"], ["case", "team_config.note"]])
        with self.assertRaises(ValueError):  # опечатка во вложенном имени типа ошибки — ошибка, а не чистая компания
            judge_adapter.generate_episode(5, {"errors": {"duplicate": 1}}, str(self.root / "gen typo"))

    # ------------------------------------------------------------------ каталог closed без эталона; episode_id
    def test_closed_episode_dir_without_truth_still_runs(self):
        d = self.root / "только открытые файлы"
        shutil.copytree(self.E / "public", d / "public")
        obs, _ = self.reset(d)
        out = self.step(J("submit", findings=oracle_findings(self.e_truth)))
        self.assertEqual((out[0]["result"], out[1], out[2]), ({"accepted": True}, 0.0, True))
        rec = self.records()[-1]
        self.assertEqual(rec["source"]["reference"], "missing")
        self.assertTrue(rec["valid"])
        self.assertEqual((rec["tp"], rec["reward"], rec["issues_total"]), (None, None, None))
        with self.assertRaises(Exception):  # train отдаёт награду политике — без эталона нельзя
            self.reset(d, mode="train")
        with self.assertRaises(ValueError):  # score_episode жюри требует эталон
            judge_adapter.score_episode(str(d), oracle_findings(self.e_truth), 0)
        self.assertEqual(self.reset(self.C)[0]["remaining_steps"], 100)

    def test_manifest_episode_id_wins_over_the_reference(self):
        d = self.root / "truth с другим episode_id"
        shutil.copytree(self.E, d)
        truth = dict(self.e_truth, episode_id="ep-OTHERONE0")
        (d / "private" / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
        answer = oracle_findings(self.e_truth)  # episode_id из manifest — то, что видит политика
        self.reset(d)
        self.step(J("submit", findings=answer))
        self.assertEqual(self.records()[-1]["tp"], 1)
        self.assertEqual(judge_adapter.score_episode(str(d), answer, 0)["tp"], 1)

    # ------------------------------------------------------------------ BOM, CRLF, порядок колонок
    def test_copy_with_bom_crlf_and_reordered_columns_gives_the_same_records_and_score(self):
        d = self.root / "копия жюри BOM CRLF"
        jury_style_copy(self.E, d)
        seen = {}
        for name, path in (("source", self.E), ("copy", d)):
            obs, _ = self.reset(path)
            records = {t: all_records(self.env, t) for t in ("ledger", "invoices", "payments", "fx")}
            seen[name] = (obs["manifest"], obs["policy"], records)
            self.step(J("submit", findings=oracle_findings(self.e_truth)))
            rec = self.records()[-1]
            seen[name] += ((rec["tp"], rec["fp"], rec["fn"], rec["reward"]),)
        self.assertEqual(seen["source"], seen["copy"])
        self.assertEqual(judge_adapter.score_episode(str(d), oracle_findings(self.e_truth), 9),
                         judge_adapter.score_episode(str(self.E), oracle_findings(self.e_truth), 9))

    # ------------------------------------------------------------------ неверный submit
    def test_malformed_submit_through_step_gets_a_bounded_structure_error(self):
        one = oracle_findings(self.e_truth)["findings"][0]
        for label, findings, needle in (
                ("31 findings", {"episode_id": self.e_truth["episode_id"], "findings": [one] * 31}, "30"),
                ("amount is a string", {"episode_id": self.e_truth["episode_id"],
                                        "findings": [{**one, "claimed_amount_minor": "10000"}]}, "claimed_amount_minor"),
                ("another episode", {"episode_id": "ep-NOTTHIS00", "findings": []}, "episode_id")):
            with self.subTest(case=label):
                obs, _ = self.reset(self.E)
                out = self.step(J("submit", findings=findings))
                self.assertIsNone(out[0]["result"])
                self.assertEqual(out[0]["error"]["code"], "invalid_submission")
                self.assertIn(needle, out[0]["error"]["detail"])
                self.assertEqual((out[1], out[2], out[3], out[4]), (0.0, True, False, {}))  # closed: награды нет
                self.assertBoundedError(out[0], 100)  # submit тоже расходует шаг
                self.assertEqual(self.records()[-1]["reward"], -1.0)

    # ------------------------------------------------------------------ ключ курса жюри
    def test_fx_record_by_the_jury_key(self):
        self.reset(self.C)
        rec = self.c_pub.tables["fx"][0]
        by_id = self.step(J("get_record", table="fx", id=rec["fx_id"]))[0]
        jury = self.step(J("get_record", table="fx", id=f"{rec['date']}:{rec['source_currency']}:{rec['target_currency']}"))[0]
        self.assertEqual(by_id["result"], jury["result"])
        self.assertEqual(jury["result"]["rate"], rec["rate"])
        missing = self.step(J("get_record", table="fx", id="1999-01-01:USD:KZT"))[0]
        self.assertEqual(missing["error"]["code"], "not_found")

    # ------------------------------------------------------------------ процессы без close()
    def test_abandoned_envs_and_sandbox_stop_quickly_without_close(self):
        policy = self.root / "trivial_policy.py"
        policy.write_text(TRIVIAL_POLICY, encoding="utf-8")
        log = self.root / "abandon-log.jsonl"
        env = {**os.environ, "AUDIT_GRADER_LOG": str(log)}
        proc = subprocess.run([sys.executable, "-c", ABANDON_SCRIPT, str(ROOT), str(self.C), str(policy)], cwd=self.root,
                              env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
        ended = time.time()
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        got = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertLess(ended - got["t"], 5.0)  # от последней строки скрипта до выхода процесса
        time.sleep(0.3)
        self.assertEqual([p for p in got["pids"] if pid_alive(p)], [])
        self.assertFalse(Path(got["workdir"]).exists())
        self.assertEqual([r["end_reason"] for r in read_log(log)], ["abandoned"] * 3)  # эпизоды закрыты, журнал дописан


class GraderLogLocation(unittest.TestCase):
    def test_default_grader_log_is_outside_the_repository_and_private(self):
        path = Path(judge_adapter._team_env("constants").default_grader_log())
        self.assertNotIn(ROOT.resolve(), path.resolve().parents)
        from grader.session import append_private_log

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "private-dir" / "log.jsonl"
            append_private_log(target, "{}\n")
            append_private_log(target, "{}\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "{}\n{}\n")
            if os.name == "posix":
                self.assertEqual(target.stat().st_mode & 0o777, 0o600)
                self.assertEqual(target.parent.stat().st_mode & 0o777, 0o700)


if __name__ == "__main__":
    unittest.main()
