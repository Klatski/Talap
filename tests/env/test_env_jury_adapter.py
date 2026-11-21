"""Протокол жюри через judge_adapter.py: блок 1 (контрактные тесты C01–C20), точки входа адаптера и парная
проверка блока 4 с НЕЗАВИСИМЫМ преобразованием (своя перестановка и своя смена ID, не generator.transform).

Фикстуры собираются вызовом generate_episode, как это делает жюри, в каталоги с произвольными именами:
  C — чистый эпизод (он же L: в ledger не меньше 101 записи, страниц больше трёх);
  E — эпизод ровно с одной ошибкой и полным доступным набором доказательств;
  U — ID, которого нет в открытых данных (выбирается после чтения данных).
Среда — make_env() (отдельный процесс среды и оценщика); журнал оценщика — файл из AUDIT_GRADER_LOG.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from env_helpers import EPISODES, ROOT  # noqa: F401 — ROOT кладёт корень репозитория в sys.path
import judge_adapter
from env.records import PUBLIC_FILES, load_public

TABLE_FILES = {"ledger": "ledger.csv", "invoices": "invoices.jsonl", "payments": "payments.jsonl", "fx": "fx.csv"}
E_CONFIG = {"size": "small", "errors": {"duplicate_invoice": 1}}
C_CONFIG = {"size": "small"}


def load_dir(d: Path):
    return load_public({n: (d / n).read_text(encoding="utf-8") for n in PUBLIC_FILES})


def read_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def oracle_findings(truth: dict) -> dict:
    """Правильные выводы из эталона — как oracle жюри (на стороне жюри, обычным submit)."""
    return {"episode_id": truth["episode_id"], "findings": [
        {"type": i["type"], "affected_ids": list(i["affected_ids"]), "claimed_amount_minor": i["expected_amount_minor"],
         "currency": i["currency"], "evidence_ids": list(i["evidence_sets"][0]), "explanation": "oracle"}
        for i in truth["issues"]]}


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def J(name: str, **args) -> dict:
    return {"name": name, "args": args}


class JuryAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="jury-adapter-")
        cls.root = Path(cls.tmp.name)
        cls.log = cls.root / "закрытый журнал оценщика.jsonl"
        cls._old_log = os.environ.get("AUDIT_GRADER_LOG")
        os.environ["AUDIT_GRADER_LOG"] = str(cls.log)
        cls.C = cls.root / "фикстура C (чистая) #1"
        cls.E = cls.root / "fixture E — одна ошибка"
        judge_adapter.generate_episode(5, {"team_config": C_CONFIG}, str(cls.C))
        judge_adapter.generate_episode(11, {"mode": "closed", "team_config": E_CONFIG}, str(cls.E))
        cls.c_pub = load_dir(cls.C / "public")
        cls.e_pub = load_dir(cls.E / "public")
        cls.e_truth = json.loads((cls.E / "private" / "truth.json").read_text(encoding="utf-8"))
        cls.c_truth = json.loads((cls.C / "private" / "truth.json").read_text(encoding="utf-8"))
        assert cls.c_truth["issues"] == [] and len(cls.e_truth["issues"]) == 1
        assert len(cls.c_pub.tables["ledger"]) >= 101
        known = {i for pub in (cls.c_pub, cls.e_pub) for ids in pub.by_id.values() for i in ids}
        known |= set(cls.c_pub.documents) | set(cls.e_pub.documents)
        cls.U = next(u for u in ("U-0000000", "NOPE-1", "D-ZZZZZZZZ") if u not in known)
        cls.env = judge_adapter.make_env()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        if cls._old_log is None:
            os.environ.pop("AUDIT_GRADER_LOG", None)
        else:
            os.environ["AUDIT_GRADER_LOG"] = cls._old_log
        cls.tmp.cleanup()

    def reset(self, d: Path, mode: str = "closed", env=None, **extra):
        return self.raw_reset(0, {"mode": mode, "episode_dir": str(d), "team_config": {}, **extra}, env)

    def raw_reset(self, seed, config, env=None):
        out = (env or self.env).reset(seed=seed, config=config)
        # прошлый незавершённый эпизод среда закрывает (abandoned) и пишет в журнал при reset — до этой отметки
        self.mark = len(read_log(self.log))
        return out

    def records(self) -> list[dict]:
        """Записи журнала оценщика после последнего reset."""
        return read_log(self.log)[self.mark:]

    def step(self, action, env=None):
        return (env or self.env).step(action)

    def assertJsonCompatible(self, value):
        self.assertEqual(json.loads(json.dumps(value, allow_nan=False)), value)

    def assertBoundedError(self, obs, before):
        self.assertIsNone(obs["result"])
        self.assertLessEqual({"code", "message"}, set(obs["error"]))
        self.assertLessEqual(set(obs["error"]), {"code", "message", "detail"})  # detail — короткое уточнение
        self.assertTrue(all(isinstance(v, str) and len(v) <= 200 for v in obs["error"].values()))
        self.assertEqual(obs["remaining_steps"], before - 1)  # шаг израсходован
        text = json.dumps(obs, ensure_ascii=False)
        for secret in (str(self.root), self.C.name, self.E.name, "truth.json", "provenance"):
            self.assertNotIn(secret, text)


class ContractBlock1(JuryAdapter):
    def test_c01_reset_returns_json_compatible_pair(self):
        out = self.reset(self.C)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        obs, info = out
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(info, dict)
        self.assertJsonCompatible(obs)
        self.assertJsonCompatible(info)

    def test_c02_first_observation(self):
        obs, info = self.reset(self.C)
        self.assertIsInstance(obs["task"], str)
        self.assertIsInstance(obs["manifest"], dict)
        self.assertIsInstance(obs["policy"], dict)
        self.assertEqual(obs["remaining_steps"], 100)
        text = json.dumps([obs, info], ensure_ascii=False)
        self.assertNotIn('"seed"', text)
        self.assertNotIn('"issues"', text)
        self.assertNotIn("expected_amount_minor", text)

    def test_c03_manifest_matches_the_data(self):
        obs, _ = self.reset(self.C)
        m = obs["manifest"]
        for key in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone", "counts",
                    "rounding"):
            self.assertIn(key, m)
        on_disk = json.loads((self.C / "public" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m, on_disk)
        t = self.c_pub.tables
        self.assertEqual(m["counts"], {"ledger_rows": len(t["ledger"]), "invoices": len(t["invoices"]),
                                       "payments": len(t["payments"]), "fx_rates": len(t["fx"]),
                                       "documents": len(t["invoices"]) + len(t["payments"])})
        with open(self.C / "public" / "ledger.csv", encoding="utf-8", newline="") as fh:
            self.assertEqual(sum(1 for _ in csv.DictReader(fh)), m["counts"]["ledger_rows"])

    def test_c04_one_list_records(self):
        self.reset(self.C)
        out = self.step(J("list_records", table="ledger", cursor=None, limit=5))
        self.assertEqual(len(out), 5)
        obs, reward, terminated, truncated, info = out
        self.assertIs(type(terminated), bool)
        self.assertIs(type(truncated), bool)
        self.assertEqual((reward, terminated, truncated), (0.0, False, False))
        self.assertJsonCompatible([obs, reward, info])
        self.assertEqual(set(obs), {"result", "error", "remaining_steps"})
        self.assertEqual(set(obs["result"]), {"records", "next_cursor"})

    def test_c05_pages_reach_every_row_once(self):
        self.reset(self.C)
        seen, cursor, pages = [], None, 0
        while True:
            obs = self.step(J("list_records", table="ledger", cursor=cursor, limit=50))[0]
            self.assertIsNone(obs["error"])
            page = obs["result"]["records"]
            self.assertLessEqual(len(page), 50)
            seen += [r["row_id"] for r in page]
            pages += 1
            cursor = obs["result"]["next_cursor"]
            if cursor is None:
                break
        self.assertGreaterEqual(pages, 3)
        with open(self.C / "public" / "ledger.csv", encoding="utf-8", newline="") as fh:
            in_file = [r["row_id"] for r in csv.DictReader(fh)]
        self.assertEqual(seen, in_file)  # все строки, без пропусков и повторов, в порядке файла

    def test_c06_same_cursor_same_page(self):
        self.reset(self.C)
        cursor = self.step(J("list_records", table="ledger", cursor=None, limit=50))[0]["result"]["next_cursor"]
        a = self.step(J("list_records", table="ledger", cursor=cursor, limit=50))[0]
        b = self.step(J("list_records", table="ledger", cursor=cursor, limit=50))[0]
        self.assertEqual(a["result"], b["result"])
        self.assertEqual(a["remaining_steps"] - 1, b["remaining_steps"])

    def test_c07_get_record_returns_the_record_object(self):
        self.reset(self.C)
        for table, key in (("ledger", "row_id"), ("invoices", "invoice_id"), ("payments", "payment_id")):
            rec = self.c_pub.tables[table][0]
            obs = self.step(J("get_record", table=table, id=rec[key]))[0]
            self.assertEqual(obs["result"], rec)
        with open(self.C / "public" / "ledger.csv", encoding="utf-8", newline="") as fh:
            raw = next(csv.DictReader(fh))
        got = self.step(J("get_record", table="ledger", id=raw["row_id"]))[0]["result"]
        self.assertEqual({k: ("" if v is None else str(v)) for k, v in got.items()}, raw)  # те же поля, что в файле

    def test_c08_get_document_returns_the_document(self):
        self.reset(self.C)
        with open(self.C / "public" / "invoices.jsonl", encoding="utf-8") as fh:
            inv = json.loads(fh.readline())
        with open(self.C / "public" / "payments.jsonl", encoding="utf-8") as fh:
            pay = json.loads(fh.readline())
        self.assertEqual(self.step(J("get_document", doc_id=inv["doc_id"]))[0]["result"], inv)
        self.assertEqual(self.step(J("get_document", doc_id=pay["doc_id"]))[0]["result"], pay)

    def test_c09_calculator_decimal(self):
        self.reset(self.C)
        self.assertEqual(self.step(J("calculator", expression="0.1+0.2"))[0]["result"], "0.3")

    def test_c10_calculator_outside_the_grammar(self):
        self.reset(self.C)
        for expr in ("open('truth.json').read()", "__import__('os').listdir('.')", "[x for x in ().__class__.__mro__]"):
            before = self.step(J("calculator", expression="1"))[0]["remaining_steps"]
            obs = self.step(J("calculator", expression=expr))[0]
            self.assertEqual(obs["error"]["code"], "calculator_error")
            self.assertBoundedError(obs, before)

    def test_c11_c12_unknown_id(self):
        obs, _ = self.reset(self.C)
        left = obs["remaining_steps"]
        for action in (J("get_record", table="ledger", id=self.U), J("get_document", doc_id=self.U),
                       J("get_document", doc_id="../private/truth.json"), J("get_record", table="invoices", id="*")):
            obs = self.step(action)[0]
            self.assertIn(obs["error"]["code"], ("not_found", "document_not_found"))
            self.assertBoundedError(obs, left)
            left = obs["remaining_steps"]
            text = json.dumps(obs)
            self.assertFalse(any(d in text for d in self.c_pub.documents))  # других документов не раскрывает
        ok = self.step(J("get_record", table="ledger", id=self.c_pub.tables["ledger"][0]["row_id"]))[0]
        self.assertEqual(ok["result"], self.c_pub.tables["ledger"][0])  # процесс среды жив

    def test_c13_unknown_action(self):
        obs, _ = self.reset(self.C)
        for action in (J("shell", cmd="ls"), J("sql", query="select 1"), {"name": "read_file", "args": {"path": "/"}}):
            before = obs["remaining_steps"]
            obs = self.step(action)[0]
            self.assertEqual(obs["error"]["code"], "unknown_action")
            self.assertBoundedError(obs, before)

    def test_c14_error_costs_one_step_and_breaks_nothing(self):
        self.reset(self.C)
        rid = self.c_pub.tables["ledger"][3]["row_id"]
        a = self.step(J("get_record", table="ledger", id=rid))[0]
        bad = self.step(J("get_record", table="ledger", id=self.U))[0]
        b = self.step(J("get_record", table="ledger", id=rid))[0]
        self.assertEqual([a["remaining_steps"], bad["remaining_steps"], b["remaining_steps"]], [99, 98, 97])
        self.assertEqual(a["result"], b["result"])

    def test_c15_submit_is_final(self):
        obs, _ = self.reset(self.E, mode="train")
        out = self.step(J("submit", findings=oracle_findings(self.e_truth)))
        self.assertEqual((out[0]["result"], out[2], out[3]), ({"accepted": True}, True, False))
        later = [self.step(J("calculator", expression="1")),
                 self.step(J("submit", findings={"episode_id": obs["episode_id"], "findings": []}))]
        for o, r, term, trunc, info in later:
            self.assertEqual((o["error"]["code"], r, info), ("episode_over", 0.0, {}))
        recs = self.records()
        self.assertEqual(len(recs), 1)
        self.assertEqual((recs[0]["tp"], recs[0]["fp"], recs[0]["fn"], recs[0]["reward"]), (1, 0, 0, 1.0))

    def test_c16_submit_as_action_100(self):
        obs, _ = self.reset(self.E)
        for k in range(99):
            o, r, term, trunc, _ = self.step(J("calculator", expression="1"))
            self.assertFalse(term or trunc, k)
        o, r, term, trunc, _ = self.step(J("submit", findings=oracle_findings(self.e_truth)))
        self.assertEqual((o["result"], term, trunc), ({"accepted": True}, True, False))
        rec = self.records()[0]
        self.assertEqual((rec["end_reason"], rec["inspection_steps"], rec["tp"]), ("submit", 99, 1))

    def test_c17_hundred_inspection_actions(self):
        self.reset(self.E)
        outs = [self.step(J("calculator", expression="2")) for _ in range(100)]
        self.assertFalse(any(o[2] or o[3] for o in outs[:99]))
        self.assertEqual((outs[-1][2], outs[-1][3]), (False, True))
        late = self.step(J("submit", findings=oracle_findings(self.e_truth)))
        self.assertEqual(late[0]["error"]["code"], "episode_over")
        recs = self.records()
        self.assertEqual(len(recs), 1)
        self.assertEqual((recs[0]["end_reason"], recs[0]["truncated"], recs[0]["tp"], recs[0]["fp"], recs[0]["fn"]),
                         ("step_limit", True, 0, 0, 1))  # итоговые findings пусты

    def test_c18_time_limit_includes_policy_time(self):
        self.reset(self.E, time_limit_s=0.5)
        self.step(J("calculator", expression="1"))
        time.sleep(1.0)  # «политика думает»: шагов нет, а таймер среды идёт
        recs = self.records()
        self.assertEqual(len(recs), 1)  # эпизод завершён и записан самой средой, без шага политики
        self.assertEqual((recs[0]["end_reason"], recs[0]["truncated"], recs[0]["tp"], recs[0]["fp"]),
                         ("time_limit", True, 0, 0))
        o, r, term, trunc, info = self.step(J("submit", findings=oracle_findings(self.e_truth)))
        self.assertEqual((o["result"], o["error"]["code"], term, trunc), (None, "time_limit", False, True))
        self.assertEqual(len(self.records()), 1)  # поздний submit ничего не добавил

    def test_c19_closed_mode_hides_the_reward(self):
        obs, info = self.reset(self.E)
        self.assertNotIn("reward", info)
        outs = [self.step(J("list_records", table="invoices", cursor=None, limit=3)),
                self.step(J("calculator", expression="1+1")),
                self.step(J("get_record", table="ledger", id=self.U)),
                self.step(J("submit", findings=oracle_findings(self.e_truth)))]
        for o, r, term, trunc, inf in outs:
            self.assertIs(type(r), float)
            self.assertEqual((r, inf), (0.0, {}))
        self.assertEqual(outs[-1][0], {"result": {"accepted": True}, "error": None, "remaining_steps": 96})
        rec = self.records()[0]
        self.assertEqual((rec["mode"], rec["tp"], rec["fp"], rec["fn"]), ("closed", 1, 0, 0))
        self.assertAlmostEqual(rec["reward"], 1 - 0.002 * 3, delta=1e-12)  # истинная награда — только у оценщика
        self.assertTrue(rec["diagnostics"])

    def test_c20_no_private_data_in_any_policy_channel(self):
        secrets = {self.E.name, self.C.name, str(self.root), "truth.json", "provenance.json", "counterfactual",
                   "expected_amount_minor", "evidence_sets", "issue_id"}
        for truth in (self.e_truth, self.c_truth):
            for issue in truth["issues"]:
                secrets.add(issue["issue_id"])
                secrets |= {v for k, v in issue.items() if k in ("scenario", "scenario_group") and v}
        seen = []
        for d, pub in ((self.C, self.c_pub), (self.E, self.e_pub)):
            for table in TABLE_FILES:  # все записи всех таблиц, по эпизоду на таблицу
                obs, info = self.reset(d)
                seen.append([obs, info])
                cursor = None
                while True:
                    out = self.step(J("list_records", table=table, cursor=cursor, limit=50))
                    seen.append(list(out))
                    cursor = out[0]["result"]["next_cursor"]
                    if cursor is None:
                        break
            docs = sorted(pub.documents)
            for start in range(0, len(docs), 95):  # все документы
                seen.append(list(self.reset(d)))
                for doc_id in docs[start:start + 95]:
                    seen.append(list(self.step(J("get_document", doc_id=doc_id))))
            seen.append(list(self.reset(d)))
            for action in (J("get_record", table="truth", id="x"), J("list_records", table="../private/truth"),
                           J("get_document", doc_id=str(d / "private" / "truth.json")), J("calculator", expression="1/0"),
                           J("submit", findings={"episode_id": "ep-OTHER00000", "findings": "all"})):
                seen.append(list(self.step(action)))
        text = json.dumps(seen, ensure_ascii=False)
        leaked = sorted(s for s in secrets if s in text)
        self.assertEqual(leaked, [])


class AdapterEntryPoints(JuryAdapter):
    def test_generate_episode_is_deterministic_and_writes_the_documented_files(self):
        a, b = self.root / "gen a", self.root / "gen b"
        judge_adapter.generate_episode(11, {"team_config": E_CONFIG}, str(a))
        judge_adapter.generate_episode(11, E_CONFIG, str(b))  # config можно дать и без обёртки team_config
        self.assertEqual(tree_digest(a), tree_digest(b))
        self.assertEqual(tree_digest(a), tree_digest(self.E))  # ключи reset (mode) в config генератору не мешают
        for name in ("manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "policy.json"):
            self.assertTrue((a / "public" / name).is_file(), name)
        for name in ("truth.json", "provenance.json"):
            self.assertTrue((a / "private" / name).is_file(), name)
        other = self.root / "gen c"
        judge_adapter.generate_episode(12, {"team_config": E_CONFIG}, str(other))
        self.assertNotEqual(json.loads((other / "public" / "manifest.json").read_text(encoding="utf-8"))["episode_id"],
                            self.e_pub.manifest["episode_id"])

    def test_closed_reset_loads_exactly_the_given_dir_and_restarts_it(self):
        first, _ = self.reset(self.E)
        self.assertEqual(first["episode_id"], self.e_pub.manifest["episode_id"])
        self.step(J("calculator", expression="1"))
        again, _ = self.reset(self.E)
        self.assertEqual(again, first)  # повторный reset начинает тот же эпизод заново, бюджет снова 100
        page = self.step(J("list_records", table="invoices", cursor=None, limit=50))[0]["result"]["records"]
        self.assertEqual(page, self.e_pub.tables["invoices"][:50])

    def test_any_directory_name_and_flat_layout(self):
        flat = self.root / "копия жюри (flat) [42] ünïcode"
        shutil.copytree(self.E / "public", flat)
        shutil.copy(self.E / "private" / "truth.json", flat / "truth.json")
        for d in (flat, self.root / "E copy with spaces"):
            if not d.exists():
                shutil.copytree(self.E, d)
            obs, _ = self.reset(d)
            self.assertEqual(obs["episode_id"], self.e_pub.manifest["episode_id"])
            self.step(J("submit", findings=oracle_findings(self.e_truth)))
            rec = self.records()[0]
            self.assertEqual((rec["tp"], rec["fp"], rec["fn"]), (1, 0, 0))
        try:
            rel = os.path.relpath(self.E)
        except ValueError:  # другой диск Windows: относительного пути нет
            rel = None
        if rel is not None:
            self.assertEqual(self.reset(Path(rel))[0]["episode_id"], self.e_pub.manifest["episode_id"])

    def test_train_without_episode_dir_uses_the_team_generator(self):
        obs, info = self.raw_reset(11, {"mode": "train", "team_config": E_CONFIG})
        self.assertEqual(obs["episode_id"], self.e_pub.manifest["episode_id"])  # тот же эпизод, что generate_episode
        page = self.step(J("list_records", table="ledger", cursor=None, limit=50))[0]["result"]["records"]
        self.assertEqual(page, self.e_pub.tables["ledger"][:50])
        o, r, term, trunc, info = self.step(J("submit", findings=oracle_findings(self.e_truth)))
        self.assertTrue(term)
        self.assertAlmostEqual(r, 1 - 0.002, delta=1e-12)  # в train терминальная награда доступна политике
        self.assertEqual((info["tp"], info["fp"], info["fn"]), (1, 0, 0))

    def test_train_with_null_team_config_still_uses_the_generator(self):
        a, _ = self.raw_reset(3, {"mode": "train", "team_config": None})
        b, _ = self.raw_reset(3, {"mode": "train", "team_config": {}})
        self.assertEqual(a["episode_id"], b["episode_id"])
        splits = {p.name for s in ("train", "dev", "stub") for p in (EPISODES / s).iterdir() if p.is_dir()}
        self.assertNotIn(a["episode_id"], splits)  # сгенерирован, а не взят из открытого сплита

    def test_bad_reset_raises_and_the_env_stays_usable(self):
        for config in ({"mode": "closed", "episode_dir": str(self.root / "нет такого каталога")},
                       {"mode": "closed", "episode_dir": str(self.E), "split": "train"},
                       {"mode": "exam", "episode_dir": str(self.E)}):
            with self.subTest(config=sorted(config)):
                with self.assertRaises(Exception):
                    self.env.reset(seed=0, config=config)
        self.assertEqual(self.reset(self.C)[0]["remaining_steps"], 100)

    def test_grader_log_records_environment_version_seed_and_config(self):
        self.raw_reset(7, {"mode": "closed", "episode_dir": str(self.E), "team_config": {"note": 1}})
        self.step(J("submit", findings=oracle_findings(self.e_truth)))
        rec = self.records()[0]
        self.assertEqual(rec["seed"], 7)
        self.assertEqual(rec["config"]["mode"], "closed")
        self.assertEqual(Path(rec["config"]["episode_dir"]), self.E)
        self.assertEqual(set(rec["environment"]), {"python", "implementation", "platform"})
        self.assertTrue(rec["env_version"] and rec["grader_version"])

    def test_flat_action_form_is_the_same_action(self):
        self.reset(self.C)
        a = self.step(J("get_record", table="ledger", id=self.c_pub.tables["ledger"][1]["row_id"]))[0]
        b = self.step({"action": "get_record", "table": "ledger", "id": self.c_pub.tables["ledger"][1]["row_id"]})[0]
        self.assertEqual(a["result"], b["result"])

    def test_score_episode_contract(self):
        good = judge_adapter.score_episode(str(self.E), oracle_findings(self.e_truth), 10)
        self.assertEqual(good, {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
                                "return": good["return"]})
        self.assertAlmostEqual(good["return"], 0.98, delta=1e-9)
        bad = judge_adapter.score_episode(str(self.E), {"episode_id": self.e_truth["episode_id"], "findings": [{}]}, 10)
        self.assertEqual(bad, {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None,
                               "return": -1.0})


# --------------------------------------------------------------------------- блок 4: независимое преобразование
ID_COLUMNS = {"ledger": {"row_id": "row", "entity_id": "entity", "invoice_id": "invoice", "payment_id": "payment"},
              "invoices": {"doc_id": "doc", "invoice_id": "invoice", "entity_id": "entity"},
              "payments": {"doc_id": "doc", "payment_id": "payment"}}
NEW_FORMAT = {"row": "row/{:05d}", "doc": "doc:{:05d}", "invoice": "bill.{:04d}", "payment": "rcpt.{:04d}",
              "entity": "party_{:03d}"}


def jury_transform(src: Path, dst: Path, seed: int, permute: bool = True, remap: bool = True) -> dict:
    """Своя копия жюри: перестановка строк/документов и взаимно однозначная смена ID во всех данных и ссылках,
    с правкой truth/provenance в закрытой части копии. Возвращает отображение старый ID -> новый."""
    rng = random.Random(seed)
    pub = src / "public"
    with open(pub / "ledger.csv", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header, ledger = reader.fieldnames, list(reader)
    docs = {t: [json.loads(x) for x in (pub / f).read_text(encoding="utf-8").splitlines() if x.strip()]
            for t, f in (("invoices", "invoices.jsonl"), ("payments", "payments.jsonl"))}
    with open(pub / "fx.csv", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fx_header, fx = reader.fieldnames, list(reader)
    spaces: dict[str, set] = {ns: set() for ns in NEW_FORMAT}
    for table, cols in ID_COLUMNS.items():
        rows = ledger if table == "ledger" else docs[table]
        for r in rows:
            for col, ns in cols.items():
                if r.get(col):
                    spaces[ns].add(r[col])
    mapping: dict[str, str] = {}
    for ns, ids in spaces.items():
        order = sorted(ids)
        numbers = list(range(1, len(order) + 1))
        rng.shuffle(numbers)
        for old, n in zip(order, numbers):
            mapping[old] = NEW_FORMAT[ns].format(n) if remap else old
    assert len(set(mapping.values())) == len(mapping)  # пространства ID не слились

    def m(value):
        return mapping.get(value, value) if isinstance(value, str) else value

    def deep(obj):
        if isinstance(obj, dict):
            return {k: deep(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [deep(v) for v in obj]
        return m(obj)

    for r in ledger:
        for col in ID_COLUMNS["ledger"]:
            r[col] = m(r[col]) if r[col] else r[col]
    for table in ("invoices", "payments"):
        for d in docs[table]:
            for col in ID_COLUMNS[table]:
                d[col] = m(d[col])
            if table == "payments" and d.get("invoice_refs"):
                d["invoice_refs"] = [{**ref, "invoice_id": m(ref["invoice_id"])} for ref in d["invoice_refs"]]
    if permute:  # порядок line_items не трогаем: он значим
        for seq in (ledger, docs["invoices"], docs["payments"], fx):
            rng.shuffle(seq)
    (dst / "public").mkdir(parents=True)
    (dst / "private").mkdir()
    for name in ("manifest.json", "policy.json"):
        shutil.copy(pub / name, dst / "public" / name)
    for name, hdr, rows in (("ledger.csv", header, ledger), ("fx.csv", fx_header, fx)):
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=hdr, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        (dst / "public" / name).write_bytes(buf.getvalue().encode("utf-8"))
    for table, name in (("invoices", "invoices.jsonl"), ("payments", "payments.jsonl")):
        body = "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs[table])
        (dst / "public" / name).write_bytes(body.encode("utf-8"))
    for name in ("truth.json", "provenance.json"):
        data = json.loads((src / "private" / name).read_text(encoding="utf-8"))
        (dst / "private" / name).write_bytes(json.dumps(deep(data), ensure_ascii=False).encode("utf-8"))
    return mapping


class PairBlock4(JuryAdapter):
    SOURCES = (("train", "ep-5C8KTAPXJA"), ("stub", "ep-SUW6BEVZNR"), ("dev", "ep-TBP8V4MB8H"))

    def _play(self, d: Path, findings: dict, steps: int) -> tuple[dict, list]:
        trace = [self.reset(d, mode="train")]
        for _ in range(steps):
            trace.append(self.step(J("calculator", expression="2+2")))
        trace.append(self.step(J("submit", findings=findings)))
        rec = self.records()[0]
        return {k: rec[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1", "reward", "inspection_steps")}, trace

    def test_original_replays_exactly(self):
        src = EPISODES / "train" / "ep-5C8KTAPXJA"
        truth = json.loads((src / "private" / "truth.json").read_text(encoding="utf-8"))
        runs = []
        for _ in range(2):
            trace = [self.reset(src, mode="train")]
            for table in ("ledger", "invoices", "payments", "fx"):
                trace.append(self.step(J("list_records", table=table, cursor=None, limit=50)))
            trace.append(self.step(J("submit", findings=oracle_findings(truth))))
            runs.append((json.dumps(trace, sort_keys=True), self.records()[0]["reward"]))
        self.assertEqual(runs[0], runs[1])

    def test_permuted_and_renamed_copy_scores_the_same(self):
        for k, (split, name) in enumerate(self.SOURCES):
            src = EPISODES / split / name
            digest = tree_digest(src)
            truth = json.loads((src / "private" / "truth.json").read_text(encoding="utf-8"))
            for variant, (permute, remap) in {"permute": (True, False), "remap": (False, True),
                                              "both": (True, True)}.items():
                with self.subTest(episode=name, variant=variant):
                    dst = self.root / f"pair {name} {variant} — копия"
                    mapping = jury_transform(src, dst, seed=100 + k, permute=permute, remap=remap)
                    t2 = json.loads((dst / "private" / "truth.json").read_text(encoding="utf-8"))
                    inverse = {v: k2 for k2, v in mapping.items()}
                    back = [{**i, "affected_ids": [inverse.get(x, x) for x in i["affected_ids"]],
                             "evidence_sets": [[inverse.get(x, x) for x in s] for s in i["evidence_sets"]]}
                            for i in t2["issues"]]
                    self.assertEqual(back, truth["issues"])  # после обратного отображения — тот же эталон
                    a, trace_a = self._play(src, oracle_findings(truth), 7)
                    b, trace_b = self._play(dst, oracle_findings(t2), 7)
                    self.assertEqual(a, b)
                    self.assertEqual((a["tp"], a["fn"]), (len(truth["issues"]), 0))
                    self.assertEqual(judge_adapter.score_episode(str(src), oracle_findings(truth), 7),
                                     judge_adapter.score_episode(str(dst), oracle_findings(t2), 7))
                    if remap and truth["issues"]:  # старые ID в новой копии не работают: доказательства настоящие
                        stale = judge_adapter.score_episode(str(dst), {**oracle_findings(truth),
                                                                       "episode_id": t2["episode_id"]}, 7)
                        self.assertEqual(stale["tp"], 0)
                        broken = oracle_findings(t2)  # без доказательств награда обязана упасть
                        for f in broken["findings"]:
                            f["evidence_ids"] = []
                        got = judge_adapter.score_episode(str(dst), broken, 7)
                        self.assertEqual(got["tp"], 0)
                        self.assertLess(got["return"], b["reward"])
            self.assertEqual(tree_digest(src), digest)  # исходный эпизод не перезаписан


if __name__ == "__main__":
    unittest.main()
