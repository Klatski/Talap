"""Карточка жюри: раздел 10 Case.md («Критерии оценки») и раздел 06, переведённые в код один в один.

    python tests/jury/scorecard.py [--salt S] [--out reports/jury_scorecard.md]
    (в составе единой проверки — этап 7 python check.py)

Блоки и формулы — как у жюри:
  Контракт и изоляция   25 × пройдено/20 фиксированных проверок reset/step, схем, лимитов, завершения,
                        неверного ввода и отсутствия утечек;
  Данные и эталон       25 × корректных/20 закрытых эпизодов: бухгалтерские равенства до внесения
                        ошибки, объявленное расхождение после, независимая достаточность доказательств;
  Надёжность награды    25 × пройдено/20 проверок сопоставления, чистых случаев, FP, дублей, стоимости
                        и атакующих ответов;
  Воспроизводимость     15 × пройдено/20 парных проверок: повтор seed/действий точен, а перестановка
                        и смена ID сохраняют ответ oracle и награду с учётом новых ID;
  Инженерия             10 = пять критериев × 0/1/2 (здесь только автоматическая самопроверка наличия).
Закрытые эпизоды — имитация (generator.holdout: соль, новые сочетания параметров, 5 чистых, 5 со
взаимодействующими ошибками, + 20 пар «перестановка + смена ID»). Настоящие seed жюри неизвестны,
поэтому 20/20 здесь — необходимое, но не достаточное условие.
Всё, что видит агент, идёт через изолированную среду (env.AuditEnv); эталон читает только эта обвязка.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import AuditEnv  # noqa: E402
from env.constants import ERRORS  # noqa: E402
from env.records import load_public  # noqa: E402
from generator.attacks import build_attacks  # noqa: E402
from generator.fileio import read_episode_dir, read_texts  # noqa: E402
from generator.generate import generate_episode  # noqa: E402
from generator.leakscan import scan_records  # noqa: E402
from generator.transform import remap_findings, transform_texts  # noqa: E402
from grader import score_submission  # noqa: E402
from grader.episodes import PUBLIC_FILES  # noqa: E402
from grader.validate import ISSUE_TYPES  # noqa: E402
from policies import make_policy  # noqa: E402
from policies.auditor import derive_from_public_texts  # noqa: E402
from policies.oracle import oracle_findings  # noqa: E402

ENGINEERING_HEADING = "## Инженерия и объяснение (0/1/2, самопроверка)"
BLOCKS = (  # (ключ, название, вес)
    ("contract", "Контракт и изоляция", 25),
    ("data", "Данные и эталон", 25),
    ("reward", "Надёжность награды", 25),
    ("pairs", "Воспроизводимость и обобщение", 15),
)


@dataclass
class Check:
    name: str
    ok: bool
    note: str = ""


def _guard(name: str, fn: Callable[[], tuple[bool, str]]) -> Check:
    try:
        ok, note = fn()
    except Exception as exc:  # noqa: BLE001 — исключение = проверка не пройдена
        ok, note = False, f"{type(exc).__name__}: {str(exc)[:160]}"
    return Check(name, bool(ok), note)


def _json_ok(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


def _canonical(findings: list[dict]) -> list:
    """Ответ без выбора среди равноправных наборов доказательств."""
    return sorted((f["type"], tuple(sorted(set(f["affected_ids"]))), f["claimed_amount_minor"], f["currency"]) for f in findings)


def _truth_key(truth: dict, rename: Callable[[str], str] = lambda x: x) -> list:
    """Эталон как множество: ошибки и их допустимые наборы доказательств (порядок в списках не важен)."""
    return sorted((i["type"], tuple(sorted(rename(x) for x in i["affected_ids"])), i["expected_amount_minor"], i["currency"],
                   i["amount_tolerance_minor"], tuple(sorted(tuple(sorted(rename(x) for x in s)) for s in i["evidence_sets"])))
                  for i in truth["issues"])


def _play(env, policy, config: dict, seed: int = 0) -> list[dict]:
    obs, info = env.reset(seed=seed, config=config)
    policy.reset(obs, info)
    trace = [{"episode_id": obs["episode_id"], "step": 0, "action": None, "observation": obs, "reward": 0.0,
              "terminated": False, "truncated": False, "info": info}]
    done = False
    while not done:
        action = policy.act(obs)
        obs, reward, term, trunc, info = env.step(action)
        trace.append({"episode_id": trace[0]["episode_id"], "step": len(trace), "action": action, "observation": obs,
                      "reward": reward, "terminated": term, "truncated": trunc, "info": info})
        done = term or trunc
    return trace


def _dump(trace: list[dict]) -> str:
    return json.dumps(trace, sort_keys=True, default=repr, ensure_ascii=False)


def _log(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []


# ====================================================================== эпизоды
class Holdout:
    def __init__(self, root: Path, salt: str) -> None:
        self.root, self.salt = Path(root), salt
        self.index = json.loads((self.root / "holdout_index.json").read_text(encoding="utf-8"))
        self.entries: list[dict] = self.index["episodes"]
        self.position = {e["episode_id"]: k for k, e in enumerate(self.entries)}  # номер пары в seed преобразования

    def episode_dir(self, entry: dict, split: str = "episodes") -> Path:
        return self.root / split / (entry["episode_id"] if split == "episodes" else entry["pair_episode_id"])

    def texts(self, entry: dict, split: str = "episodes") -> tuple[dict, dict]:
        return read_episode_dir(self.episode_dir(entry, split))

    def counterfactual(self, entry: dict) -> dict:
        return read_texts(self.episode_dir(entry) / "private" / "counterfactual", PUBLIC_FILES)

    def config(self, entry: dict, mode: str = "train", split: str = "episodes", **extra) -> dict:
        ep = entry["episode_id"] if split == "episodes" else entry["pair_episode_id"]
        return {"split": split, "episode_id": ep, "mode": mode, **extra}


# ====================================================================== бухгалтерия (своя реализация B)
def accounting_problems(texts: dict[str, str]) -> list[str]:
    """Бухгалтерские равенства по открытым записям, без кода генератора."""
    ep = load_public(texts)
    t, m = ep.tables, ep.manifest
    out = []
    ledger = t["ledger"]
    if sum(r["debit_minor"] for r in ledger) != sum(r["credit_minor"] for r in ledger):
        out.append("ledger: debit != credit")
    if any(r["currency"] != m["base_currency"] for r in ledger):
        out.append("ledger: row not in base currency")
    if any((r["debit_minor"] > 0) == (r["credit_minor"] > 0) for r in ledger):
        out.append("ledger: row must have exactly one of debit/credit")
    per_doc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in ledger:
        key = r["payment_id"] or r["invoice_id"]
        if key:
            per_doc[key][0] += r["debit_minor"]
            per_doc[key][1] += r["credit_minor"]
    if any(d != c for d, c in per_doc.values()):
        out.append("ledger: postings of a document do not balance")
    inv_ids = {i["invoice_id"] for i in t["invoices"]}
    pay_ids = {p["payment_id"] for p in t["payments"]}
    if any(r["invoice_id"] and r["invoice_id"] not in inv_ids for r in ledger) or \
            any(r["payment_id"] and r["payment_id"] not in pay_ids for r in ledger):
        out.append("ledger: reference to a missing document (guessed link)")
    for inv in t["invoices"]:
        if inv["total_minor"] != sum(li["amount_minor"] for li in inv["line_items"]):
            out.append(f"invoice total != sum of lines ({inv['doc_id']})")
            break
        if any(li["amount_minor"] != li["quantity"] * li["unit_price_minor"] for li in inv["line_items"]):
            out.append(f"line amount != quantity * price ({inv['doc_id']})")
            break
    for pay in t["payments"]:
        refs = pay["invoice_refs"]
        if refs is not None and (sum(r["amount_minor"] for r in refs) > pay["amount_minor"]
                                 or any(r["invoice_id"] not in inv_ids for r in refs)):
            out.append(f"payment allocation inconsistent ({pay['doc_id']})")
            break
    exps = m["currency_exponents"]
    if any(d["currency"] not in exps for d in t["invoices"] + t["payments"]):
        out.append("document currency without exponent in manifest")
    if m["counts"]["ledger_rows"] != len(ledger) or m["counts"]["documents"] != len(t["invoices"]) + len(t["payments"]):
        out.append("manifest counts differ from the files")
    return out


# ====================================================================== блок 1: контракт и изоляция
def contract_block(h: Holdout, work: Path) -> list[Check]:
    clean = next(e for e in h.entries if e["kind"] == "clean")
    rich = next(e for e in h.entries if e["kind"] == "interacting")
    pub, priv = h.texts(rich)
    data = load_public(pub)
    truth = json.loads(priv["truth.json"])
    env = AuditEnv(episodes_root=h.root)
    checks: list[Check] = []

    def reset(entry=rich, mode="train", **extra):
        return env.reset(seed=0, config=h.config(entry, mode, **extra))

    def c01():
        obs, info = reset()
        ok = ({"task", "manifest", "policy", "remaining_steps"} <= set(obs) and obs["remaining_steps"] == 100
              and obs["manifest"] == json.loads(pub["manifest.json"]) and obs["policy"] == json.loads(pub["policy.json"])
              and isinstance(info, dict) and _json_ok(obs) and _json_ok(info))
        return ok, "first observation: task, manifest, policy, budget"

    def c02():
        reset()
        out = env.step({"action": "list_records", "table": "ledger"})
        obs, r, term, trunc, info = out
        ok = (len(out) == 5 and set(obs) == {"result", "error", "remaining_steps"} and type(r) is float
              and type(term) is bool and type(trunc) is bool and isinstance(info, dict) and obs["remaining_steps"] == 99)
        return ok, "step -> (observation, reward, terminated, truncated, info); observation = result/error/budget"

    def c03():
        obs, info = reset()
        blob = json.dumps([obs, info], ensure_ascii=False)
        hits = [w for w in (h.root.name, "truth.json", "provenance", "counterfactual", str(rich["seed"])) if w in blob]
        leaks = scan_records([{"episode_id": obs["episode_id"], "observation": obs, "info": info}], h.root)
        return not hits and not leaks, f"no seed/paths/truth in reset output ({len(hits) + len(leaks)} hits)"

    def c04():
        reset()
        rows, cursor, sizes = [], None, []
        while True:
            res = env.step({"action": "list_records", "table": "ledger", "cursor": cursor, "limit": 50})[0]["result"]
            rows += res["records"]
            sizes.append(len(res["records"]))
            cursor = res["next_cursor"]
            if cursor is None:
                break
        big = env.step({"action": "list_records", "table": "invoices", "limit": 500})[0]["result"]
        ok = rows == data.tables["ledger"] and max(sizes) <= 50 and len(big["records"]) == min(50, len(data.tables["invoices"]))
        return ok, f"{len(rows)} ledger rows in {len(sizes)} pages of <= 50, file order, limit 500 -> 50"

    def c05():
        reset()
        c = env.step({"action": "list_records", "table": "ledger", "limit": 20})[0]["result"]["next_cursor"]
        p1 = env.step({"action": "list_records", "table": "ledger", "cursor": c, "limit": 20})[0]["result"]
        p2 = env.step({"action": "list_records", "table": "ledger", "cursor": c, "limit": 20})[0]["result"]
        other = env.step({"action": "list_records", "table": "invoices", "cursor": c})[0]["error"]["code"]
        forged = env.step({"action": "list_records", "table": "ledger", "cursor": c[:-1] + ("0" if c[-1] != "0" else "1")})[0]
        reset()
        again = env.step({"action": "list_records", "table": "ledger", "limit": 20})[0]["result"]["next_cursor"]
        ok = p1 == p2 and other == "invalid_cursor" and forged["error"]["code"] == "invalid_cursor" and again == c
        return ok, "same cursor -> same page; bound to table/episode; deterministic"

    def c06():
        reset()
        inv, pay, row = data.tables["invoices"][0], data.tables["payments"][0], data.tables["ledger"][0]
        got = [env.step({"name": "get_record", "args": {"table": "invoices", "id": inv["invoice_id"]}})[0]["result"] == inv,
               env.step({"name": "get_document", "args": {"doc_id": pay["doc_id"]}})[0]["result"] == pay,
               env.step({"name": "get_document", "args": {"doc_id": inv["doc_id"]}})[0]["result"] == inv,
               env.step({"name": "get_record", "args": {"table": "ledger", "id": row["row_id"]}})[0]["result"] == row]
        if data.tables["fx"]:
            fx = data.tables["fx"][0]
            got.append(env.step({"name": "get_record", "args": {"table": "fx", "id": fx["fx_id"]}})[0]["result"] == fx)
        return all(got), "get_record/get_document return exactly the public record object (jury envelope)"

    def c07():
        reset()
        cases = {"0.1 + 0.2": "0.3", "10 / 4": "2.5", "round_half_up(2.5)": "3", "round_half_even(2.5)": "2",
                 "round_half_up(1.005, 2)": "1.01", "1000000 * 1000000": "1000000000000", "abs(-3.25)": "3.25"}
        bad = [e for e, v in cases.items() if env.step({"name": "calculator", "args": {"expression": e}})[0]["result"] != v]
        return not bad, f"exact decimal arithmetic and rounding ({len(cases) - len(bad)}/{len(cases)})"

    def c08():
        reset()
        exprs = ["__import__('os').system('dir')", "9**9**9", "open('truth.json').read()", "x + 1", "1e999", "().__class__"]
        outs = [env.step({"action": "calculator", "expression": e})[0] for e in exprs]
        ok = all(o["error"] and o["error"]["code"] == "calculator_error" for o in outs) and outs[-1]["remaining_steps"] == 100 - len(exprs)
        return ok, "code, power towers, names and exponent literals rejected; each step used"

    def c09():
        reset()
        outs = [env.step(a)[0] for a in ({"action": "shell", "cmd": "cat truth.json"}, {"action": "sql", "query": "select *"},
                                         {"action": "read_file", "path": "/etc/passwd"})]
        ok = (all(o["error"]["code"] == "unknown_action" for o in outs) and outs[-1]["remaining_steps"] == 97
              and not any(w in json.dumps(outs) for w in ("cat truth", "select", "passwd")))
        return ok, "no shell/SQL/FS: bounded unknown_action, step used, input not echoed"

    def c10():
        reset()
        acts = [({"action": "get_record", "table": "ledger", "id": "L-ZZZZZZZZ"}, "not_found"),
                ({"action": "get_document", "doc_id": "D-ZZZZZZZZ"}, "document_not_found"),
                ({"action": "get_document", "doc_id": data.tables["invoices"][0]["invoice_id"]}, "document_not_found"),
                ({"action": "get_record", "table": "truth", "id": "x"}, "unknown_table")]
        outs = [(env.step(a)[0], code) for a, code in acts]
        ok = all(o["error"]["code"] == code for o, code in outs) and outs[-1][0]["remaining_steps"] == 96 \
            and "ZZZZ" not in json.dumps([o for o, _ in outs])
        return ok, "unknown IDs/tables: bounded errors, steps used, IDs not echoed"

    def c11():
        reset()
        deep: Any = {"action": "calculator", "expression": "1"}
        for _ in range(20_000):
            deep = {"a": deep}
        bad = [None, [], 42, "garbage", "{}", {"action": 5}, {"action": "get_record"},
               {"action": "get_record", "table": "ledger", "id": "x", "path": "/"},
               {"action": "list_records", "table": "ledger", "limit": -1}, {"action": "list_records", "table": "ledger", "limit": True},
               {"action": "calculator", "expression": None}, deep, {"action": "calculator", "expression": "1", "n": float("nan")},
               {"action": "calculator", "expression": "1", "n": 10 ** 5000}, "x" * 300_000, '{"action": "calculator", "x": NaN}']
        outs = [env.step(a)[0] for a in bad]
        ok = all(o["error"] and o["error"]["code"] in ERRORS and o["error"]["code"] != "internal_error" for o in outs) \
            and outs[-1]["remaining_steps"] == 100 - len(bad)
        return ok, f"{len(bad)} malformed inputs -> bounded catalogue errors, no exception, each step used"

    def c12():
        obs, _ = reset()
        o, r, term, trunc, _ = env.step({"name": "submit", "args": {"findings": {"episode_id": obs["episode_id"], "findings": []}}})
        return term is True and trunc is False and o["result"] == {"accepted": True}, "submit -> {accepted: true}, terminated=true"

    def c13():
        log = work / "c13.jsonl"
        with AuditEnv(episodes_root=h.root, grader_log=log) as e2:
            e2.reset(seed=0, config=h.config(clean))
            outs = [e2.step({"action": "calculator", "expression": "1+1"}) for _ in range(100)]
        rec = _log(log)[-1]
        last = outs[-1]
        ok = (not any(o[2] or o[3] for o in outs[:99]) and last[3] and not last[2] and abs(last[1] - 0.8) < 1e-12
              and rec["end_reason"] == "step_limit" and rec["f1"] == 1.0)
        return ok, "100 steps incl. submit; step 100 -> truncated, empty final answer scored (clean book: 0.8)"

    def c14():
        reset(time_limit_s=0.5)
        env.step({"action": "calculator", "expression": "1"})
        time.sleep(0.8)
        o, r, term, trunc, _ = env.step({"action": "list_records", "table": "ledger"})
        return trunc and not term and o["error"]["code"] == "time_limit" and o["result"] is None, "time limit -> truncated, action not executed"

    def c15():
        log = work / "c15.jsonl"
        with AuditEnv(episodes_root=h.root, grader_log=log) as e2:
            obs, _ = e2.reset(seed=0, config=h.config(rich))
            first = e2.step({"action": "submit", "findings": oracle_findings(truth)})
            after = e2.step({"action": "calculator", "expression": "1"})
            resubmit = e2.step({"action": "submit", "findings": {"episode_id": obs["episode_id"], "findings": []}})
        recs = [x for x in _log(log) if x["episode_id"] == rich["episode_id"]]
        ok = (first[1] == 1.0 and after[0]["error"]["code"] == "episode_over" and resubmit[0]["error"]["code"] == "episode_over"
              and resubmit[1] == 0.0 and len(recs) == 1 and recs[0]["reward"] == 1.0)
        return ok, "after the end: defined error; a second submit is not scored"

    def c16():
        trace = _play(env, make_policy("auditor"), h.config(rich))
        mid = trace[1:-1]
        ok = all(s["reward"] == 0.0 and s["info"] == {} and set(s["observation"]) == {"result", "error", "remaining_steps"} for s in mid)
        return ok and len(mid) > 5, f"{len(mid)} exploration steps: reward 0, empty info, no hints"

    def c17():
        log = work / "c17.jsonl"
        with AuditEnv(episodes_root=h.root, grader_log=log) as e2:
            train = _play(e2, make_policy("oracle", truth=truth), h.config(rich, "train"))
            ev = _play(e2, make_policy("oracle", truth=truth), h.config(rich, "eval"))
        recs = _log(log)
        ok = (train[-1]["reward"] == 1.0 == recs[0]["reward"] and train[-1]["info"].get("f1") == 1.0
              and all(s["reward"] == 0.0 and s["info"] == {} for s in ev[1:]) and recs[1]["reward"] == 1.0 and recs[1]["mode"] == "eval")
        return ok, "train: terminal reward to the policy; eval: 0 and {} — result only in the grader log"

    def c18():
        log = work / "c18.jsonl"
        with AuditEnv(episodes_root=h.root, grader_log=log) as e2:
            obs, _ = e2.reset(seed=0, config=h.config(rich))
            good = oracle_findings(truth)["findings"]
            o, r, term, trunc, info = e2.step({"action": "submit", "findings": {"episode_id": obs["episode_id"],
                                                                                "findings": good + [{"type": "fraud"}]}})
        rec = _log(log)[-1]
        ok = term and r == -1.0 and info["valid"] is False and info["errors"] and rec["valid"] is False and rec["errors"]
        return ok, "wrong structure (even with correct findings inside) -> -1, error in the grader log"

    def c19():
        records = []
        for entry in h.entries[:6]:
            truth_e = json.loads(h.texts(entry)[1]["truth.json"])
            for name in ("random", "reconcile", "flag_all", "auditor"):
                records += _play(env, make_policy(name, seed=1, truth=truth_e), h.config(entry))
        leaks = scan_records(json.loads(_dump(records)), h.root)
        return not leaks, f"{len(records)} transitions of 4 agent policies on 6 closed episodes: {len(leaks)} leaks"

    def c20():
        policy = make_policy("random", seed=4)
        a1 = _dump(_play(env, policy, h.config(rich), seed=3))
        _play(env, policy, h.config(clean), seed=4)
        a2 = _dump(_play(env, policy, h.config(rich), seed=3))
        code = "import sys, env.server; print(sorted({m.split('.')[0] for m in sys.modules} & {'grader', 'generator', 'policies'}))"
        mods = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=60).stdout.strip()
        return a1 == a2 and mods == "[]", "reset isolates episodes (A->B->A identical); env server never loads grader code"

    try:
        for i, fn in enumerate((c01, c02, c03, c04, c05, c06, c07, c08, c09, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20), 1):
            checks.append(_guard(f"C{i:02d}", fn))
    finally:
        env.close()
    return checks


# ====================================================================== блок 2: данные и эталон
def evidence_problems(public: dict[str, str], truth: dict) -> list[str]:
    ep = load_public(public)
    known = set(ep.documents) | set(ep.by_id["ledger"]) | set(ep.by_id["fx"])
    out = []
    for issue in truth["issues"]:
        if not issue["evidence_sets"]:
            out.append(f"{issue['type']}: no evidence set")
        for ev in issue["evidence_sets"]:
            missing = [x for x in ev if x not in known]
            if missing:
                out.append(f"{issue['type']}: evidence refers to records that are not public ({missing[0]})")
    return out


def data_block(h: Holdout) -> list[Check]:
    checks = []
    for entry in h.entries:
        def fn(entry=entry):
            public, private = h.texts(entry)
            truth = json.loads(private["truth.json"])
            cf = h.counterfactual(entry)
            problems = [f"before injection: {p}" for p in accounting_problems(cf)]
            if derive_from_public_texts(cf):
                problems.append("before injection: independent audit already finds discrepancies")
            problems += [f"after injection: {p}" for p in accounting_problems(public)]
            derived = derive_from_public_texts(public)
            r = score_submission(truth, {"episode_id": truth["episode_id"], "findings": derived}, 0)
            if r["fp"] or r["fn"]:
                problems.append(f"declared discrepancies != independent derivation (fp {r['fp']}, fn {r['fn']})")
            problems += evidence_problems(public, truth)
            m = json.loads(public["manifest.json"])
            if not (200 <= m["counts"]["ledger_rows"] <= 2000 and 20 <= m["counts"]["documents"] <= 200 and len(truth["issues"]) <= 12):
                problems.append("episode size outside 200-2000 rows / 20-200 documents / 0-12 issues")
            if entry["kind"] == "clean" and truth["issues"]:
                problems.append("clean company has issues")
            if entry["kind"] == "interacting" and not any(i["scenario"] for i in truth["issues"]):
                problems.append("no interacting scenario")
            note = "; ".join(problems[:3]) or (f"{len(truth['issues'])} issues re-derived with evidence; books balance before "
                                               f"and after; {m['counts']['ledger_rows']} rows")
            return not problems, note
        checks.append(_guard(f"{entry['episode_id']} ({entry['kind']})", fn))
    return checks


# ====================================================================== блок 3: надёжность награды
def _score(truth: dict, findings: list[dict], steps: int = 0, **sub) -> dict:
    return score_submission(truth, {"episode_id": truth["episode_id"], "findings": findings, **sub}, steps)


def _plain_finding(doc: dict, kind: str) -> dict:
    is_inv = "invoice_id" in doc
    return {"type": kind, "affected_ids": [doc["invoice_id"] if is_inv else doc["payment_id"]],
            "claimed_amount_minor": doc["total_minor"] if is_inv else doc["amount_minor"], "currency": doc["currency"],
            "evidence_ids": [doc["doc_id"]], "explanation": "flagged"}


def reward_block(h: Holdout) -> list[Check]:
    eps = []
    for entry in h.entries:
        public, private = h.texts(entry)
        truth = json.loads(private["truth.json"])
        eps.append((entry, public, truth, oracle_findings(truth)["findings"], load_public(public)))
    rng = random.Random("jury-reward")

    def over(pred: Callable[[Any], tuple[bool, bool]], what: str) -> Callable[[], tuple[bool, str]]:
        """pred(ep) -> (применимо, выполнено); проверка пройдена, если выполнена на всех применимых эпизодах."""
        def fn():
            applicable = failed = 0
            for ep in eps:
                app, ok = pred(ep)
                applicable += app
                failed += app and not ok
            return applicable > 0 and failed == 0, f"{what}: {applicable - failed}/{applicable} episodes"
        return fn

    def r01(ep):
        r = _score(ep[2], ep[3])
        return True, r["reward"] == 1.0 and r["f1"] == 1.0

    def r02(ep):
        if len(ep[3]) < 2:
            return False, True
        shuffled = rng.sample(ep[3], len(ep[3]))
        return True, _score(ep[2], shuffled)["reward"] == 1.0 == _score(ep[2], ep[3][::-1])["reward"]

    def r03(ep):
        if not ep[3]:
            return False, True
        norm = [{**f, "affected_ids": f["affected_ids"][::-1] + f["affected_ids"][:1],
                 "evidence_ids": f["evidence_ids"][::-1] + f["evidence_ids"][:1]} for f in ep[3]]
        return True, _score(ep[2], norm)["reward"] == 1.0

    def r04(ep):
        if not ep[3]:
            return False, True
        n = len(ep[3])
        r = _score(ep[2], ep[3] + copy.deepcopy(ep[3]))
        r1 = _score(ep[2], ep[3] + copy.deepcopy(ep[3][:1]))
        return True, (r["tp"], r["fp"]) == (n, n) and r["reward"] < 1.0 and r1["fp"] == 1 and r1["reward"] < 1.0

    def r05(ep):
        issues = ep[2]["issues"]
        pair = next(((a, b) for a in range(len(issues)) for b in range(a + 1, len(issues))
                     if set(issues[a]["affected_ids"]) != set(issues[b]["affected_ids"])
                     and not set(issues[b]["affected_ids"]) <= set(issues[a]["affected_ids"])), None)
        if pair is None:
            return False, True
        a, b = pair
        fa, fb = ep[3][a], ep[3][b]
        merged = {**fa, "affected_ids": sorted(set(fa["affected_ids"]) | set(fb["affected_ids"])),
                  "evidence_ids": sorted(set(fa["evidence_ids"]) | set(fb["evidence_ids"]))[:10]}
        rest = [f for k, f in enumerate(ep[3]) if k not in pair]
        r = _score(ep[2], [merged] + rest)
        return True, (r["tp"], r["fp"], r["fn"]) == (len(issues) - 2, 1, 2)

    def r06():
        a = {"issue_id": "ISS-SYNTH001", "type": "fx_rate_error", "affected_ids": ["PAY-SYNTH01"], "expected_amount_minor": 100,
             "currency": "KZT", "amount_tolerance_minor": 1, "evidence_sets": [["D-SYNTH001"]], "scenario": None, "scenario_group": None}
        b = {**a, "issue_id": "ISS-SYNTH002", "expected_amount_minor": 101}
        x = {"type": "fx_rate_error", "affected_ids": ["PAY-SYNTH01"], "claimed_amount_minor": 100, "currency": "KZT",
             "evidence_ids": ["D-SYNTH001"], "explanation": ""}
        y = {**x, "claimed_amount_minor": 99}
        tps = {score_submission({"episode_id": "ep-SYNTH00001", "issues": issues},
                                {"episode_id": "ep-SYNTH00001", "findings": fs}, 0)["tp"]
               for fs in ([x, y], [y, x]) for issues in ([a, b], [b, a])}
        return tps == {2}, "maximum one-to-one matching (greedy would give 1 of 2), independent of order"

    def r07(ep):
        if ep[2]["issues"]:
            return False, True
        return True, _score(ep[2], [])["reward"] == 1.0

    def r08(ep):
        if ep[2]["issues"]:
            return False, True
        inv = ep[4].tables["invoices"][0]
        one = _score(ep[2], [_plain_finding(inv, "cutoff_error")])
        many = _score(ep[2], [_plain_finding(d, t) for d, t in zip(ep[4].tables["invoices"][:5], ISSUE_TYPES * 2)])
        return True, one["f1"] == 0.0 and one["reward"] == -1.0 and many["reward"] == -1.0

    def r09(ep):
        if not ep[2]["issues"]:
            return False, True
        return True, _score(ep[2], ep[3], 40)["reward"] > _score(ep[2], [], 0)["reward"]

    def r10(ep):
        docs = ep[4].tables["invoices"] + ep[4].tables["payments"]
        flags = [_plain_finding(d, ISSUE_TYPES[k % 4]) for k, d in enumerate(docs[:30])]
        return True, _score(ep[2], flags)["reward"] <= -0.9

    def r11(ep):
        if not ep[3]:
            return False, True
        guessed = [{**f, "affected_ids": [x.split("-")[0] + "-ZZZZZZZ" for x in f["affected_ids"]]} for f in ep[3]]
        return True, _score(ep[2], guessed)["tp"] == 0

    def r12(ep):
        if not ep[3]:
            return False, True
        in_sets = {x for i in ep[2]["issues"] for s in i["evidence_sets"] for x in s}
        unrelated = [r["row_id"] for r in ep[4].tables["ledger"] if r["row_id"] not in in_sets][:3]
        fake = [{**f, "evidence_ids": ["L-ZZZZZZZZ", "D-ZZZZZZZZ"]} for f in ep[3]]
        real_but_wrong = [{**f, "evidence_ids": unrelated} for f in ep[3]]
        return True, _score(ep[2], fake)["tp"] == 0 and _score(ep[2], real_but_wrong)["tp"] == 0

    def r13(ep):
        expected_tp, findings = 0, []
        for issue, f in zip(ep[2]["issues"], ep[3]):
            ev = list(f["evidence_ids"])
            if len(ev) >= 2:
                rest = set(ev[1:])
                still = any(set(s) <= rest for s in issue["evidence_sets"])
                findings.append({**f, "evidence_ids": ev[1:]})
                expected_tp += still
            else:
                findings.append(f)
                expected_tp += 1
        if len(findings) == expected_tp:
            return False, True
        return True, _score(ep[2], findings)["tp"] == expected_tp

    def r14(ep):
        if not ep[3]:
            return False, True
        swap = [{**f, "currency": "XTS" if f["currency"] != "XTS" else "XXX"} for f in ep[3]]
        return True, _score(ep[2], swap)["tp"] == 0

    def r15(ep):
        if not ep[3]:
            return False, True
        tol = {i["issue_id"]: i["amount_tolerance_minor"] for i in ep[2]["issues"]}
        tols = [tol[i["issue_id"]] for i in ep[2]["issues"]]
        inside = [{**f, "claimed_amount_minor": f["claimed_amount_minor"] + t} for f, t in zip(ep[3], tols)]
        outside = [{**f, "claimed_amount_minor": f["claimed_amount_minor"] + t + 1} for f, t in zip(ep[3], tols)]
        return True, _score(ep[2], inside)["tp"] == len(ep[3]) and _score(ep[2], outside)["tp"] == 0

    def r16(ep):
        if not ep[3]:
            return False, True
        rot = {t: ISSUE_TYPES[(k + 1) % 4] for k, t in enumerate(ISSUE_TYPES)}
        moved = [{**f, "type": rot[f["type"]]} for f in ep[3]]
        coincide = sum(any(i["type"] == g["type"] and set(i["affected_ids"]) == set(g["affected_ids"])
                           and abs(i["expected_amount_minor"] - g["claimed_amount_minor"]) <= i["amount_tolerance_minor"]
                           and i["currency"] == g["currency"] for i in ep[2]["issues"]) for g in moved)
        return True, _score(ep[2], moved)["tp"] <= coincide

    def r17(ep):
        rewards = [_score(ep[2], ep[3], k)["reward"] for k in range(100)]
        exact = all(r == float(max(Fraction(-1), min(Fraction(1), 1 - Fraction(2, 1000) * k))) for k, r in enumerate(rewards))
        monotone = all(a > b for a, b in zip(rewards, rewards[1:]))
        return True, exact and monotone and rewards[0] - rewards[99] < 0.2

    def r18(ep):
        good = ep[3]
        base = good[0] if good else _plain_finding(ep[4].tables["invoices"][0], "cutoff_error")
        # неверная структура (тесты 18-20 и M05/M09 жюри): -1 и null-метрики, даже с верными выводами внутри
        structural = [(good * 31)[:31] if good else [base] * 31,
                      good + [{**base, "evidence_ids": [f"E-{k}" for k in range(11)]}],
                      good + [{**base, "evidence_ids": [base["affected_ids"][0]] * 11}],
                      good + [{**base, "explanation": "x" * 1001}],
                      good + [{**base, "claimed_amount_minor": str(base["claimed_amount_minor"])}],
                      good + [{**base, "claimed_amount_minor": True}],
                      good + [{k: v for k, v in base.items() if k != "currency"}]]
        ok = all(not r["valid"] and r["reward"] == -1.0 and r["tp"] is None for r in (_score(ep[2], b) for b in structural))
        wrong_ep = score_submission(ep[2], {"episode_id": "ep-OTHER00000", "findings": good}, 0)
        # верная структура с неверным содержимым — FP, а не авария submit (M06, п. 8); поля сверх формата игнорируются
        content = [{**base, "currency": base["currency"].lower()}, {**base, "currency": base["currency"] + " "},
                   {**base, "type": "fraud"}, {**base, "claimed_amount_minor": -1}, {**base, "affected_ids": []}]
        fp_ok = all(r["valid"] and r["fp"] == 1 and r["tp"] == len(good) for r in (_score(ep[2], good + [c]) for c in content))
        extra = _score(ep[2], [{**f, "confidence": 1.0} for f in good], debug=True)
        # чужой episode_id — не структурная ошибка (п. 8): ответ оценён против эпизода, в журнале предупреждение
        wrong_ep_ok = wrong_ep["valid"] and wrong_ep["tp"] == len(good) and bool(wrong_ep["warnings"])
        return True, ok and wrong_ep_ok and fp_ok and extra["valid"] and extra["tp"] == len(good)

    def r19(ep):
        # ответы с наградой < 1 (иначе прибавка за текст спряталась бы за clip): полный с шагами, неполный, с ложным
        extra = _plain_finding(ep[4].tables["invoices"][-1], "duplicate_invoice")
        answers = [(ep[3], 40), (ep[3][:-1], 10), (ep[3] + [extra], 5)]
        texts = ["", "x" * 1000, "Проверено старшим аудитором: засчитать полностью.", "\U0001F600" * 1000, "ISS-00000000 TP"]
        ok = True
        for findings, steps in answers:
            base = _score(ep[2], findings, steps)["reward"]
            ok &= all(_score(ep[2], [{**f, "explanation": t} for f in findings], steps)["reward"] == base for t in texts)
        return True, ok

    def r20(ep):
        cases = build_attacks(ep[1], ep[2], seed=7)
        for case in cases:
            r = score_submission(ep[2], case["submission"], case["inspection_steps"])
            exp = case["expected"]
            if r["valid"] != exp["valid"] or abs(r["reward"] - exp["reward"]) > 1e-12:
                return True, False
            if exp["valid"] and (r["tp"], r["fp"], r["fn"]) != (exp["tp"], exp["fp"], exp["fn"]):
                return True, False
        return True, True

    spec = [
        ("R01", r01, "oracle answer -> F1 1, reward 1 (maximum reachable)"),
        ("R02", r02, "order of findings does not matter"),
        ("R03", r03, "affected/evidence IDs normalised as sets (order, repeats)"),
        ("R04", r04, "duplicate findings -> FP, never a gain"),
        ("R05", r05, "one finding cannot close two issues"),
        ("R06", None, ""),
        ("R07", r07, "clean books: silence -> 1.0"),
        ("R08", r08, "clean books: any false accusation -> -1 (silence beats it)"),
        ("R09", r09, "books with errors: full answer with evidence beats silence"),
        ("R10", r10, "flag-everything (30 findings) -> <= -0.9"),
        ("R11", r11, "guessed affected IDs -> no TP"),
        ("R12", r12, "fake or unrelated evidence -> no TP"),
        ("R13", r13, "incomplete evidence set -> no TP"),
        ("R14", r14, "right amount in another currency -> no TP"),
        ("R15", r15, "amount tolerance is exact: +tol TP, +tol+1 no TP"),
        ("R16", r16, "wrong type -> no TP"),
        ("R17", r17, "step cost exact (0.002 per inspection step, 0..99), strictly decreasing, at most 0.198"),
        ("R18", r18, "wrong structure (31 findings, 11 evidence, 1001 chars, amount not int) -> -1; wrong content -> FP"),
        ("R19", r19, "explanation text never changes the reward (checked on answers scoring < 1)"),
        ("R20", r20, "author's attack suite (36 attacks/episode) matches the spec"),
    ]
    checks = []
    for code, pred, what in spec:
        if code == "R06":
            checks.append(_guard(code, r06))
        else:
            checks.append(_guard(code, over(pred, what)))
    return checks


# ====================================================================== блок 4: воспроизводимость и пары
def pairs_block(h: Holdout, rewards: dict | None = None) -> list[Check]:
    """rewards: {(policy, episode_id): награда} из этапа 6 (eval через изолированную среду); иначе считаем сами."""
    if rewards is None:
        rewards = {}
        with AuditEnv(episodes_root=h.root) as env:
            for entry in h.entries:
                truth = json.loads(h.texts(entry)[1]["truth.json"])
                truth2 = json.loads(h.texts(entry, "pairs")[1]["truth.json"])
                for split, t in (("episodes", truth), ("pairs", truth2)):
                    ep_id = entry["episode_id"] if split == "episodes" else entry["pair_episode_id"]
                    for name in ("oracle", "auditor", "reconcile", "flag_all"):
                        trace = _play(env, make_policy(name, truth=t), h.config(entry, "train", split=split))
                        rewards[(name, ep_id)] = trace[-1]["reward"]
    env1, env2 = AuditEnv(episodes_root=h.root), AuditEnv(episodes_root=h.root)  # два разных процесса
    checks = []
    try:
        for entry in h.entries:
            def fn(entry=entry):
                i = h.position[entry["episode_id"]]
                problems = []
                public, private = h.texts(entry)
                truth = json.loads(private["truth.json"])
                regen = generate_episode(entry["seed"], entry["config"], salt=h.salt)
                if regen.public != public or {k: v for k, v in regen.private.items() if "/" not in k} != private:
                    problems.append("same seed+config+version+salt does not reproduce the files byte for byte")
                pair_pub, pair_priv = h.texts(entry, "pairs")
                core_private = {k: v for k, v in private.items() if "/" not in k}
                pub2, priv2, mapping = transform_texts(public, core_private, f"{h.salt}|pair|{i}")
                if pub2 != pair_pub or priv2.get("truth.json") != pair_priv["truth.json"]:
                    problems.append("pair on disk != transform(episode)")
                truth2 = json.loads(pair_priv["truth.json"])
                if _truth_key(truth, lambda x: mapping.get(x, x)) != _truth_key(truth2):
                    problems.append("pair truth != renamed truth (issues, amounts, evidence sets)")
                remapped = remap_findings(oracle_findings(truth), mapping)
                if _canonical(remapped["findings"]) != _canonical(oracle_findings(truth2)["findings"]):
                    problems.append("oracle answer (type, IDs, amount, currency) not preserved under permutation + renaming")
                if score_submission(truth2, remapped, 0)["reward"] != 1.0:
                    problems.append("remapped oracle answer does not score 1.0 on the pair")
                cfg = h.config(entry, "train")
                t1 = _play(env1, make_policy("random", seed=i), cfg, seed=i)
                t2 = _play(env1, make_policy("random", seed=i), cfg, seed=i)
                if _dump(t1) != _dump(t2):
                    problems.append("same seed + same policy seed: different trajectory")
                obs, _ = env2.reset(seed=i, config=cfg)  # повтор записанных действий в другом процессе
                replay = [obs] + [env2.step(s["action"])[0] for s in t1[1:]]
                if _dump(replay) != _dump([s["observation"] for s in t1]):
                    problems.append("replaying the recorded actions in another process gives different observations")
                for name in ("oracle", "auditor", "reconcile", "flag_all"):
                    a, b = rewards.get((name, entry["episode_id"])), rewards.get((name, entry["pair_episode_id"]))
                    if a is None or a != b:
                        problems.append(f"{name}: reward {a} on the episode vs {b} on its pair")
                if rewards.get(("oracle", entry["pair_episode_id"])) != 1.0:
                    problems.append("oracle is not 1.0 on the pair")
                note = "; ".join(problems[:3]) or (f"regen byte-identical; replay identical ({len(t1) - 1} steps, 2 processes); "
                                                   f"pair truth = renamed truth ({len(truth['issues'])} issues); "
                                                   "renamed oracle = 1.0; 4 policies' rewards equal")
                return not problems, note
            checks.append(_guard(f"pair {h.position[entry['episode_id']] + 1:02d}: {entry['episode_id']} -> {entry['pair_episode_id']}", fn))
    finally:
        env1.close()
        env2.close()
    return checks


# ====================================================================== инженерия (самопроверка наличия)
def engineering_block() -> list[tuple[str, int, str]]:
    out = []
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8") if (ROOT / "Dockerfile").is_file() else ""
    make = (ROOT / "Makefile").read_text(encoding="utf-8") if (ROOT / "Makefile").is_file() else ""
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").is_file() else ""
    single = (ROOT / "check.py").is_file()
    container = "check.py" in docker and "--network none" in readme and "check:" in make
    out.append(("one command", (2 if container else 1) if single else 0, "python check.py = make check = docker run --network none"))
    manifests = list((ROOT / "episodes").glob("*/*/public/manifest.json"))
    mv = all({"schema_version", "generator_version"} <= set(json.loads(p.read_text(encoding="utf-8"))) for p in manifests)
    idx = all(all({"config", "config_digest", "seed"} <= set(e) for e in json.loads(p.read_text(encoding="utf-8"))["episodes"])
              for p in (ROOT / "episodes").glob("*/index.json"))
    cv = all("config_version" in json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "generator" / "configs").glob("*.json")
             if p.name != "splits.json")
    out.append(("config versions", mv + (idx and cv), f"{len(manifests)} manifests with schema/generator version; split indexes with seed+config digest"))
    traj = ROOT / "trajectories" / "trajectories.jsonl"
    fields = {"episode_id", "step", "action", "observation", "reward", "terminated", "truncated"}
    tj = traj.is_file() and all(fields <= set(json.loads(x)) for x in traj.read_text(encoding="utf-8").splitlines()[:50])
    out.append(("trajectory logs", 2 if tj else 0, "trajectories/trajectories.jsonl with the required fields (leak scan: stage 5)"))
    base = (ROOT / "reports" / "baselines.md").read_text(encoding="utf-8") if (ROOT / "reports" / "baselines.md").is_file() else ""
    out.append(("measured failure analysis", ("Failure analysis" in base) + (ROOT / "reports" / "holdout.md").is_file(),
                "reports/baselines.md (P/R/F1/return/steps/time + FP reasons by count), reports/holdout.md"))
    design = (ROOT / "docs" / "design.md").read_text(encoding="utf-8") if (ROOT / "docs" / "design.md").is_file() else ""
    lim = design.split("Честные ограничения", 1)[1] if "Честные ограничения" in design else ""
    bullets = sum(1 for ln in lim.splitlines() if ln.startswith("* "))
    out.append(("honest limitations", 2 if bullets >= 5 else (1 if bullets else 0), f"docs/design.md: {bullets} stated limitations"))
    return out


# ====================================================================== сборка
def run_scorecard(root: Path, salt: str, rewards: dict | None = None) -> dict:
    h = Holdout(root, salt)
    with tempfile.TemporaryDirectory(prefix="jury-") as tmp:
        t0 = time.perf_counter()
        blocks = {"contract": contract_block(h, Path(tmp))}
        blocks["data"] = data_block(h)
        blocks["reward"] = reward_block(h)
        blocks["pairs"] = pairs_block(h, rewards)
        seconds = time.perf_counter() - t0
    eng = engineering_block()
    total = 0.0
    summary = {}
    for key, title, weight in BLOCKS:
        passed = sum(c.ok for c in blocks[key])
        n = len(blocks[key])
        points = weight * passed / 20
        total += points
        summary[key] = {"title": title, "passed": passed, "of": n, "points": points, "weight": weight}
    eng_points = sum(p for _, p, _ in eng)
    total += eng_points
    return {"blocks": blocks, "summary": summary, "engineering": eng, "engineering_points": eng_points,
            "total": total, "seconds": round(seconds, 1), "episodes": len(h.entries)}


def markdown(card: dict) -> str:
    lines = ["# Jury scorecard (emulation)", "",
             "Case.md section 10 translated into code one to one (`tests/jury/scorecard.py`, stage 7 of `python check.py`).",
             f"Closed set emulated by `generator.holdout`: {card['episodes']} salted episodes with new parameter combinations "
             "(5 clean, 5 with interacting errors) and their permuted + renamed pairs. The jury's real seeds are unknown, so "
             "20/20 here is necessary, not sufficient.", "",
             "| block | passed | points |", "|---|---|---|"]
    for key, _, weight in BLOCKS:
        s = card["summary"][key]
        lines.append(f"| {s['title']} | {s['passed']}/{s['of']} | {s['points']:.2f} / {weight} |")
    lines.append(f"| Инженерия и объяснение (самопроверка наличия) | {card['engineering_points']}/10 | {card['engineering_points']:.2f} / 10 |")
    lines.append(f"| **Итого (оценка снизу вверх)** | | **{card['total']:.2f} / 100** |")
    for key, title, _ in BLOCKS:
        lines += ["", f"## {title}", "", "| check | result | what is verified |", "|---|---|---|"]
        for c in card["blocks"][key]:
            lines.append(f"| {c.name} | {'PASS' if c.ok else '**FAIL**'} | {c.note.replace('|', '/')} |")
    lines += ["", ENGINEERING_HEADING, "", "| criterion | score | evidence |", "|---|---|---|"]
    for name, pts, note in card["engineering"]:
        lines.append(f"| {name} | {pts} | {note} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    from generator.holdout import build_holdout

    parser = argparse.ArgumentParser(prog="python tests/jury/scorecard.py")
    parser.add_argument("--salt", default="check-emulation-salt")
    parser.add_argument("--out", default=None, help="write the markdown scorecard here")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="holdout-") as tmp:
        build_holdout(args.salt, 20, Path(tmp))
        card = run_scorecard(Path(tmp), args.salt)
    md = markdown(card)
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8", newline="")
    full = all(s["passed"] == 20 for s in card["summary"].values())
    return 0 if full else 1


if __name__ == "__main__":
    raise SystemExit(main())
