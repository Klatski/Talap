from __future__ import annotations

from collections import defaultdict
import random
from typing import Any

from .base import BasePolicy


def _extract_id(row: dict[str, Any], fallback: str) -> str:
    for key in ("row_id", "record_id", "invoice_id", "payment_id", "_id", "id"):
        val = row.get(key)
        if val is not None and str(val).strip() != "":
            return str(val)
    return fallback


def _extract_amount_minor(row: dict[str, Any]) -> int:
    for key in ("amount_minor", "total_minor", "debit_minor", "credit_minor", "amount"):
        val = row.get(key)
        if val is not None:
            try:
                return abs(int(str(val).strip()))
            except ValueError:
                try:
                    return abs(int(round(float(str(val)) * 100)))
                except ValueError:
                    pass
    return 0


class RandomBaselinePolicy(BasePolicy):
    name = "random"

    def __init__(self, seed: int = 0, max_actions_before_submit: int = 8) -> None:
        super().__init__(seed=seed)
        self._rng = random.Random(seed)
        self._max_actions_before_submit = max(1, max_actions_before_submit)

    def reset(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> None:
        ep_id = ""
        if isinstance(obs, dict) and isinstance(obs.get("manifest"), dict):
            ep_id = str(obs["manifest"].get("episode_id") or "")
        self._rng = random.Random(self.seed + (sum(ord(ch) for ch in ep_id) % 10000))
        super().reset(obs, info)

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> dict[str, Any]:
        self.ingest_obs(obs)
        self.step_count += 1
        steps_left = int(obs.get("steps_remaining", 100 - self.step_count))

        if steps_left <= 1 or self.step_count >= self._max_actions_before_submit:
            findings: list[dict[str, Any]] = []
            if self.seen_records["ledger"] and self._rng.random() < 0.3:
                row = self._rng.choice(self.seen_records["ledger"])
                rid = _extract_id(row, "row_0")
                findings.append(
                    {
                        "type": "anomalous_posting",
                        "affected_ids": [rid],
                        "claimed_amount_minor": _extract_amount_minor(row),
                        "currency": str(row.get("currency") or "USD"),
                        "evidence_ids": [rid],
                        "explanation": "Random baseline candidate",
                    }
                )
            return self.make_submit(findings)

        choice = self._rng.choice(["list_records", "calculator"])
        if choice == "calculator":
            a = self._rng.randint(10, 500)
            b = self._rng.randint(1, 20)
            return {
                "name": "calculator",
                "args": {"expression": f"{a} * {b}"},
            }

        table = self._rng.choice(["ledger", "invoices", "payments", "fx"])
        cursor = self._rng.choice([0, 0, 50])
        return {
            "name": "list_records",
            "args": {"table": table, "cursor": cursor, "limit": 50},
        }


class ReconcileBaselinePolicy(BasePolicy):
    name = "reconcile"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._tables_to_scan = ["ledger", "invoices", "payments", "fx"]
        self._scan_idx = 0
        self._cursor = 0
        self._calc_done = False
        self._awaiting_table_page = False

    def reset(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> None:
        self._scan_idx = 0
        self._cursor = 0
        self._calc_done = False
        self._awaiting_table_page = False
        super().reset(obs, info)

    def _advance_cursor_from_obs(self, obs: dict[str, Any]) -> None:
        if not self._awaiting_table_page or not isinstance(obs, dict):
            return
        self._awaiting_table_page = False

        next_cur = obs.get("next_cursor")
        returned = obs.get("returned")
        has_more = obs.get("has_more")

        if isinstance(returned, int) and returned <= 0:
            self._scan_idx += 1
            self._cursor = 0
            return

        if has_more is False:
            self._scan_idx += 1
            self._cursor = 0
            return

        if isinstance(next_cur, int) and next_cur > self._cursor:
            if isinstance(returned, int) and returned < 50:
                self._scan_idx += 1
                self._cursor = 0
            else:
                self._cursor = next_cur
        else:
            self._scan_idx += 1
            self._cursor = 0

    def _build_reconciled_findings(self, limit: int | None = None) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        seen_signatures: set[tuple[Any, ...]] = set()

        ledger = self.seen_records["ledger"]
        invoices = self.seen_records["invoices"]
        payments = self.seen_records["payments"]

        by_invoice_id: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for idx, row in enumerate(ledger):
            rid = _extract_id(row, f"ledger_{idx}")
            inv_id = row.get("invoice_id")
            if inv_id:
                by_invoice_id[str(inv_id)].append((rid, row))

        for inv_id, entries in by_invoice_id.items():
            if len(entries) > 1:
                ids = [rid for rid, _ in entries]
                first_row = entries[0][1]
                amt = _extract_amount_minor(first_row)
                curr = str(first_row.get("currency") or "USD")
                sig = ("duplicate_invoice_posting", tuple(sorted(ids)), amt, curr)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    findings.append(
                        {
                            "type": "duplicate_invoice_posting",
                            "affected_ids": ids,
                            "claimed_amount_minor": amt,
                            "currency": curr,
                            "evidence_ids": list(dict.fromkeys(ids + [inv_id])),
                            "explanation": f"Multiple ledger entries reference invoice {inv_id}",
                        }
                    )

        inv_map: dict[str, tuple[str, dict[str, Any]]] = {}
        for idx, inv in enumerate(invoices):
            iid = _extract_id(inv, f"inv_{idx}")
            inv_map[iid] = (iid, inv)

        for idx, pay in enumerate(payments):
            pid = _extract_id(pay, f"pay_{idx}")
            ref_inv = pay.get("invoice_id")
            if ref_inv and str(ref_inv) in inv_map:
                _, inv = inv_map[str(ref_inv)]
                inv_amt = _extract_amount_minor(inv)
                pay_amt = _extract_amount_minor(pay)
                if inv_amt != pay_amt:
                    curr = str(pay.get("currency") or inv.get("currency") or "USD")
                    aff = sorted([str(ref_inv), pid])
                    sig = ("payment_invoice_mismatch", tuple(aff), abs(inv_amt - pay_amt), curr)
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        findings.append(
                            {
                                "type": "payment_invoice_mismatch",
                                "affected_ids": aff,
                                "claimed_amount_minor": abs(inv_amt - pay_amt),
                                "currency": curr,
                                "evidence_ids": [str(ref_inv), pid],
                                "explanation": f"Payment {pid} amount differs from invoice {ref_inv}",
                            }
                        )

        if limit is not None:
            return findings[:limit]
        return findings

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> dict[str, Any]:
        self.ingest_obs(obs)
        self._advance_cursor_from_obs(obs)
        self.step_count += 1
        steps_left = int(obs.get("steps_remaining", 100 - self.step_count))

        if steps_left <= 1 or self._scan_idx >= len(self._tables_to_scan):
            if not self._calc_done and steps_left > 2:
                self._calc_done = True
                candidates = self._build_reconciled_findings(limit=5)
                if candidates:
                    amt = candidates[0]["claimed_amount_minor"]
                    return {
                        "name": "calculator",
                        "args": {"expression": f"{amt}"},
                    }
            return self.make_submit(self._build_reconciled_findings())

        table = self._tables_to_scan[self._scan_idx]
        self._awaiting_table_page = True
        return {
            "name": "list_records",
            "args": {"table": table, "cursor": self._cursor, "limit": 50},
        }


class AllFindingsBaselinePolicy(ReconcileBaselinePolicy):
    name = "all"

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> dict[str, Any]:
        self.ingest_obs(obs)
        self._advance_cursor_from_obs(obs)
        self.step_count += 1
        steps_left = int(obs.get("steps_remaining", 100 - self.step_count))

        if steps_left <= 1 or self._scan_idx >= len(self._tables_to_scan):
            findings = self._build_reconciled_findings()
            seen_ids = {
                fid
                for f in findings
                for fid in f.get("affected_ids", [])
            }
            for idx, row in enumerate(self.seen_records["ledger"][:15]):
                rid = _extract_id(row, f"row_{idx}")
                if rid in seen_ids:
                    continue
                amt = _extract_amount_minor(row)
                curr = str(row.get("currency") or "USD")
                findings.append(
                    {
                        "type": "candidate_ledger_anomaly",
                        "affected_ids": [rid],
                        "claimed_amount_minor": amt,
                        "currency": curr,
                        "evidence_ids": [rid],
                        "explanation": "All-findings baseline broad candidate",
                    }
                )
            return self.make_submit(findings)

        table = self._tables_to_scan[self._scan_idx]
        self._awaiting_table_page = True
        return {
            "name": "list_records",
            "args": {"table": table, "cursor": self._cursor, "limit": 50},
        }


class OracleBaselinePolicy(BasePolicy):
    name = "oracle"

    def __init__(self, seed: int = 0, truth_provider: Any = None) -> None:
        super().__init__(seed=seed)
        self._truth_provider = truth_provider

    def _truth_to_findings(self, truth_items: list[Any]) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for idx, item in enumerate(truth_items):
            if isinstance(item, dict):
                issue_type = str(item.get("issue_type") or item.get("type") or "audit_issue")
                affected = list(item.get("affected_ids") or [])
                amt = int(item.get("expected_amount_minor") or item.get("claimed_amount_minor") or 0)
                curr = str(item.get("currency") or "USD")
                ev_sets = item.get("valid_evidence_sets")
                evidence = list(item.get("evidence_ids") or [])
                if not evidence and ev_sets:
                    first_set = next(iter(ev_sets))
                    evidence = sorted(str(x) for x in first_set)
                if not evidence:
                    evidence = list(affected)
                findings.append(
                    {
                        "type": issue_type,
                        "affected_ids": affected,
                        "claimed_amount_minor": amt,
                        "currency": curr,
                        "evidence_ids": evidence,
                        "explanation": f"Oracle truth issue #{idx + 1}",
                    }
                )
            else:
                issue_type = str(getattr(item, "issue_type", "audit_issue"))
                affected = list(getattr(item, "affected_ids", ()))
                amt = int(getattr(item, "expected_amount_minor", 0))
                curr = str(getattr(item, "currency", "USD"))
                ev_sets = getattr(item, "valid_evidence_sets", ())
                evidence: list[str] = []
                if ev_sets:
                    first_set = next(iter(ev_sets))
                    evidence = sorted(str(x) for x in first_set)
                if not evidence:
                    evidence = list(affected)
                findings.append(
                    {
                        "type": issue_type,
                        "affected_ids": affected,
                        "claimed_amount_minor": amt,
                        "currency": curr,
                        "evidence_ids": evidence,
                        "explanation": f"Oracle truth issue #{idx + 1}",
                    }
                )
        return findings

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> dict[str, Any]:
        self.ingest_obs(obs)
        self.step_count += 1

        truth_items: list[Any] = []
        if isinstance(info, dict) and isinstance(info.get("truth"), list):
            truth_items = info["truth"]
        elif isinstance(obs, dict) and isinstance(obs.get("truth"), list):
            truth_items = obs["truth"]
        elif callable(self._truth_provider):
            loaded = self._truth_provider(self.episode_id, info or {})
            if isinstance(loaded, list):
                truth_items = loaded

        return self.make_submit(self._truth_to_findings(truth_items))


def build_policy(policy_name: str, seed: int = 0, truth_provider: Any = None) -> BasePolicy:
    key = policy_name.strip().lower()
    if key == "random":
        return RandomBaselinePolicy(seed=seed)
    if key in ("reconcile", "recon", "reconciliation"):
        return ReconcileBaselinePolicy(seed=seed)
    if key in ("all", "all_findings"):
        return AllFindingsBaselinePolicy(seed=seed)
    if key == "oracle":
        return OracleBaselinePolicy(seed=seed, truth_provider=truth_provider)
    raise ValueError(f"Unknown baseline policy: {policy_name!r}")