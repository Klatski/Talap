from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import dataclass, field
from decimal import InvalidOperation
from pathlib import Path
from typing import Any, Callable

from env.calculator import safe_calculate
from grader import (
    SubmissionSchemaError,
    TruthIssue,
    evaluate_findings,
    validate_findings_payload,
)


MAX_STEPS = 100
MAX_PAGE_LIMIT = 50
DEFAULT_TIMEOUT_SEC = 60.0

ALLOWED_TABLES = ("ledger", "invoices", "payments", "fx")
ALLOWED_TOOLS = (
    "list_records(table, cursor, limit)",
    "get_record(table, id)",
    "get_document(doc_id)",
    "calculator(expression)",
    "submit(findings)",
)


@dataclass
class EpisodeBundle:
    episode_id: str
    manifest: dict[str, Any]
    policy: dict[str, Any]
    ledger: list[dict[str, Any]]
    invoices: list[dict[str, Any]]
    payments: list[dict[str, Any]]
    fx: list[dict[str, Any]]
    truth_data: dict[str, Any] | None = field(default=None, repr=False)
    truth_issues: list[TruthIssue] | None = field(default=None, repr=False)


def load_episode_from_dir(episode_dir: str | Path, include_truth: bool = False) -> EpisodeBundle:
    root = Path(episode_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    policy = json.loads((root / "policy.json").read_text(encoding="utf-8"))

    with (root / "ledger.csv").open(encoding="utf-8", newline="") as fh:
        ledger = list(csv.DictReader(fh))

    invoices = [
        json.loads(line)
        for line in (root / "invoices.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payments = [
        json.loads(line)
        for line in (root / "payments.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    with (root / "fx.csv").open(encoding="utf-8", newline="") as fh:
        fx = list(csv.DictReader(fh))

    truth_data: dict[str, Any] | None = None
    truth_path = root / "truth.json"
    if include_truth and truth_path.exists():
        truth_data = json.loads(truth_path.read_text(encoding="utf-8"))

    return EpisodeBundle(
        episode_id=str(manifest["episode_id"]),
        manifest=manifest,
        policy=policy,
        ledger=ledger,
        invoices=invoices,
        payments=payments,
        fx=fx,
        truth_data=truth_data,
    )


class AuditEnv:
    """
    Контракт среды финансового аудита:
      reset(seed, config) -> (observation, info)
      step(action) -> (observation, reward, terminated, truncated, info)
    """

    def __init__(
        self,
        episode_loader: Callable[..., EpisodeBundle],
        submit_evaluator: Callable[..., dict[str, Any]] | None = None,
        max_steps: int = MAX_STEPS,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        if max_steps < 1 or max_steps > MAX_STEPS:
            raise ValueError(f"max_steps must be between 1 and {MAX_STEPS}")
        self._episode_loader = episode_loader
        self._submit_evaluator = submit_evaluator
        self.max_steps = max_steps
        self.timeout_sec = timeout_sec

        self._mode: str = "eval"
        self._seed: int | None = None
        self._episode_id: str = ""
        self._manifest: dict[str, Any] = {}
        self._policy_summary: dict[str, Any] = {}
        self._tables: dict[str, list[dict[str, Any]]] = {}
        self._record_index: dict[tuple[str, str], dict[str, Any]] = {}
        self._doc_index: dict[str, dict[str, Any]] = {}
        self._truth_issues: list[TruthIssue] | None = None

        self._step_count: int = 0
        self._start_monotonic: float = 0.0
        self._terminated: bool = False
        self._truncated: bool = False
        self._initialized: bool = False
        self._last_grader_result: dict[str, Any] | None = None

    def reset(
        self,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cfg = copy.deepcopy(config) if isinstance(config, dict) else {}
        mode = str(cfg.get("mode", "eval"))
        if mode not in {"train", "eval"}:
            raise ValueError("config['mode'] must be 'train' or 'eval'")

        # Закрытый эталон не загружается в рабочую память среды агента во время eval
        bundle = self._episode_loader(seed=seed, config=cfg, include_truth=(mode == "train"))

        self._mode = mode
        self._seed = seed
        self._episode_id = str(bundle.episode_id)
        self._manifest = copy.deepcopy(bundle.manifest)
        self._truth_issues = copy.deepcopy(bundle.truth_issues)
        self._policy_summary = {
            "fx_date_rule": bundle.policy.get("fx_date_rule"),
            "period_boundaries": bundle.policy.get("period_boundaries"),
            "rounding": bundle.policy.get("rounding"),
        }

        self._tables = {
            "ledger": copy.deepcopy(bundle.ledger),
            "invoices": copy.deepcopy(bundle.invoices),
            "payments": copy.deepcopy(bundle.payments),
            "fx": copy.deepcopy(bundle.fx),
        }

        self._record_index = {}
        for row in self._tables["ledger"]:
            rid = row.get("row_id")
            if rid is not None:
                self._record_index[("ledger", str(rid))] = row
        for doc in self._tables["invoices"]:
            iid = doc.get("invoice_id")
            if iid is not None:
                self._record_index[("invoices", str(iid))] = doc
        for doc in self._tables["payments"]:
            pid = doc.get("payment_id")
            if pid is not None:
                self._record_index[("payments", str(pid))] = doc

        self._doc_index = {}
        for doc in self._tables["invoices"]:
            did = doc.get("doc_id")
            if did is not None:
                self._doc_index[str(did)] = doc
        for doc in self._tables["payments"]:
            did = doc.get("doc_id")
            if did is not None:
                self._doc_index[str(did)] = doc

        self._step_count = 0
        self._start_monotonic = time.monotonic()
        self._terminated = False
        self._truncated = False
        self._initialized = True
        self._last_grader_result = None

        observation = {
            "episode_id": self._episode_id,
            "task": (
                "Investigate company records and documents for the reporting period "
                "and submit verified financial audit findings with evidence IDs."
            ),
            "manifest": self._manifest,
            "policy_summary": self._policy_summary,
            "step": 0,
            "steps_remaining": self.max_steps,
            "tools": list(ALLOWED_TOOLS),
        }
        info = {
            "episode_id": self._episode_id,
            "mode": self._mode,
            "seed": self._seed,
        }
        return observation, info

    def _obs(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "episode_id": self._episode_id,
            "step": self._step_count,
            "steps_remaining": max(0, self.max_steps - self._step_count),
            **body,
        }

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError("Environment must be reset() before calling step()")

        if self._terminated or self._truncated:
            return (
                self._obs({"status": "finished", "error": "Episode has already ended"}),
                0.0,
                self._terminated,
                self._truncated,
                {"episode_id": self._episode_id, "mode": self._mode},
            )

        self._step_count += 1
        elapsed = time.monotonic() - self._start_monotonic

        if self._step_count > self.max_steps or elapsed > self.timeout_sec:
            self._truncated = True
            self._terminated = False
            grader_out = (
                self._submit_evaluator(
                    episode_id=self._episode_id,
                    findings_payload={"episode_id": self._episode_id, "findings": []},
                    inspection_steps=self._step_count,
                    mode=self._mode,
                    truncated=True,
                )
                if self._submit_evaluator is not None
                else None
            )
            self._last_grader_result = grader_out
            reward = float(grader_out["reward"]) if (self._mode == "train" and grader_out) else 0.0
            info: dict[str, Any] = {
                "episode_id": self._episode_id,
                "mode": self._mode,
                "truncated_reason": "timeout" if elapsed > self.timeout_sec else "max_steps_exceeded",
            }
            if self._mode == "train" and grader_out is not None:
                info["evaluation"] = grader_out
            return (
                self._obs({"status": "truncated", "findings": []}),
                reward,
                False,
                True,
                info,
            )

        if not isinstance(action, dict) or not isinstance(action.get("name"), str):
            return (
                self._obs({"error": "Invalid action: expected dict with string 'name'"}),
                0.0,
                False,
                False,
                {"episode_id": self._episode_id, "error": "invalid_action"},
            )

        name = action["name"]
        args = action.get("args", {})
        if not isinstance(args, dict):
            return (
                self._obs({"error": "Invalid action: 'args' must be an object"}),
                0.0,
                False,
                False,
                {"episode_id": self._episode_id, "error": "invalid_args"},
            )

        if name == "list_records":
            return self._act_list_records(args)
        if name == "get_record":
            return self._act_get_record(args)
        if name == "get_document":
            return self._act_get_document(args)
        if name == "calculator":
            return self._act_calculator(args)
        if name == "submit":
            return self._act_submit(args)

        return (
            self._obs({"error": f"Unknown tool '{name}'"}),
            0.0,
            False,
            False,
            {"episode_id": self._episode_id, "error": "unknown_tool"},
        )

    def _act_list_records(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        table = args.get("table")
        if not isinstance(table, str) or table not in self._tables:
            return (
                self._obs({"error": f"Invalid table '{table}'"}),
                0.0,
                False,
                False,
                {"episode_id": self._episode_id, "error": "invalid_table"},
            )

        cursor = args.get("cursor", 0)
        limit = args.get("limit", MAX_PAGE_LIMIT)

        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            return (
                self._obs({"error": "cursor must be a non-negative integer"}),
                0.0,
                False,
                False,
                {"episode_id": self._episode_id, "error": "invalid_cursor"},
            )
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return (
                self._obs({"error": "limit must be a positive integer"}),
                0.0,
                False,
                False,
                {"episode_id": self._episode_id, "error": "invalid_limit"},
            )

        effective_limit = min(limit, MAX_PAGE_LIMIT)
        rows = self._tables[table]
        slice_rows = copy.deepcopy(rows[cursor : cursor + effective_limit])
        next_cursor = cursor + len(slice_rows) if (cursor + len(slice_rows)) < len(rows) else None

        return (
            self._obs(
                {
                    "table": table,
                    "cursor": cursor,
                    "limit": effective_limit,
                    "returned": len(slice_rows),
                    "next_cursor": next_cursor,
                    "records": slice_rows,
                }
            ),
            0.0,
            False,
            False,
            {"episode_id": self._episode_id},
        )

    def _act_get_record(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        table = args.get("table")
        record_id = args.get("id")
        if not isinstance(table, str) or table not in ALLOWED_TABLES:
            return (
                self._obs({"error": f"Invalid table '{table}'"}),
                0.0,
                False,
                False,
                {"episode_id": self._episode_id, "error": "invalid_table"},
            )
        if not isinstance(record_id, str) or not record_id:
            return (
                self._obs({"error": "id must be a non-empty string"}),
                0.0,
                False,
                False,
                {"episode_id": self._episode_id, "error": "invalid_id"},
            )

        rec = self._record_index.get((table, record_id))
        if rec is None:
            return (
                self._obs({"error": f"Record '{record_id}' not found in '{table}'"}),
                0.0,
                False,
                False,
                {"episode_id": self._episode_id, "error": "not_found"},
            )

        return (
            self._obs({"table": table, "id": record_id, "record": copy.deepcopy(rec)}),
            0.0,
            False,
            False,
            {"episode_id": self._episode_id},
        )

    def _act_get_document(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        doc_id = args.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            return (
                self._obs({"error": "doc_id must be a non-empty string"}),
                0.0,
                False,
                False,
                {"episode_id": self._episode_id, "error": "invalid_doc_id"},
            )

        doc = self._doc_index.get(doc_id)
        if doc is None:
            return (
                self._obs({"error": f"Document '{doc_id}' not found"}),
                0.0,
                False,
                False,
                {"episode_id": self._episode_id, "error": "not_found"},
            )

        return (
            self._obs({"doc_id": doc_id, "document": copy.deepcopy(doc)}),
            0.0,
            False,
            False,
            {"episode_id": self._episode_id},
        )

    def _act_calculator(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        expression = args.get("expression")
        try:
            result = safe_calculate(expression)
        except (ValueError, ZeroDivisionError, InvalidOperation, SyntaxError) as exc:
            return (
                self._obs({"error": f"Calculator error: {exc}"}),
                0.0,
                False,
                False,
                {"episode_id": self._episode_id, "error": "calculator_error"},
            )

        return (
            self._obs({"expression": expression, "result": result}),
            0.0,
            False,
            False,
            {"episode_id": self._episode_id},
        )

    def _act_submit(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._terminated = True
        self._truncated = False

        payload = args.get("findings_payload", args)
        try:
            normalized = validate_findings_payload(payload, self._episode_id)
        except SubmissionSchemaError as exc:
            grader_out = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "inspection_steps": self._step_count,
                "reward": -1.0,
                "schema_error": str(exc),
            }
            self._last_grader_result = grader_out
            reward = -1.0 if self._mode == "train" else 0.0
            info: dict[str, Any] = {
                "episode_id": self._episode_id,
                "mode": self._mode,
                "inspection_steps": self._step_count,
                "schema_error": str(exc),
            }
            if self._mode == "train":
                info["evaluation"] = grader_out
            return (
                self._obs({"status": "terminated_invalid_submission", "error": str(exc)}),
                reward,
                True,
                False,
                info,
            )

        if self._submit_evaluator is not None:
            grader_out = self._submit_evaluator(
                episode_id=self._episode_id,
                findings_payload=payload,
                normalized_findings=normalized,
                inspection_steps=self._step_count,
                mode=self._mode,
                truncated=False,
            )
        elif self._truth_issues is not None:
            grader_out = evaluate_findings(
                normalized,
                self._truth_issues,
                inspection_steps=self._step_count,
            )
        else:
            grader_out = None
        self._last_grader_result = grader_out

        # В открытом train-режиме возвращаем терминальную награду; в eval — 0.0 до конца внешнего прогона
        reward = float(grader_out["reward"]) if (self._mode == "train" and grader_out) else 0.0
        info: dict[str, Any] = {
            "episode_id": self._episode_id,
            "mode": self._mode,
            "inspection_steps": self._step_count,
            "submitted_count": len(normalized),
        }
        if self._mode == "train" and grader_out is not None:
            info["evaluation"] = grader_out

        return (
            self._obs({"status": "submitted", "submitted_count": len(normalized)}),
            reward,
            True,
            False,
            info,
        )