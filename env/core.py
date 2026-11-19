"""Ядро среды: состояние одного эпизода, действия, лимиты, завершение.

EnvCore никогда не видит эталон. На submit / усечении он вызывает `finish_handler`
(транспорт к оценщику), передавая только ответ агента и счётчики шагов. Что вернуть
политике — решает оценщик (train: награда и метрики; eval: 0 и пустой info), а ядро
дополнительно обнуляет награду и info в eval (защита в глубину).

Контракт шага:
  * каждый вызов step() во время эпизода расходует шаг, в том числе ошибочный;
  * submit -> terminated=True;
  * последний разрешённый шаг (max_steps) без submit -> truncated=True, итоговый ответ пустой;
  * вышел лимит времени -> действие не выполняется, truncated=True, итоговый ответ пустой;
  * шаг после конца эпизода -> ошибка episode_over, без изменений состояния;
  * inspection_steps = число шагов без submit (ошибочные шаги считаются).
Наблюдение шага содержит ровно {result, error, remaining_steps}.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .calculator import CalculatorError, evaluate
from .constants import ACTIONS, CALC_DETAILS, ERRORS, MAX_ACTION_BYTES, MAX_ID_CHARS, PAGE_LIMIT, TABLES
from .cursor import CursorError, decode_cursor, encode_cursor
from .observation import initial_observation
from .records import PublicEpisode

# finish_handler(findings_payload, inspection_steps, steps_used, end_reason) -> {"reward": float, "info": dict}
FinishHandler = Callable[[Any, int, int, str], dict]


class ActionError(Exception):
    def __init__(self, code: str, detail: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


def error_obj(code: str, detail: str | None = None) -> dict:
    out = {"code": code, "message": ERRORS[code]}
    if detail is not None:
        out["detail"] = detail
    return out


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


@dataclass
class _Episode:
    data: PublicEpisode
    mode: str
    max_steps: int
    time_limit_s: float
    deadline: float
    steps_used: int = 0
    done: bool = False
    terminated: bool = False
    truncated: bool = False
    pending: dict | None = None  # терминальный переход, который ещё не отдан политике (watchdog)
    end_reason: str | None = None
    terminal_info: dict = field(default_factory=dict)
    terminal_reward: float = 0.0


class EnvCore:
    def __init__(self, finish_handler: FinishHandler, clock: Callable[[], float] = time.monotonic) -> None:
        self._finish_handler = finish_handler
        self._clock = clock
        self._ep: _Episode | None = None
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ жизненный цикл
    def reset(self, data: PublicEpisode, mode: str, max_steps: int, time_limit_s: float) -> tuple[dict, dict]:
        with self.lock:
            self.abandon()
            self._ep = _Episode(
                data=data, mode=mode, max_steps=max_steps, time_limit_s=time_limit_s,
                deadline=self._clock() + time_limit_s,
            )
            obs = initial_observation(data, max_steps, time_limit_s)
            info = {"env_version": obs["env_version"], "mode": mode}
            return _json_copy(obs), info

    def abandon(self) -> None:
        """reset/close во время эпизода: эпизод засчитывается как усечённый с пустым ответом."""
        with self.lock:
            ep = self._ep
            if ep is not None and not ep.done:
                self._finish(ep, None, ep.steps_used, "abandoned", truncated=True)

    def expire_if_due(self) -> bool:
        """Вызывается сторожевым таймером: по истечении времени оценщик получает пустой ответ сразу."""
        with self.lock:
            ep = self._ep
            if ep is None or ep.done or self._clock() < ep.deadline:
                return False
            self._finish(ep, None, ep.steps_used, "time_limit", truncated=True)
            ep.pending = {"error": error_obj("time_limit")}
            return True

    @property
    def deadline(self) -> float | None:
        ep = self._ep
        return None if ep is None or ep.done else ep.deadline

    @property
    def active(self) -> bool:
        return self._ep is not None and not self._ep.done

    # ------------------------------------------------------------------ шаг
    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        with self.lock:
            ep = self._ep
            if ep is None:
                return self._obs(None, error_obj("not_reset"), 0), 0.0, False, False, {}
            if ep.done:
                if ep.pending is not None:  # время вышло между шагами: отдать терминальный переход один раз
                    pending, ep.pending = ep.pending, None
                    return self._terminal(ep, None, pending["error"])
                return self._obs(None, error_obj("episode_over"), 0), 0.0, ep.terminated, ep.truncated, {}
            if self._clock() >= ep.deadline:
                self._finish(ep, None, ep.steps_used, "time_limit", truncated=True)
                return self._terminal(ep, None, error_obj("time_limit"))

            ep.steps_used += 1
            try:
                name, args = self._parse(action)
            except ActionError as exc:
                return self._after_tool(ep, None, error_obj(exc.code, exc.detail))
            if name == "submit":
                payload = args.get("findings") if set(args) == {"findings"} else _MalformedSubmit()
                self._finish(ep, payload, ep.steps_used - 1, "submit", truncated=False)
                return self._terminal(ep, {"status": "submitted"}, None)
            try:
                result = self._dispatch(ep, name, args)
            except ActionError as exc:
                return self._after_tool(ep, None, error_obj(exc.code, exc.detail))
            return self._after_tool(ep, result, None)

    # ------------------------------------------------------------------ внутреннее
    def _after_tool(self, ep: _Episode, result: Any, error: dict | None):
        if ep.steps_used >= ep.max_steps:
            self._finish(ep, None, ep.steps_used, "step_limit", truncated=True)
            return self._terminal(ep, result, error)
        return self._obs(result, error, ep.max_steps - ep.steps_used), 0.0, False, False, {}

    def _terminal(self, ep: _Episode, result: Any, error: dict | None):
        remaining = max(0, ep.max_steps - ep.steps_used)
        reward = ep.terminal_reward if ep.mode == "train" else 0.0
        info = _json_copy(ep.terminal_info) if ep.mode == "train" else {}
        return self._obs(result, error, remaining), reward, ep.terminated, ep.truncated, info

    def _finish(self, ep: _Episode, payload: Any, inspection_steps: int, reason: str, truncated: bool) -> None:
        ep.done = True
        ep.terminated = not truncated
        ep.truncated = truncated
        ep.end_reason = reason
        if isinstance(payload, _MalformedSubmit):
            payload = {"__malformed_submit_action__": True}
        response = self._finish_handler(payload, inspection_steps, ep.steps_used, reason) or {}
        reward = response.get("reward", 0.0)
        ep.terminal_reward = float(reward) if isinstance(reward, (int, float)) and not isinstance(reward, bool) else 0.0
        info = response.get("info", {})
        ep.terminal_info = info if isinstance(info, dict) else {}

    @staticmethod
    def _obs(result: Any, error: dict | None, remaining: int) -> dict:
        return {"result": _json_copy(result), "error": error, "remaining_steps": remaining}

    @staticmethod
    def _parse(action: Any) -> tuple[str, dict]:
        if isinstance(action, str):
            if len(action.encode("utf-8", "replace")) > MAX_ACTION_BYTES:
                raise ActionError("action_too_large")
            try:
                action = json.loads(action)
            except (json.JSONDecodeError, RecursionError):
                raise ActionError("invalid_action") from None
        if not isinstance(action, dict):
            raise ActionError("invalid_action")
        try:
            size = len(json.dumps(action, allow_nan=False).encode("utf-8"))
        except (TypeError, ValueError, RecursionError):
            raise ActionError("invalid_action") from None
        if size > MAX_ACTION_BYTES:
            raise ActionError("action_too_large")
        name = action.get("action")
        if not isinstance(name, str):
            raise ActionError("invalid_action")
        if name not in ACTIONS:
            raise ActionError("unknown_action")
        args = {k: v for k, v in action.items() if k != "action"}
        required, optional = ACTIONS[name]
        if name == "submit":
            return name, args  # любая форма submit завершает эпизод; структуру проверяет оценщик
        if not required <= set(args):
            raise ActionError("missing_argument")
        if set(args) - required - optional:
            raise ActionError("unexpected_argument")
        return name, args

    def _dispatch(self, ep: _Episode, name: str, args: dict) -> Any:
        if name == "list_records":
            return self._list_records(ep, args.get("table"), args.get("cursor"), args.get("limit", PAGE_LIMIT))
        if name == "get_record":
            return self._get_record(ep, args["table"], args["id"])
        if name == "get_document":
            return self._get_document(ep, args["doc_id"])
        if name == "calculator":
            try:
                return {"value": evaluate(args["expression"])}
            except CalculatorError as exc:
                raise ActionError("calculator_error", CALC_DETAILS.get(exc.detail, CALC_DETAILS["syntax"])) from None
        raise ActionError("unknown_action")  # pragma: no cover

    @staticmethod
    def _table(table: Any) -> str:
        if not isinstance(table, str):
            raise ActionError("invalid_argument", "table must be a string")
        if table not in TABLES:
            raise ActionError("unknown_table")
        return table

    def _list_records(self, ep: _Episode, table: Any, cursor: Any, limit: Any) -> dict:
        table = self._table(table)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ActionError("invalid_limit")
        limit = min(limit, PAGE_LIMIT)
        rows = ep.data.tables[table]
        try:
            offset = decode_cursor(cursor, ep.data.episode_id, table, len(rows))
        except CursorError:
            raise ActionError("invalid_cursor") from None
        end = min(len(rows), offset + limit)
        return {
            "table": table,
            "records": rows[offset:end],
            "offset": offset,
            "total": len(rows),
            "next_cursor": encode_cursor(ep.data.episode_id, table, end) if end < len(rows) else None,
        }

    def _get_record(self, ep: _Episode, table: Any, rid: Any) -> dict:
        table = self._table(table)
        if not isinstance(rid, str) or not 0 < len(rid) <= MAX_ID_CHARS:
            raise ActionError("invalid_argument", "id must be a non-empty string")
        record = ep.data.by_id[table].get(rid)
        if record is None:
            raise ActionError("not_found")
        return {"table": table, "record": record}

    def _get_document(self, ep: _Episode, doc_id: Any) -> dict:
        if not isinstance(doc_id, str) or not 0 < len(doc_id) <= MAX_ID_CHARS:
            raise ActionError("invalid_argument", "doc_id must be a non-empty string")
        found = ep.data.documents.get(doc_id)
        if found is None:
            raise ActionError("document_not_found")
        kind, record = found
        return {"doc_type": kind, "document": record}


class _MalformedSubmit:
    """Маркер: submit с лишними/отсутствующими полями обёртки — неверная структура (-1)."""
