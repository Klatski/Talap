"""RL-среда ИИ-аудитора (зона B): reset/step, инструменты расследования, лимиты, изоляция.

    from env import AuditEnv          # официальный режим: среда и оценщик в отдельных процессах
    from env import LocalAuditEnv     # всё в одном процессе (тесты/отладка, без изоляции)

Импорты ленивые, чтобы `import env.protocol` из процесса оценщика не тянул клиент.
"""

from .constants import CONTRACT_VERSION, ENV_VERSION

__all__ = ["AuditEnv", "LocalAuditEnv", "EnvError", "EnvConfigError", "ENV_VERSION", "CONTRACT_VERSION"]


def __getattr__(name: str):
    if name in ("AuditEnv", "EnvError", "EnvConfigError"):
        from . import client

        return getattr(client, name)
    if name == "LocalAuditEnv":
        from .local import LocalAuditEnv

        return LocalAuditEnv
    raise AttributeError(name)
