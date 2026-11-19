"""Генератор синтетических компаний со скрытыми финансовыми ошибками (зона человека A).

Публичный вход: generator.generate.generate_episode / write_episode,
CLI: python -m generator --help
"""

from .version import GENERATOR_VERSION, SCHEMA_VERSION

__all__ = ["GENERATOR_VERSION", "SCHEMA_VERSION"]
