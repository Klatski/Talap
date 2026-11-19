# RL-среда для ИИ-аудитора (FinTech, кейс 1)

Воспроизводимый симулятор для дообучения моделей финансового аудита. Генератор создаёт
синтетическую компанию за один отчётный период со скрытыми ошибками четырёх типов, двумя
сценариями взаимодействующих ошибок, чистыми компаниями и правдоподобными отвлекающими случаями.
Среда в стиле Gymnasium даёт агенту инструменты расследования, оценщик в отдельном процессе
сопоставляет выводы с эталоном и считает награду. Все данные вымышлены. Сеть, GPU и сторонние
пакеты не нужны: только Python 3.11 stdlib.

## Единая команда проверки

```bash
python check.py            # или: make check
python check.py --quick    # без имитации закрытого прогона (этап 6)
```

Этапы (около 70 с на ноутбуке): окружение → все тесты (generator, env, grader) → побайтовая
воспроизводимость, схемы и эталон 21 открытого эпизода → базовые политики через изолированные
процессы в режиме eval (`reports/baselines.md`) → пересборка `trajectories/trajectories.jsonl`
байт в байт и сканер утечек → имитация закрытой проверки жюри: 20 эпизодов с солью на новых
сочетаниях параметров и 20 пар «перестановка + смена ID» (`reports/holdout.md`).
Код возврата 0 только если прошли все этапы.

## Контейнер

```bash
docker build -t audit-env .
docker run --rm --network none --cpus 2 --memory 4g audit-env                 # = python check.py
docker run --rm --network none audit-env python -m env.demo                   # демо
docker run --rm --network none audit-env python -m env.runner --splits dev    # таблица P/R/F1/возврат/шаги/время
```

## Демо (скриптовая политика)

```bash
python -m env.demo                                   # аудитор на заглушке, train: шаги, награда, выводы
python -m env.demo --policy reconcile --split dev --index 2
python -m env.demo --policy random --mode eval       # в eval политика видит 0 и {}, награда в журнале оценщика
```

## Использование среды

```python
from env import AuditEnv                      # env.LocalAuditEnv — то же в одном процессе (тесты, без изоляции)
from policies import make_policy

with AuditEnv(grader_log="reports/grader_log.jsonl") as env:
    obs, info = env.reset(seed=3, config={"split": "train", "mode": "train"})   # или {"generator": "all_types"}
    policy = make_policy("auditor"); policy.reset(obs, info)
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(policy.act(obs))
```

Действия: `list_records(table, cursor, limit)`, `get_record(table, id)`, `get_document(doc_id)`,
`calculator(expression)`, `submit(findings)`. Лимит — 100 шагов вместе с submit и 60 с на эпизод.
Полный контракт — [`docs/contract.md`](docs/contract.md).

## Что где (соответствие ТЗ)

| Требование | Где |
|---|---|
| Генератор: 4 типа ошибок, 2 сценария, чистые компании, отвлекающие случаи | `generator/`, [`docs/generator.md`](docs/generator.md) |
| Эталон с наборами доказательств, история генерации | `episodes/*/*/private/{truth,provenance}.json` |
| 10 train + 10 dev эпизодов (+ заглушка) | `episodes/train`, `episodes/dev`, `episodes/stub` |
| reset/step, действия, курсоры, калькулятор, лимиты, тайм-аут, изоляция процессов | `env/` |
| Валидация submit, matcher (паросочетание), метрики, награда, train/eval | `grader/` |
| Схемы | `schemas/*.schema.json` |
| 4 базовые политики: random, reconcile, flag_all, oracle (+ демо-аудитор) | `policies/` |
| Траектории без эталона | `trajectories/trajectories.jsonl` |
| Тесты среды, сопоставления, награды, атак | `tests/env`, `tests/grader`, `tests/generator` |
| Диагностика P, R, F1, возврат, шаги, время; разбор сбоев | `reports/baselines.md`, `reports/holdout.md` |
| Записка о дизайне и честные ограничения | [`docs/design.md`](docs/design.md) |
| Контейнер и единая команда | `Dockerfile`, `check.py`, `Makefile` |
| Лицензия, зависимости | `LICENSE` (MIT), `requirements.txt` (только stdlib) |

## Команды по частям

```bash
python -m unittest discover -s tests/env -p "test_env_*.py"          # среда (62 теста)
python -m unittest discover -s tests/grader -p "test_grader_*.py"    # оценщик (33 теста)
python -m unittest discover -s tests/generator -p "test_gen_*.py"    # генератор и перекрёстные проверки (68 тестов)
python -m env.runner --policies random,reconcile,flag_all,oracle,auditor --splits train,dev --mode eval
python -m generator check-splits                       # и build-splits, audit <ep>, stress, leakcheck, holdout
python -m generator scan-trajectories trajectories/trajectories.jsonl
```

## Разделение труда

Работают два человека, подробности — в `CLAUDE.md`.

**Человек A — генератор и эталон:** скрытая история компании и рендер ledger, invoices, payments,
fx, policy, manifest; бухгалтерские инварианты до и после внесения ошибки; 4 типа ошибок,
2 взаимодействующих сценария, чистые компании, отвлекающие случаи; `truth.json` с наборами
доказательств, `provenance.json`; преобразование «перестановка и смена ID» и переотображение
truth; oracle; 10 train + 10 dev эпизодов.

**Человек B — среда и оценщик:** reset/step, изоляция процессов, лимит 100 шагов, тайм-аут,
truncated; действия, пагинация с курсорами, безопасный калькулятор, ограниченные ошибки;
валидация submit, matcher с паросочетанием, метрики, награда, режимы train и eval; тесты награды
и атак; политики random, flag_all, reconcile; trajectories.jsonl, Docker, единая команда проверки.

**Перекрёстная проверка (с 21 ноября).** B проверяет данные A: независимый аудитор
`policies/auditor.py` заново выводит все расхождения из открытых записей, а простая сверка
служит детектором утечек (`tests/grader/test_grader_crosscheck_data.py`). A проверяет среду B:
36 атак через оценщик и через `step(submit)`, oracle через изолированную среду, сканер утечек
траекторий (`tests/generator/test_gen_crosscheck_tools.py`, `test_gen_against_env.py`).
Детерминизм между процессами каждый проверяет на своей части.
