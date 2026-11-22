# RL-среда для ИИ-аудитора (FinTech, кейс 1)

Воспроизводимый симулятор для дообучения моделей финансового аудита. Генератор создаёт
синтетическую компанию за один отчётный период со скрытыми ошибками четырёх типов, двумя
сценариями взаимодействующих ошибок, чистыми компаниями и правдоподобными отвлекающими случаями.
Среда в стиле Gymnasium даёт агенту инструменты расследования, оценщик в отдельном процессе
сопоставляет выводы с эталоном и считает награду. Все данные вымышлены. Сеть, GPU и сторонние
пакеты не нужны: только Python 3.11 stdlib.

Полное описание проекта в одном файле: [`docs/overview.md`](docs/overview.md).

## Единая команда проверки

```bash
python check.py            # или: make check
python check.py --quick    # без имитации закрытого прогона и карточки жюри (этапы 6–7)
python check.py --update   # после намеренных изменений: перезаписать траектории и снимки reports/*
python tools/clean_run.py  # перед сдачей: то же с нуля — свежий клон, новый venv, 2 CPU, сеть запрещена
```

Этапы (обычно 3–5 минут на ноутбуке, при нехватке памяти дольше):
1. окружение;
2. все тесты (generator, env, grader, jury);
3. побайтовая воспроизводимость, схемы и эталон 21 открытого эпизода;
4. базовые политики через изолированные процессы в режиме eval (`reports/baselines.{md,json}`);
5. пересборка `trajectories/trajectories.jsonl` байт в байт и сканер утечек;
6. имитация закрытой проверки жюри: 20 эпизодов с солью на новых сочетаниях параметров и 20 пар
   «перестановка + смена ID» (`reports/holdout.{md,json}`);
7. карточка жюри: раздел 10 ТЗ один в один, 20 проверок контракта, 20 эпизодов, 20 проверок награды,
   20 парных проверок и баллы по формуле жюри (`reports/jury_scorecard.md`).

Результаты этапов 4, 6 и 7 сравниваются с закоммиченными снимками (golden, без времени).
Код возврата 0 только если прошли все этапы.

## Контейнер

```bash
docker build -t audit-env .
docker run --rm --network none --cpus 2 --memory 4g audit-env                 # = python check.py
docker run --rm --network none audit-env python -m env.demo                   # демо
docker run --rm --network none audit-env python -m env.runner --splits dev    # таблица P/R/F1/возврат/шаги/время
```

## Интерфейс жюри (`judge_adapter.py`)

Рабочая директория — корень репозитория (в контейнере `/app`), нужен только Python 3.11 stdlib.
Точки входа — модуль [`judge_adapter.py`](judge_adapter.py) в корне, адаптер ничего не выбирает за политику.
Адаптер самодостаточен: он не меняет `sys.path` и не импортирует в процесс жюри пакеты `env`, `grader`,
`generator`, `policies` под их именами (свои пакеты жюри с такими именами остаются его собственными).
`generate_episode` и `score_episode` выполняет отдельный процесс `python -E -s -m grader.judge` из корня
репозитория; клиентская часть среды грузится под частным именем `_talaps_team_env`. Среды и песочницы,
брошенные без `close()`, останавливаются при выходе процесса жюри (не дольше 2 с).

```python
import judge_adapter as ja

ja.generate_episode(seed, config, output_dir)       # -> None: обёртка над генератором команды
env = ja.make_env()                                   # среда и оценщик в отдельных процессах
obs, info = env.reset(seed, {"mode": "closed", "episode_dir": output_dir, "team_config": {}})
obs, reward, terminated, truncated, info = env.step(
    {"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}})
ja.score_episode(episode_dir, findings, inspection_steps)   # закрытый вызов оценщика
policy = ja.load_policy("my_policy.py")              # своя политика в отдельном ограниченном процессе
```

```bash
docker run --rm --network none audit-env python -c "import judge_adapter as ja; \
  ja.generate_episode(1, {'size': 'small'}, '/tmp/ep'); env = ja.make_env(); \
  print(env.reset(0, {'mode': 'closed', 'episode_dir': '/tmp/ep'})[0]['remaining_steps'])"     # -> 100
python -m env.runner --episodes-root DIR --splits S --mode closed --policy-file my_policy.py  # прогон в песочнице
```

| Вызов | Поведение |
|---|---|
| `generate_episode(seed, config, output_dir)` | `config` = `{"team_config": {...}}` или сам конфиг генератора: имя пресета (`"heavy"`), `{"preset": "…", …переопределения}` или объект из [`docs/generator.md`](docs/generator.md) (например `{"size": "small", "errors": {"duplicate_invoice": 1}}`). Ключи reset (`mode`, `episode_dir`, `max_steps`, `time_limit_s`) и ключи, которых генератор не знает, игнорируются и пишутся в журнал оценщика. Опечатка — `ValueError`, а не молча чистая компания: и ключ верхнего уровня, похожий на известный (`{"Errors": …}`, `{"error": …}`, `{"errors ": …}`, `team_confg`; то же внутри `team_config`), и вложенное имя типа ошибки или сценария. Явно чужие ключи (`kind`, `label`, `note`) игнорируются. Пишет `output_dir/public/{manifest.json, ledger.csv, invoices.jsonl, payments.jsonl, fx.csv, policy.json}` и `output_dir/private/{truth.json, provenance.json, counterfactual/}`. Те же seed и config дают побайтово те же файлы. Соль закрытого прогона — переменная окружения `AUDIT_GRADER_SALT`. |
| `reset(seed, config)` | `{"mode": "closed", "episode_dir": D, "team_config": {...}}` загружает ровно каталог `D`: имя любое, путь абсолютный или от рабочей директории жюри; раскладка `public/` + `private/` или плоская (шесть открытых файлов и `truth.json` в одном каталоге). ID не меняются, повторный reset начинает тот же эпизод заново, `team_config` при `episode_dir` не используется. `truth.json` в каталоге `closed` не обязателен: без него эпизод проводится, а метрики жюри считает своим `score_episode` (в журнале `reference: "missing"`). `episode_id` берётся из `manifest.json`. Файлы читаются и с BOM, и с CRLF, колонки CSV — по заголовку в любом порядке; файл из 0 байт (например, `fx.csv` фикстуры без курсов) — пустая таблица. `seed` — JSON-значение; целое не типа `int` (`numpy.int64`) приводится к `int`. `{"mode": "train", "team_config": {...}}` без `episode_dir` генерирует эпизод генератором команды с этим seed (тот же, что `generate_episode`). Режимы: `train` — терминальная награда и метрики политике; `closed` (синоним `eval`) — политика всегда получает `0.0` и `{}`. Необязательно: `max_steps` 1..100, `time_limit_s` (0, 60]; 60 с считаются от начала reset. Лишние ключи config (и `split`/`index` рядом с `episode_dir`) игнорируются и пишутся в журнал оценщика (`source.ignored_config_keys`); ключ, похожий на известный (`Mode`, `episode-dir`, `team_confg`, `Errors`, `error`), — ошибка, а не молча другой эпизод. Неверный config (неизвестный `mode`, нет каталога, неверные значения, опечатка в ключе) — исключение `ValueError` (у `make_env()` — его подкласс `EnvConfigError`), среда остаётся рабочей. |
| `step(action)` | `{"name": "<действие>", "args": {...}}`; плоская форма `{"action": "…", …}` — то же действие. |
| `score_episode(episode_dir, findings, inspection_steps)` | `{"tp", "fp", "fn", "precision", "recall", "f1", "return"}`; невалидный submit — `return = -1`, остальные поля `null`, ошибка в журнале оценщика. `findings` проходит через `json.dumps`/`json.loads`, как через `step`: кортежи — массивы, не-JSON значение (NaN, `set`) — неверная структура. |

**Наблюдения и результаты.** Первое наблюдение: `task`, `manifest`, `policy`, `remaining_steps = 100`
и справочные `episode_id`, `tables`, `issue_types`, `matching`, `actions`, `action_format`, `limits`,
`env_version`. Каждое следующее — ровно `{"result", "error", "remaining_steps"}`. `error` равен `null` или
`{"code", "message"[, "detail"]}` из фиксированного каталога `env/constants.py`. В нём нет путей, seed и
закрытых данных, шаг с ошибкой расходуется.

| Действие (`args`) | `result` |
|---|---|
| `list_records` (`table`, `cursor` — первый `null`, `limit` ≤ 50, `null` = 50) | `{"records": [объекты], "next_cursor": null \| "строка"}`; порядок = порядок файла |
| `get_record` (`table`, `id`) | объект записи |
| `get_document` (`doc_id`) | объект документа (счёт из `invoices.jsonl` или платёж из `payments.jsonl`) |
| `calculator` (`expression`) | десятичная строка: `"0.1+0.2"` → `"0.3"` (Decimal, без eval) |
| `submit` (`findings` — объект findings.json) | `{"accepted": true}`, `terminated = true`; оценка только у оценщика. Неверная структура ответа — `result: null` и ограниченная ошибка `invalid_submission` с первой причиной в `detail` (эпизод тоже завершён) |

Таблицы и ключи (канонические имена жюри совпадают с нашими): `ledger` — `row_id`; `invoices` — `invoice_id`,
документ `doc_id`; `payments` — `payment_id`, документ `doc_id`; `fx` — `fx_id` = `FX:<source>:<target>:<date>`.
`fx_id` вычисляется средой из полей строки, это не surrogate ID. Ключ `<date>:<source>:<target>` (как в
правилах жюри) — синоним: `get_record` и `evidence_ids` принимают обе формы. Поля записей — §03 кейса
([`docs/contract.md`](docs/contract.md), раздел 2).

**Соответствие четырёх типов ошибок кейса** (подробно — `docs/contract.md`, раздел 4):

| Ошибка кейса | `type` | `affected_ids` | сумма / валюта | наборы доказательств |
|---|---|---|---|---|
| дубль счёта | `duplicate_invoice` | оба `invoice_id` пары | `total_minor` счёта / валюта счёта | `{doc_id A, doc_id B}` |
| пропущенное или частичное распределение платежа | `unapplied_payment` | `{payment_id, invoice_id}` | строка 2300 платежа / базовая | `{doc_id платежа, row_id строки 2300}`; у платежа на несколько счетов ещё все строки 1200 платежа, `doc_id` каждого счёта и (не базовая валюта) FX-id курса на дату каждого счёта |
| неверная дата или курс конвертации | `fx_rate_error` | `{invoice_id}` или `{payment_id}` | разница с верной конвертацией / базовая | `{doc_id, row_id, fx_id верной даты}` |
| ошибка отнесения к периоду | `cutoff_error` | `{invoice_id}` или `{payment_id}` | сумма документа / валюта документа | `{doc_id, row_id проводки}` |

Сценарии взаимодействия: `fx_partial_boundary_payment` (`fx_rate_error` + `unapplied_payment`) и
`cutoff_fx_boundary_invoice` (`cutoff_error` + `fx_rate_error`).

**truth.json для своих фикстур:** `{"episode_id", "issues": [{"issue_id", "type", "affected_ids",
"expected_amount_minor", "currency", "evidence_sets": [[...], ...], "amount_tolerance_minor"}]}`.
`evidence_sets` — допустимые наборы, хватает любого целиком. τ = `amount_tolerance_minor`: без поля 0,
генератор публикует 0 для дубля, cut-off и неразнесённого платежа и 1 для курсовой разницы (округление). `provenance.json` необязателен.

**Сопоставление** (`grader/matcher.py`, точные соглашения жюри):
* ID сравниваются точно. `affected_ids` — множество, оно должно совпасть с каноническим.
* `evidence_ids` — множество (после проверки исходного лимита 10). Оно должно включать хотя бы один набор
  целиком; лишние существующие ID допустимы, несуществующий ID делает вывод FP.
* `type` и `currency` сравниваются точно, без смены регистра и обрезки. Неизвестный тип или валюта — FP.
* `claimed_amount_minor` — JSON-целое; bool, строка или дробное число — ошибка структуры.
* Ошибка структуры всего submit (−1): больше 30 выводов, больше 10 `evidence_ids`, `explanation` длиннее
  1000 code points, неверный тип значения, нет обязательного поля. Поля сверх формата игнорируются.
* `episode_id` — строка из `manifest.json`. Другая строка — не ошибка структуры: ответ оценивается против
  текущего эпизода, несовпадение пишется в журнал оценщика (`warnings`).
* Ключ курса жюри `<date>:<source>:<target>` равен `fx_id` `FX:<source>:<target>:<date>` и в ответе, и в
  `evidence_sets` своего truth.
* Паросочетание максимальной мощности. При равенстве выбирается лексикографически наименьшая
  последовательность пар (`issue_id`, индекс вывода); повторы — FP.

**Журнал оценщика:** файл из `AUDIT_GRADER_LOG` (относительный путь — от рабочего каталога жюри). По умолчанию
он вне репозитория, каталог 0700, файл 0600:
* Linux и macOS — `<временная папка>/talaps-grader-log-<uid>/grader_log_judge.jsonl`; в нашем контейнере
  (пользователь `auditor`, uid 1000) — `/tmp/talaps-grader-log-1000/grader_log_judge.jsonl`;
* Windows — `<временная папка>\talaps-grader-log\grader_log_judge.jsonl`.

Точный путь печатает `python -c "import judge_adapter as ja; print(ja.grader_log_path())"` (у среды — `env.grader_log`).
Одна JSON-строка на эпизод или вызов `generate_episode` / `score_episode`; параллельные среды пишут в один файл
без потери строк. В строке: окружение (python, реализация, платформа), версии среды и оценщика, `code_version`
(sha256 исходников команды; его же печатает `python -m grader.codeversion`), seed, config, `episode_id`, режим,
причина завершения, ошибки структуры, предупреждения (`warnings`, например чужой `episode_id`), TP/FP/FN/P/R/F1,
шаги, награда и разбор по выводам. Строка дописывается
сразу по завершении эпизода.

**Граница доступа:**
* Среда (только открытые записи) и оценщик (эталон) работают в отдельных процессах.
* Политика работает в отдельном ограниченном процессе ([`env/sandbox.py`](env/sandbox.py)):
  `python -I -S -B` в пустом временном каталоге с очищенным окружением. Код политики приходит текстом
  и грузится из памяти. До первой строки политики ставятся необратимые ограничения:
  - аудит-хук: `open`, перечисление каталогов, процессы, сеть, `ctypes`, субинтерпретаторы, импорт
    `grader`/`generator`/`env` → `PermissionError`;
  - на Linux — seccomp-фильтр ядра: открытие и `stat` файлов по пути, exec, сокеты, ptrace, сигналы,
    новые процессы (`fork`, `clone` без `CLONE_THREAD`; потоки можно) и смена лимитов → `EPERM`;
  - на Linux — лимит адресного пространства 1 ГиБ (`RLIMIT_AS`) и без core-файлов;
  - дополнительный слой: `_imp.create_builtin` / `create_dynamic` отказывают запрещённым модулям. Это не
    граница: на уровне Python обёртку можно разобрать интроспекцией (например, через `__kwdefaults__`).
    Настоящая граница — ядро Linux: полученный так модуль ничего не даёт, fork, exec и сокеты → `EPERM`.
    На Windows и macOS ядерного фильтра нет, и граница там мягкая (`docs/design.md`).
* Oracle строится только на стороне оценщика.
* Враждебная политика, которая знает точный путь к `truth.json`, в песочнице не получает ничего.
  Без песочницы она же получает 12/12 и 1.0 — это негативный контроль (`tests/env/test_env_sandbox.py`).
* Зависшая политика ждётся не дольше 60 с эпизода; эпизод завершает среда (truncated, пустые findings).

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

## Закрытая проверка на своих seed и новых сочетаниях параметров

Генератор — чистая функция `(seed, config, версия[, соль]) → файлы`. Конфиг — пресет из
`generator/configs/` или свой JSON, параметры описаны в [`docs/generator.md`](docs/generator.md):
число ошибок каждого типа независимо, вариант ошибки (`variants`, например
`{"unapplied_payment": ["missing_allocation", "partial_allocation"]}` — один полный пропуск и одно частичное
разнесение), сценарии, размер, валюты, правило даты курса, округление.
Эталон и история генерации пишутся в `private/`, агент их не видит.

```bash
# 1) сгенерировать эпизоды: каталог эпизода — <корень>/<split>/ep-<4..32 букв/цифр>
python -m generator generate --seed 9001 --config interactions_both --overrides '{"size": "small"}' \
       --salt "$SECRET" --out /data/jury/hidden/ep-s9001
python -m generator generate --seed 9002 --config clean_small --salt "$SECRET" --out /data/jury/hidden/ep-s9002
# или сразу 20 эпизодов (5 чистых, 5 со взаимодействиями) + 20 пар «перестановка + смена ID»:
python -m generator holdout --salt "$SECRET" --n 20 --out /data/jury          # -> /data/jury/{episodes,pairs}/

# 2) прогнать политики через изолированную среду; в eval награда остаётся у оценщика
python -m env.runner --episodes-root /data/jury --splits hidden --mode eval \
       --policies random,reconcile,flag_all,oracle --report-md /data/jury/report.md
# после holdout сплиты называются episodes и pairs:
python -m env.runner --episodes-root /data/jury --splits episodes,pairs --mode eval \
       --policies random,reconcile,flag_all,oracle --report-md /data/jury/holdout.md
```

Из своей обвязки: `AuditEnv(episodes_root="/data/jury", grader_log="grader_log.jsonl")`, затем
`reset(seed, {"split": "hidden", "index": i, "mode": "eval"})`. Эпизод можно сгенерировать и прямо в
процессе оценщика: `reset(seed, {"generator": {...конфиг...}, "mode": "eval"})`, соль передаётся через
переменную окружения `AUDIT_GRADER_SALT`. Итоги (P, R, F1, награда, причина завершения, ошибки структуры)
дописываются в журнал оценщика сразу по завершении эпизода. JSON-lines сервер `python -m env.server` (методы
`reset`, `step`, `close`) подходит для политики на любом языке.

## Что где (соответствие ТЗ)

| Требование | Где |
|---|---|
| Генератор: 4 типа ошибок, 2 сценария, чистые компании, отвлекающие случаи | `generator/`, [`docs/generator.md`](docs/generator.md) |
| Эталон с наборами доказательств, история генерации | `episodes/*/*/private/{truth,provenance}.json` |
| 10 train + 10 dev эпизодов (+ заглушка) | `episodes/train`, `episodes/dev`, `episodes/stub` |
| Интерфейс жюри: `generate_episode`, `make_env`, `score_episode`, `{"name","args"}`, `closed` + `episode_dir` | `judge_adapter.py`, `tests/env/test_env_jury_adapter.py` (C01–C20, парные проверки) |
| reset/step, действия, курсоры, калькулятор, лимиты, тайм-аут, изоляция процессов | `env/` |
| Политика в отдельном ограниченном процессе (граница доступа) | `env/sandbox.py`, `env/sandbox_child.py`, `tests/env/test_env_sandbox.py` |
| Валидация submit, matcher (паросочетание), метрики, награда, train/eval/closed | `grader/`, `tests/grader/test_grader_jury_rules.py` (блок 3 и M01–M12) |
| Схемы | `schemas/*.schema.json` |
| 4 базовые политики: random, reconcile, flag_all, oracle (+ демо-аудитор) | `policies/` |
| Траектории без эталона | `trajectories/trajectories.jsonl` |
| Тесты среды, сопоставления, награды, атак | `tests/env`, `tests/grader`, `tests/generator` |
| Критерии оценки (раздел 10) как проверки: 4 × 20 + баллы | `tests/jury/scorecard.py` → `reports/jury_scorecard.md` |
| Диагностика P, R, F1, возврат, шаги, время; разбор сбоев | `reports/baselines.md`, `reports/holdout.md` |
| Записка о дизайне и честные ограничения | [`docs/design.md`](docs/design.md) |
| Контейнер и единая команда | `Dockerfile`, `check.py`, `Makefile`; CI `.github/workflows/check.yml` (Linux + Docker `--network none`) |
| Лицензия, зависимости | `LICENSE` (MIT), `requirements.txt` (только stdlib) |

## Команды по частям

```bash
python -m unittest discover -s tests/env -p "test_env_*.py"          # среда, адаптер жюри, песочница, файлы CI (141 тест)
python tests/jury/scorecard.py [--salt S] [--out card.md]            # карточка жюри на своей соли (≈45 с)
python -m unittest discover -s tests/grader -p "test_grader_*.py"    # оценщик и правила жюри (78 тестов)
python tests/grader/test_grader_handworked.py --show                 # ручные примеры: «вручную | программа»
python -m grader --episode episodes/stub/ep-SUW6BEVZNR --submission answer.json --steps 3   # свой ответ: TP/FP/FN, формула награды
python -m unittest discover -s tests/generator -p "test_gen_*.py"    # генератор и перекрёстные проверки (79 тестов)
python -m env.runner --policies random,reconcile,flag_all,oracle,auditor --splits train,dev --mode eval
python -m env.runner --splits train --mode closed --policy-file tests/env/attack_policies/read_truth.py  # атака в песочнице
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
