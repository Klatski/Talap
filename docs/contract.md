# Общий контракт A ↔ B

> Статус: **согласовано** (20.11). Разделы 1–6 реализованы генератором (A), разделы 7–10 —
> средой и оценщиком (B); обе стороны проверены тестами друг друга (`tests/generator/test_gen_against_env.py`,
> `tests/grader/test_grader_crosscheck_data.py`). Машиночитаемые схемы — `schemas/*.schema.json`.
> Любое изменение этого файла затрагивает обоих — только по согласию.

## 1. Эпизод на диске

```
episodes/<split>/<episode_id>/
  public/   manifest.json  policy.json  ledger.csv  invoices.jsonl  payments.jsonl  fx.csv
  private/  truth.json  provenance.json          ← только оценщик, агенту недоступно
  private/counterfactual/  те же 6 файлов без внесённых ошибок (проверка данных, агенту недоступно)
episodes/<split>/index.json                       ← seed, конфиг, дайджест (открыто для train/dev)
```

Сплиты: `train` (10), `dev` (10), `stub` (1 — заглушка для разработки среды: все 4 типа
ошибок + сценарий взаимодействия, KZT/USD/EUR, +05:00, март 2026).
Сборка: `python -m generator build-splits`, проверка воспроизводимости: `python -m generator check-splits`.

## 2. Файлы и поля

| Файл | Поля | Ключ записи |
|---|---|---|
| `ledger.csv` | row_id, timestamp, account, debit_minor, credit_minor, currency, entity_id, invoice_id, payment_id | `row_id` (`L-XXXXXXXX`) |
| `invoices.jsonl` | doc_id, invoice_id, entity_id, issued_at, due_at, currency, total_minor, line_items[] | `invoice_id` (`INV-…`), документ `doc_id` (`D-…`) |
| `payments.jsonl` | doc_id, payment_id, timestamp, currency, amount_minor, invoice_refs | `payment_id` (`PAY-…`), документ `doc_id` (`D-…`) |
| `fx.csv` | date, source_currency, target_currency, rate | `FX:<source>:<target>:<date>` (вычисляемый) |
| `manifest.json` | schema_version, generator_version, episode_id, base_currency, currency_exponents, period_start/end, timezone, counts, rounding, tables | — |
| `policy.json` | часовой пояс, границы периода, правило даты курса, округление, разнесение платежей, план счетов | — |

`line_items[]`: `{line_no, sku, description, quantity, unit_price_minor, amount_minor, order_ref}`.
`invoice_refs`: `null` (плательщик не указал счёт) или `[{invoice_id, amount_minor}]` в валюте платежа.
`amount_minor < 0` у платежа = возврат клиенту (refund), `invoice_refs = null`.

### Правила значений
- Время — ISO-8601 с секундами и явным смещением (`+00:00`, не `Z`). **Смещения у разных
  записей разные** (компания, UTC, другие) — сравнивать только моменты времени.
- Деньги — `int` в минимальных единицах; разрядность — `manifest.currency_exponents` (JPY 0, KWD 3, остальные 2).
- Курсы — десятичные строки. `rate` = сколько `target_currency` за 1 `source_currency`; в `fx.csv`
  только пары «валюта → базовая», только рабочие дни (выходные и пара праздников пропущены).
- В CSV пустое поле = `null`. **Среда (B) должна отдавать его как JSON `null`**, суммы — как int.
- Книга ведётся в базовой валюте: `ledger.currency == manifest.base_currency` всегда.
- Документы берутся из окна вокруг периода (~30 дней до, ~5 после); в книгу периода должны
  попадать только документы периода. Начальных остатков нет.

## 3. Политика учёта (`policy.json`)
- **Период**: `[period_start, period_end)` в `company_timezone`. Документ принадлежит периоду
  по своему времени (`issued_at` / `timestamp`), переведённому в пояс компании.
- **Дата курса**: `document_date` (дата документа в поясе компании) или `previous_day`
  (день до неё); если курса на дату нет — последняя более ранняя дата из `fx.csv`.
- **Конвертация**: `base_minor = round(src_minor * rate * 10^(exp(base) - exp(src)))`,
  округление `half_up` (от нуля) или `half_even` — по политике, каждая строка отдельно.
- **Проводка счёта**: Дт 1200 / Кт 4000 — обе по курсу счёта.
- **Проводка платежа**: Дт 1000 по курсу платежа; Кт 1200 по каждой ссылке — по курсу *счёта*;
  неразнесённое (нет ссылок или превышение над суммой ссылок) — Кт 2300 по курсу платежа;
  разница — 7900 (курсовая разница). Возврат: Дт 2300 / Кт 1000.
- **Счета**: каждая пара `(order_ref, sku)` выставляется не более одного раза; повторяющиеся
  услуги получают новый `order_ref`.

## 4. Типы ошибок: каноничные ID, сумма, доказательства

Допуск суммы — `amount_tolerance_minor` в truth (сейчас **1** минимальная единица для всех типов).
Валюта должна совпадать точно. Все `affected_ids` — бизнес-ID (`INV-…`/`PAY-…`), сравниваются как множество.

| type | Что произошло | affected_ids | expected_amount_minor / currency | Допустимые наборы доказательств |
|---|---|---|---|---|
| `duplicate_invoice` | Счёт повторно введён (тот же клиент, тот же момент `issued_at`, идентичные строки вкл. `order_ref`) и проведён второй раз | **оба** `invoice_id` пары | `total_minor` одного счёта / валюта счёта | `{doc_id A, doc_id B}` |
| `unapplied_payment` | В `invoice_refs` платёж оплачивает счёт X (сумма ссылок = сумме платежа), но в книге Кт 1200 по X нет или меньше — остаток на 2300 | `{payment_id, invoice_id X}` | сумма строки 2300 этого платежа **как проведено** / базовая | `{doc_id платежа, row_id строки 2300}` |
| `fx_rate_error` | Сумма в книге посчитана не по курсу даты из политики | `{invoice_id}` или `{payment_id}` | `|проведено − конвертация по верному курсу|` по строке 1200 (счёт) или 1000 (платёж) / базовая | счёт: `{doc_id, row 1200, FX-id верной даты}` или `{doc_id, row 4000, FX-id}`; платёж: `{doc_id, row 1000, FX-id}` |
| `cutoff_error` | Документ другого периода (по поясу компании) проведён в книге этого периода | `{invoice_id}` или `{payment_id}` | `|total_minor|` или `|amount_minor|` / валюта документа | `{doc_id, любая строка проводки этого документа}` |

TP требует, чтобы `evidence_ids` агента **содержали хотя бы один набор целиком** (лишние ID допустимы).
Наборы минимальны и независимо достаточны: по одним этим записям + manifest/policy можно
восстановить тип и сумму (это проверяет `generator/audit.py::check_evidence_set`).

Варианты внутри типов (в `provenance.injections[].variant`):
`fx_rate_error`: `adjacent_date`, `other_rule`, `due_date`, `document_offset_date`, `utc_date`;
`cutoff_error`: `same_instant` (время проводки = времени документа), `shifted_into_period`
(проводка датирована внутри периода), `document_offset_misread`;
`unapplied_payment`: `missing_allocation`, `partial_allocation`; `duplicate_invoice`: `reentered_document`.

### Взаимодействующие ошибки
Каждый сценарий даёт **две отдельные записи** в truth (по одной на тип) с общим `scenario_group`.
Сумма каждой считается по правилу своего типа, независимо от второй (по проведённым строкам).
- `fx_partial_boundary_payment`: платёж в иностранной валюте у границы периода; время записано
  в смещении, где дата «перескакивает» границу. Бухгалтер взял курс по дате из записи (а не по
  поясу компании) **и** разнёс платёж частично → `fx_rate_error` + `unapplied_payment`.
  Платёж в периоде, поэтому `cutoff_error` здесь нет (ловушка для наивной сверки дат).
- `cutoff_fx_boundary_invoice`: счёт в иностранной валюте из соседнего периода, который по
  записи выглядит текущим; проведён в этом периоде по курсу даты из записи →
  `cutoff_error` + `fx_rate_error`.

Если на выходных обе даты сводятся к одному курсу, сценарий берёт другой вариант неверного курса.

### Чистые случаи и отвлекающие факторы (не попадают в truth)
Повторяющиеся счета (одинаковые сумма и строки, новый `order_ref` каждую неделю), частичные
отгрузки (один `order_ref`, разные `sku`), разбитые платежи, частичные оплаты, пакетные платежи
на несколько счетов, переплата + возврат, платежи без ссылок (законно на 2300), документы у
границы периода в «обманчивых» смещениях (и в периоде, и вне), задержка проводки внутри
периода, курсы выходных (откат на пятницу). Полностью чистые компании (0 ошибок).
Список отвлекающих случаев эпизода — в `provenance.distractors` (для разбора FP).

## 5. truth.json (закрытый)

```json
{"schema_version": "1.0", "episode_id": "ep-…",
 "issues": [{"issue_id": "ISS-…", "type": "fx_rate_error", "affected_ids": ["INV-…"],
             "expected_amount_minor": 1234, "currency": "KZT", "amount_tolerance_minor": 1,
             "evidence_sets": [["D-…", "FX:USD:KZT:2026-03-10", "L-…"], ["D-…", "FX:…", "L-…"]],
             "scenario": null, "scenario_group": null}]}
```
`provenance.json`: seed, конфиг и дайджест, параметры, проверки инвариантов до/после внесения,
по каждой ошибке — `variant`, `cause`, `before`/`after`; список отвлекающих случаев; история transform.

## 6. Генератор и преобразование (зона A)
- `generate_episode(seed, config, salt="")` — чистая функция; одинаковые seed+config+версия →
  побайтово одинаковые файлы в любом процессе. `salt` — секрет для закрытых прогонов.
- Эпизод не выпускается, если `audit(public) != truth` или набор доказательств недостаточен.
- `transform_texts / transform_episode(src, dst, seed)` — перестановка строк + смена всех ID,
  truth/provenance переотображаются; `remap_findings(findings, mapping)` — для парных проверок.
- Oracle: `policies/oracle.py` (`oracle_findings(truth)`, `OraclePolicy`).

## 7. findings.json (submit) и сопоставление — реализовано B (`grader/`)
Схема: `schemas/findings.schema.json`; строгая валидация — `grader/validate.py` (зеркало схемы,
проверено фаззингом на согласие с ней). Неверная структура (лишние поля, неверные типы,
bool/float вместо integer, отрицательные суммы, неизвестный `type`, валюта не `[A-Z]{3}`,
больше 30 выводов, `affected_ids` вне 1..10, больше 10 `evidence_ids`, `explanation` длиннее 1000,
ID-строки вне 1..64 символов, чужой `episode_id`) → эпизод завершён, награда **−1**, ошибки — в журнале оценщика.
```json
{"episode_id": "ep-…", "findings": [{"type": "…", "affected_ids": ["…"], "claimed_amount_minor": 0,
  "currency": "KZT", "evidence_ids": ["…"], "explanation": "…"}]}
```
**Нормализация** (`grader/matcher.py`): `affected_ids` и `evidence_ids` → множества (порядок и повторы
не важны); строки сравниваются точно (без смены регистра и обрезки); `type`, `currency` — точно;
сумма — `|claimed − expected| ≤ amount_tolerance_minor` (из truth, сейчас 1).
**TP** ⇔ все пять условий: type, множество affected_ids, валюта, сумма в допуске,
`set(evidence_ids) ⊇` хотя бы один набор из `evidence_sets` целиком (лишние ID допустимы).
**Паросочетание** — максимальное взаимно однозначное (алгоритм Куна), не жадное; обход в каноническом
порядке (выводы — по отсортированным полям, ошибки — по `(type, affected_ids, сумма, валюта, issue_id)`),
поэтому результат не зависит от порядка выводов. Несопоставленные и повторные выводы — FP, пропущенные — FN.
**Метрики и награда** (`grader/metrics.py`, точная арифметика `Fraction`): P, R, F1 по ТЗ (нулевой
знаменатель → 0; пустой эталон + пустой ответ → F1 = 1; пустой эталон + выводы → F1 = 0);
`reward = clip(2·F1 − 1 − 0.002·inspection_steps, −1, 1)`. Экспериментальная награда
(`reward_experimental`, только в журнале и отчётах, общую не заменяет): как общая, но усечённый эпизод → −1.
**API для A:** `grader.score_submission(truth, submission, inspection_steps, *, provenance=None, truncated=False, malformed_reason=None)
-> {"valid", "errors", "tp", "fp", "fn", "precision", "recall", "f1", "inspection_steps", "reward",
"reward_experimental", "diagnostics"}` (для неверной структуры tp/fp/fn = None, reward = −1).

## 8. Действия и наблюдения — реализовано B (`env/`)
```json
{"action": "list_records", "table": "ledger", "cursor": null, "limit": 50}
{"action": "get_record", "table": "invoices", "id": "INV-…"}
{"action": "get_document", "doc_id": "D-…"}
{"action": "calculator", "expression": "round_half_up(23208 * 516.3571 / 100)"}
{"action": "submit", "findings": { … findings.json … }}
```
* Формат действия — **плоский** `{"action": …, …}`; строка с JSON тоже принимается. Лишние или отсутствующие
  аргументы, неверные типы, неизвестные действие/таблица/ID → ограниченная ошибка
  `{"code", "message"[, "detail"]}` из фиксированного каталога (`env/constants.py::ERRORS`), шаг расходуется.
  Тексты ошибок не содержат пользовательского ввода, путей и сведений об эталоне.
  Нечитаемое действие (не JSON: NaN/Infinity, целое длиннее 4300 цифр, смешанные типы ключей, глубокая
  вложенность, не объект) → `invalid_action`; больше 256 КБ → `action_too_large`. Размер считается по
  `json.dumps(sort_keys=True)` с ASCII-экранированием; строка больше 2 МБ (UTF-8) не разбирается.
  Классификация одна для ядра и клиента — `env/jsonsafe.py::classify_action`.
* **Страховка step.** Никакое исключение из `step` наружу не выходит. Непредвиденный дефект среды при разборе
  или выполнении действия (в том числе результат, который нельзя сериализовать в JSON) → `internal_error`, шаг
  расходуется, текст исключения не раскрывается. Если оценщик упал или недоступен при завершении (submit, лимит
  шагов, тайм-аут), эпизод всё равно завершается: `grader_unavailable`, награда 0, `info` = `{}`.
  Фаззинг (3000 случайных действий) не даёт ни одного `internal_error` (`tests/env/test_env_fault_tolerance.py`).
* **Любой submit завершает эпизод.** Объект с `"action": "submit"` (или JSON-строка такого объекта; строка
  больше 2 МБ узнаётся по началу `{"action": "submit"`) всегда даёт `terminated=true`. Если обёртка не ровно
  `{action, findings}`, полезная нагрузка не JSON или больше 1 МБ (`MAX_SUBMIT_BYTES`), это неверная
  структура: награда −1, причина (`wrapper` / `not_json` / `too_large`) в журнале оценщика. 1 МБ — с запасом
  выше худшего **валидного** ответа (30 выводов, ID и explanation из четырёхбайтовых символов, суммы по
  4300 цифр ≈ 0.96 МБ), поэтому размер текста explanation никогда не меняет награду валидного ответа.
  Причину передаёт оценщику среда отдельным полем `finish.malformed`, а не внутри ответа агента.
  Одинаково в `AuditEnv` и `LocalAuditEnv` (`tests/env/test_env_hostile_input.py`).
* Таблицы и ключи: `ledger` (`row_id`), `invoices` (`invoice_id`), `payments` (`payment_id`),
  `fx` (`fx_id` = `FX:<src>:<dst>:<date>`, поле добавляется средой в каждую запись fx).
  `get_document(doc_id)` → `{"doc_type": "invoice"|"payment", "document": …}`; бизнес-ID там не принимаются.
* `list_records`: `limit` — целое ≥ 1, больше 50 → 50; результат `{table, records, offset, total, next_cursor}`;
  курсор `"<table>.<offset>.<tag>"`, tag = sha256(episode_id, table, offset) — детерминирован, стабилен,
  привязан к таблице и эпизоду (после reset старые курсоры недействительны).
* Записи: пустое поле CSV → `null`, суммы → int, курсы → строки; порядок строк = порядок файла.
* `calculator`: только `+ - * / ( )`, десятичные литералы (ASCII-цифры, без экспоненты), `abs`,
  `round_half_up(x[, n])`, `round_half_even(x[, n])`; Decimal, 34 знака; ≤ 256 символов, ≤ 128 токенов,
  вложенность ≤ 32, |значение| < 1e30. Результат `{"value": "<десятичная строка>"}`.
* Наблюдение шага — ровно `{"result", "error", "remaining_steps"}`. Первое наблюдение: `task`,
  `episode_id`, `manifest`, `policy`, `tables` (размеры и поля), `issue_types` (раздел 4 кратко),
  `matching`, `actions`, `limits`, `remaining_steps`, `env_version`. Seed, путей и сведений об эталоне нет.
* Лимиты: ≤ 100 шагов **включая submit**; 60 с на эпизод от reset. `submit` → `terminated=true`.
  Последний разрешённый шаг без submit → `truncated=true`, итоговый ответ пустой (оценщик считает
  `findings: []`). Время вышло → действие не выполняется, `truncated=true`, ошибка `time_limit`;
  сторожевой таймер закрывает эпизод у оценщика, даже если агент больше не вызывает step.
  reset/close посреди эпизода → эпизод записывается как `abandoned` (усечён, пустой ответ).
  Шаг после конца → ошибка `episode_over`, состояние не меняется, повторный submit не учитывается.
* **inspection_steps** = все шаги, кроме submit (шаги с ошибкой считаются). Награда по ходу = 0.
* Режимы (`config.mode`): `train` — терминальная награда и агрегированные метрики в `info`
  (`valid, errors, tp, fp, fn, precision, recall, f1, inspection_steps, truncated_reason, reward_experimental`);
  `eval` — политика получает 0 и `{}` на каждом шаге, результат — только в журнале оценщика (после close).
* `reset(seed, config)`; ключи config: `split` + (`episode_id` | `index` | ничего → `seed mod N`)
  либо `generator` (имя пресета или объект-конфиг; эпизод генерируется в процессе оценщика по seed и соли
  обвязки `AUDIT_GRADER_SALT`); `mode`, `max_steps` (1..100), `time_limit_s` (0..60], `config_version`.
  Корень данных задаёт обвязка при запуске (`AuditEnv(episodes_root=…)`), а не reset.
* Изоляция: агент ↔ `python -m env.server` (только открытые записи) ↔ `python -m grader.service`
  (эталон); JSON-lines по stdin/stdout. У агента нет канала к оценщику.

## 9. Интерфейс политики (адаптеры базовых политик) — `policies/`
```python
class Policy:
    name: str
    def reset(self, observation: dict, info: dict) -> None: ...
    def act(self, observation: dict) -> dict: ...   # одно действие из раздела 8
```
`policies.make_policy(name, seed=0, truth=None)`: `random` (seed фиксирован), `reconcile`, `flag_all`,
`auditor` (B), `oracle` (A; только с `truth`, создаётся обвязкой на стороне оценщика — `env/runner.py`).

## 10. Перекрёстная проверка A ↔ B (готовые инструменты)
- `python -m generator attacks <episode_dir> --out attacks.json` — 36 атакующих submit'ов на эпизод
  с ожидаемыми по ТЗ `valid/tp/fp/fn/reward`. Прогоняются через оценщик B
  (`tests/generator/test_gen_crosscheck_tools.py::AgainstGraderTests`, `tests/grader/test_grader_reward_properties.py`
  на всех 21 эпизоде) и через `step(submit)` среды (`tests/generator/test_gen_against_env.py`).
- `python -m generator scan-trajectories trajectories/trajectories.jsonl` — сканер утечек A; входит в `python check.py`.
- B → A: `policies/auditor.py` — независимый (не импортирует generator/) вывод всех расхождений из открытых
  записей; `tests/grader/test_grader_crosscheck_data.py` — на открытых эпизодах, контрфактах (0 находок),
  50 новых сочетаниях параметров и их парах; бухгалтерские равенства; простая сверка как детектор утечек.
- Имитация закрытого прогона жюри целиком: `python check.py` (этап 6).

## Решения по открытым вопросам (закрыты 20.11)
1. Формат действия — плоский `{"action": …, …}` (раздел 8).
2. Бизнес-ID (`INV-…`) как синоним `doc_id` в `evidence_ids` **не принимаются**: наборы доказательств
   однозначны (`doc_id` / `row_id` / `fx_id`), формат ID указан в первом наблюдении.
3. `submit` в `inspection_steps` не входит, шаги с ошибкой входят.
4. Функция оценки — `grader.score_submission(truth, submission, inspection_steps)` (раздел 7).
5. Находка B при перекрёстной проверке (исправлена в `generator/attacks.py`): в атаке
   `one_finding_for_two_issues` пара ошибок бралась как `issues[0], issues[1]`. У сценария
   `cutoff_fx_boundary_invoice` обе записи относятся к одному счёту, и «склейка» оказывалась верным выводом:
   ожидание (n−2, 1, 2) было неверным, верно (n−1, 0, 1). Теперь берётся первая пара с разными `affected_ids`.
6. Находка повторной проверки B (20.11, исправлена в `env/`): действие-строка с целым длиннее 4300 цифр
   роняло `step` (ValueError Python 3.11), а submit больше 256 КБ (например, explanation в 300 000 символов)
   давал `action_too_large` без завершения эпизода вместо −1. Теперь см. раздел 8 «Любой submit завершает эпизод».
   Уточнение после ревью кода: порог для submit поднят до 1 МБ, иначе валидный ответ с эмодзи/кириллицей в
   explanation получал −1 (ASCII-экранирование раздувает символ до 12 байт); причина неверной структуры
   больше не читается из ответа агента (её нельзя подделать ключом в findings).
