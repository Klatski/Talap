# Общий контракт A ↔ B

> Статус: **черновик от A** (19.11). Разделы 1–6 реализованы генератором и проверены тестами.
> Разделы 7–9 — зона B (среда/оценщик): здесь предложение, B может менять, но тогда
> правим этот файл вместе. Машиночитаемые схемы — `schemas/*.schema.json`.

## 1. Эпизод на диске

```
episodes/<split>/<episode_id>/
  public/   manifest.json  policy.json  ledger.csv  invoices.jsonl  payments.jsonl  fx.csv
  private/  truth.json  provenance.json          ← только оценщик, агенту недоступно
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

## 7. findings.json (submit) — предложение для B
Схема: `schemas/findings.schema.json` (строгая: лишние поля, неверные типы, отрицательные суммы,
неизвестный `type` → неверная структура → награда −1).
```json
{"episode_id": "ep-…", "findings": [{"type": "…", "affected_ids": ["…"], "claimed_amount_minor": 0,
  "currency": "KZT", "evidence_ids": ["…"], "explanation": "…"}]}
```
Лимиты: ≤30 выводов, ≤10 `evidence_ids`, `explanation` ≤1000 символов, `affected_ids` 1..10 (наш выбор).
Предложение по сопоставлению: нормализация `affected_ids` → множество; TP если совпали type,
множество affected_ids, валюта, `|claimed − expected| ≤ amount_tolerance_minor` и
`set(evidence_ids) ⊇ один из evidence_sets`; паросочетание один-к-одному (венгерский алгоритм
или полный перебор при ≤30×12), тай-брейк по отсортированным ID. Чужой `episode_id` → неверная структура.

## 8. Действия и наблюдения — предложение для B
```json
{"action": "list_records", "table": "ledger", "cursor": null, "limit": 50}
{"action": "get_record", "table": "invoices", "id": "INV-…"}
{"action": "get_document", "doc_id": "D-…"}
{"action": "calculator", "expression": "123.45 * 498.12"}
{"action": "submit", "findings": { … findings.json … }}
```
Таблицы: `ledger` (ключ `row_id`), `invoices` (`invoice_id`), `payments` (`payment_id`),
`fx` (ключ `FX:<src>:<dst>:<date>`; среде стоит добавлять поле `fx_id` в каждую запись fx).
`get_document(doc_id)` — счёт или платёж по `doc_id`.

## 9. Интерфейс политики (адаптеры базовых политик)
```python
class Policy:
    name: str
    def reset(self, observation: dict, info: dict) -> None: ...
    def act(self, observation: dict) -> dict: ...   # одно действие из раздела 8
```
Oracle реализует этот интерфейс (одним `submit`), но создаётся на стороне оценщика из truth.

## Открытые вопросы к B
1. Формат действия: плоский `{"action": …, …}` как выше или `{"name": …, "args": {…}}`?
2. Принимать ли в `evidence_ids` бизнес-ID (`INV-…`) как синоним `doc_id`? Сейчас наборы — только `doc_id`/`row_id`/`FX:`.
3. Считается ли submit в `inspection_steps` (предлагаю: нет) и шаги с ошибкой (предлагаю: да).
