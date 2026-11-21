# Jury scorecard (emulation)

Case.md section 10 translated into code one to one (`tests/jury/scorecard.py`, stage 7 of `python check.py`).
Closed set emulated by `generator.holdout`: 20 salted episodes with new parameter combinations (5 clean, 5 with interacting errors) and their permuted + renamed pairs. The jury's real seeds are unknown, so 20/20 here is necessary, not sufficient.

| block | passed | points |
|---|---|---|
| Контракт и изоляция | 20/20 | 25.00 / 25 |
| Данные и эталон | 20/20 | 25.00 / 25 |
| Надёжность награды | 20/20 | 25.00 / 25 |
| Воспроизводимость и обобщение | 20/20 | 15.00 / 15 |
| Инженерия и объяснение (самопроверка наличия) | 10/10 | 10.00 / 10 |
| **Итого (оценка снизу вверх)** | | **100.00 / 100** |

## Контракт и изоляция

| check | result | what is verified |
|---|---|---|
| C01 | PASS | first observation: task, manifest, policy, budget |
| C02 | PASS | step -> (observation, reward, terminated, truncated, info); observation = result/error/budget |
| C03 | PASS | no seed/paths/truth in reset output (0 hits) |
| C04 | PASS | 258 ledger rows in 6 pages of <= 50, file order, limit 500 -> 50 |
| C05 | PASS | same cursor -> same page; bound to table/episode; deterministic |
| C06 | PASS | get_record/get_document return exactly the public record object (jury envelope) |
| C07 | PASS | exact decimal arithmetic and rounding (7/7) |
| C08 | PASS | code, power towers, names and exponent literals rejected; each step used |
| C09 | PASS | no shell/SQL/FS: bounded unknown_action, step used, input not echoed |
| C10 | PASS | unknown IDs/tables: bounded errors, steps used, IDs not echoed |
| C11 | PASS | 16 malformed inputs -> bounded catalogue errors, no exception, each step used |
| C12 | PASS | submit -> {accepted: true}, terminated=true |
| C13 | PASS | 100 steps incl. submit; step 100 -> truncated, empty final answer scored (clean book: 0.8) |
| C14 | PASS | time limit -> truncated, action not executed |
| C15 | PASS | after the end: defined error; a second submit is not scored |
| C16 | PASS | 10 exploration steps: reward 0, empty info, no hints |
| C17 | PASS | train: terminal reward to the policy; eval: 0 and {} — result only in the grader log |
| C18 | PASS | wrong structure (even with correct findings inside) -> -1, error in the grader log |
| C19 | PASS | 531 transitions of 4 agent policies on 6 closed episodes: 0 leaks |
| C20 | PASS | reset isolates episodes (A->B->A identical); env server never loads grader code |

## Данные и эталон

| check | result | what is verified |
|---|---|---|
| ep-TQ3U2MFGS5 (clean) | PASS | 0 issues re-derived with evidence; books balance before and after; 409 rows |
| ep-29ZTDSSPJT (clean) | PASS | 0 issues re-derived with evidence; books balance before and after; 337 rows |
| ep-BQT3JZPMKG (clean) | PASS | 0 issues re-derived with evidence; books balance before and after; 284 rows |
| ep-QDS5JVZJTU (clean) | PASS | 0 issues re-derived with evidence; books balance before and after; 224 rows |
| ep-2ERTJDPP8W (clean) | PASS | 0 issues re-derived with evidence; books balance before and after; 711 rows |
| ep-HJ732WN98H (interacting) | PASS | 5 issues re-derived with evidence; books balance before and after; 258 rows |
| ep-X2HHPCC8MK (interacting) | PASS | 11 issues re-derived with evidence; books balance before and after; 1364 rows |
| ep-JGM6XVN2HZ (interacting) | PASS | 9 issues re-derived with evidence; books balance before and after; 279 rows |
| ep-VKECDDHZ69 (interacting) | PASS | 11 issues re-derived with evidence; books balance before and after; 287 rows |
| ep-N85UJST2TC (interacting) | PASS | 12 issues re-derived with evidence; books balance before and after; 225 rows |
| ep-MPQQGE4JY6 (mixed) | PASS | 6 issues re-derived with evidence; books balance before and after; 330 rows |
| ep-DPXQWCHGJ9 (mixed) | PASS | 4 issues re-derived with evidence; books balance before and after; 212 rows |
| ep-UHMXQ8NF2Q (mixed) | PASS | 7 issues re-derived with evidence; books balance before and after; 1499 rows |
| ep-HV2VVZWEQG (mixed) | PASS | 3 issues re-derived with evidence; books balance before and after; 307 rows |
| ep-SM67D7V8TE (mixed) | PASS | 4 issues re-derived with evidence; books balance before and after; 494 rows |
| ep-CBRWSZKUUJ (mixed) | PASS | 5 issues re-derived with evidence; books balance before and after; 1475 rows |
| ep-WH9FCZSWDR (mixed) | PASS | 3 issues re-derived with evidence; books balance before and after; 586 rows |
| ep-6VJ2EYPD5D (mixed) | PASS | 4 issues re-derived with evidence; books balance before and after; 1080 rows |
| ep-BQ2MJ6GQYV (mixed) | PASS | 3 issues re-derived with evidence; books balance before and after; 609 rows |
| ep-BEWRJNF9NV (mixed) | PASS | 7 issues re-derived with evidence; books balance before and after; 1490 rows |

## Надёжность награды

| check | result | what is verified |
|---|---|---|
| R01 | PASS | oracle answer -> F1 1, reward 1 (maximum reachable): 20/20 episodes |
| R02 | PASS | order of findings does not matter: 15/15 episodes |
| R03 | PASS | affected/evidence IDs normalised as sets (order, repeats): 15/15 episodes |
| R04 | PASS | duplicate findings -> FP, never a gain: 15/15 episodes |
| R05 | PASS | one finding cannot close two issues: 15/15 episodes |
| R06 | PASS | maximum one-to-one matching (greedy would give 1 of 2), independent of order |
| R07 | PASS | clean books: silence -> 1.0: 5/5 episodes |
| R08 | PASS | clean books: any false accusation -> -1 (silence beats it): 5/5 episodes |
| R09 | PASS | books with errors: full answer with evidence beats silence: 15/15 episodes |
| R10 | PASS | flag-everything (30 findings) -> <= -0.9: 20/20 episodes |
| R11 | PASS | guessed affected IDs -> no TP: 15/15 episodes |
| R12 | PASS | fake or unrelated evidence -> no TP: 15/15 episodes |
| R13 | PASS | incomplete evidence set -> no TP: 15/15 episodes |
| R14 | PASS | right amount in another currency -> no TP: 15/15 episodes |
| R15 | PASS | amount tolerance is exact: +tol TP, +tol+1 no TP: 15/15 episodes |
| R16 | PASS | wrong type -> no TP: 15/15 episodes |
| R17 | PASS | step cost exact (0.002 per inspection step, 0..99), strictly decreasing, at most 0.198: 20/20 episodes |
| R18 | PASS | wrong structure (31 findings, 11 evidence, 1001 chars, amount not int) -> -1; wrong content -> FP: 20/20 episodes |
| R19 | PASS | explanation text never changes the reward (checked on answers scoring < 1): 20/20 episodes |
| R20 | PASS | author's attack suite (36 attacks/episode) matches the spec: 20/20 episodes |

## Воспроизводимость и обобщение

| check | result | what is verified |
|---|---|---|
| pair 01: ep-TQ3U2MFGS5 -> ep-9ECM67QB83 | PASS | regen byte-identical; replay identical (44 steps, 2 processes); pair truth = renamed truth (0 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 02: ep-29ZTDSSPJT -> ep-T2WJSHSFKQ | PASS | regen byte-identical; replay identical (55 steps, 2 processes); pair truth = renamed truth (0 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 03: ep-BQT3JZPMKG -> ep-JG2JHEC833 | PASS | regen byte-identical; replay identical (86 steps, 2 processes); pair truth = renamed truth (0 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 04: ep-QDS5JVZJTU -> ep-6Q5PCN8HJC | PASS | regen byte-identical; replay identical (90 steps, 2 processes); pair truth = renamed truth (0 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 05: ep-2ERTJDPP8W -> ep-NX43UJ9H8G | PASS | regen byte-identical; replay identical (17 steps, 2 processes); pair truth = renamed truth (0 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 06: ep-HJ732WN98H -> ep-X5TPGPZWXA | PASS | regen byte-identical; replay identical (100 steps, 2 processes); pair truth = renamed truth (5 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 07: ep-X2HHPCC8MK -> ep-VBPKT9WR52 | PASS | regen byte-identical; replay identical (80 steps, 2 processes); pair truth = renamed truth (11 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 08: ep-JGM6XVN2HZ -> ep-ZGGXB9JTQV | PASS | regen byte-identical; replay identical (100 steps, 2 processes); pair truth = renamed truth (9 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 09: ep-VKECDDHZ69 -> ep-AU4SPCVSRT | PASS | regen byte-identical; replay identical (56 steps, 2 processes); pair truth = renamed truth (11 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 10: ep-N85UJST2TC -> ep-HQB4PXW659 | PASS | regen byte-identical; replay identical (38 steps, 2 processes); pair truth = renamed truth (12 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 11: ep-MPQQGE4JY6 -> ep-49TYBRJTRU | PASS | regen byte-identical; replay identical (73 steps, 2 processes); pair truth = renamed truth (6 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 12: ep-DPXQWCHGJ9 -> ep-KASUUCVDG8 | PASS | regen byte-identical; replay identical (68 steps, 2 processes); pair truth = renamed truth (4 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 13: ep-UHMXQ8NF2Q -> ep-GXZMNBW7T2 | PASS | regen byte-identical; replay identical (80 steps, 2 processes); pair truth = renamed truth (7 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 14: ep-HV2VVZWEQG -> ep-GFJX8XP5NG | PASS | regen byte-identical; replay identical (7 steps, 2 processes); pair truth = renamed truth (3 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 15: ep-SM67D7V8TE -> ep-22CRHZT4TT | PASS | regen byte-identical; replay identical (100 steps, 2 processes); pair truth = renamed truth (4 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 16: ep-CBRWSZKUUJ -> ep-NMTVXT8TFN | PASS | regen byte-identical; replay identical (96 steps, 2 processes); pair truth = renamed truth (5 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 17: ep-WH9FCZSWDR -> ep-9UTXFQSJAW | PASS | regen byte-identical; replay identical (29 steps, 2 processes); pair truth = renamed truth (3 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 18: ep-6VJ2EYPD5D -> ep-R93YMQT3VA | PASS | regen byte-identical; replay identical (95 steps, 2 processes); pair truth = renamed truth (4 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 19: ep-BQ2MJ6GQYV -> ep-BQ39KB6WYZ | PASS | regen byte-identical; replay identical (42 steps, 2 processes); pair truth = renamed truth (3 issues); renamed oracle = 1.0; 4 policies' rewards equal |
| pair 20: ep-BEWRJNF9NV -> ep-ACNV286CZY | PASS | regen byte-identical; replay identical (52 steps, 2 processes); pair truth = renamed truth (7 issues); renamed oracle = 1.0; 4 policies' rewards equal |

## Инженерия и объяснение (0/1/2, самопроверка)

| criterion | score | evidence |
|---|---|---|
| one command | 2 | python check.py = make check = docker run --network none |
| config versions | 2 | 21 manifests with schema/generator version; split indexes with seed+config digest |
| trajectory logs | 2 | trajectories/trajectories.jsonl with the required fields (leak scan: stage 5) |
| measured failure analysis | 2 | reports/baselines.md (P/R/F1/return/steps/time + FP reasons by count), reports/holdout.md |
| honest limitations | 2 | docs/design.md: 10 stated limitations |
