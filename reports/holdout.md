# Closed-check emulation: 20 salted episodes (5 clean, 5 interacting, 10 mixed)

env 1.0.0, grader 1.0.0, mode `eval`, splits holdout (salted, new configs), 20 episodes per policy, isolated processes: True, policy sandbox: separate restricted process (env/sandbox.py); oracle grader-side.
Values are means over episodes; `micro F1` pools TP/FP/FN over all episodes. `return` is the shared reward; `return_exp` is the experimental one (truncation = -1), shown separately. By the spec P and R are 0 on a clean book with an empty answer (zero denominators) while F1 = 1, so mean P/R stay below 1 even for the oracle.

| policy | P | R | F1 | micro F1 | return | return_exp | return on clean | steps | time, s (mean / max) | TP | FP | FN | truncated | invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.000 | 0.000 | 0.100 | 0.000 | -0.814 | -0.904 | -0.257 | 36.6 | 0.118 / 0.194 | 0 | 47 | 94 | 1 | 1 |
| reconcile | 0.154 | 0.452 | 0.223 | 0.226 | -0.584 | -0.584 | -1.000 | 19.9 | 0.134 / 0.253 | 55 | 337 | 39 | 0 | 0 |
| flag_all | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | -1.000 | -1.000 | 3.7 | 0.072 / 0.165 | 0 | 600 | 94 | 0 | 0 |
| oracle | 0.750 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.0 | 0.059 / 0.254 | 94 | 0 | 0 | 0 | 0 |
| auditor | 0.750 | 0.750 | 1.000 | 1.000 | 0.962 | 0.962 | 0.972 | 19.9 | 0.135 / 0.253 | 94 | 0 | 0 | 0 | 0 |

## Failure analysis (measured)

| policy | FP reasons | FP on correct distractors | FN by type |
|---|---|---|---|
| random | no_such_issue: 46, wrong_type+currency+amount+evidence: 1 | batch_payment: 1, boundary_document: 1, on_account_payment: 1, overpayment: 3, partial_payment: 8, partial_shipments: 1, posting_lag: 7, recurring_invoices: 4, split_payments: 4 | cutoff_error: 23, duplicate_invoice: 13, fx_rate_error: 25, unapplied_payment: 22 |
| reconcile | no_such_issue: 323, wrong_affected_ids+evidence: 2, wrong_amount+evidence: 7, wrong_type+currency+amount: 1, wrong_type+currency+amount+evidence: 4 | batch_payment: 2, boundary_document: 90, on_account_payment: 2, overpayment: 11, partial_payment: 18, partial_shipments: 11, posting_lag: 69, recurring_invoices: 76, split_payments: 15 | cutoff_error: 7, fx_rate_error: 21, unapplied_payment: 11 |
| flag_all | no_such_issue: 583, wrong_affected_ids+evidence: 2, wrong_amount+evidence: 1, wrong_currency+amount+evidence: 2, wrong_evidence: 4, wrong_type+currency+amount+evidence: 2, wrong_type+evidence: 6 | batch_payment: 14, boundary_document: 26, on_account_payment: 2, overpayment: 18, partial_payment: 57, partial_shipments: 2, posting_lag: 24, recurring_invoices: 10, split_payments: 55 | cutoff_error: 26, duplicate_invoice: 15, fx_rate_error: 29, unapplied_payment: 24 |
| oracle | - | - | - |
| auditor | - | - | - |

## Paired episodes (rows permuted, all IDs renamed)

| policy | return (original) | return (pair) | equal per pair |
|---|---|---|---|
| random | -0.8143 | -1.0000 | 18/20 |
| reconcile | -0.5842 | -0.5842 | 20/20 |
| flag_all | -1.0000 | -1.0000 | 20/20 |
| oracle | 1.0000 | 1.0000 | 20/20 |
| auditor | 0.9621 | 0.9621 | 20/20 |

Episode kinds: 5 clean, 5 interacting, 10 mixed. Salt and seeds are not published; this is an emulation of the jury's closed run, not the real one.
