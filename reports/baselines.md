# Baseline diagnostics (train + dev, eval mode)

env 1.0.0, grader 1.0.0, mode `eval`, splits train, dev, 20 episodes per policy, isolated processes: True, policy sandbox: separate restricted process (env/sandbox.py); oracle grader-side.
Values are means over episodes; `micro F1` pools TP/FP/FN over all episodes. `return` is the shared reward; `return_exp` is the experimental one (truncation = -1), shown separately. By the spec P and R are 0 on a clean book with an empty answer (zero denominators) while F1 = 1, so mean P/R stay below 1 even for the oracle.

| policy | P | R | F1 | micro F1 | return | return_exp | return on clean | steps | time, s (mean / max) | TP | FP | FN | truncated | invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.000 | 0.000 | 0.100 | 0.000 | -0.816 | -0.906 | -0.078 | 55.0 | 0.146 / 0.266 | 0 | 30 | 87 | 3 | 3 |
| reconcile | 0.140 | 0.488 | 0.207 | 0.201 | -0.622 | -0.622 | -1.000 | 23.4 | 0.191 / 0.300 | 53 | 388 | 34 | 0 | 0 |
| flag_all | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | -1.000 | -1.000 | 4.2 | 0.102 / 0.219 | 0 | 600 | 87 | 0 | 0 |
| oracle | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.0 | 0.080 / 0.346 | 87 | 0 | 0 | 0 | 0 |
| auditor | 0.800 | 0.800 | 1.000 | 1.000 | 0.955 | 0.955 | 0.957 | 23.4 | 0.206 / 0.345 | 87 | 0 | 0 | 0 | 0 |

## Failure analysis (measured)

| policy | FP reasons | FP on correct distractors | FN by type |
|---|---|---|---|
| random | no_such_issue: 25, unknown_evidence: 3, wrong_affected_ids+amount+evidence: 1, wrong_type+currency+amount+evidence: 1 | boundary_document: 2, overpayment: 1, partial_payment: 2, partial_shipments: 2, posting_lag: 3, recurring_invoices: 1, split_payments: 2 | cutoff_error: 18, duplicate_invoice: 11, fx_rate_error: 21, unapplied_payment: 16 |
| reconcile | no_such_issue: 370, wrong_affected_ids: 1, wrong_affected_ids+evidence: 2, wrong_amount+evidence: 4, wrong_type+currency+amount: 4, wrong_type+currency+amount+evidence: 7 | batch_payment: 4, boundary_document: 120, on_account_payment: 4, overpayment: 6, partial_payment: 14, partial_shipments: 8, posting_lag: 67, recurring_invoices: 80, split_payments: 28 | cutoff_error: 9, fx_rate_error: 16, unapplied_payment: 9 |
| flag_all | no_such_issue: 588, wrong_affected_ids+currency+amount+evidence: 1, wrong_affected_ids+evidence: 1, wrong_currency+amount+evidence: 2, wrong_evidence: 3, wrong_type+evidence: 5 | batch_payment: 11, boundary_document: 30, on_account_payment: 2, overpayment: 3, partial_payment: 53, posting_lag: 9, recurring_invoices: 2, split_payments: 76 | cutoff_error: 23, duplicate_invoice: 15, fx_rate_error: 28, unapplied_payment: 21 |
| oracle | - | - | - |
| auditor | - | - | - |
