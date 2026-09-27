# Final measured results

Selected model: lightgbm_31; MIT license; threshold 0.750.
240 trees, 14,640 nodes; conservative tree-parameter bound 468,480, below 8 billion.

| Metric | Held-out evaluation |
|---|---:|
| Macro per-S1 F0.5 (challenge metric) | 0.891918 |
| Micro F0.5 (diagnostic only) | 0.930619 |
| Pooled pair precision | 0.980943 |
| Pooled pair recall | 0.772163 |
| Pair candidate recall | 0.862289 |
| Zero-match S1 accuracy | 0.962121 |

Sample: 12,000 S1s; fit 7,265, tune 2,376, held-out 2,359.
F0.5 is averaged over individual S1 entities, with 1 for a correctly empty singleton and 0 for a singleton false merge.
Model/threshold/retrieval decisions use tuning data. Held-out results are reporting only; this split was also evaluated during earlier development, so it is not a fresh blind benchmark.

Official validator: PASS, ID existence check enabled: True.
Strict streaming validation: PASS. Both outputs have 1,732,544 S1 rows.
Candidate pairs: 40,106,707; predicted pairs: 5,134,678; empty matches: 142,107.
Average candidates/S1: 23.149027; maximum: 24.
Candidates are precisely those passed to the model; matching lists are subsets. SHA-256 hashes bind validation to these files.

Earlier pooled scores and long-form output are superseded. Local backups remain under reports/history/pre_compliance and output/history.
No test score is claimed; only the portal can score hidden test labels.
