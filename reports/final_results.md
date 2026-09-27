# Final measured results

Model: hist_gradient_boosting. Features: 30. Threshold: 0.800.

| Metric | Held-out evaluation |
|---|---:|
| candidate_recall | 0.862289 |
| precision | 0.986011 |
| recall | 0.753965 |
| F0.5 | 0.928838 |
| singleton_accuracy | 0.962121 |

Sampled S1: 12000; fit 7265; tune 2376; held-out evaluation 2359.
Sampled S1s sharing a truth target are grouped before splitting. Candidates search the full training target corpus.
The model is retained exactly as evaluated, with no post-threshold refit. No test labels or external business data were used.

Local integrity validator: PASS. Official validator: unavailable and NOT run.
Test S1 rows: 1,732,544; candidate pairs: 40,106,707; predicted pairs: 4,950,360; empty lists: 152,975.

Candidate generation uses auxiliary transliteration, exact/core names, token/prefix postings and address/number keys across countries.
Postings larger than 1200 are skipped, 80 targets per source (160 total) are pre-ranked, and the best 24 reach the matcher per S1. These limits are included in the reported candidate recall.

Archive: Hacksmiths_submission.zip. Official schema and model/license constraints remain unverified because their documents were not supplied.
