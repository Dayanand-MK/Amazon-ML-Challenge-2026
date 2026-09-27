# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Hacksmiths
**Team Members:** Daya — team lead, ML pipeline and integration; James — EDA and error analysis; Vijay — normalization and blocking; Dhanush — features, evaluation and threshold optimization.
**Submission Date:** 2026-09-27

## 1. Executive Summary

We resolve each S1 business to zero, one or multiple S2/S3 records using a disk-backed candidate index and a locally trained MIT-licensed LightGBM classifier. Unicode-preserving text processing, auxiliary local transliteration and bounded retrieval support multilingual data while limiting final candidates to 24 per S1. Threshold selection uses the challenge's macro per-S1 F0.5, including singleton credit.

## 2. Methodology

### 2.1 Problem Analysis

Only the organizers' unchanged TSV files are used. Train S1 contains 2,206,821 rows; test S1 contains 1,732,544. EDA covers full-file counts, missingness, duplicate IDs/rows, countries, field lengths and Unicode scripts in reports/eda_statistics.json and eda_report.md.
Training contains US and India; test additionally includes France. Countries remain open strings, with no country hard filter or fixed one-hot country vocabulary. Every test S1 is emitted, including France and empty-match entities.

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier.
**Core Contribution:** Bounded disk-backed multi-key retrieval with Unicode-aware and transliterated similarity features, followed by macro-calibrated matching.

NFKC, casefold and whitespace/punctuation cleanup preserve Unicode letters, combining marks and numbers. Original TSV text is unchanged. Compact/token/transliterated forms are auxiliary; transliteration can collide and does not replace the original representation. No external business databases, registries, APIs, geocoding, remote embeddings or test labels are used. text-unidecode is a generic local character mapping dependency.

## 3. Candidate Generation (Blocking)

- **Blocking keys:** Exact/core transliterated names, name tokens and prefixes, full addresses, address-token pairs, and number/token combinations. Common legal/address words are discounted only in retrieval keys.
- **Search:** All training or test S2/S3 targets as appropriate. Hashed sorted posting arrays and record offsets are memory-mapped. Postings over 1200 are skipped; 80 targets per source (160 total) are preselected by key overlap, then pre-ranked by name/address similarity and key hits. The best 24 are fed into the model, without a one-to-one constraint.
- **Final test candidates:** 40,106,707; average 23.149027/S1; maximum 24.
- **Recall measurement:** True matches can be lost. Held-out candidate pair recall is 0.862289; all blocking losses count as false negatives. No gold positives are injected. Exact-name and multi-key comparisons, source-level recall and source-balanced preselection are recorded in experiments/.

The optional src/rescore.py path reuses an already generated final candidate set only after input/code/configuration provenance checks and a fresh-retrieval audit. It applies the same feature and model scoring as fresh inference; normal infer reproduces the outputs without this cache.

candidate_pairs.tsv exports the final candidate list actually scored by the ML model, not the larger intermediate preselection. It has exactly one row per test S1 and exact headers source1_entity_id and candidate_entity_ids. Lists are comma-separated without quotes; empty lists are empty fields.

## 4. Matching Model

**Features:** 30 ordered features: Unicode name/address exact, ratio, partial, token-sort, token-set, Jaccard, length ratio, bigram Dice and missingness; auxiliary transliteration similarities; country equality/missingness; script equality; target source; number overlap/conflict; first-number/last-long-number/name-number agreement; and a name-address interaction. Numeric features are heuristics, not country-specific postcode parsers.

**Model type:** lightgbm_31, trained locally from scratch with LightGBM 4.6.0 (MIT). The original final model artifact is released under the project MIT LICENSE. No pretrained model is used. The selected model has 240 trees and 14,640 nodes. A conservative bound of 32 stored scalar parameters per tree node gives 468,480, far below 8 billion; it is a tree model, not a neural network.

**Training:** Seed 2026; uniform reservoir sample of 12,000 S1 entities including zero-match rows. Sampled S1s sharing a truth target are grouped before 60/20/20 fit/tune/holdout splitting: 7265/2376/2359. Retrieval searches the full target corpus. No target supplies positive labels to multiple splits. Candidate negatives use only provided training truth.

**Selection:** Two LightGBM capacities and a text-feature-only LightGBM ablation are compared; standardized logistic regression is a diagnostic baseline, ineligible as the final model. Exact candidate/feature/model settings are in source and experiments. Source-balanced retrieval is adopted only if tuning macro F0.5 improves. The evaluated fit model is retained without refitting after threshold selection.

**Threshold:** 0.750, selected on tuning macro F0.5 from 0.05 through 0.975 in 0.025 increments plus 0.99, 0.995 and 0.999. Pooled precision and threshold break ties. For each nonempty-truth entity, F0.5 = 1.25 TP / (predicted_count + 0.25 true_count). Empty-truth entities receive 1 for an empty prediction and 0 otherwise. Scores are averaged across every S1, including those with no candidates. Micro scores are explicitly diagnostic.

## 5. Results & Error Analysis

| Metric | Held-out evaluation |
|---|---:|
| Macro per-S1 F0.5 (challenge metric) | 0.891918 |
| Micro F0.5 (diagnostic only) | 0.930619 |
| Pooled pair precision | 0.980943 |
| Pooled pair recall | 0.772163 |
| Pair candidate recall | 0.862289 |
| Zero-match S1 accuracy | 0.962121 |

Tuning macro F0.5: 0.888748. These are training holdout measurements, not leaderboard scores. Holdout is used for reporting rather than selection; prior development also evaluated this same split. France has no labeled validation records, so no France-specific accuracy claim is made.

Common failure categories include blocking misses, missing addresses, cross-script variation, numeric conflicts, similar names with conflicting addresses and ambiguous text. Counts and full holdout FP/FN examples, including original Unicode fields and blocked-out truth, are in reports/error_analysis.md and experiments/error_analysis.csv. These labels are diagnostic heuristics rather than verified causes. Blocking remains a recall ceiling; high-frequency names and severe text corruption can be missed.

## 6. Conclusion

The pipeline produces grouped candidate and matching TSVs for every test S1 with bounded candidate volume. Macro scoring rewards correctly empty singletons while penalizing false merges, and the selected model meets the MIT license and parameter-size requirements. Official validation is PASS; stricter full ID/subset validation is PASS.

## Appendix

### A. Code Artefacts

The submission ZIP contains output/matching_results.tsv and output/candidate_pairs.tsv, this completed official-template document, and code/business_entity_resolution/ with src/, README.md, pinned requirements, run_pipeline.py, tests/, the unchanged official validator and the final fitted model. README.md gives setup, data placement, full regeneration, model-only inference, validation and packaging commands. Datasets and rebuildable caches are omitted. Output hashes are in reports/submission_validation.json and ZIP manifest.json; the package is CRC-checked.

### B. Additional Results

Both output files have 1,732,544 rows. There are 5,134,678 matches and 142,107 empty predictions. Full EDA, tuning model/threshold/blocking comparisons, country results and error analysis are included. Historical notebook cells are preserved and marked separately from current report-reading cells. Earlier pooled F0.5 reports are superseded, not treated as macro scores. Final predictions have not been uploaded to the challenge portal by this pipeline.
