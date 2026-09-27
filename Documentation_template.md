# Hacksmiths — Business Entity Resolution

## 1. Problem Understanding
Resolve each reference S1 business to zero or more S2/S3 records. Optimize precision-focused pairwise F0.5; do not force matches.

## 2. Dataset Description
Only supplied TSV data is used. Full-file counts, missingness, uniqueness, countries, lengths and Unicode scripts are in reports/eda_statistics.json and reports/eda_report.md.
Training S1 rows: 2,206,821. Test S1 rows: 1,732,544. Test ground truth is unavailable.

## 3. Data Preprocessing
Preserved existing config and loading interfaces. NFKC, casefold, whitespace and punctuation normalization retain Unicode letters, marks and numbers. Original text remains in unchanged source TSV files.
No unjustified repeated-character collapse or destructive suffix removal is applied to primary text.

## 4. Multilingual Handling
Raw, normalized Unicode, compact, token and auxiliary local transliterated forms are available. Combining marks are preserved.
Scripts are detected from Unicode character names; mixed-script records are counted. Transliteration is approximate and can collide.
Countries are open-set features, never a hard retrieval filter. No external registries, geocoding, APIs, embeddings or business-data augmentation are used.

## 5. Candidate Generation / Blocking
Disk-backed hashed postings cover all target records. Keys include exact/core transliterated names, name tokens and prefixes, full addresses, address-token pairs and number/token combinations.
Common postings over 1200 targets are skipped. Up to 80 targets per source (160 total) are pre-ranked with name/address similarity; the top 24 become candidates, without one-to-one constraints.
Global and source-balanced preselection, the exact-name comparator and combined blocker are measured on the same tuning S1s in experiments/blocking_results.csv.
No character TF-IDF experiment was run; the bounded posting index is the implemented retrieval method.

## 6. Feature Engineering
30 features cover Unicode exact/character/partial/token similarities, Jaccard, relative lengths, character bigram Dice, missingness, auxiliary transliteration, country equality, scripts, source and numeric agreement/conflict.
First-number and last-long-number agreement are explicit heuristics; they are not claimed to be reliable house/postcode parsers for every country.
Feature order is stored with the serialized model.

## 7. Model Architecture
Selected hist_gradient_boosting against standardized logistic regression. Histogram gradient boosting is a small locally trained tree model with no pretrained parameters.
Models use scikit-learn (BSD-3-Clause); RapidFuzz is MIT and text-unidecode metadata reports Artistic License. The exact challenge license/parameter restrictions were not supplied and compliance cannot be certified.

## 8. Training / Validation
Uniform reservoir sample of 12,000 S1 records, including zero-match entities. Labels come only from provided training truth.
Truth-connected sampled entities are grouped into fit (7265), tuning (2376), and final holdout (2359) with seed 2026.
Retrieval runs against all targets; missed truth pairs count as false negatives. Singleton accuracy means fraction of zero-match S1s correctly left empty.
No artificial positive pairs are injected into candidates. Source-balanced preselection was chosen on tuning F0.5; baseline and selected retrieval were both evaluated on holdout, which was never used for selection. The selected model is frozen at its evaluated fit rather than refitted after threshold selection.

## 9. Experiments
See experiments/experiment_log.csv, model_results.csv and blocking_results.csv. Same-split comparisons cover logistic regression versus histogram gradient boosting, basic 18-text-feature versus full 30-feature boosting, exact-name versus multi-key retrieval, and fixed versus tuned thresholds.
Original notebook history is preserved. It reported 55.04% S2 and 56.48% S3 recall on a different 100,000-positive-S1 sample; those are historical, not directly comparable to the new validation cohort.
No existing trained Daya model was present in the inspected checkout; model.py and models/ were empty.

## 10. Threshold Selection
Threshold 0.800 selected by maximum tuning F0.5, then precision on ties, over a recorded grid from 0.05 to 0.999.
Held-out evaluation results:

| Metric | Held-out evaluation |
|---|---:|
| candidate_recall | 0.862289 |
| precision | 0.986011 |
| recall | 0.753965 |
| F0.5 | 0.928838 |
| singleton_accuracy | 0.962121 |

## 11. Error Analysis
All holdout false positives and false negatives are in experiments/error_analysis.csv, including blocking losses.
reports/error_analysis.md categorizes script variation, missing addresses, numeric conflicts, similar names/addresses and ambiguous text using disclosed heuristics.
The holdout findings were not used to retune the selected model. Changes based on them require fresh validation.

## 12. Final Approach
Raw TSV → Unicode and auxiliary representations → full-corpus bounded postings → pair features → selected classifier → tuned threshold → grouped matches and long-form candidate pairs.
All scored test pairs appear in candidate_pairs.tsv. Every test S1 appears exactly once, with an empty TSV field for no matches. No ground-truth labels are used in test inference.
Local validation: PASS; 1,732,544 S1 rows, 40,106,707 candidate pairs and 4,950,360 predicted pairs.

## 13. Limitations
Official validation script, exact candidate-file schema and detailed licensing/parameter limits were absent. A clearly labeled local validator checks the user-specified invariants, but cannot certify official acceptance.
Training uses a reproducible sample rather than all labels. High-frequency names, severe typos and transliteration failures can be excluded by bounded retrieval.
France is present only in test; no France-specific labeled validation score is claimed. Random grouped holdout does not fully measure this country shift.
Pairwise F0.5 is implemented from the user brief; no unseen official scoring implementation is assumed.
The archive is prepared for review; no external upload was performed.

## 14. Conclusion
The missing supervised stages are implemented and measured, and complete test outputs are locally validated. Official acceptance remains pending the supplied challenge tools/rules.
