# Hacksmiths: Business Entity Resolution

Local-only, reproducible S1 → S2/S3 matching over the supplied Amazon ML Challenge data.
Start with `reports/inspection_report.md` for the preserved baseline and `reports/final_results.md` for measured results once the run finishes.

## Setup and execution

Python 3.12 was used. Create a virtual environment and install pinned dependencies:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe run_pipeline.py --stage all --team Hacksmiths
```

Put original data under `dataset/train` and `dataset/test`. The pipeline never changes those files.
Individual stages: `eda`, `index-train`, `train`, `index-test`, `infer`, `validate`, `report`, `package`.
Default fitting uses 12,000 uniformly sampled S1s with all training targets as retrieval distractors. Seed 2026; 60/20/20 truth-component splits.
`--workers 3` controls inference processes. Disk-backed indexes bound memory; allow tens of GB of free disk and substantial runtime for the multi-million-record files.
Keep `cache/` for repeat runs. Index manifests detect input size/mtime changes. Training caches are tied to sample size; remove the derived training cache if changing features/retrieval parameters before rerunning training.
Inference writes partial outputs and per-batch checkpoints; rerunning `infer` resumes the same model/data combination. Final filenames are published only on completion.

## Validate and inspect

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --report reports/submission_validation.json
```

**This validator is project-local, not the missing official Amazon validator.**
Matching headers mirror supplied ground truth: `source1_entity_id`, `matched_entity_ids`; lists are comma-separated, with an empty field for no match.
Chosen long-form candidate headers: `source1_entity_id`, `target_entity_id`. Official schema and detailed license/parameter rules were not provided; confirm them before upload.
Local validation checks exact S1 coverage, valid test target IDs, uniqueness, matching-as-candidate subset and TSV structure.

## Reproduce packaged predictions

The ZIP includes the small fitted model, source and pinned requirements; it excludes datasets and large indexes.
With test data restored to the expected paths, run `index-test`, `infer` and `validate` to reproduce predictions without retraining.
Only load the included pickle from this trusted project; pickle is not a safe format for arbitrary third-party files.

## Implementation

Existing config, loaders, helper interfaces and historical notebooks are preserved. Unicode marks are retained. Transliteration supplies extra features and keys, never replaces raw text.
Production blocking is exposed through `src/blocking.py` and implemented in `src/retrieval.py`; original dataframe joins remain for historical notebook compatibility.
No country hard filter, one-to-one constraint, pretrained model, external business registry or remote matching service is used.
Model and F0.5 threshold come from measured tuning comparisons; a separate holdout is used only for reporting. Candidate losses count as false negatives.
New notebook supplements inspect saved measurements and are explicitly marked unexecuted.
