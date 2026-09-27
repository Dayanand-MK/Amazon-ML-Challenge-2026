# Hacksmiths — Amazon ML Challenge 2026

Local business entity resolution: S1 to zero or more S2/S3 records. See [measured results](reports/final_results.md), [methodology](Documentation_template.md), and [model card](MODEL_CARD.md).

## Setup

Use Python 3.12 (64-bit). From this repository root, or the ZIP's code/business_entity_resolution directory:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Place the organizers' original files here:

```text
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

All input is UTF-8 TSV. No external business data or services are used. Data files remain unchanged.

## Run from data through final ZIP

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe run_pipeline.py --stage all --workers 6 --team Hacksmiths
```

Stages run in order: eda, index-train, train, index-test, infer, validate, report, package. Each is also available separately with --stage. Linux/macOS use .venv/bin/python. Default training samples 12,000 S1s with seed 2026 and a component-grouped 60/20/20 split. The full training target corpus supplies candidates. Model and threshold selection use macro per-S1 F0.5; the held-out split reports performance.

Allow tens of GB free disk and substantial runtime. Full test inference can take an hour or more, depending on memory pressure ; index construction adds time. Six workers were used locally; reduce --workers for lower memory. Official validation with both files and --check-ids can require several GB RAM. Keep cache/ for repeat runs. Input fingerprints and feature/retrieval hashes guard cache reuse. Inference checkpoints resume only with the same data/model/code; preserve incompatible partial outputs before starting a changed pipeline.

## Reproduce using the included final model

```powershell
.venv/Scripts/python.exe run_pipeline.py --stage index-test
.venv/Scripts/python.exe run_pipeline.py --stage infer --workers 6
.venv/Scripts/python.exe run_pipeline.py --stage validate
.venv/Scripts/python.exe run_pipeline.py --stage report
.venv/Scripts/python.exe run_pipeline.py --stage package --team Hacksmiths
```

Only test data is needed for index-test/infer/validate. Packaged reports contain training measurements for report generation. The trusted final model pickle includes its threshold and feature order. Do not load arbitrary third-party pickles.

## Output and validation

Both files have one row per test S1 in input order, including France and entities with no candidates/matches:

- output/matching_results.tsv: source1_entity_id TAB matched_entity_ids
- output/candidate_pairs.tsv: source1_entity_id TAB candidate_entity_ids

Lists are comma-separated, unquoted, duplicate-free, and empty when appropriate. Candidate lists are exactly those scored by the final model. Matches must be subsets of candidates. Every target must exist in test Source 2/3.

The validate stage first runs strict streaming checks, then runs the **unchanged official validator** with ID checking enabled:

```powershell
.venv/Scripts/python.exe utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

The pipeline records its result, validator SHA-256 and output hashes under reports/. Packaging requires official PASS, no warnings and unchanged output hashes. The ZIP is CRC-checked and contains a manifest. The challenge portal receives matching_results.tsv; the final submission is Hacksmiths_submission.zip. These files remain local; this code does not upload to the portal.

## GitHub contents and license

Git includes source, notebooks, measured reports, the small final model, licenses, tests and README guides for every working folder. Multi-gigabyte datasets, indexes, generated TSVs and the submission ZIP stay local and can be regenerated. The final model and original project code use MIT; LightGBM is MIT-licensed. See THIRD_PARTY_NOTICES.md for other materials and dependencies.

Original notebook cells and legacy helper interfaces are preserved. Production retrieval is src/retrieval.py; legacy dataframe blocking is not used by run_pipeline.py and its country hard filter has also been removed. Current notebook supplements read current measurements. Pre-correction pooled F0.5 results are historical and must not be confused with the macro challenge metric.

## Optional candidate-cache rescoring

src/rescore.py accelerates a model-only update by reusing a previous completed final candidate set. It verifies source fingerprints, blocking/normalization/feature hashes and retrieval limits, checks target IDs, and audits a deterministic spread against fresh retrieval. It changes neither candidate membership nor final scoring. The normal infer stage remains the standalone reproduction route from supplied data and does not require earlier outputs. This accelerator accepts only canonical numeric target IDs; normal inference supports opaque IDs.
