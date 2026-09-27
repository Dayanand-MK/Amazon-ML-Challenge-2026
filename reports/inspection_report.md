# Initial project inspection

Inspected 2026-09-27 before code changes.

- Existing nonempty source: config.py, data_loader.py, normalization.py and blocking.py. Existing `_init_.py` was misspelled; a proper `__init__.py` is added.
- Notebooks 01–03 contained code and saved outputs. Notebooks 04–06 were zero-byte placeholders.
- model.py, features.py, evaluate.py, inference.py, submission.py, run_pipeline.py, README.md, requirements.txt, Documentation_template.md and all reports were empty. models/ and experiments/ held no files.
- Two historical candidate TSVs existed and are preserved: blocking_sample_s1_s2.tsv and blocking_sample_s1_s3.tsv. They are not final predictions.
- No AGENTS.md, Git metadata, official validator, challenge rules document or fitted model was present.

The brief states that Daya's part is complete, but this checkout supplies no completed model/integration artifacts. Authorship cannot be established from files. Reusable foundations are loading, configuration, Unicode/case/whitespace normalization and blocking helpers.

Historical notebook outputs report 2,206,821 training S1s, 5,034,616 S2s, 5,285,603 S3s and 7,638,365 true pairs. Its 100,000-S1 sample excluded zero-match records. It recorded 31,784,568 S2 and 38,935,086 S3 candidates, with 55.04% and 56.48% recall respectively. Saved output labels differ from some current notebook code, so these are historical observations rather than a rerun of the current source.

Problems found: regex `\w` normalization drops Indic combining marks; script detection returns only the first detected script; every legacy blocker hard-filters by country; loose address/name keys create very large joins. No supervised end-to-end pipeline or measured model baseline existed to preserve.

Original normalization.py and blocking.py were copied to reports/original before editing. Original datasets and historical candidates remain unchanged. Production retrieval uses bounded disk-backed postings; old helper APIs remain for notebook compatibility.
