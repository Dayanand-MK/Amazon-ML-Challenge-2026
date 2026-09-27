"""Generate current measured reports in the supplied official template structure."""
import json
from pathlib import Path
from datetime import date


def cell(kind,text):
    result={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':result.update(execution_count=None,outputs=[])
    return result


def report(root):
    root=Path(root)
    meta=json.loads((root/'reports/validation.json').read_text())
    eda=json.loads((root/'reports/eda_statistics.json').read_text())
    validation=json.loads((root/'reports/submission_validation.json').read_text())
    m=meta['holdout']
    names={'F0.5':'Macro per-S1 F0.5 (challenge metric)','micro_F0.5':'Micro F0.5 (diagnostic only)',
        'precision':'Pooled pair precision','recall':'Pooled pair recall','candidate_recall':'Pair candidate recall',
        'singleton_accuracy':'Zero-match S1 accuracy'}
    table='| Metric | Held-out evaluation |\n|---|---:|\n'+'\n'.join(f'| {label} | {m[name]:.6f} |' for name,label in names.items())
    members=(root/'reports/team_members.txt').read_text(encoding='utf-8').strip() if (root/'reports/team_members.txt').exists() else 'Not provided by the team'
    balanced=meta.get('source_balanced',False)
    retrieval='80 targets per source (160 total)' if balanced else '160 targets globally'
    final=f"""# Final measured results

Selected model: {meta['selected_model']}; MIT license; threshold {meta['tuning']['threshold']:.3f}.
{meta['tree_count']} trees, {meta['tree_node_count']:,} nodes; conservative tree-parameter bound {meta['parameter_upper_bound']:,}, below 8 billion.

{table}

Sample: {meta['sample_s1']:,} S1s; fit {meta['fit_s1']:,}, tune {meta['tune_s1']:,}, held-out {meta['holdout_s1']:,}.
F0.5 is averaged over individual S1 entities, with 1 for a correctly empty singleton and 0 for a singleton false merge.
Model/threshold/retrieval decisions use tuning data. Held-out results are reporting only; this split was also evaluated during earlier development, so it is not a fresh blind benchmark.

Official validator: {validation['official_validator']}, ID existence check enabled: {validation.get('official_check_ids',False)}.
Strict streaming validation: {validation['local_integrity']}. Both outputs have {validation['s1_rows']:,} S1 rows.
Candidate pairs: {validation['candidate_pairs']:,}; predicted pairs: {validation['predicted_pairs']:,}; empty matches: {validation['empty_matches']:,}.
Average candidates/S1: {validation['average_candidates_per_s1']:.6f}; maximum: {validation['max_candidates']}.
Candidates are precisely those passed to the model; matching lists are subsets. SHA-256 hashes bind validation to these files.

Earlier pooled scores and long-form output are superseded. Local backups remain under reports/history/pre_compliance and output/history.
No test score is claimed; only the portal can score hidden test labels.
"""
    (root/'reports/final_results.md').write_text(final,encoding='utf-8')
    (root/'reports/run_status.md').write_text('# Run status\n\n'+final.split('# Final measured results\n\n')[1],encoding='utf-8')
    doc=f"""# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Hacksmiths
**Team Members:** {members}
**Submission Date:** {date.today().isoformat()}

## 1. Executive Summary

We resolve each S1 business to zero, one or multiple S2/S3 records using a disk-backed candidate index and a locally trained MIT-licensed LightGBM classifier. Unicode-preserving text processing, auxiliary local transliteration and bounded retrieval support multilingual data while limiting final candidates to {meta['max_candidates']} per S1. Threshold selection uses the challenge's macro per-S1 F0.5, including singleton credit.

## 2. Methodology

### 2.1 Problem Analysis

Only the organizers' unchanged TSV files are used. Train S1 contains {eda['train_source1']['rows']:,} rows; test S1 contains {eda['test_source1']['rows']:,}. EDA covers full-file counts, missingness, duplicate IDs/rows, countries, field lengths and Unicode scripts in reports/eda_statistics.json and eda_report.md.
Training contains US and India; test additionally includes France. Countries remain open strings, with no country hard filter or fixed one-hot country vocabulary. Every test S1 is emitted, including France and empty-match entities.

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier.
**Core Contribution:** Bounded disk-backed multi-key retrieval with Unicode-aware and transliterated similarity features, followed by macro-calibrated matching.

NFKC, casefold and whitespace/punctuation cleanup preserve Unicode letters, combining marks and numbers. Original TSV text is unchanged. Compact/token/transliterated forms are auxiliary; transliteration can collide and does not replace the original representation. No external business databases, registries, APIs, geocoding, remote embeddings or test labels are used. text-unidecode is a generic local character mapping dependency.

## 3. Candidate Generation (Blocking)

- **Blocking keys:** Exact/core transliterated names, name tokens and prefixes, full addresses, address-token pairs, and number/token combinations. Common legal/address words are discounted only in retrieval keys.
- **Search:** All training or test S2/S3 targets as appropriate. Hashed sorted posting arrays and record offsets are memory-mapped. Postings over {meta['max_posting']} are skipped; {retrieval} are preselected by key overlap, then pre-ranked by name/address similarity and key hits. The best {meta['max_candidates']} are fed into the model, without a one-to-one constraint.
- **Final test candidates:** {validation['candidate_pairs']:,}; average {validation['average_candidates_per_s1']:.6f}/S1; maximum {validation['max_candidates']}.
- **Recall measurement:** True matches can be lost. Held-out candidate pair recall is {m['candidate_recall']:.6f}; all blocking losses count as false negatives. No gold positives are injected. Exact-name and multi-key comparisons, source-level recall and source-balanced preselection are recorded in experiments/.

The optional src/rescore.py path reuses an already generated final candidate set only after input/code/configuration provenance checks and a fresh-retrieval audit. It applies the same feature and model scoring as fresh inference; normal infer reproduces the outputs without this cache.

candidate_pairs.tsv exports the final candidate list actually scored by the ML model, not the larger intermediate preselection. It has exactly one row per test S1 and exact headers source1_entity_id and candidate_entity_ids. Lists are comma-separated without quotes; empty lists are empty fields.

## 4. Matching Model

**Features:** {len(meta['features'])} ordered features: Unicode name/address exact, ratio, partial, token-sort, token-set, Jaccard, length ratio, bigram Dice and missingness; auxiliary transliteration similarities; country equality/missingness; script equality; target source; number overlap/conflict; first-number/last-long-number/name-number agreement; and a name-address interaction. Numeric features are heuristics, not country-specific postcode parsers.

**Model type:** {meta['selected_model']}, trained locally from scratch with LightGBM 4.6.0 (MIT). The original final model artifact is released under the project MIT LICENSE. No pretrained model is used. The selected model has {meta['tree_count']} trees and {meta['tree_node_count']:,} nodes. A conservative bound of 32 stored scalar parameters per tree node gives {meta['parameter_upper_bound']:,}, far below 8 billion; it is a tree model, not a neural network.

**Training:** Seed 2026; uniform reservoir sample of {meta['sample_s1']:,} S1 entities including zero-match rows. Sampled S1s sharing a truth target are grouped before 60/20/20 fit/tune/holdout splitting: {meta['fit_s1']}/{meta['tune_s1']}/{meta['holdout_s1']}. Retrieval searches the full target corpus. No target supplies positive labels to multiple splits. Candidate negatives use only provided training truth.

**Selection:** Two LightGBM capacities and a text-feature-only LightGBM ablation are compared; standardized logistic regression is a diagnostic baseline, ineligible as the final model. Exact candidate/feature/model settings are in source and experiments. Source-balanced retrieval is adopted only if tuning macro F0.5 improves. The evaluated fit model is retained without refitting after threshold selection.

**Threshold:** {meta['tuning']['threshold']:.3f}, selected on tuning macro F0.5 from 0.05 through 0.975 in 0.025 increments plus 0.99, 0.995 and 0.999. Pooled precision and threshold break ties. For each nonempty-truth entity, F0.5 = 1.25 TP / (predicted_count + 0.25 true_count). Empty-truth entities receive 1 for an empty prediction and 0 otherwise. Scores are averaged across every S1, including those with no candidates. Micro scores are explicitly diagnostic.

## 5. Results & Error Analysis

{table}

Tuning macro F0.5: {meta['tuning']['F0.5']:.6f}. These are training holdout measurements, not leaderboard scores. Holdout is used for reporting rather than selection; prior development also evaluated this same split. France has no labeled validation records, so no France-specific accuracy claim is made.

Common failure categories include blocking misses, missing addresses, cross-script variation, numeric conflicts, similar names with conflicting addresses and ambiguous text. Counts and full holdout FP/FN examples, including original Unicode fields and blocked-out truth, are in reports/error_analysis.md and experiments/error_analysis.csv. These labels are diagnostic heuristics rather than verified causes. Blocking remains a recall ceiling; high-frequency names and severe text corruption can be missed.

## 6. Conclusion

The pipeline produces grouped candidate and matching TSVs for every test S1 with bounded candidate volume. Macro scoring rewards correctly empty singletons while penalizing false merges, and the selected model meets the MIT license and parameter-size requirements. Official validation is {validation['official_validator']}; stricter full ID/subset validation is {validation['local_integrity']}.

## Appendix

### A. Code Artefacts

The submission ZIP contains output/matching_results.tsv and output/candidate_pairs.tsv, this completed official-template document, and code/business_entity_resolution/ with src/, README.md, pinned requirements, run_pipeline.py, tests/, the unchanged official validator and the final fitted model. README.md gives setup, data placement, full regeneration, model-only inference, validation and packaging commands. Datasets and rebuildable caches are omitted. Output hashes are in reports/submission_validation.json and ZIP manifest.json; the package is CRC-checked.

### B. Additional Results

Both output files have {validation['s1_rows']:,} rows. There are {validation['predicted_pairs']:,} matches and {validation['empty_matches']:,} empty predictions. Full EDA, tuning model/threshold/blocking comparisons, country results and error analysis are included. Historical notebook cells are preserved and marked separately from current report-reading cells. Earlier pooled F0.5 reports are superseded, not treated as macro scores. Final predictions have not been uploaded to the challenge portal by this pipeline.
"""
    (root/'Documentation_template.md').write_text(doc,encoding='utf-8')
    modelcard=f"""# Final model card

Model: {meta['selected_model']}. Framework: LightGBM 4.6.0 (MIT). Artifact: models/final_model.pkl (MIT; see LICENSE).
Locally trained from supplied training data only; no pretrained weights. {meta['tree_count']} trees, {meta['tree_node_count']:,} nodes, conservative tree parameter bound {meta['parameter_upper_bound']:,} (<8 billion).
Threshold {meta['tuning']['threshold']:.3f}; source-balanced retrieval {balanced}; max {meta['max_candidates']} scored targets/S1.

{table}

See Documentation_template.md for split, selection, features, provenance and limitations. Country shift to France is not quantified by this US/India holdout. Only load trusted pickle files. Model serialization includes the ordered features, threshold and measured metadata.
"""
    (root/'MODEL_CARD.md').write_text(modelcard,encoding='utf-8')
    setup="from pathlib import Path\nimport json, sys\nROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\nsys.path.insert(0, str(ROOT))\n"
    notebooks={
        '01_eda.ipynb':('Full-file EDA supplement', "stats = json.loads((ROOT / 'reports/eda_statistics.json').read_text(encoding='utf-8'))\nstats"),
        '02_normalization.ipynb':('Unicode correction and representations', "from src.normalization import representations\n[representations(s) for s in ['राम मार्केटिंग', 'Café', 'Москва', '北京']]"),
        '03_blocking.ipynb':('Production bounded blocking results', "import pandas as pd\npd.read_csv(ROOT / 'experiments/blocking_results.csv')"),
        '04_features.ipynb':('Pairwise features', "from src.features import FEATURE_NAMES\nfrom src.model import load_model\nbundle = load_model(ROOT / 'models/final_model.pkl')\nassert bundle['feature_names'] == FEATURE_NAMES\nFEATURE_NAMES"),
        '05_model.ipynb':('Measured model and thresholds', "import pandas as pd\ndisplay(pd.read_csv(ROOT / 'experiments/model_results.csv'))\ndisplay(pd.read_csv(ROOT / 'experiments/threshold_results.csv'))\njson.loads((ROOT / 'reports/validation.json').read_text())"),
        '06_error_analysis.ipynb':('Holdout errors', "import pandas as pd\nerrors = pd.read_csv(ROOT / 'experiments/error_analysis.csv')\ndisplay(errors.groupby(['kind','category']).size())\nerrors.head(30)")}
    marker='<!-- measured-pipeline-supplement -->'
    for name,(title,code) in notebooks.items():
        path=root/'notebooks'/name
        notebook=json.loads(path.read_text(encoding='utf-8')) if path.stat().st_size else {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[]}
        old=notebook['cells']
        for i,c in enumerate(old):
            if marker in ''.join(c.get('source',[])):
                old=old[:i];break
        notebook['cells']=old+[cell('markdown',f'{marker}\n# {title}\n\nReproducible views of measured CLI artifacts. Historical cells above are preserved; do not rerun the original full-memory blocking experiment on a memory-limited machine. New cells are supplied unexecuted.'),cell('code',setup+code)]
        for i,c in enumerate(notebook['cells']):c.setdefault('id',f'cell-{i:04d}')
        path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    print('Measured reports and notebooks written',flush=True)
