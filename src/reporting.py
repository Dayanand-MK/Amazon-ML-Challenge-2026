"""Generate reports from measured artifacts; preserve historical notebook cells."""
import json
from pathlib import Path


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
    table='| Metric | Held-out evaluation |\n|---|---:|\n'+'\n'.join(
        f'| {name} | {m[name]:.6f} |' for name in ('candidate_recall','precision','recall','F0.5','singleton_accuracy'))
    final=f'''# Final measured results

Model: {meta['selected_model']}. Features: {len(meta['features'])}. Threshold: {meta['tuning']['threshold']:.3f}.

{table}

Sampled S1: {meta['sample_s1']}; fit {meta['fit_s1']}; tune {meta['tune_s1']}; held-out evaluation {meta['holdout_s1']}.
Sampled S1s sharing a truth target are grouped before splitting. Candidates search the full training target corpus.
The model is retained exactly as evaluated, with no post-threshold refit. No test labels or external business data were used.

Local integrity validator: {validation['local_integrity']}. Official validator: unavailable and NOT run.
Test S1 rows: {validation['s1_rows']:,}; candidate pairs: {validation['candidate_pairs']:,}; predicted pairs: {validation['predicted_pairs']:,}; empty lists: {validation['empty_matches']:,}.

Candidate generation uses auxiliary transliteration, exact/core names, token/prefix postings and address/number keys across countries.
Postings larger than {meta['max_posting']} are skipped, 80 targets per source (160 total) are pre-ranked, and the best {meta['max_candidates']} reach the matcher per S1. These limits are included in the reported candidate recall.

Archive: Hacksmiths_submission.zip. Official schema and model/license constraints remain unverified because their documents were not supplied.
'''
    (root/'reports/final_results.md').write_text(final,encoding='utf-8')
    (root/'reports/run_status.md').write_text('# Run status\n\nImplementation, EDA, experiments, held-out evaluation, full test inference and local submission validation are complete. See final_results.md for measured results. Official validation remains pending because its script and rules were not supplied.\n',encoding='utf-8')
    doc=f'''# Hacksmiths — Business Entity Resolution

## 1. Problem Understanding
Resolve each reference S1 business to zero or more S2/S3 records. Optimize precision-focused pairwise F0.5; do not force matches.

## 2. Dataset Description
Only supplied TSV data is used. Full-file counts, missingness, uniqueness, countries, lengths and Unicode scripts are in reports/eda_statistics.json and reports/eda_report.md.
Training S1 rows: {eda['train_source1']['rows']:,}. Test S1 rows: {eda['test_source1']['rows']:,}. Test ground truth is unavailable.

## 3. Data Preprocessing
Preserved existing config and loading interfaces. NFKC, casefold, whitespace and punctuation normalization retain Unicode letters, marks and numbers. Original text remains in unchanged source TSV files.
No unjustified repeated-character collapse or destructive suffix removal is applied to primary text.

## 4. Multilingual Handling
Raw, normalized Unicode, compact, token and auxiliary local transliterated forms are available. Combining marks are preserved.
Scripts are detected from Unicode character names; mixed-script records are counted. Transliteration is approximate and can collide.
Countries are open-set features, never a hard retrieval filter. No external registries, geocoding, APIs, embeddings or business-data augmentation are used.

## 5. Candidate Generation / Blocking
Disk-backed hashed postings cover all target records. Keys include exact/core transliterated names, name tokens and prefixes, full addresses, address-token pairs and number/token combinations.
Common postings over {meta['max_posting']} targets are skipped. Up to 80 targets per source (160 total) are pre-ranked with name/address similarity; the top {meta['max_candidates']} become candidates, without one-to-one constraints.
Global and source-balanced preselection, the exact-name comparator and combined blocker are measured on the same tuning S1s in experiments/blocking_results.csv.
No character TF-IDF experiment was run; the bounded posting index is the implemented retrieval method.

## 6. Feature Engineering
{len(meta['features'])} features cover Unicode exact/character/partial/token similarities, Jaccard, relative lengths, character bigram Dice, missingness, auxiliary transliteration, country equality, scripts, source and numeric agreement/conflict.
First-number and last-long-number agreement are explicit heuristics; they are not claimed to be reliable house/postcode parsers for every country.
Feature order is stored with the serialized model.

## 7. Model Architecture
Selected {meta['selected_model']} against standardized logistic regression. Histogram gradient boosting is a small locally trained tree model with no pretrained parameters.
Models use scikit-learn (BSD-3-Clause); RapidFuzz is MIT and text-unidecode metadata reports Artistic License. The exact challenge license/parameter restrictions were not supplied and compliance cannot be certified.

## 8. Training / Validation
Uniform reservoir sample of {meta['sample_s1']:,} S1 records, including zero-match entities. Labels come only from provided training truth.
Truth-connected sampled entities are grouped into fit ({meta['fit_s1']}), tuning ({meta['tune_s1']}), and final holdout ({meta['holdout_s1']}) with seed 2026.
Retrieval runs against all targets; missed truth pairs count as false negatives. Singleton accuracy means fraction of zero-match S1s correctly left empty.
No artificial positive pairs are injected into candidates. Source-balanced preselection was chosen on tuning F0.5; baseline and selected retrieval were both evaluated on holdout, which was never used for selection. The selected model is frozen at its evaluated fit rather than refitted after threshold selection.

## 9. Experiments
See experiments/experiment_log.csv, model_results.csv and blocking_results.csv. Same-split comparisons cover logistic regression versus histogram gradient boosting, basic 18-text-feature versus full 30-feature boosting, exact-name versus multi-key retrieval, and fixed versus tuned thresholds.
Original notebook history is preserved. It reported 55.04% S2 and 56.48% S3 recall on a different 100,000-positive-S1 sample; those are historical, not directly comparable to the new validation cohort.
No existing trained Daya model was present in the inspected checkout; model.py and models/ were empty.

## 10. Threshold Selection
Threshold {meta['tuning']['threshold']:.3f} selected by maximum tuning F0.5, then precision on ties, over a recorded grid from 0.05 to 0.999.
Held-out evaluation results:

{table}

## 11. Error Analysis
All holdout false positives and false negatives are in experiments/error_analysis.csv, including blocking losses.
reports/error_analysis.md categorizes script variation, missing addresses, numeric conflicts, similar names/addresses and ambiguous text using disclosed heuristics.
The holdout findings were not used to retune the selected model. Changes based on them require fresh validation.

## 12. Final Approach
Raw TSV → Unicode and auxiliary representations → full-corpus bounded postings → pair features → selected classifier → tuned threshold → grouped matches and long-form candidate pairs.
All scored test pairs appear in candidate_pairs.tsv. Every test S1 appears exactly once, with an empty TSV field for no matches. No ground-truth labels are used in test inference.
Local validation: {validation['local_integrity']}; {validation['s1_rows']:,} S1 rows, {validation['candidate_pairs']:,} candidate pairs and {validation['predicted_pairs']:,} predicted pairs.

## 13. Limitations
Official validation script, exact candidate-file schema and detailed licensing/parameter limits were absent. A clearly labeled local validator checks the user-specified invariants, but cannot certify official acceptance.
Training uses a reproducible sample rather than all labels. High-frequency names, severe typos and transliteration failures can be excluded by bounded retrieval.
France is present only in test; no France-specific labeled validation score is claimed. Random grouped holdout does not fully measure this country shift.
Pairwise F0.5 is implemented from the user brief; no unseen official scoring implementation is assumed.
The archive is prepared for review; no external upload was performed.

## 14. Conclusion
The missing supervised stages are implemented and measured, and complete test outputs are locally validated. Official acceptance remains pending the supplied challenge tools/rules.
'''
    (root/'Documentation_template.md').write_text(doc,encoding='utf-8')
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
