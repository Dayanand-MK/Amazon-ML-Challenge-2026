"""Sample S1 uniformly, retrieve against ALL training targets, split by truth component."""
import csv
import hashlib
import json
import pickle
import time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from .retrieval import CandidateIndex, prepared, fingerprint
from .features import FEATURE_NAMES, feature_matrix
from .model import estimators, save_model
from .evaluate import metrics, sweep
from .normalization import detect_script


def sample_training(root, size=12000):
    # Reservoir sampling includes zero-match entities and is independent of truth.
    rng=np.random.default_rng(2026)
    sample=[]
    with (root/"dataset/train/train_source1.tsv").open(encoding="utf-8-sig",newline="") as f:
        for i,row in enumerate(csv.reader(f,delimiter="\t")):
            if i==0:continue
            if len(sample)<size:sample.append(tuple(row))
            else:
                j=int(rng.integers(i))
                if j<size:sample[j]=tuple(row)
    ids={r[0] for r in sample}
    truth={}
    with (root/"dataset/train/train_ground_truth.tsv").open(encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f,delimiter="\t"):
            if row['source1_entity_id'] in ids:
                truth[row['source1_entity_id']]=set(x.strip() for x in row['matched_entity_ids'].split(',') if x.strip())
    assert len(truth)==len(sample), "Missing S1 truth labels"
    # Union S1s sharing any target ID so a target never supplies labels to two splits.
    parent=list(range(size))
    def find(i):
        while i!=parent[i]:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    owner={}
    for i,row in enumerate(sample):
        for eid in truth[row[0]]:
            if eid in owner:parent[find(i)]=find(owner[eid])
            else:owner[eid]=i
    roots={find(i) for i in range(size)}
    split_map={r:int(rng.choice(3,p=[.6,.2,.2])) for r in sorted(roots)}
    splits=np.array([split_map[find(i)] for i in range(size)])
    return sample,truth,splits


def build_training(root, size=12000):
    root=Path(root)
    cache=root/f"cache/training_{size}.pkl"
    signature={'inputs':fingerprint(sorted((root/'dataset/train').glob('*.tsv'))),
               'code':{name:hashlib.sha256((root/'src'/name).read_bytes()).hexdigest()
                       for name in ('features.py','normalization.py','retrieval.py')},'sample_size':size,'seed':2026}
    if cache.exists():
        with cache.open('rb') as f:
            cached=pickle.load(f)
        if cached.get('signature')==signature:return cached
    sample,truth,splits=sample_training(root,size)
    index=CandidateIndex([root/"dataset/train/train_source2.tsv",root/"dataset/train/train_source3.tsv"],root/"cache/train_index")
    xs,ys,groups,pairs=[],[],[],[]
    exact_stats=[0,0]
    start=time.time()
    for i,row in enumerate(sample):
        left=prepared(row)
        ids=index.retrieve(left)
        rights=[index.record(r) for r in ids]
        xs.append(feature_matrix(left,rights))
        ys.extend(int(r[0] in truth[row[0]]) for r in rights)
        groups.extend([i]*len(ids))
        pairs.extend((row[0],r[0],rid) for r,rid in zip(rights,ids))
        if splits[i]==1:
            exact=index.retrieve(left,mode="exact")
            exact_stats[0]+=sum(index.record(r)[0] in truth[row[0]] for r in exact)
            exact_stats[1]+=len(exact)
        if (i+1)%500==0:print(f"Training features: {i+1:,}/{size:,}, {time.time()-start:.0f}s",flush=True)
    data={"X":np.concatenate(xs),"y":np.asarray(ys,dtype=np.uint8),"groups":np.asarray(groups),"pairs":pairs,
          "sample":sample,"truth":truth,"splits":splits,"exact_stats":exact_stats,"signature":signature}
    with cache.open('wb') as f:pickle.dump(data,f)
    index.close()
    return data


def subset(data,split):
    ids=np.flatnonzero(data['splits']==split)
    mask=np.isin(data['groups'],ids)
    mapping=np.full(len(data['sample']),-1,dtype=int)
    mapping[ids]=np.arange(len(ids))
    counts=[len(data['truth'][data['sample'][i][0]]) for i in ids]
    return mask,mapping[data['groups'][mask]],counts,ids


def train(root,size=12000):
    root=Path(root)
    data=build_training(root,size)
    trainmask,_,_,_=subset(data,0)
    tune,groups,counts,ids=subset(data,1)
    hold,hgroups,hcounts,hids=subset(data,2)
    models=estimators()
    summaries=[]
    thresholds=[]
    fitted={}
    for name,model in models.items():
        print('Fitting',name,flush=True)
        model.fit(data['X'][trainmask],data['y'][trainmask])
        scores=model.predict_proba(data['X'][tune])[:,1]
        rows=sweep(data['y'][tune],scores,groups,counts)
        for row in rows:row['model']=name
        thresholds.extend(rows)
        best=max(rows,key=lambda r:(r['F0.5'],r['precision'],r['threshold']))
        summaries.append(best)
        fitted[name]=model
        print(name,best,flush=True)
        if name=='logistic_regression':
            save_model(root/'models/baseline_model.pkl',model,best['threshold'],FEATURE_NAMES,{'split':'fit only'})
    chosen=max([s for s in summaries if s["model"].startswith("lightgbm")],key=lambda r:(r['F0.5'],r['precision']))
    name,threshold=chosen['model'],chosen['threshold']
    model=fitted[name]
    hscores=model.predict_proba(data['X'][hold])[:,1]
    held=metrics(data['y'][hold],hscores,hgroups,hcounts,threshold)
    country_rows=[]
    for country in sorted({data['sample'][i][3] for i in hids}):
        local=np.asarray([j for j,i in enumerate(hids) if data['sample'][i][3]==country])
        cmask=np.isin(hgroups,local)
        result=metrics(data['y'][hold][cmask],hscores[cmask],np.searchsorted(local,hgroups[cmask]),
                       np.asarray(hcounts)[local],threshold)
        country_rows.append({'country':country,**result})
    pd.DataFrame(country_rows).to_csv(root/'experiments/country_results.csv',index=False)
    pd.DataFrame(thresholds).to_csv(root/'experiments/threshold_results.csv',index=False)
    pd.DataFrame(summaries).to_csv(root/'experiments/model_results.csv',index=False)
    # Keep the evaluated model unchanged: avoids a probability shift after threshold tuning.
    meta={'sample_s1':size,'fit_s1':int(sum(data['splits']==0)),'tune_s1':len(ids),'holdout_s1':len(hids),
          'selected_model':name,'tuning':chosen,'holdout':held,'seed':2026,
          'refit':False,'split':'60/20/20 by sampled ground-truth connected component',
          'model_license':'MIT', 'implementation_license':'MIT (LightGBM)',
          'metric':'macro per-S1 F0.5 including singleton credit',
          'max_candidates':24,'max_posting':1200,'features':FEATURE_NAMES,
          'effective_features':FEATURE_NAMES[:18] if name.endswith('text_only') else FEATURE_NAMES}
    booster = model.steps[-1][1].booster_ if hasattr(model, 'steps') else model.booster_
    trees = booster.dump_model()['tree_info']
    meta['tree_count'] = len(trees)
    meta['tree_node_count'] = sum(2*t['num_leaves']-1 for t in trees)
    meta['parameter_upper_bound'] = 32*meta['tree_node_count']
    assert meta['parameter_upper_bound'] < 8_000_000_000
    save_model(root/'models/final_model.pkl',model,threshold,FEATURE_NAMES,meta)
    (root/'reports/validation.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    candidate_count=int(tune.sum())
    retained=int(data['y'][tune].sum())
    true=sum(counts)
    blocking_rows=[{'method':'exact_transliterated_name','source_pair':'combined','split':'tune','true_pairs':true,'retained_true_pairs':data['exact_stats'][0],
        'lost_true_pairs':true-data['exact_stats'][0],'candidate_recall':data['exact_stats'][0]/true,'candidate_count':data['exact_stats'][1],
        'average_candidates_per_s1':data['exact_stats'][1]/len(ids)},
        {'method':'multi_key_bounded','source_pair':'combined','split':'tune','true_pairs':true,'retained_true_pairs':retained,'lost_true_pairs':true-retained,
        'candidate_recall':retained/true,'candidate_count':candidate_count,'average_candidates_per_s1':candidate_count/len(ids)}]
    for source in ('S2','S3'):
        source_mask=tune & np.asarray([p[1].startswith(source+'-') for p in data['pairs']])
        source_true=sum(sum(t.startswith(source+'-') for t in data['truth'][data['sample'][i][0]]) for i in ids)
        source_retained=int(data['y'][source_mask].sum())
        blocking_rows.append({'method':'multi_key_bounded','source_pair':'S1-'+source,'split':'tune',
            'true_pairs':source_true,'retained_true_pairs':source_retained,'lost_true_pairs':source_true-source_retained,
            'candidate_recall':source_retained/source_true,'candidate_count':int(source_mask.sum()),
            'average_candidates_per_s1':float(source_mask.sum()/len(ids))})
    pd.DataFrame(blocking_rows).to_csv(root/'experiments/blocking_results.csv',index=False)
    experiments=[]
    for i,s in enumerate(summaries):
        experiments.append({'experiment_id':f'E0{i+1}','normalization':'NFKC Unicode + auxiliary transliteration',
            'blocking':'multi-key bounded postings','features':18 if s['model'].endswith('text_only') else len(FEATURE_NAMES), **s,
            'notes':'Same component-separated tuning split and candidates; E01 is new baseline, no preexisting model found.'})
    # Controlled threshold comparison for the selected model.
    base=metrics(data['y'][tune],model.predict_proba(data['X'][tune])[:,1],groups,counts,.5)
    experiments.append({'experiment_id':'E05','normalization':'same','blocking':'same','features':len(meta['effective_features']),'model':name,**base,
        'notes':'Fixed 0.50 threshold comparator; final threshold selected by tuning F0.5.'})
    pd.DataFrame(experiments).to_csv(root/'experiments/experiment_log.csv',index=False)
    error_analysis(root,data,hold,hscores,threshold,hids)
    print('Held-out evaluation:',held,flush=True)
    return meta


def error_analysis(root,data,mask,scores,threshold,ids):
    index=CandidateIndex([root/"dataset/train/train_source2.tsv",root/"dataset/train/train_source3.tsv"],root/"cache/train_index")
    predicted={int(i):set() for i in ids}
    candidates={int(i):set() for i in ids}
    errors=[]
    positions=np.flatnonzero(mask)
    pair_lookup={}
    for pos,score in zip(positions,scores):
        group=int(data['groups'][pos])
        s1,target,rid=data['pairs'][pos]
        candidates[group].add(target)
        pair_lookup[(group,target)]=(rid,float(score))
        if score>=threshold:predicted[group].add(target)
    needed=set()
    for i in ids:
        true=data['truth'][data['sample'][i][0]]
        needed |= (true-predicted[int(i)])-candidates[int(i)]
    lost_records={}
    for path in index.paths:
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.reader(f,delimiter='\t'):
                if row[0] in needed:lost_records[row[0]]=tuple(row)
    transitions=Counter()
    for i in ids:
        left_raw=data['sample'][i]
        true=data['truth'][left_raw[0]]
        for target in true:
            if (int(i),target) in pair_lookup:
                right_raw=index.raw(pair_lookup[(int(i),target)][0])
            elif target in lost_records:right_raw=lost_records[target]
            else:continue
            transitions[f'{detect_script(left_raw[1])} -> {detect_script(right_raw[1])}']+=1
        for kind,targets in [('false_positive',predicted[int(i)]-true),('false_negative',true-predicted[int(i)])]:
            for target in sorted(targets):
                lookup=pair_lookup.get((int(i),target))
                right_raw=index.raw(lookup[0]) if lookup else lost_records.get(target, (target,'','',''))
                left,right=prepared(left_raw),prepared(right_raw)
                f=feature_matrix(left,[right])[0]
                if not lookup:category='blocking_miss'
                elif not left[2] or not right[2]:category='missing_address'
                elif detect_script(left_raw[1])!=detect_script(right_raw[1]):category='script_variation'
                elif f[25]:category='numeric_conflict'
                elif f[1]>.85 and f[10]<.5:category='similar_name_different_address'
                elif f[1]<.5 and f[10]>.8:category='similar_address_different_name'
                else:category='text_variation_or_ambiguous'
                errors.append({'kind':kind,'category':category,'source1_entity_id':left[0],'target_entity_id':target,
                    'probability':lookup[1] if lookup else None,'candidate':bool(lookup),'left_name':left_raw[1],
                    'right_name':right_raw[1],'left_address':left_raw[2],'right_address':right_raw[2]})
    pd.DataFrame(errors).to_csv(root/'experiments/error_analysis.csv',index=False)
    summary=Counter((r['kind'],r['category']) for r in errors)
    text=['# Holdout error analysis','','Categories are diagnostic heuristics, not verified causal explanations.',
          'Held-out labels are used for reporting only; model and retrieval choices use the tuning split. No model was retuned on these errors.','',
          '| Type | Category | Count |','|---|---|---:|']
    text += [f'| {kind} | {category} | {count} |' for (kind,category),count in sorted(summary.items())]
    text += ['','## True-pair name script transitions','', '```json',json.dumps(transitions,indent=2), '```',
        '', 'Full error pairs, original Unicode text and probabilities are in experiments/error_analysis.csv.',
        'Cross-script truth pairs demonstrate script variation; local transliteration is an auxiliary representation and can create collisions.',
        'First/last numeric features are heuristics, not country-specific house/postcode parsers. No external address data is used.']
    (root/'reports/error_analysis.md').write_text('\n'.join(text),encoding='utf-8')
    index.close()
