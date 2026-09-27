"""Controlled tuning-only experiment: change global preselection to 80 per source.

This experimental implementation deliberately leaves the production retrieval
unchanged until a same-cohort comparison demonstrates improvement.
"""
import json
import pickle
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .retrieval import CandidateIndex,prepared
from .features import feature_matrix
from .evaluate import sweep,metrics
from .model import load_model,save_model


def balanced_retrieve(index,left):
    previous=index.balance_sources
    index.balance_sources=True
    try:return index.retrieve(left)
    finally:index.balance_sources=previous


def run(root,adopt=True):
    root=Path(root)
    with (root/'cache/training_12000.pkl').open('rb') as f:data=pickle.load(f)
    index=CandidateIndex([root/f'dataset/train/train_source{i}.tsv' for i in (2,3)],root/'cache/train_index')
    ids=np.flatnonzero(data['splits']==1)
    xs,ys,groups,counts,pairs,global_groups=[],[],[],[],[],[]
    start=time.time()
    for j,i in enumerate(ids):
        row=data['sample'][i]
        left=prepared(row)
        retrieved=balanced_retrieve(index,left)
        rights=[index.record(r) for r in retrieved]
        xs.append(feature_matrix(left,rights))
        ys.extend(int(r[0] in data['truth'][row[0]]) for r in rights)
        groups.extend([j]*len(rights))
        global_groups.extend([int(i)]*len(rights))
        counts.append(len(data['truth'][row[0]]))
        pairs.extend((row[0],r[0],rid) for r,rid in zip(rights,retrieved))
        if (j+1)%500==0:print(f'Balanced experiment: {j+1}/{len(ids)}, {time.time()-start:.0f}s',flush=True)
    bundle=load_model(root/'models/final_model.pkl')
    model=bundle['model']
    scores=model.predict_proba(np.concatenate(xs))[:,1]
    results=sweep(ys,scores,groups,counts)
    pd.DataFrame(results).to_csv(root/'experiments/balanced_threshold_results.csv',index=False)
    best=max(results,key=lambda x:(x['F0.5'],x['precision'],x['threshold']))
    old=json.loads((root/'reports/validation.json').read_text())['tuning']
    report={'baseline':old,'balanced':best,'F0.5_gain':best['F0.5']-old['F0.5'],
            'selection_split':'tuning only; existing frozen model; same keys, posting cap and final 24-candidate cap'}
    (root/'reports/source_balance_experiment.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if adopt and report['F0.5_gain']>0:
        from .training import subset,error_analysis
        # Preserve the earlier global-pool model/measurements before replacing selected artifacts.
        shutil.copy2(root/'models/final_model.pkl',root/'models/global_pool_model.pkl')
        shutil.copy2(root/'reports/validation.json',root/'reports/global_pool_validation.json')
        shutil.copy2(root/'experiments/error_analysis.csv',root/'experiments/global_pool_error_analysis.csv')
        for j,i in enumerate(np.flatnonzero(data['splits']==2)):
            row=data['sample'][i]
            left=prepared(row)
            retrieved=balanced_retrieve(index,left)
            rights=[index.record(r) for r in retrieved]
            xs.append(feature_matrix(left,rights))
            ys.extend(int(r[0] in data['truth'][row[0]]) for r in rights)
            global_groups.extend([int(i)]*len(rights))
            pairs.extend((row[0],r[0],rid) for r,rid in zip(rights,retrieved))
            if (j+1)%500==0:print('Selected retrieval holdout:',j+1,flush=True)
        evaluated=dict(data)
        evaluated.update(X=np.concatenate(xs),y=np.asarray(ys,dtype=np.uint8),groups=np.asarray(global_groups),pairs=pairs)
        hold,hgroups,hcounts,hids=subset(evaluated,2)
        hscores=model.predict_proba(evaluated['X'][hold])[:,1]
        held=metrics(evaluated['y'][hold],hscores,hgroups,hcounts,best['threshold'])
        meta=dict(bundle['metadata'])
        best['model']=meta['selected_model']
        meta.update(source_balanced=True,tuning=best,holdout=held,
            retrieval='same keys and final cap; 80 preselected targets per source instead of global top 160',
            evaluation_note='Retrieval selected only by tuning F0.5. Both baseline and selected retrieval evaluated on held-out S1s; holdout never used for selection.')
        save_model(root/'models/final_model.pkl',model,best['threshold'],bundle['feature_names'],meta)
        (root/'reports/validation.json').write_text(json.dumps(meta,indent=2))
        tune,tgroups,tcounts,tids=subset(evaluated,1)
        added=[{'method':'source_balanced_80','source_pair':'combined','split':'tune','true_pairs':sum(tcounts),
                'retained_true_pairs':int(evaluated['y'][tune].sum()),'lost_true_pairs':sum(tcounts)-int(evaluated['y'][tune].sum()),
                'candidate_recall':best['candidate_recall'],'candidate_count':int(tune.sum()),'average_candidates_per_s1':float(tune.sum()/len(tids))}]
        for source in ('S2','S3'):
            smask=tune & np.asarray([p[1].startswith(source+'-') for p in pairs])
            total=sum(sum(t.startswith(source+'-') for t in data['truth'][data['sample'][i][0]]) for i in tids)
            retained=int(evaluated['y'][smask].sum())
            added.append({'method':'source_balanced_80','source_pair':'S1-'+source,'split':'tune','true_pairs':total,
                'retained_true_pairs':retained,'lost_true_pairs':total-retained,'candidate_recall':retained/total,
                'candidate_count':int(smask.sum()),'average_candidates_per_s1':float(smask.sum()/len(tids))})
        pd.concat([pd.read_csv(root/'experiments/blocking_results.csv'),pd.DataFrame(added)],ignore_index=True).to_csv(root/'experiments/blocking_results.csv',index=False)
        experiment={'experiment_id':'E05','normalization':'same','blocking':'source-balanced top 80 per source; final top 24',
            'features':len(bundle['feature_names']),**best,'notes':'Existing frozen model. Retrieval and threshold selected on tuning cohort only.'}
        pd.concat([pd.read_csv(root/'experiments/experiment_log.csv'),pd.DataFrame([experiment])],ignore_index=True).to_csv(root/'experiments/experiment_log.csv',index=False)
        pd.concat([pd.read_csv(root/'experiments/model_results.csv'),pd.DataFrame([{**best,'model':best['model']+'_balanced'}])],ignore_index=True).to_csv(root/'experiments/model_results.csv',index=False)
        country_rows=[]
        for country in sorted({data['sample'][i][3] for i in hids}):
            local=np.asarray([j for j,i in enumerate(hids) if data['sample'][i][3]==country])
            cmask=np.isin(hgroups,local)
            result=metrics(evaluated['y'][hold][cmask],hscores[cmask],np.searchsorted(local,hgroups[cmask]),np.asarray(hcounts)[local],best['threshold'])
            country_rows.append({'country':country,**result})
        pd.DataFrame(country_rows).to_csv(root/'experiments/country_results.csv',index=False)
        with (root/'cache/selected_evaluation.pkl').open('wb') as f:pickle.dump(evaluated,f)
        error_analysis(root,evaluated,hold,hscores,best['threshold'],hids)
        print('Selected balanced holdout:',held,flush=True)
    index.close()


if __name__=='__main__':run(Path(__file__).resolve().parent.parent)
