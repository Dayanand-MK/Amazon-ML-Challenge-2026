"""Rescore verified previous final candidates; fresh reproduction uses normal infer.

This optional accelerator requires the old inference summary, original model,
and its long-form candidate TSV. Input fingerprints, blocking source hashes,
retrieval settings, source IDs and pair counts must agree. No candidates are
added or removed. Candidate IDs are mapped exactly to current target records.
"""
import csv
import hashlib
import json
import os
from array import array
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby, islice
from pathlib import Path
import time
import numpy as np
from . import inference
from .model import load_model
from .retrieval import fingerprint
from .submission import sha256

_ID_MAPS=None

def build_id_map(root):
    directory=root/'cache/test_id_map'
    directory.mkdir(exist_ok=True)
    paths=[root/f'dataset/test/test_source{i}.tsv' for i in (2,3)]
    signature=fingerprint(paths)
    manifest=directory/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())==signature:return
    offset=0
    for source,path in zip((2,3),paths):
        ids=array('Q')
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'):
                eid=row['entity_id']
                n=int(eid.split('-')[1])
                if eid!=f'S{source}-{n}':raise ValueError('Cache accelerator requires canonical numeric IDs; use normal infer for opaque IDs')
                ids.append(n)
        keys=np.asarray(ids,dtype=np.uint64)
        order=np.argsort(keys)
        np.save(directory/f'keys{source}.npy',keys[order])
        np.save(directory/f'rows{source}.npy',(order+offset).astype(np.uint32))
        offset+=len(keys)
    manifest.write_text(json.dumps(signature))


def init_worker(root):
    global _ID_MAPS
    inference.worker_init(root)
    directory=Path(root)/'cache/test_id_map'
    _ID_MAPS={source:(np.load(directory/f'keys{source}.npy',mmap_mode='r'),np.load(directory/f'rows{source}.npy',mmap_mode='r')) for source in (2,3)}


def resolve(ids):
    result=[]
    for source in (2,3):
        selected=[eid for eid in ids if eid.startswith(f'S{source}-')]
        if not selected:continue
        numbers=np.array([int(eid.split('-')[1]) for eid in selected],dtype=np.uint64)
        assert all(eid==f'S{source}-{int(n)}' for eid,n in zip(selected,numbers))
        keys,rows=_ID_MAPS[source]
        positions=np.searchsorted(keys,numbers)
        assert np.all(positions<len(keys)) and np.array_equal(keys[positions],numbers),'Unknown cached target'
        result.extend(int(r) for r in rows[positions])
    assert len(result)==len(ids)==len(set(ids)), 'Invalid or duplicate cached ID'
    return result


def score_cached(batch):
    rows,targets=zip(*batch)
    return inference.process_batch(rows,[resolve(ids) for ids in targets])


def cached_batches(root,candidates,batch_size=500):
    with candidates.open(encoding='utf-8',newline='') as cf, (root/'dataset/test/test_source1.tsv').open(encoding='utf-8-sig',newline='') as sf:
        reader=csv.reader(cf,delimiter='\t')
        assert next(reader)==['source1_entity_id','target_entity_id']
        grouped=iter(groupby(reader,key=lambda r:r[0]))
        pending=next(grouped,None)
        source=csv.reader(sf,delimiter='\t');next(source)
        batch=[]
        for row in source:
            ids=[]
            if pending is not None and pending[0]==row[0]:
                ids=[r[1] for r in pending[1]]
                pending=next(grouped,None)
            batch.append((tuple(row),ids))
            if len(batch)==batch_size:yield batch;batch=[]
        if batch:yield batch
        assert pending is None,'Cached S1 order/coverage invalid'


def verify(root,history,old_model):
    previous=json.loads((history/'inference_summary.json').read_text())
    current=load_model(root/'models/final_model.pkl')
    old=load_model(old_model)
    assert sha256(old_model)==previous['signature']['model_sha256']
    assert previous['signature']['inputs']==fingerprint([root/f'dataset/test/test_source{i}.tsv' for i in (1,2,3)])
    for name in ('normalization.py','retrieval.py','features.py'):
        assert previous['signature']['pipeline_sha256'][name]==sha256(root/'src'/name),'Blocking/feature code changed'
    for name in ('max_candidates','max_posting','source_balanced'):
        assert current['metadata'].get(name)==old['metadata'].get(name),'Retrieval configuration changed'
    return previous


def run(root,history,old_model,workers=6):
    root,history,old_model=Path(root).resolve(),Path(history).resolve(),Path(old_model).resolve()
    previous=verify(root,history,old_model)
    build_id_map(root)
    candidates=history/'candidate_pairs.tsv'
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    # Audit a deterministic spread of 64 S1 rows against fresh retrieval before reuse.
    init_worker(root)
    audited=0
    for position,batch in enumerate(cached_batches(root,candidates)):
        if position%53==0:
            row,ids=batch[0]
            fresh=inference._INDEX.retrieve(inference.prepared(row))
            assert set(fresh)==set(resolve(ids)),'Candidate cache differs from fresh retrieval'
            audited+=1
        if audited==64:break
    inference._INDEX.close()
    inference._INDEX=None
    signature={'inputs':previous['signature']['inputs'],'model_sha256':sha256(root/'models/final_model.pkl'),
        'pipeline_sha256':{n:sha256(root/'src'/n) for n in ('normalization.py','retrieval.py','features.py','inference.py','rescore.py')},
        'candidate_cache_sha256':sha256(candidates),'fresh_retrieval_audit_s1':audited}
    state={'signature':signature,'s1_rows':0,'predicted_pairs':0,'empty_matches':0,'candidate_pairs':0}
    output=root/'output'; start=time.time()
    mpath,cpath=output/'matching_results.rescore.partial.tsv',output/'candidate_pairs.rescore.partial.tsv'
    with mpath.open('wb') as mf,cpath.open('wb') as cf,ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(str(root),)) as pool:
        mf.write(b'source1_entity_id\tmatched_entity_ids\n');cf.write(b'source1_entity_id\tcandidate_entity_ids\n')
        batches=iter(cached_batches(root,candidates));pending=deque()
        for _ in range(workers*2):
            batch=next(batches,None)
            if batch is not None:pending.append(pool.submit(score_cached,batch))
        while pending:
            m,c,n,p,e,cp=pending.popleft().result()
            mf.write(m);cf.write(c)
            for key,value in zip(('s1_rows','predicted_pairs','empty_matches','candidate_pairs'),(n,p,e,cp)):state[key]+=value
            if state['s1_rows']%10000==0:
                print(f"Rescoring: {state['s1_rows']:,} S1, {state['candidate_pairs']:,} candidates, {state['s1_rows']/(time.time()-start):.1f} S1/s",flush=True)
            batch=next(batches,None)
            if batch is not None:pending.append(pool.submit(score_cached,batch))
    assert state['s1_rows']==previous['s1_rows'] and state['candidate_pairs']==previous['candidate_pairs'],'Cache count mismatch'
    mpath.replace(output/'matching_results.tsv');cpath.replace(output/'candidate_pairs.tsv')
    (output/'inference_summary.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(state,indent=2),flush=True)
    return state

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    run(root,root/'output/history/pre_compliance',root/'reports/history/pre_compliance/models/final_model.pkl')
