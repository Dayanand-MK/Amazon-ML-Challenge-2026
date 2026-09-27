"""Batched, deterministic test inference with bounded multiprocessing and resume."""
import csv
import hashlib
import json
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .features import feature_matrix, FEATURE_NAMES
from .model import load_model
from .retrieval import CandidateIndex, prepared, fingerprint

_INDEX = None
_BUNDLE = None


def worker_init(root):
    global _INDEX, _BUNDLE
    import warnings
    warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names", category=UserWarning)
    root=Path(root)
    _BUNDLE=load_model(root/'models/final_model.pkl')
    assert _BUNDLE['feature_names']==FEATURE_NAMES
    _INDEX=CandidateIndex([root/'dataset/test/test_source2.tsv',root/'dataset/test/test_source3.tsv'],root/'cache/test_index',
        max_posting=_BUNDLE['metadata']['max_posting'],max_candidates=_BUNDLE['metadata']['max_candidates'],
        balance_sources=_BUNDLE['metadata'].get('source_balanced',False))


def process_batch(rows, candidate_rids=None):
    import numpy as np
    _INDEX.try_prepared()
    candidates,matrices=[],[]
    for position,row in enumerate(rows):
        left=prepared(row)
        rids = _INDEX.retrieve(left) if candidate_rids is None else candidate_rids[position]
        rights=[_INDEX.record(r) for r in rids]
        candidates.append([r[0] for r in rights])
        matrices.append(feature_matrix(left,rights))
    x=np.concatenate(matrices)
    scores=_BUNDLE['model'].predict_proba(x)[:,1] if len(x) else np.empty(0)
    match_lines,candidate_lines=[],[]
    pos=0
    matched=0
    empty=0
    for row,targets in zip(rows,candidates):
        selected=sorted(t for t,s in zip(targets,scores[pos:pos+len(targets)]) if s>=_BUNDLE['threshold'])
        match_lines.append(row[0]+'\t'+','.join(selected)+'\n')
        candidate_lines.append(row[0]+'\t'+','.join(sorted(targets))+'\n')
        pos+=len(targets)
        matched+=len(selected)
        empty+=not selected
    return ''.join(match_lines).encode(),''.join(candidate_lines).encode(),len(rows),matched,empty,pos


def batches(path,skip,batch_size):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f,delimiter='\t')
        next(reader)
        batch=[]
        for i,row in enumerate(reader):
            if i<skip:continue
            batch.append(tuple(row))
            if len(batch)==batch_size:
                yield batch
                batch=[]
        if batch:yield batch


def infer(root,workers=3,batch_size=500):
    root=Path(root).resolve()
    output=root/'output'
    checkpoint=output/'inference_checkpoint.json'
    sig={'inputs':fingerprint([root/f'dataset/test/test_source{i}.tsv' for i in (1,2,3)]),
         'model_sha256':hashlib.sha256((root/'models/final_model.pkl').read_bytes()).hexdigest(),
         'pipeline_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                            for name in ('normalization.py','retrieval.py','features.py','inference.py')}}
    done_path=output/'inference_summary.json'
    if done_path.exists():
        done=json.loads(done_path.read_text())
        if done.get('signature')==sig and all((output/n).exists() for n in ('matching_results.tsv','candidate_pairs.tsv')):
            print('Reusing completed inference',flush=True)
            return done
    state={'signature':sig,'s1_rows':0,'predicted_pairs':0,'empty_matches':0,'candidate_pairs':0}
    mpath,cpath=output/'matching_results.partial.tsv',output/'candidate_pairs.partial.tsv'
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text())
        if state['signature']!=sig:
            raise ValueError('Inference checkpoint belongs to different model/data; preserve or relocate partial outputs before restarting')
        mf,cf=mpath.open('r+b'),cpath.open('r+b')
        mf.truncate(state['matching_offset']); mf.seek(state['matching_offset'])
        cf.truncate(state['candidate_offset']); cf.seek(state['candidate_offset'])
    else:
        mf,cf=mpath.open('wb'),cpath.open('wb')
        mf.write(b'source1_entity_id\tmatched_entity_ids\n')
        cf.write(b'source1_entity_id\tcandidate_entity_ids\n')
    start=time.time()
    initial=state['s1_rows']
    for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[var]='1'
    try:
        with ProcessPoolExecutor(max_workers=workers,initializer=worker_init,initargs=(str(root),)) as pool:
            pending=deque()
            stream=iter(batches(root/'dataset/test/test_source1.tsv',state['s1_rows'],batch_size))
            for _ in range(workers*2):
                batch=next(stream,None)
                if batch is not None:pending.append(pool.submit(process_batch,batch))
            while pending:
                m,c,n,p,e,cp=pending.popleft().result()
                mf.write(m);cf.write(c)
                state['s1_rows']+=n;state['predicted_pairs']+=p;state['empty_matches']+=e;state['candidate_pairs']+=cp
                mf.flush();cf.flush()
                state['matching_offset']=mf.tell();state['candidate_offset']=cf.tell()
                temp=checkpoint.with_suffix('.tmp')
                temp.write_text(json.dumps(state,indent=2))
                temp.replace(checkpoint)
                if state['s1_rows']%10000==0:
                    elapsed=time.time()-start
                    print(f"Inference: {state['s1_rows']:,} S1, {state['candidate_pairs']:,} candidates, {(state['s1_rows']-initial)/max(elapsed,1):.1f} S1/s",flush=True)
                batch=next(stream,None)
                if batch is not None:pending.append(pool.submit(process_batch,batch))
    finally:
        mf.close();cf.close()
    mpath.replace(output/'matching_results.tsv')
    cpath.replace(output/'candidate_pairs.tsv')
    done_path.write_text(json.dumps(state,indent=2))
    checkpoint.unlink(missing_ok=True)
    return state
