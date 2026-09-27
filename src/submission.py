"""Local integrity validation and packaging. This is NOT the official validator."""
import csv
import hashlib
import json
import re
import zipfile
from array import array
from itertools import groupby
from pathlib import Path
import numpy as np


def id_arrays(test_dir):
    arrays={}
    for source in (2,3):
        ids=array('I')
        with (test_dir/f'test_source{source}.tsv').open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'):
                eid=row['entity_id']
                if not re.fullmatch(f'S{source}-[0-9]+',eid):
                    raise ValueError('Unexpected entity ID syntax; extend validator instead of coercing')
                assert eid == f'S{source}-{int(eid.split("-")[1])}', 'Noncanonical source IDs require a string-ID validator'
                ids.append(int(eid.split('-')[1]))
        a=np.asarray(ids,dtype=np.uint32)
        a.sort()
        assert not np.any(a[1:]==a[:-1]),'Duplicate test target IDs'
        arrays[f'S{source}']=a
    return arrays


def valid_targets(ids,arrays):
    for source in ('S2','S3'):
        chosen=[]
        for eid in ids:
            assert re.fullmatch(r'S[23]-[0-9]+',eid),f'Invalid target ID: {eid}'
            assert eid == eid.split('-')[0]+'-'+str(int(eid.split('-')[1])), 'Noncanonical predicted ID'
            if eid.startswith(source+'-'):chosen.append(int(eid.split('-')[1]))
        if not chosen:continue
        v=np.asarray(chosen,dtype=np.uint64)
        a=arrays[source]
        pos=np.searchsorted(a,v)
        assert np.all(pos<len(a)),'Unknown target IDs'
        assert np.array_equal(a[pos],v),'Unknown target IDs'


def validate(matching,candidate,test_dir,report=None):
    arrays=id_arrays(Path(test_dir))
    counts={'s1_rows':0,'candidate_pairs':0,'predicted_pairs':0,'empty_matches':0}
    seen_s1=array('I')
    with open(matching,encoding='utf-8',newline='') as mf,open(candidate,encoding='utf-8',newline='') as cf, \
         (Path(test_dir)/'test_source1.tsv').open(encoding='utf-8-sig',newline='') as sf:
        matches=csv.DictReader(mf,delimiter='\t')
        pairs=csv.DictReader(cf,delimiter='\t')
        assert matches.fieldnames==['source1_entity_id','matched_entity_ids'],'Wrong matching header'
        assert pairs.fieldnames==['source1_entity_id','target_entity_id'],'Wrong candidate header (local schema)'
        grouped=iter(groupby(pairs,key=lambda r:r['source1_entity_id']))
        pending=next(grouped,None)
        targets_to_validate=[]
        for source_row in csv.DictReader(sf,delimiter='\t'):
            row=next(matches,None)
            assert row is not None,'Missing S1 output rows'
            s1=source_row['entity_id']
            assert row['source1_entity_id']==s1,'Missing, duplicate, or out-of-order S1'
            assert re.fullmatch(r'S1-[0-9]+',s1),'Unexpected S1 syntax'
            seen_s1.append(int(s1.split('-')[1]))
            predicted=row['matched_entity_ids'].split(',') if row['matched_entity_ids'] else []
            assert len(predicted)==len(set(predicted)),'Duplicate predicted IDs'
            proposed=[]
            if pending is not None and pending[0]==s1:
                proposed=[r['target_entity_id'] for r in pending[1]]
                pending=next(grouped,None)
            assert len(proposed)==len(set(proposed)),'Duplicate candidate pair'
            assert set(predicted)<=set(proposed),'Match absent from candidates'
            targets_to_validate.extend(proposed)
            if len(targets_to_validate)>100000:
                valid_targets(targets_to_validate,arrays)
                targets_to_validate=[]
            counts['s1_rows']+=1
            counts['candidate_pairs']+=len(proposed)
            counts['predicted_pairs']+=len(predicted)
            counts['empty_matches']+=not predicted
        valid_targets(targets_to_validate,arrays)
        assert next(matches,None) is None,'Extra matching rows'
        assert pending is None,'Unknown or out-of-order candidate S1 rows'
    a=np.asarray(seen_s1,dtype=np.uint32)
    a.sort()
    assert not np.any(a[1:]==a[:-1]),'Duplicate S1 IDs'
    result={'local_integrity':'PASS','official_validator':'NOT AVAILABLE',
            'schema':'matching mirrors supplied training truth; candidates are long-form S1,target pairs',**counts}
    if report:Path(report).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result


def package(root,team='Hacksmiths'):
    root=Path(root)
    validation=json.loads((root/'reports/submission_validation.json').read_text())
    assert validation['local_integrity']=='PASS'
    archive=root/f'{team}_submission.zip'
    files=[]
    for name in ('matching_results.tsv','candidate_pairs.tsv'):
        files.append((root/'output'/name,f'output/{name}'))
    prefix='code/business_entity_resolution/'
    for path in (root/'src').glob('*.py'):files.append((path,prefix+'src/'+path.name))
    for name in ('README.md','requirements.txt','run_pipeline.py'):
        files.append((root/name,prefix+name))
    files.append((root/'utils/validate_submission.py',prefix+'utils/validate_submission.py'))
    for path in (root/'tests').glob('*.py'):files.append((path,prefix+'tests/'+path.name))
    files.append((root/'models/final_model.pkl',prefix+'models/final_model.pkl'))
    files.append((root/'Documentation_template.md','Documentation_template.md'))
    files.append((root/'reports/submission_validation.json','submission_validation.json'))
    for folder in ('reports','experiments','notebooks'):
        for path in (root/folder).iterdir():
            if path.is_file() and path.suffix in ('.md','.json','.csv','.ipynb') and path.stat().st_size<10_000_000:
                files.append((path,prefix+folder+'/'+path.name))
    manifest={}
    for path,arc in files:
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
        manifest[arc]=digest.hexdigest()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path,arc in files:z.write(path,arc)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
        z.writestr('OFFICIAL_VALIDATION_PENDING.txt',
            'Local integrity checks passed. Official validator and detailed challenge licensing/parameter and output-schema rules were not supplied. Confirm those requirements before upload.\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None,'ZIP CRC failure'
    return archive
