"""Strict streaming validation plus reproducible, hash-bound packaging."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path):
    digest=hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def validate(matching, candidate, test_dir, report=None):
    test_dir=Path(test_dir)
    targets=set()
    for source in (2,3):
        with (test_dir/f'test_source{source}.tsv').open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'):
                eid=row['entity_id']
                assert eid.startswith(f'S{source}-') and eid not in targets, 'Invalid or duplicate test target ID'
                targets.add(eid)
    counts={'s1_rows':0,'candidate_rows':0,'candidate_pairs':0,'predicted_pairs':0,'empty_matches':0,'empty_candidates':0,'max_candidates':0}
    seen=set()
    with open(matching,encoding='utf-8',newline='') as mf, open(candidate,encoding='utf-8',newline='') as cf, (test_dir/'test_source1.tsv').open(encoding='utf-8-sig',newline='') as sf:
        assert mf.readline().rstrip('\r\n')=='source1_entity_id\tmatched_entity_ids', 'Wrong matching header'
        assert cf.readline().rstrip('\r\n')=='source1_entity_id\tcandidate_entity_ids', 'Wrong candidate header'
        for source_row in csv.DictReader(sf,delimiter='\t'):
            s1=source_row['entity_id']
            assert s1.startswith('S1-') and s1 not in seen, 'Duplicate or invalid S1'
            seen.add(s1)
            lists=[]
            for stream in (mf,cf):
                fields=stream.readline().rstrip('\r\n').split('\t')
                assert len(fields)==2 and fields[0]==s1, 'Missing, extra, duplicate, or out-of-order S1 output row'
                ids=fields[1].split(',') if fields[1] else []
                assert len(ids)==len(set(ids)), 'Duplicate target IDs'
                assert all(eid in targets for eid in ids), 'Unknown or invalid target IDs'
                lists.append(set(ids))
            predicted,proposed=lists
            assert predicted<=proposed, 'Match absent from candidates'
            counts['s1_rows']+=1
            counts['candidate_rows']+=1
            counts['candidate_pairs']+=len(proposed)
            counts['predicted_pairs']+=len(predicted)
            counts['empty_matches']+=not predicted
            counts['empty_candidates']+=not proposed
            counts['max_candidates']=max(counts['max_candidates'],len(proposed))
        assert not mf.read(1) and not cf.read(1), 'Extra output rows'
    counts['average_candidates_per_s1']=counts['candidate_pairs']/max(counts['s1_rows'],1)
    result={'local_integrity':'PASS','official_validator':'NOT RUN',
        'schema':'one row per S1; comma-separated target lists; exact challenge headers',
        'output_sha256':{Path(p).name:sha256(p) for p in (matching,candidate)},**counts}
    if report: Path(report).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return result


def validate_all(root):
    """Run strict streaming checks and the unchanged official validator."""
    import subprocess
    import sys
    root=Path(root)
    report=root/'reports/submission_validation.json'
    summary=json.loads((root/'output/inference_summary.json').read_text())
    signature=summary['signature']
    assert sha256(root/'models/final_model.pkl')==signature['model_sha256'], 'Model changed after inference'
    for name,digest in signature['pipeline_sha256'].items():
        assert sha256(root/'src'/name)==digest, 'Inference code changed after output generation'
    result=validate(root/'output/matching_results.tsv',root/'output/candidate_pairs.tsv',root/'dataset/test',report)
    for name in ('s1_rows','predicted_pairs','empty_matches','candidate_pairs'):
        assert summary[name]==result[name], 'Output count differs from inference summary'
    result['inference_signature']=signature
    command=[sys.executable,str(root/'utils/validate_submission.py'),
        '--matching',str(root/'output/matching_results.tsv'),
        '--candidate',str(root/'output/candidate_pairs.tsv'),
        '--test-dir',str(root/'dataset/test'),'--check-ids']
    env=dict(__import__('os').environ, PYTHONIOENCODING='utf-8')
    completed=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',env=env)
    log=completed.stdout+completed.stderr
    (root/'reports/official_validation.log').write_text(log,encoding='utf-8')
    print(log,flush=True)
    if completed.returncode or 'PASS — no blocking issues found.' not in log or 'WARNING:' in log:
        raise RuntimeError('Official validation failed or warned; see reports/official_validation.log')
    result.update(official_validator='PASS',official_check_ids=True,
        official_validator_sha256=sha256(root/'utils/validate_submission.py'),official_exit_code=completed.returncode)
    report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def package(root,team='Hacksmiths'):
    root=Path(root)
    validation=json.loads((root/'reports/submission_validation.json').read_text())
    assert validation['local_integrity']=='PASS'
    assert validation.get('official_validator')=='PASS', 'Run official validation before packaging'
    for name, digest in validation['output_sha256'].items():
        assert sha256(root/'output'/name)==digest, 'Output changed after validation'
    archive=root/f'{team}_submission.zip'
    files=[]
    for name in ('matching_results.tsv','candidate_pairs.tsv'):
        files.append((root/'output'/name,f'output/{name}'))
    prefix='code/business_entity_resolution/'
    for path in (root/'src').glob('*.py'):files.append((path,prefix+'src/'+path.name))
    for path in (root/'licenses').glob('*.txt'):files.append((path,prefix+'licenses/'+path.name))
    for name in ('README.md','requirements.txt','run_pipeline.py','LICENSE','THIRD_PARTY_NOTICES.md','MODEL_CARD.md'):
        files.append((root/name,prefix+name))
    for path in (root/'utils').glob('*.py'):files.append((path,prefix+'utils/'+path.name))
    for path in (root/'tests').glob('*.py'):files.append((path,prefix+'tests/'+path.name))
    files.append((root/'models/final_model.pkl',prefix+'models/final_model.pkl'))
    files.append((root/'Documentation_template.md','Documentation_template.md'))
    files.append((root/'Documentation_template.md',prefix+'Documentation_template.md'))
    files.append((root/'reports/submission_validation.json','submission_validation.json'))
    for folder in ('reports','experiments','notebooks'):
        for path in (root/folder).iterdir():
            if path.is_file() and path.suffix in ('.md','.json','.csv','.ipynb','.log','.txt') and path.stat().st_size<10_000_000:
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

    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None,'ZIP CRC failure'
    return archive
