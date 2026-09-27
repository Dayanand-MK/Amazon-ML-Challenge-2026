import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.normalization import normalize_text, representations, detect_scripts
from src.retrieval import build_index, build_prepared, CandidateIndex, prepared
from src.features import pair_features, FEATURE_NAMES
from src.evaluate import metrics
from src.submission import validate


def write_tsv(path,header,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,delimiter='\t')
        writer.writerow(header)
        writer.writerows(rows)


class PipelineTests(unittest.TestCase):
    def test_unicode_marks_preserved(self):
        text='राम मार्केटिंग प्राइवेट लिमिटेड'
        self.assertEqual(normalize_text(text),text)
        self.assertEqual(normalize_text('CAFÉ'),normalize_text('CAFE\u0301'))
        self.assertEqual(representations(text)['raw'],text)
        self.assertIn('Cyrillic',detect_scripts('Москва'))
        self.assertEqual(detect_scripts('Café राम'),['Devanagari','Latin'])
        self.assertIn('CJK',detect_scripts('北京'))
        self.assertEqual(normalize_text('  X—Y__Z  '),'x y z')

    def test_missing_does_not_match(self):
        f=pair_features(prepared(('S1-1','','','')),prepared(('S2-1','','','')))
        self.assertEqual(len(f),len(FEATURE_NAMES))
        for i in (0,9,18,20):self.assertEqual(f[i],0)
        self.assertTrue(np.all(np.isfinite(f)))

    def test_lost_truth_counts_as_false_negative(self):
        m=metrics([1,0],[.9,.8],[0,1],[2,0],.5)
        self.assertEqual(m['candidate_recall'],.5)
        self.assertEqual(m['false_negatives'],1)
        for key in ('precision','recall','F0.5'):self.assertEqual(m[key],.5)
        self.assertEqual(m['singleton_accuracy'],0)

    def test_cross_country_retrieval_and_submission(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            header=['entity_id','business_name','business_address','country']
            paths=[root/'test_source2.tsv',root/'test_source3.tsv']
            write_tsv(paths[0],header,[('S2-2','Café Zenith','10 Main Road','France')])
            write_tsv(paths[1],header,[('S3-3','राम मार्केटिंग','','India')])
            left=('S1-1','Café Zenith','10 Main Road','unknown')
            write_tsv(root/'test_source1.tsv',header,[left,('S1-4','No match','','')])
            build_index(paths,root/'index')
            build_prepared(paths,root/'index')
            index=CandidateIndex(paths,root/'index')
            for rid in range(2):self.assertEqual(index.record(rid),prepared(index.raw(rid)))
            self.assertIn('S2-2',[index.record(r)[0] for r in index.retrieve(prepared(left))])
            index.close()
            del index
            write_tsv(root/'matching.tsv',['source1_entity_id','matched_entity_ids'],[('S1-1','S2-2'),('S1-4','')])
            write_tsv(root/'candidate.tsv',['source1_entity_id','target_entity_id'],[('S1-1','S2-2')])
            result=validate(root/'matching.tsv',root/'candidate.tsv',root)
            self.assertEqual(result['local_integrity'],'PASS')
            self.assertEqual(result['empty_matches'],1)
            write_tsv(root/'matching.tsv',['source1_entity_id','matched_entity_ids'],[('S1-1','S2-999'),('S1-4','')])
            with self.assertRaises(AssertionError):validate(root/'matching.tsv',root/'candidate.tsv',root)

    def test_batched_inference_end_to_end(self):
        from src.inference import infer
        from src.model import estimators,save_model
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for folder in ('dataset/test','cache','models','output'):(root/folder).mkdir(parents=True,exist_ok=True)
            header=['entity_id','business_name','business_address','country']
            left=('S1-1','Zenith Workshop','10 Main Road','France')
            right=('S2-2','Zenith Workshop','10 Main Rd','France')
            other=('S3-3','Other store','20 Side Street','US')
            paths=[root/f'dataset/test/test_source{i}.tsv' for i in (2,3)]
            write_tsv(root/'dataset/test/test_source1.tsv',header,[left,('S1-4','Unrelated name','','')])
            write_tsv(paths[0],header,[right]);write_tsv(paths[1],header,[other])
            build_index(paths,root/'cache/test_index')
            x=np.asarray([pair_features(prepared(left),prepared(right)),pair_features(prepared(left),prepared(other))]*30)
            model=estimators()['logistic_regression'].fit(x,[1,0]*30)
            save_model(root/'models/final_model.pkl',model,.8,FEATURE_NAMES,{'max_posting':1200,'max_candidates':24,'source_balanced':True})
            result=infer(root,workers=2,batch_size=1)
            self.assertEqual(result['s1_rows'],2)
            checked=validate(root/'output/matching_results.tsv',root/'output/candidate_pairs.tsv',root/'dataset/test')
            self.assertEqual(checked['local_integrity'],'PASS')
            self.assertEqual(result,infer(root,workers=2,batch_size=1))


if __name__=='__main__':unittest.main()
