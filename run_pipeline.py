"""Run stages explicitly or the full resumable pipeline."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','eda','index-train','train','index-test','infer','validate','report','package'],default='all')
    parser.add_argument('--sample-size',type=int,default=12000)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--team',default='Hacksmiths')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    for name in ('cache','output','reports','experiments','models'):(root/name).mkdir(exist_ok=True)
    stages=['eda','index-train','train','index-test','infer','validate','report','package'] if args.stage=='all' else [args.stage]
    for stage in stages:
        if stage=='eda':
            from src.eda import run_eda
            run_eda(root)
        elif stage.startswith('index-'):
            from src.retrieval import build_index,build_prepared
            split=stage.split('-')[1]
            build_index([root/f'dataset/{split}/{split}_source{i}.tsv' for i in (2,3)],root/f'cache/{split}_index')
            if split=='test':
                build_prepared([root/f'dataset/{split}/{split}_source{i}.tsv' for i in (2,3)],root/f'cache/{split}_index')
        elif stage=='train':
            from src.training import train
            train(root,args.sample_size)
            if args.sample_size==12000:
                from src.source_balance_experiment import run
                run(root)
        elif stage=='infer':
            from src.inference import infer
            infer(root,args.workers)
        elif stage=='validate':
            from src.submission import validate
            validate(root/'output/matching_results.tsv',root/'output/candidate_pairs.tsv',root/'dataset/test',root/'reports/submission_validation.json')
        elif stage=='report':
            from src.reporting import report
            report(root)
        elif stage=='package':
            from src.submission import package
            print(package(root,args.team))


if __name__=='__main__':main()
