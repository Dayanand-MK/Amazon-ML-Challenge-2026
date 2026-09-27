"""Project-local validation CLI, NOT the missing Amazon official script."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.submission import validate

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matching',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--test-dir',required=True)
    parser.add_argument('--report')
    args=parser.parse_args()
    validate(args.matching,args.candidate,args.test_dir,args.report)
