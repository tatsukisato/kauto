"""Benchmark generate_submission and template matching costs.

Measures:
- repo.fit time
- generate_submission total time (for a sample)
- scorer call count and total time (wraps seq_identity)

Usage:
python competitions/rna2/scripts/benchmark_generate_submission.py --sample 50
"""
import time
import sys, os
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from baseline.data import load_sequences, load_labels
from baseline.template_model import TemplateRepository
from baseline.search import seq_identity
from baseline.predict import generate_submission


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=os.path.join('competitions','rna2','data','stanford-rna-3d-folding-2'))
    p.add_argument('--sample', type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    sample_n = args.sample

    train_seq = load_sequences(os.path.join(data_dir, 'train_sequences.csv'))
    train_labels = load_labels(os.path.join(data_dir, 'train_labels.csv'))
    val_seq = load_sequences(os.path.join(data_dir, 'validation_sequences.csv'))

    repo = TemplateRepository()
    t0 = time.perf_counter()
    repo.fit(train_seq, train_labels)
    fit_time = time.perf_counter() - t0

    # wrapper scorer to count and time
    call_count = 0
    total_scorer_time = 0.0

    def wrapped_scorer(a,b):
        nonlocal call_count, total_scorer_time
        t1 = time.perf_counter()
        res = seq_identity(a,b)
        dt = time.perf_counter() - t1
        call_count += 1
        total_scorer_time += dt
        return res

    sample = val_seq.head(sample_n)
    t1 = time.perf_counter()
    sub = generate_submission(sample, repo, wrapped_scorer, n_structures=1)
    gen_time = time.perf_counter() - t1

    print(f'fit_time: {fit_time:.3f}s')
    print(f'generate_submission on {len(sample)} queries: {gen_time:.3f}s')
    print(f'scorer calls: {call_count}, total_scorer_time: {total_scorer_time:.3f}s')
    if gen_time>0:
        print(f'scorer_time fraction: {total_scorer_time/gen_time:.2%}')
    print('output rows:', len(sub))


if __name__ == '__main__':
    main()
