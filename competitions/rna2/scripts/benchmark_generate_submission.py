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
import multiprocessing

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from baseline.data import load_sequences, load_labels
from baseline.template_model import TemplateRepository
from baseline.search import seq_identity
from baseline.predict import generate_submission


class WrappedScorer:
    def __init__(self, call_count_proxy, total_time_proxy):
        self.call_count = call_count_proxy
        self.total_time = total_time_proxy

    def __call__(self, a, b):
        t1 = time.perf_counter()
        res = seq_identity(a, b)
        dt = time.perf_counter() - t1
        try:
            self.call_count.value += 1
            self.total_time.value += dt
        except Exception:
            pass
        return res


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=os.path.join('competitions','rna2','data','stanford-rna-3d-folding-2'))
    p.add_argument('--sample', type=int, default=50)
    p.add_argument('--n-jobs', type=int, default=1, help='Number of worker processes to use for generate_submission')
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

    # wrapper scorer to count and time (use multiprocessing.Manager proxies so workers update counts)
    manager = multiprocessing.Manager()
    call_count = manager.Value('i', 0)
    total_scorer_time = manager.Value('d', 0.0)
    wrapped = WrappedScorer(call_count, total_scorer_time)

    sample = val_seq.head(sample_n)
    t1 = time.perf_counter()
    # call the parallel version with requested number of jobs
    try:
        sub = generate_submission(sample, repo, wrapped, n_structures=1, n_jobs=args.n_jobs)
    except TypeError:
        # fallback if generate_submission is the old API: import and call parallel directly
        from baseline.predict import generate_submission_parallel
        sub = generate_submission_parallel(sample, repo, wrapped, n_structures=1, n_jobs=args.n_jobs)
    gen_time = time.perf_counter() - t1

    print(f'fit_time: {fit_time:.3f}s')
    print(f'generate_submission on {len(sample)} queries: {gen_time:.3f}s')
    print(f'scorer calls: {call_count.value}, total_scorer_time: {total_scorer_time.value:.3f}s')
    if gen_time>0:
        print(f'scorer_time fraction: {total_scorer_time.value/gen_time:.2%}')
    print('output rows:', len(sub))


if __name__ == '__main__':
    main()
