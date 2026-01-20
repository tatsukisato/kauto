"""Lightweight RMSD evaluation pipeline (Kabsch + C1' extraction).

Usage (from repo root):
python competitions/rna2/scripts/run_rmsd_evaluate.py --data-dir competitions/rna2/data/stanford-rna-3d-folding-2 
"""
import os
import sys
import argparse
import json
import pandas as pd
import numpy as np

# ensure local baseline package is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from baseline.data import load_sequences, load_labels
from baseline.template_model import TemplateRepository
from baseline.predict import generate_submission

from kabsch_utils import align_and_rmsd
from backbone_utils import extract_C1p_coords, extract_coords_from_submission


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=os.path.join('competitions','rna2','data','stanford-rna-3d-folding-2'))
    p.add_argument('--out-dir', default=os.path.join('competitions','rna2','experiments'))
    p.add_argument('--train-sequences', default='train_sequences.csv')
    p.add_argument('--train-labels', default='train_labels.csv')
    p.add_argument('--validation-sequences', default='validation_sequences.csv')
    p.add_argument('--validation-labels', default='validation_labels.csv')
    p.add_argument('--n-structures', type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    train_seq = load_sequences(os.path.join(data_dir, args.train_sequences))
    train_labels = load_labels(os.path.join(data_dir, args.train_labels))
    val_seq = load_sequences(os.path.join(data_dir, args.validation_sequences))
    val_labels = load_labels(os.path.join(data_dir, args.validation_labels))

    repo = TemplateRepository()
    repo.fit(train_seq, train_labels)

    print('Generating predictions for validation set...')
    pred_df = generate_submission(val_seq, repo, scorer=lambda a,b: 0.0, n_structures=args.n_structures)
    if pred_df.empty:
        print('No predictions generated; exiting')
        return

    true_groups = extract_C1p_coords(val_labels)
    pred_groups = extract_coords_from_submission(pred_df)

    results = []
    for tid, (true_coords, true_resids) in true_groups.items():
        pred_entry = pred_groups.get(tid)
        if pred_entry is None:
            results.append({'target_id': tid, 'n_matched': 0, 'mean_rmsd': None, 'median_rmsd': None, 'skipped': True})
            continue
        pred_coords, pred_resids = pred_entry
        # build index maps
        true_map = {r: i for i,r in enumerate(true_resids)}
        pred_map = {r: i for i,r in enumerate(pred_resids)}
        common = sorted(set(true_map.keys()) & set(pred_map.keys()), key=lambda x:int(x) if x.isdigit() else x)
        pairs = []
        for r in common:
            t_idx = true_map[r]
            p_idx = pred_map[r]
            tc = true_coords[t_idx]
            pc = pred_coords[p_idx]
            if np.any(np.isnan(tc)) or np.any(np.isnan(pc)):
                continue
            pairs.append((tc, pc))
        if len(pairs) < 3:
            results.append({'target_id': tid, 'n_matched': len(pairs), 'mean_rmsd': None, 'median_rmsd': None, 'skipped': True})
            continue
        P = np.vstack([t for t,p in pairs])
        Q = np.vstack([p for t,p in pairs])
        Q_aligned, rmsd = align_and_rmsd(P, Q)
        results.append({'target_id': tid, 'n_matched': len(pairs), 'mean_rmsd': float(rmsd), 'median_rmsd': float(np.median(np.linalg.norm(P-Q_aligned,axis=1))), 'skipped': False})

    out_csv = os.path.join(out_dir, 'validation_rmsd_per_target.csv')
    out_json = os.path.join(out_dir, 'validation_rmsd_summary.json')
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)

    summary = {
        'n_targets': int(df.shape[0]),
        'n_evaluated': int((~df['skipped']).sum()),
        'mean_rmsd': None if df['mean_rmsd'].dropna().empty else float(df['mean_rmsd'].dropna().mean()),
        'median_rmsd': None if df['mean_rmsd'].dropna().empty else float(df['mean_rmsd'].dropna().median())
    }
    with open(out_json,'w') as f:
        json.dump(summary, f, indent=2)

    print('Wrote:', out_csv, out_json)


if __name__ == '__main__':
    main()
