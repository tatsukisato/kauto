import numpy as np
from sklearn.metrics import f1_score
from typing import Dict, Sequence
from numpy.typing import ArrayLike


def compute_evaluation_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    bg_label: int = 11,
    player_labels: Sequence[int] = list(range(11)),
) -> Dict[str, float]:
    """
    Compute evaluation metrics used across experiments.

    Returns dict with keys:
      - macro_f1_all: overall macro F1 (all classes)
      - macro_f1_player: macro F1 restricted to player labels (0..10)
      - bg_tp: true positives for background class
      - bg_fp: false positives for background class
      - bg_recall: bg recall
      - bg_precision: bg precision
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Overall Macro F1
    try:
        macro_f1_all = float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        macro_f1_all = 0.0

    # Player-only macro F1 (exclude background)
    player_mask = y_true != bg_label
    if np.sum(player_mask) > 0:
        try:
            macro_f1_player = float(
                f1_score(
                    y_true[player_mask],
                    y_pred[player_mask],
                    average="macro",
                    labels=list(player_labels),
                )
            )
        except Exception:
            macro_f1_player = 0.0
    else:
        macro_f1_player = 0.0

    # Background stats
    bg_mask = y_true == bg_label
    bg_pred_mask = y_pred == bg_label
    bg_tp = int(np.sum(bg_mask & bg_pred_mask))
    bg_fp = int(np.sum((~bg_mask) & bg_pred_mask))
    bg_total = int(np.sum(bg_mask))
    bg_recall = float(bg_tp / bg_total) if bg_total > 0 else 0.0
    bg_precision = float(bg_tp / (bg_tp + bg_fp)) if (bg_tp + bg_fp) > 0 else 0.0

    return {
        "macro_f1_all": macro_f1_all,
        "macro_f1_player": macro_f1_player,
        "bg_tp": bg_tp,
        "bg_fp": bg_fp,
        "bg_recall": bg_recall,
        "bg_precision": bg_precision,
    }
