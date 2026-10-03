from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class BoundaryCounts:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denom if denom else 0.0


def one_to_one_counts(predicted: Iterable[int], truth: Iterable[int], tolerance: int) -> BoundaryCounts:
    predicted = sorted(set(int(x) for x in predicted))
    truth = sorted(set(int(x) for x in truth))
    candidates = sorted((abs(p - t), p, t) for p in predicted for t in truth if abs(p - t) <= tolerance)
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    for _, pred, true in candidates:
        if pred not in used_pred and true not in used_truth:
            used_pred.add(pred)
            used_truth.add(true)
    tp = len(used_pred)
    return BoundaryCounts(tp=tp, fp=len(predicted) - tp, fn=len(truth) - tp)


def evaluate_piece(piece_id: str, probabilities: np.ndarray, labels: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, float | int | str]:
    valid = mask.astype(bool)
    predicted = np.flatnonzero((probabilities >= threshold) & valid)
    truth = np.flatnonzero((labels > 0.5) & valid)
    row: dict[str, float | int | str] = {
        "piece_id": piece_id,
        "beats": int(valid.sum()),
        "true_boundaries": int(len(truth)),
        "predicted_boundaries": int(len(predicted)),
        "threshold": float(threshold),
    }
    for tolerance in (0, 1, 2):
        counts = one_to_one_counts(predicted, truth, tolerance)
        row[f"precision_tol{tolerance}"] = counts.precision
        row[f"recall_tol{tolerance}"] = counts.recall
        row[f"f1_tol{tolerance}"] = counts.f1
        row[f"tp_tol{tolerance}"] = counts.tp
        row[f"fp_tol{tolerance}"] = counts.fp
        row[f"fn_tol{tolerance}"] = counts.fn
    if labels[valid].min(initial=0) != labels[valid].max(initial=0):
        row["pr_auc"] = float(average_precision_score(labels[valid], probabilities[valid]))
    else:
        row["pr_auc"] = float("nan")
    return row


def evaluate_predictions(predictions: dict[str, np.ndarray], targets: dict[str, tuple[np.ndarray, np.ndarray]], threshold: float) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = [evaluate_piece(piece_id, probabilities, *targets[piece_id], threshold) for piece_id, probabilities in sorted(predictions.items())]
    frame = pd.DataFrame(rows)
    summary: dict[str, float] = {"pieces": float(len(frame)), "threshold": float(threshold)}
    for metric in ["precision_tol0", "recall_tol0", "f1_tol0", "precision_tol1", "recall_tol1", "f1_tol1", "precision_tol2", "recall_tol2", "f1_tol2", "pr_auc"]:
        summary[f"macro_{metric}"] = float(frame[metric].mean())
    for tolerance in (0, 1, 2):
        tp = int(frame[f"tp_tol{tolerance}"].sum())
        fp = int(frame[f"fp_tol{tolerance}"].sum())
        fn = int(frame[f"fn_tol{tolerance}"].sum())
        counts = BoundaryCounts(tp, fp, fn)
        summary[f"micro_precision_tol{tolerance}"] = counts.precision
        summary[f"micro_recall_tol{tolerance}"] = counts.recall
        summary[f"micro_f1_tol{tolerance}"] = counts.f1
    return frame, summary


def choose_threshold(predictions: dict[str, np.ndarray], targets: dict[str, tuple[np.ndarray, np.ndarray]], grid: Iterable[float]) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in grid:
        _, summary = evaluate_predictions(predictions, targets, float(threshold))
        rows.append({"threshold": float(threshold), "macro_f1_tol1": summary["macro_f1_tol1"], "macro_precision_tol1": summary["macro_precision_tol1"], "macro_recall_tol1": summary["macro_recall_tol1"]})
    frame = pd.DataFrame(rows)
    best = frame.sort_values(["macro_f1_tol1", "macro_precision_tol1", "threshold"], ascending=[False, False, False]).iloc[0]
    return float(best.threshold), frame


def bootstrap_macro_ci(frame: pd.DataFrame, metric: str = "f1_tol1", iterations: int = 1000, seed: int = 42) -> tuple[float, float]:
    values = frame[metric].dropna().to_numpy(float)
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(iterations)])
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def non_maximum_suppression(probabilities: np.ndarray, threshold: float, radius: int = 1) -> np.ndarray:
    selected = []
    for index in np.flatnonzero(probabilities >= threshold):
        left = max(0, index - radius)
        right = min(len(probabilities), index + radius + 1)
        if probabilities[index] >= probabilities[left:right].max():
            selected.append(index)
    result = np.zeros_like(probabilities, dtype=bool)
    result[selected] = True
    return result
