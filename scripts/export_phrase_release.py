"""Export small, non-private metadata/examples for the phrase-boundary code release."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "phrase_boundary_overnight"
REPORT = PROJECT / "reports/mentor_continuation_20260916"
OUT = PROJECT / "release"
sys.path[:0] = [str(REPORT), str(PROJECT)]

from score_only_export_training import data_u
from src.phase3_features import SCORE_CUE_NAMES


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    frame, data = data_u()
    assert set(frame.columns) == {"fold", "piece_id", "opus", "split"}
    for fold, part in frame.groupby("fold"):
        train = set(part.loc[part.split == "train", "opus"])
        validation = set(part.loc[part.split == "validation", "opus"])
        assert train.isdisjoint(validation), f"opus leakage in fold {fold}"
    frame.sort_values(["fold", "split", "piece_id"]).to_csv(OUT / "current_splits.csv", index=False)

    pid = "chopin_op06_no1"
    item = data[pid]
    performance_id = str(item["performance_ids"][0])
    probability_file = REPORT / "demo_masked_candidate_op06_no1/probabilities.npz"
    starts_file = REPORT / "demo_masked_candidate_op06_no1/phrase_starts.csv"
    with np.load(probability_file, allow_pickle=False) as values:
        index = list(values["performance_ids"]).index(performance_id)
        probabilities = values["probabilities"][index]
    decoded = pd.read_csv(starts_file)
    decoded = set(decoded.loc[decoded.performance_id == performance_id, "beat_index"].astype(int))

    first_positive = int(np.flatnonzero(item["labels"] * item["label_mask"])[0])
    selected = np.arange(first_positive - 6, first_positive + 7)
    feature_names = [str(x) for x in item["curve_feature_names"]]
    feature_names += list(SCORE_CUE_NAMES)
    feature_names += ["tempo_level_1", "tempo_level_3", "tempo_level_6", "tempo_level_12",
                      "tempo_level_24", "tempo_quality", "lag_available",
                      "tempo_direction", "direction_available"]
    for staff in (0, 1):
        for window in (3, 6, 12):
            feature_names += [f"staff{staff}_w{window}_{name}" for name in
                              ("forward_similarity", "backward_similarity", "novelty", "available")]
    assert len(feature_names) == 58 and item["curves"].shape[-1] == 58
    rows = pd.DataFrame(item["curves"][0, selected], columns=feature_names)
    rows.insert(0, "beat_index", selected)
    rows.insert(1, "measure_number", item["measure_number"][selected])
    rows.insert(2, "beat_in_measure", item["beat_number"][selected])
    rows.insert(3, "label_phrase_start", item["labels"][selected].astype(int))
    rows.insert(4, "label_is_scored", item["label_mask"][selected].astype(int))
    rows.insert(5, "mean_model_probability", probabilities[selected])
    rows.insert(6, "decoded_phrase_start", [int(i in decoded) for i in selected])
    assert rows.label_phrase_start.sum() == 1 and rows.decoded_phrase_start.sum() == 1
    rows.to_csv(OUT / "example_input_label_prediction.csv", index=False)

    summary_file = REPORT / "locked_holdout_test/primary_summary.json"
    means_file = REPORT / "temporal_mask_replication/means.csv"
    export_file = REPORT / "temporal_mask_replication/export.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    means = pd.read_csv(means_file)
    exported = json.loads(export_file.read_text(encoding="utf-8"))
    deployment_folds = [{
        "fold": fold["fold"],
        "ensemble_threshold": fold["threshold"],
        "prior_mean_interval": fold["prior"]["mean"],
        "prior_interval_count": fold["prior"]["intervals"],
        "components": [{"seed": component["seed"], "threshold": component["threshold"],
                        "checkpoint_sha256": component["sha256"]}
                       for component in fold["components"]],
    } for fold in exported["folds"]]
    internal = means.query("arm == 'B' and mode == 'ensemble' and policy == 'B10'").iloc[0]
    config = {
        "status": {"code": "implemented", "local_results": "reproduced", "generalization": "preliminary"},
        "task": "DCML new phrase-start event detection",
        "grid": "one row per quarter-note duration (quarterbeat)",
        "input_groups": {"performance_curves": 9, "score_cues": 16,
                         "tempo_hierarchy": 9, "ordered_motif_recurrence": 24},
        "model": {"input": 58, "projection": 32, "bidirectional_gru_hidden_per_direction": 14,
                  "gru_layers": 1, "dropout": 0.2, "parameters_per_model": 6005,
                  "ensemble_seeds": [42, 43, 44, 45]},
        "training": {"window_quarterbeats": 64, "batch_size": 8, "optimizer_steps": 600,
                     "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
                     "positive_weight": 10, "augmentation": "contiguous 4-position normalized-zero time mask"},
        "decoding": {"ensemble": "equal probability mean", "policy": "B10 fixed train-derived inter-start prior",
                     "matching": "distance-first greedy one-to-one", "tolerance_quarterbeats": 1,
                     "deployment_folds": deployment_folds},
        "results": {"internal_development_macro_f1_tol1": float(internal.macro_f1_tol1),
                    "internal_development_exact_f1": float(internal.macro_f1_tol0),
                    "internal_development_raw_ap": float(internal.raw_ap),
                    "historical_holdout_macro_f1_tol1": summary["macro_f1"],
                    "historical_holdout_exact_f1": summary["exact_f1"],
                    "historical_holdout_strict_blind": summary["strict_blind_test"]},
        "example": {"piece_id": pid, "performance_id": performance_id,
                    "row_range": [int(selected[0]), int(selected[-1])],
                    "source_probability_sha256": sha256(probability_file)},
    }
    (OUT / "current_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "splits": len(frame), "example_rows": len(rows),
                      "features": 58, "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
