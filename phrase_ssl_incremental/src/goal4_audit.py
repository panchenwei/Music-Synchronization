"""Recompute saved validation predictions using NumPy, independent of train loss."""
import json

import numpy as np
import pandas as pd

from src.goal1_unlabeled import FEATURE_NAMES, sha256
from src.goal2_smoke import GROUPS, METADATA
from src.goal4_pretrain import OUT, PLAN


def recompute():
    contract = json.loads((OUT / "registered_protocol.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    split = contract["split"]
    for a, b in (("train", "validation"), ("train", "reserved_not_loaded"), ("validation", "reserved_not_loaded")):
        assert not set(split[a]) & set(split[b])
    access = pd.read_csv(OUT / "cache_access.csv")
    assert set(access.piece_id) == set(split["train"] + split["validation"])
    assert not set(access.piece_id) & set(split["reserved_not_loaded"])
    raw = pd.read_csv(OUT / "raw_source_access.csv")
    assert len(raw) == 76 and set(raw.role) == {"mazurkabl_beat_time", "mazurkabl_beat_dyn"}
    rows, feature_rows, baseline_rows = [], [], []
    reference = {}
    for directory in sorted((OUT / "predictions").iterdir()):
        seed_string, condition = directory.name.split("_", 1)
        seed = int(seed_string.removeprefix("seed"))
        for path in sorted(directory.glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                prediction, target, valid, spans, lengths = (z[k] for k in ("prediction", "target", "valid", "spans", "lengths"))
                assert prediction.shape == target.shape == valid.shape
                assert target.shape[1] == lengths.max() and target.shape[-1] == 58
                assert np.isfinite(prediction).all()
                assert not valid[:, :, METADATA].any()
                assert not spans[np.arange(target.shape[1])[None] >= lengths[:, None]].any()
                identity = (target.tobytes(), valid.tobytes(), spans.tobytes(), lengths.tobytes())
                if path.name in reference:
                    assert reference[path.name] == identity
                else:
                    reference[path.name] = identity
                selected = valid & spans[:, :, None] & (np.arange(target.shape[1])[None, :, None] < lengths[:, None, None])
                hidden = np.broadcast_to(spans[:, :, None], target.shape).copy()
                for shift in range(1, 5):
                    hidden[:, shift:] |= spans[:, :-shift, None]
                    hidden[:, :-shift] |= spans[:, shift:, None]
                hidden[:, :, 25:] = True
                hidden[:, :, METADATA] = True
                linear = np.zeros_like(target)
                for row, length in enumerate(lengths):
                    for col in range(58):
                        visible = np.flatnonzero(valid[row, :length, col] & ~hidden[row, :length, col])
                        if len(visible):
                            linear[row, :length, col] = np.interp(np.arange(length), visible, target[row, visible, col])
                for row, piece in enumerate(z["piece_ids"].tolist()):
                    identifier = {"piece_id": piece, "performance_id": str(z["performance_ids"][row]), "view": int(z["view"])}
                    for group, (start, stop) in GROUPS.items():
                        use = selected[row, :, start:stop]
                        if not use.any():
                            continue
                        difference = prediction[row, :, start:stop][use] - target[row, :, start:stop][use]
                        rows.append({**identifier, "seed": seed, "condition": condition, "group": group, "mse": float(np.mean(difference ** 2))})
                        if seed == PLAN["seeds"][0] and condition == "music":
                            for method, output in (("mean", np.zeros_like(target)), ("interpolation", linear)):
                                error = output[row, :, start:stop][use] - target[row, :, start:stop][use]
                                baseline_rows.append({**identifier, "method": method, "group": group, "mse": float(np.mean(error ** 2))})
                        for col in range(start, stop):
                            use_col = selected[row, :, col]
                            if use_col.any():
                                error = prediction[row, :, col][use_col] - target[row, :, col][use_col]
                                baseline = target[row, :, col][use_col]
                                feature_rows.append({**identifier, "seed": seed, "condition": condition, "group": group,
                                                     "feature": FEATURE_NAMES[col], "mse": float(np.mean(error ** 2)), "mean_mse": float(np.mean(baseline ** 2))})
    frame = pd.DataFrame(rows)
    summaries = pd.read_csv(OUT / "run_summary.csv")
    errors = []
    for record in summaries.itertuples(index=False):
        selected = frame[(frame.seed == record.seed) & (frame.condition == record.condition)]
        value = selected.groupby(["piece_id", "group"]).mse.mean().groupby("piece_id").mean().mean()
        errors.append(abs(value - record.best_validation))
        assert np.isclose(value, record.best_validation, rtol=1e-6, atol=1e-7)
    for method in ("mean", "interpolation"):
        selected = pd.DataFrame(baseline_rows).query("method == @method")
        value = selected.groupby(["piece_id", "group"]).mse.mean().groupby("piece_id").mean().mean()
        assert np.isclose(value, result["baselines"][method], rtol=1e-6, atol=1e-7)
    for seed in PLAN["seeds"]:
        pair = summaries[summaries.seed == seed]
        assert pair.train_order_mask_sha256.nunique() == 1
        assert pair.effective_beat_exposure.nunique() == 1
    frame.to_csv(OUT / "independent_metrics.csv", index=False)
    features = pd.DataFrame(feature_rows)
    features.to_csv(OUT / "feature_metrics.csv", index=False)
    aggregate = features.groupby(["condition", "group", "feature", "piece_id"])[["mse", "mean_mse"]].mean().groupby(["condition", "group", "feature"]).mean().reset_index()
    aggregate["relative_gain_vs_mean"] = (aggregate.mean_mse - aggregate.mse) / aggregate.mean_mse
    aggregate.to_csv(OUT / "feature_summary.csv", index=False)
    group_summary = frame.groupby(["condition", "group", "piece_id"]).mse.mean().groupby(["condition", "group"]).mean().unstack(0)
    base = pd.DataFrame(baseline_rows).groupby(["method", "group", "piece_id"]).mse.mean().groupby(["method", "group"]).mean().unstack(0)
    group_summary.join(base).to_csv(OUT / "group_summary.csv")
    audit = {"status": "passed", "six_saved_model_predictions_recomputed": True,
             "max_macro_difference": max(errors), "identical_validation_targets_and_masks": True,
             "both_baselines_independently_recomputed": True, "matched_train_order_masks_and_beat_exposure": True,
             "train_validation_reserved_disjoint": True, "reserved_cache_not_accessed": True,
             "raw_reads_allowed_only": True, "prediction_directories": len(summaries),
             "script_sha256_matches_registration": contract["source_sha256"] == sha256(OUT.parents[1] / "src/goal4_pretrain.py")}
    assert audit["script_sha256_matches_registration"]
    (OUT / "independent_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    print(group_summary.join(base).to_string())


if __name__ == "__main__":
    recompute()
