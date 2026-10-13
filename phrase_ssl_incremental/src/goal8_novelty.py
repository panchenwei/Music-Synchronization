"""Dev-only multiscale before/after novelty input study."""
from __future__ import annotations

import copy
import json
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import src.goal5_downstream as goal5
from src.goal1_unlabeled import sha256
from src.goal2_smoke import METADATA, ROOT
from src.goal4_pretrain import normalization, prepare


OUT = ROOT / "reports/goal8"
CHECKPOINTS = ROOT / "checkpoints/goal8_novelty"
SCALES = (2, 4, 8)
SAFE_CONTENT = tuple(range(0, 7)) + tuple(range(12, 25))
PLAN = {
    "seeds": [17, 43, 89],
    "updates": 600,
    "eval_every": 100,
    "lr": 0.0003,
    "dropout": 0.1,
    "thresholds": [round(i * 0.05, 2) for i in range(1, 20)],
    "novelty_scales": list(SCALES),
    "novelty_source_columns": list(SAFE_CONTENT),
    "reference": "Goal6 scratch Transformer, matched by seed",
    "test_policy": "Goal5 test works are not loaded or scored",
    "promotion": "mean AP and F1 gains >=2pp; both positive in at least 2/3 seeds",
    "max_seconds": 900,
}


def novelty_features(values: torch.Tensor, valid: torch.Tensor, lengths: torch.Tensor):
    output = torch.zeros(values.shape[0], values.shape[1], len(SCALES), dtype=values.dtype)
    observed = torch.zeros_like(output, dtype=torch.bool)
    columns = torch.tensor(SAFE_CONTENT)
    for row, length in enumerate(lengths.tolist()):
        source = values[row, :length, columns]
        source_valid = valid[row, :length, columns]
        for channel, scale in enumerate(SCALES):
            for boundary in range(scale, length - scale + 1):
                left_valid = source_valid[boundary - scale:boundary]
                right_valid = source_valid[boundary:boundary + scale]
                use = left_valid.any(0) & right_valid.any(0)
                if not use.any():
                    continue
                left = (source[boundary - scale:boundary] * left_valid).sum(0) / left_valid.sum(0).clamp_min(1)
                right = (source[boundary:boundary + scale] * right_valid).sum(0) / right_valid.sum(0).clamp_min(1)
                output[row, boundary, channel] = (left[use] - right[use]).square().mean().sqrt()
                observed[row, boundary, channel] = True
    return output, observed


def add_novelty(batches, mean=None, scale=None):
    raw = []
    for values, valid, lengths, ids in batches:
        novelty, observed = novelty_features(values, valid, lengths)
        raw.append((values, valid, lengths, ids, novelty, observed))
    if mean is None or scale is None:
        count = torch.stack([row[5].sum((0, 1)) for row in raw]).sum(0).clamp_min(1)
        total = torch.stack([torch.where(row[5], row[4], 0).sum((0, 1)) for row in raw]).sum(0)
        mean = total / count
        square = torch.stack([torch.where(row[5], row[4].square(), 0).sum((0, 1)) for row in raw]).sum(0) / count
        scale = (square - mean.square()).clamp_min(0).sqrt().clamp_min(1e-3)
    augmented = []
    for values, valid, lengths, ids, novelty, observed in raw:
        values, valid = values.clone(), valid.clone()
        values[:, :, 25:28] = torch.where(observed, (novelty - mean) / scale, 0)
        valid[:, :, 25:28] = observed
        augmented.append((values, valid, lengths, ids))
    return augmented, mean, scale


def novelty_supervision(x, valid, lengths, ids, labels):
    clean, hidden, missing, y, use = goal5.supervision(x, valid, lengths, ids, labels)
    hidden[:, :, 25:58] = True
    hidden[:, :, 25:28] = False
    hidden[:, :, METADATA] = True
    clean = clean.masked_fill(hidden, 0)
    return clean, hidden, missing, y, use


@torch.no_grad()
def predict(model, batches, labels, device):
    model.eval()
    records = []
    for x, valid, lengths, ids in batches:
        clean, hidden, missing, y, coverage = novelty_supervision(x, valid, lengths, ids, labels)
        probability = torch.sigmoid(model(clean.to(device), hidden.to(device), missing.to(device), lengths.to(device))).cpu().numpy()
        if not np.isfinite(probability).all():
            raise RuntimeError("Non-finite validation probability")
        for row, info in enumerate(ids):
            n = int(lengths[row])
            records.append({**info, "probability": probability[row, :n].copy(), "labels": y[row, :n].numpy(),
                            "coverage": coverage[row, :n].numpy()})
    return records


def run():
    began = time.monotonic()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / "reports/goal6/registered_protocol.json").read_text(encoding="utf-8"))["split"]
    reference = pd.read_csv(ROOT / "reports/goal6/validation_summary.csv")
    reference = reference[reference.condition == "scratch"].set_index("seed")
    contract = {
        "plan": PLAN,
        "split": split,
        "model": "same 71029-parameter BoundaryTransformer as Goal6",
        "reference_sha256": sha256(ROOT / "reports/goal6/validation_summary.csv"),
        "status": "registered_before_training",
        "source_sha256": sha256(Path(__file__).resolve()),
    }
    (OUT / "registered_protocol.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    config = tomllib.loads((ROOT / "configs/data_paths.toml").read_text(encoding="utf-8"))
    guard = goal5.DownstreamGuard(Path(config["raw"]["dcml_chopin"]), Path(config["raw"]["mazurkabl"]),
                                  Path(config["legacy_read_only"]["experiment_root"]))
    guard.install()
    goal5.OUT = OUT
    labels = {}
    train_samples = goal5.load_subset("train", split, guard, config, labels)
    validation_samples = goal5.load_subset("validation", split, guard, config, labels)
    base_mean, base_scale, varying = normalization(train_samples)
    train, novelty_mean, novelty_scale = add_novelty(prepare(train_samples, base_mean, base_scale, varying))
    validation = add_novelty(prepare(validation_samples, base_mean, base_scale, varying), novelty_mean, novelty_scale)[0]
    positive = sum(float(labels[piece][0].sum()) for piece in split["train"])
    negative = sum(float(labels[piece][1].sum() - labels[piece][0].sum()) for piece in split["train"])
    pos_weight = min(negative / max(positive, 1), 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="none")

    history, rows, pieces_all = [], [], []
    for seed in PLAN["seeds"]:
        model = goal5.make_model(seed, device, "scratch")
        optimizer = torch.optim.Adam(model.parameters(), lr=PLAN["lr"])
        order = np.random.default_rng(seed)
        available, best, best_path, exposure = [], -1.0, None, 0
        for step in range(1, PLAN["updates"] + 1):
            if time.monotonic() - began > PLAN["max_seconds"]:
                raise RuntimeError("Registered resource deadline reached")
            if not available:
                available = order.permutation(len(train)).tolist()
            x, valid, lengths, ids = train[available.pop()]
            clean, hidden, missing, y, use = novelty_supervision(x, valid, lengths, ids, labels)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(clean.to(device), hidden.to(device), missing.to(device), lengths.to(device))
            loss = criterion(logits, y.to(device))[use.to(device)].mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite supervised loss")
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient):
                raise RuntimeError("Non-finite supervised gradient")
            optimizer.step()
            exposure += int(lengths.sum())
            if step % PLAN["eval_every"] == 0:
                metrics = goal5.score_records(predict(model, validation, labels, device), 0.5)[0]
                history.append({"seed": seed, "step": step, "train_batch_loss": float(loss.detach()), **metrics})
                print(f"{seed} novelty step{step} validationAP={metrics['AP']:.4f}", flush=True)
                if metrics["AP"] > best:
                    best = metrics["AP"]
                    best_path = CHECKPOINTS / f"seed{seed}.pt"
                    torch.save({"model": copy.deepcopy(model.state_dict()), "step": step, "plan": PLAN,
                                "split": split, "base_mean": base_mean, "base_scale": base_scale, "varying": varying,
                                "novelty_mean": novelty_mean, "novelty_scale": novelty_scale,
                                "pos_weight": pos_weight}, best_path)
        state = torch.load(best_path, weights_only=False, map_location=device)
        model.load_state_dict(state["model"])
        records = predict(model, validation, labels, device)
        scans = [(threshold, goal5.score_records(records, threshold)[0]["F1"]) for threshold in PLAN["thresholds"]]
        threshold, f1 = max(scans, key=lambda pair: (pair[1], pair[0]))
        metrics, performance, pieces = goal5.score_records(records, threshold)
        performance.to_csv(OUT / f"seed{seed}_validation_performance_metrics.csv", index=False)
        for piece, values in pieces.iterrows():
            pieces_all.append({"seed": seed, "piece_id": piece, **values.to_dict()})
        baseline = reference.loc[seed]
        rows.append({"seed": seed, "best_step": state["step"], "threshold": threshold,
                     "checkpoint": best_path.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256(best_path),
                     "P": metrics["P"], "R": metrics["R"], "validation_F1": f1,
                     "Exact_F1": metrics["Exact_F1"], "validation_AP": best,
                     "delta_F1_vs_control": f1 - float(baseline.validation_F1),
                     "delta_AP_vs_control": best - float(baseline.validation_AP),
                     "effective_beat_exposure": exposure})

    frame = pd.DataFrame(rows)
    mean_delta_f1 = float(frame.delta_F1_vs_control.mean())
    mean_delta_ap = float(frame.delta_AP_vs_control.mean())
    positive = {"F1": int((frame.delta_F1_vs_control > 0).sum()),
                "AP": int((frame.delta_AP_vs_control > 0).sum())}
    promoted = bool(mean_delta_f1 >= 0.02 and mean_delta_ap >= 0.02 and positive["F1"] >= 2 and positive["AP"] >= 2)
    result = {
        "status": "completed",
        "scientific_status": "preliminary_reused_validation_representation_diagnostic",
        "test_loaded": False,
        "novelty_mean": novelty_mean.tolist(),
        "novelty_scale": novelty_scale.tolist(),
        "runs": rows,
        "mean_delta_F1_vs_control": mean_delta_f1,
        "mean_delta_AP_vs_control": mean_delta_ap,
        "positive_seed_counts": positive,
        "promoted": promoted,
        "seconds": time.monotonic() - began,
        "limitation": "Repeated development validation only; adapted novelty mechanism is not a reproduction of section-level SSM papers.",
    }
    pd.DataFrame(history).to_csv(OUT / "learning_curves.csv", index=False)
    pd.DataFrame(pieces_all).to_csv(OUT / "validation_piece_metrics.csv", index=False)
    frame.to_csv(OUT / "validation_summary.csv", index=False)
    guard.access_frame().to_csv(OUT / "source_access.csv", index=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
