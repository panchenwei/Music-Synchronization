"""Dev-only diagnosis of pretrain/downstream input consistency."""
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


OUT = ROOT / "reports/goal6"
CHECKPOINTS = ROOT / "checkpoints/goal6_input_consistency"
PLAN = {
    "seeds": [17, 43, 89],
    "updates": 600,
    "eval_every": 100,
    "lr": 0.0003,
    "dropout": 0.1,
    "thresholds": [round(i * 0.05, 2) for i in range(1, 20)],
    "visible_input": "Goal4-consistent: columns 25:58 and METADATA always mask-token",
    "test_policy": "Goal5 test works are not loaded or scored",
    "promotion": "mean AP and F1 gains >=2pp; both positive in at least 2/3 seeds",
    "max_seconds": 1800,
}


def consistent_supervision(x, valid, lengths, ids, labels):
    clean, hidden, missing, y, use = goal5.supervision(x, valid, lengths, ids, labels)
    hidden[:, :, 25:58] = True
    hidden[:, :, METADATA] = True
    clean = clean.masked_fill(hidden, 0)
    return clean, hidden, missing, y, use


@torch.no_grad()
def predict(model, batches, labels, device):
    model.eval()
    records = []
    for x, valid, lengths, ids in batches:
        clean, hidden, missing, y, coverage = consistent_supervision(x, valid, lengths, ids, labels)
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
    prior = json.loads((ROOT / "reports/goal4/registered_protocol.json").read_text(encoding="utf-8"))
    split = {key: prior["split"][key] for key in ("train", "validation")}
    contract = {
        "plan": PLAN,
        "split": split,
        "pretrain_sources": {str(seed): sha256(ROOT / "checkpoints/goal4_pretrain" / f"seed{seed}_music.pt") for seed in PLAN["seeds"]},
        "model": "same 71029-parameter BoundaryTransformer as Goal5",
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
    mean, scale, varying = normalization(train_samples)
    train = prepare(train_samples, mean, scale, varying)
    validation = prepare(validation_samples, mean, scale, varying)
    positive = sum(float(labels[piece][0].sum()) for piece in split["train"])
    negative = sum(float(labels[piece][1].sum() - labels[piece][0].sum()) for piece in split["train"])
    pos_weight = min(negative / max(positive, 1), 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="none")

    history, choices, piece_rows = [], [], []
    for seed in PLAN["seeds"]:
        for condition in ("scratch", "ssl"):
            model = goal5.make_model(seed, device, condition)
            optimizer = torch.optim.Adam(model.parameters(), lr=PLAN["lr"])
            order = np.random.default_rng(seed)
            available, best, best_path, exposure = [], -1.0, None, 0
            for step in range(1, PLAN["updates"] + 1):
                if time.monotonic() - began > PLAN["max_seconds"]:
                    raise RuntimeError("Registered resource deadline reached")
                if not available:
                    available = order.permutation(len(train)).tolist()
                x, valid, lengths, ids = train[available.pop()]
                clean, hidden, missing, y, use = consistent_supervision(x, valid, lengths, ids, labels)
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
                    history.append({"seed": seed, "condition": condition, "step": step,
                                    "train_batch_loss": float(loss.detach()), **metrics})
                    print(f"{seed} {condition} step{step} validationAP={metrics['AP']:.4f}", flush=True)
                    if metrics["AP"] > best:
                        best = metrics["AP"]
                        best_path = CHECKPOINTS / f"seed{seed}_{condition}.pt"
                        torch.save({"model": copy.deepcopy(model.state_dict()), "step": step, "plan": PLAN,
                                    "split": split, "mean": mean, "scale": scale, "varying": varying,
                                    "pos_weight": pos_weight}, best_path)
            state = torch.load(best_path, weights_only=False, map_location=device)
            model.load_state_dict(state["model"])
            records = predict(model, validation, labels, device)
            scans = [(threshold, goal5.score_records(records, threshold)[0]["F1"]) for threshold in PLAN["thresholds"]]
            threshold, f1 = max(scans, key=lambda pair: (pair[1], pair[0]))
            metrics, performance, pieces = goal5.score_records(records, threshold)
            performance.to_csv(OUT / f"seed{seed}_{condition}_validation_performance_metrics.csv", index=False)
            for piece, row in pieces.iterrows():
                piece_rows.append({"seed": seed, "condition": condition, "piece_id": piece, **row.to_dict()})
            choices.append({"seed": seed, "condition": condition, "best_step": state["step"],
                            "checkpoint": best_path.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256(best_path),
                            "threshold": threshold, "validation_AP": best, "validation_F1": f1,
                            "Exact_F1": metrics["Exact_F1"], "P": metrics["P"], "R": metrics["R"],
                            "effective_beat_exposure": exposure})

    frame = pd.DataFrame(choices)
    deltas = []
    for seed in PLAN["seeds"]:
        pair = frame[frame.seed == seed].set_index("condition")
        deltas.append({"seed": seed, **{metric: float(pair.loc["ssl", metric] - pair.loc["scratch", metric])
                                         for metric in ("validation_F1", "Exact_F1", "validation_AP")}})
    delta = pd.DataFrame(deltas)
    mean = delta[["validation_F1", "Exact_F1", "validation_AP"]].mean()
    positive_seeds = {metric: int((delta[metric] > 0).sum()) for metric in ("validation_F1", "validation_AP")}
    promoted = bool(mean.validation_F1 >= 0.02 and mean.validation_AP >= 0.02 and
                    positive_seeds["validation_F1"] >= 2 and positive_seeds["validation_AP"] >= 2)
    result = {
        "status": "completed",
        "scientific_status": "preliminary_reused_validation_diagnostic",
        "test_loaded": False,
        "choices": choices,
        "ssl_minus_scratch": deltas,
        "mean_delta": mean.to_dict(),
        "positive_seed_counts": positive_seeds,
        "promoted": promoted,
        "seconds": time.monotonic() - began,
        "limitation": "The same eight validation works were used in prior method development; this is not terminal generalization evidence.",
    }
    pd.DataFrame(history).to_csv(OUT / "learning_curves.csv", index=False)
    pd.DataFrame(piece_rows).to_csv(OUT / "validation_piece_metrics.csv", index=False)
    frame.to_csv(OUT / "validation_summary.csv", index=False)
    delta.to_csv(OUT / "paired_seed_deltas.csv", index=False)
    guard.access_frame().to_csv(OUT / "source_access.csv", index=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
