"""Train-only interval-ranking auxiliary loss on the Goal 6 baseline."""
from __future__ import annotations

import copy
import json
import math
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import src.goal5_downstream as goal5
from src.goal1_unlabeled import sha256
from src.goal2_smoke import ROOT
from src.goal4_pretrain import normalization, prepare
from src.goal6_input_consistency import consistent_supervision, predict
from src.goal9_interval_decoder import fit_prior, prior_log_ratio, score


OUT = ROOT / "reports/goal11"
CHECKPOINTS = ROOT / "checkpoints/goal11_interval_ranking"
PLAN = {
    "seeds": [17, 43, 89],
    "updates": 600,
    "eval_every": 100,
    "lr": 0.0003,
    "ranking_weight": 0.2,
    "minimum_gap": 2,
    "maximum_gap": 256,
    "prior_strength": 0.5,
    "thresholds": [round(index * 0.05, 2) for index in range(1, 20)],
    "promotion": "mean AP and interval-decoded F1 gains >=2pp; both positive in >=2/3 seeds",
    "max_seconds": 1800,
}


def interval_ranking_loss(logits, labels, coverage, prior_potential,
                          minimum_gap=PLAN["minimum_gap"], maximum_gap=PLAN["maximum_gap"],
                          strength=PLAN["prior_strength"]):
    losses = []
    for row in range(logits.shape[0]):
        starts = torch.where((labels[row] > 0.5) & coverage[row])[0]
        for left, right in zip(starts[:-1], starts[1:]):
            gap = int(right - left)
            if not minimum_gap <= gap <= maximum_gap:
                continue
            indices = torch.arange(int(left) + minimum_gap,
                                   min(int(left) + maximum_gap, logits.shape[1] - 1) + 1,
                                   device=logits.device)
            indices = indices[coverage[row, indices]]
            target = torch.where(indices == right)[0]
            if len(target) != 1:
                raise ValueError("True next boundary is absent from ranking candidates")
            distance = indices - left
            candidate_score = logits[row, indices] + strength * prior_potential[distance - 1]
            losses.append(F.cross_entropy(candidate_score[None], target))
    if not losses:
        return logits.sum() * 0, 0
    return torch.stack(losses).mean(), len(losses)


def run():
    began = time.monotonic()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    previous = json.loads((ROOT / "reports/goal6/registered_protocol.json").read_text(encoding="utf-8"))
    split = previous["split"]
    contract = {"plan": PLAN, "split": split,
                "model": "Goal6-consistent 25-D scratch BoundaryTransformer",
                "baseline": "Goal9 replay of Goal6 scratch checkpoints",
                "test_policy": "Goal5 test works not loaded",
                "status": "registered_before_training", "source_sha256": sha256(Path(__file__).resolve())}
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
    prior = fit_prior(labels, split["train"])
    potential = torch.tensor(prior_log_ratio(prior, np.arange(1, PLAN["maximum_gap"] + 1)), dtype=torch.float32)
    positive = sum(float(labels[piece][0].sum()) for piece in split["train"])
    negative = sum(float(labels[piece][1].sum() - labels[piece][0].sum()) for piece in split["train"])
    pos_weight = min(negative / max(positive, 1), 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    potential = potential.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="none")

    baseline = pd.read_csv(ROOT / "reports/goal9/summary.csv")
    baseline = baseline[baseline.decoder == "interval"].set_index("seed")
    histories, summaries, deltas = [], [], []
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
            clean, hidden, missing, y, use = consistent_supervision(x, valid, lengths, ids, labels)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(clean.to(device), hidden.to(device), missing.to(device), lengths.to(device))
            y_device, use_device = y.to(device), use.to(device)
            bce = criterion(logits, y_device)[use_device].mean()
            ranking, intervals = interval_ranking_loss(logits, y_device, use_device, potential)
            loss = bce + PLAN["ranking_weight"] * ranking
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite interval-ranking loss")
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient):
                raise RuntimeError("Non-finite supervised gradient")
            optimizer.step()
            exposure += int(lengths.sum())
            if step % PLAN["eval_every"] == 0:
                metrics = goal5.score_records(predict(model, validation, labels, device), 0.5)[0]
                histories.append({"seed": seed, "step": step, "loss": float(loss.detach()),
                                  "bce": float(bce.detach()), "ranking": float(ranking.detach()),
                                  "ranking_intervals": intervals, **metrics})
                print(f"{seed} step{step} AP={metrics['AP']:.4f} BCE={float(bce.detach()):.4f} "
                      f"rank={float(ranking.detach()):.4f}", flush=True)
                if metrics["AP"] > best:
                    best = metrics["AP"]
                    best_path = CHECKPOINTS / f"seed{seed}.pt"
                    torch.save({"model": copy.deepcopy(model.state_dict()), "step": step, "plan": PLAN,
                                "split": split, "mean": mean, "scale": scale, "varying": varying,
                                "pos_weight": pos_weight, "prior": prior}, best_path)

        state = torch.load(best_path, weights_only=False, map_location=device)
        model.load_state_dict(state["model"])
        records = predict(model, validation, labels, device)
        scans = [(threshold, goal5.score_records(records, threshold)[0]["F1"]) for threshold in PLAN["thresholds"]]
        threshold, _ = max(scans, key=lambda pair: (pair[1], pair[0]))
        for decoder in ("local", "interval"):
            metrics, performance = score(records, threshold, prior, decoder)
            summaries.append({"seed": seed, "decoder": decoder, "threshold": threshold,
                              "best_step": state["step"], "checkpoint": best_path.relative_to(ROOT).as_posix(),
                              "checkpoint_sha256": sha256(best_path), **metrics})
            performance.to_csv(OUT / f"seed{seed}_{decoder}_validation_performance_metrics.csv", index=False)
        candidate = summaries[-1]
        delta = {metric: float(candidate[metric] - baseline.loc[seed, metric])
                 for metric in ("P", "R", "F1", "Exact_F1", "AP")}
        deltas.append({"seed": seed, **delta})

    delta = pd.DataFrame(deltas)
    promoted = bool(delta.F1.mean() >= 0.02 and delta.AP.mean() >= 0.02 and
                    int((delta.F1 > 0).sum()) >= 2 and int((delta.AP > 0).sum()) >= 2)
    result = {"status": "completed", "scientific_status": "preliminary_reused_validation_mechanism_diagnostic",
              "test_loaded": False, "new_training_runs": len(PLAN["seeds"]),
              "interval_ranking_minus_bce_interval_decoder": deltas,
              "mean_delta": delta.mean(numeric_only=True).to_dict(),
              "positive_seed_counts": {metric: int((delta[metric] > 0).sum()) for metric in ("F1", "AP")},
              "promoted": promoted, "seconds": time.monotonic() - began,
              "limitation": "Reuses the eight-work development split; must pass nested OOF before scientific promotion."}
    pd.DataFrame(histories).to_csv(OUT / "learning_curves.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "validation_summary.csv", index=False)
    delta.to_csv(OUT / "paired_seed_deltas.csv", index=False)
    guard.access_frame().to_csv(OUT / "source_access.csv", index=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
