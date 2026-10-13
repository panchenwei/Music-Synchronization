"""Numerical smoke test for a globally normalized interval-aware CRF loss."""
from __future__ import annotations

import json
import math
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import src.goal5_downstream as goal5
from src.goal1_unlabeled import sha256
from src.goal2_smoke import ROOT
from src.goal4_pretrain import normalization, prepare
from src.goal6_input_consistency import consistent_supervision
from src.goal9_interval_decoder import fit_prior, prior_log_ratio


OUT = ROOT / "reports/goal12"
PLAN = {"seed": 43, "updates": 60, "lr": 0.001, "minimum_gap": 2, "maximum_gap": 128,
        "prior_strength": 0.5, "promotion": "DP==brute force; finite nonzero gradients; final loss <50% initial",
        "max_seconds": 600}


def path_log_partition(unary, transition, minimum_gap=PLAN["minimum_gap"],
                       maximum_gap=PLAN["maximum_gap"]):
    """Log-sum-exp over the empty path and all legal ordered boundary subsets."""
    endings = []
    zero = unary.new_zeros(1)
    for right in range(len(unary)):
        lower = max(0, right - maximum_gap)
        upper = right - minimum_gap + 1
        if upper > lower:
            previous = torch.stack(endings[lower:upper])
            distance = right - torch.arange(lower, upper, device=unary.device)
            choices = torch.cat([zero, previous + transition[distance - 1]])
        else:
            choices = zero
        endings.append(unary[right] + torch.logsumexp(choices, dim=0))
    return torch.logsumexp(torch.cat([zero, torch.stack(endings)]), dim=0)


def sequence_nll(unary, labels, transition, minimum_gap=PLAN["minimum_gap"],
                 maximum_gap=PLAN["maximum_gap"]):
    starts = torch.where(labels > 0.5)[0]
    if len(starts) > 1:
        gaps = starts[1:] - starts[:-1]
        if bool(((gaps < minimum_gap) | (gaps > maximum_gap)).any()):
            raise ValueError("Ground-truth interval is outside registered CRF support")
        truth = unary[starts].sum() + transition[gaps - 1].sum()
    elif len(starts) == 1:
        truth = unary[starts].sum()
    else:
        truth = unary.new_zeros(())
    return (path_log_partition(unary, transition, minimum_gap, maximum_gap) - truth) / len(unary)


def batch_nll(logits, labels, coverage, transition):
    losses = []
    for row in range(logits.shape[0]):
        positions = torch.where(coverage[row])[0]
        if not len(positions):
            continue
        if len(positions) > 1 and not bool((positions[1:] - positions[:-1] == 1).all()):
            raise ValueError("CRF smoke requires contiguous annotation coverage")
        losses.append(sequence_nll(logits[row, positions], labels[row, positions], transition))
    if not losses:
        raise ValueError("No covered sequence for CRF loss")
    return torch.stack(losses).mean()


def run():
    began = time.monotonic()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    previous = json.loads((ROOT / "reports/goal6/registered_protocol.json").read_text(encoding="utf-8"))
    split = {"train": previous["split"]["train"]}
    contract = {"plan": PLAN, "split": split, "validation_loaded": False, "test_loaded": False,
                "status": "registered_before_smoke", "source_sha256": sha256(Path(__file__).resolve())}
    (OUT / "registered_protocol.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    config = tomllib.loads((ROOT / "configs/data_paths.toml").read_text(encoding="utf-8"))
    guard = goal5.DownstreamGuard(Path(config["raw"]["dcml_chopin"]), Path(config["raw"]["mazurkabl"]),
                                  Path(config["legacy_read_only"]["experiment_root"]))
    guard.install()
    goal5.OUT = OUT
    labels = {}
    samples = goal5.load_subset("train", split, guard, config, labels)
    mean, scale, varying = normalization(samples)
    batches = prepare(samples, mean, scale, varying)
    batch = min(batches, key=lambda item: int(item[2].max()))
    prior = fit_prior(labels, split["train"])
    transition_np = PLAN["prior_strength"] * prior_log_ratio(prior, np.arange(1, PLAN["maximum_gap"] + 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transition = torch.tensor(transition_np, dtype=torch.float32, device=device)
    model = goal5.make_model(PLAN["seed"], device, "scratch")
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=PLAN["lr"])
    x, valid, lengths, ids = batch
    clean, hidden, missing, y, coverage = consistent_supervision(x, valid, lengths, ids, labels)
    clean, hidden, missing = clean.to(device), hidden.to(device), missing.to(device)
    lengths, y, coverage = lengths.to(device), y.to(device), coverage.to(device)
    history, nonzero_gradient = [], False
    for step in range(PLAN["updates"] + 1):
        if time.monotonic() - began > PLAN["max_seconds"]:
            raise RuntimeError("Registered smoke deadline reached")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(clean, hidden, missing, lengths)
        loss = batch_nll(logits, y, coverage, transition)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite global CRF loss")
        history.append({"step": step, "loss": float(loss.detach())})
        if step == PLAN["updates"]:
            break
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError("Missing or non-finite global CRF gradients")
        nonzero_gradient |= any(bool((gradient != 0).any()) for gradient in gradients)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    initial, final = history[0]["loss"], history[-1]["loss"]
    passed = bool(nonzero_gradient and final < 0.5 * initial)
    result = {"status": "completed" if passed else "failed_gate", "scientific_status": "numerical_smoke_only",
              "validation_loaded": False, "test_loaded": False, "batch_performances": len(ids),
              "batch_max_length": int(lengths.max()), "initial_loss": initial, "final_loss": final,
              "loss_ratio": final / initial, "finite_nonzero_gradients": nonzero_gradient,
              "promotion_gate_passed": passed, "seconds": time.monotonic() - began,
              "limitation": "Single training batch tiny-overfit; no accuracy or generalization claim."}
    pd.DataFrame(history).to_csv(OUT / "tiny_overfit_curve.csv", index=False)
    guard.access_frame().to_csv(OUT / "source_access.csv", index=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
