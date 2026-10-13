"""Dev-only small-BiGRU comparison against the Goal 6 Transformer."""
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import src.goal5_downstream as goal5
from src.goal1_unlabeled import sha256
from src.goal2_smoke import ROOT
from src.goal4_pretrain import normalization, prepare
from src.goal6_input_consistency import consistent_supervision, predict


OUT = ROOT / "reports/goal7"
CHECKPOINTS = ROOT / "checkpoints/goal7_small_bigru"
PLAN = {
    "seeds": [17, 43, 89],
    "updates": 600,
    "eval_every": 100,
    "lr": 0.0003,
    "dropout": 0.1,
    "thresholds": [round(i * 0.05, 2) for i in range(1, 20)],
    "input": "same Goal6-consistent visible columns",
    "reference": "Goal6 scratch Transformer, matched by seed",
    "test_policy": "Goal5 test works are not loaded or scored",
    "promotion": "mean AP and F1 gains >=2pp; both positive in at least 2/3 seeds",
    "max_seconds": 900,
}


class SmallBiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(58))
        self.missing_token = nn.Parameter(torch.zeros(58))
        self.projection = nn.Linear(58, 32)
        self.encoder = nn.GRU(32, 14, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(PLAN["dropout"])
        self.normalization = nn.LayerNorm(28)
        self.classifier = nn.Linear(28, 1)

    def forward(self, clean, hidden, missing, lengths):
        x = torch.where(hidden, self.mask_token, clean)
        x = torch.where(missing & ~hidden, self.missing_token, x)
        packed = pack_padded_sequence(self.projection(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.shape[1])
        return self.classifier(self.normalization(self.dropout(encoded))).squeeze(-1)


def run():
    began = time.monotonic()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    prior = json.loads((ROOT / "reports/goal6/registered_protocol.json").read_text(encoding="utf-8"))
    split = prior["split"]
    reference = pd.read_csv(ROOT / "reports/goal6/validation_summary.csv")
    reference = reference[reference.condition == "scratch"].set_index("seed")
    contract = {
        "plan": PLAN,
        "split": split,
        "model": "58->32; one-layer bidirectional GRU hidden14/direction; dropout0.1; LayerNorm28; 28->1",
        "parameters": sum(parameter.numel() for parameter in SmallBiGRU().parameters()),
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
    mean, scale, varying = normalization(train_samples)
    train = prepare(train_samples, mean, scale, varying)
    validation = prepare(validation_samples, mean, scale, varying)
    positive = sum(float(labels[piece][0].sum()) for piece in split["train"])
    negative = sum(float(labels[piece][1].sum() - labels[piece][0].sum()) for piece in split["train"])
    pos_weight = min(negative / max(positive, 1), 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="none")

    history, rows, pieces_all = [], [], []
    for seed in PLAN["seeds"]:
        torch.manual_seed(seed)
        model = SmallBiGRU().to(device)
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
                history.append({"seed": seed, "step": step, "train_batch_loss": float(loss.detach()), **metrics})
                print(f"{seed} BiGRU step{step} validationAP={metrics['AP']:.4f}", flush=True)
                if metrics["AP"] > best:
                    best = metrics["AP"]
                    best_path = CHECKPOINTS / f"seed{seed}.pt"
                    torch.save({"model": copy.deepcopy(model.state_dict()), "step": step, "plan": PLAN,
                                "split": split, "mean": mean, "scale": scale, "varying": varying,
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
                     "delta_F1_vs_transformer": f1 - float(baseline.validation_F1),
                     "delta_AP_vs_transformer": best - float(baseline.validation_AP),
                     "effective_beat_exposure": exposure})

    frame = pd.DataFrame(rows)
    mean_delta_f1 = float(frame.delta_F1_vs_transformer.mean())
    mean_delta_ap = float(frame.delta_AP_vs_transformer.mean())
    positive = {"F1": int((frame.delta_F1_vs_transformer > 0).sum()),
                "AP": int((frame.delta_AP_vs_transformer > 0).sum())}
    promoted = bool(mean_delta_f1 >= 0.02 and mean_delta_ap >= 0.02 and positive["F1"] >= 2 and positive["AP"] >= 2)
    result = {
        "status": "completed",
        "scientific_status": "preliminary_reused_validation_architecture_diagnostic",
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "test_loaded": False,
        "runs": rows,
        "mean_delta_F1_vs_transformer": mean_delta_f1,
        "mean_delta_AP_vs_transformer": mean_delta_ap,
        "positive_seed_counts": positive,
        "promoted": promoted,
        "seconds": time.monotonic() - began,
        "limitation": "Repeated development validation only; not terminal generalization evidence.",
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
