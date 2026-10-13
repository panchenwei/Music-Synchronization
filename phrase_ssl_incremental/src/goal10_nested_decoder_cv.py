"""Nested work-level CV for the frozen train-only interval decoder."""
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

import src.goal5_downstream as goal5
from src.goal1_unlabeled import sha256
from src.goal2_smoke import ROOT
from src.goal4_pretrain import normalization, prepare
from src.goal6_input_consistency import consistent_supervision, predict
from src.goal9_interval_decoder import fit_prior, score


OUT = ROOT / "reports/goal10"
CHECKPOINTS = ROOT / "checkpoints/goal10_nested_decoder_cv"
PLAN = {
    "fold_seed": 20261013,
    "folds": 5,
    "model_seed": 43,
    "updates": 600,
    "eval_every": 100,
    "lr": 0.0003,
    "thresholds": [round(index * 0.05, 2) for index in range(1, 20)],
    "interval_strength": 0.5,
    "promotion": "OOF F1 gain >=2pp and positive in at least 4/5 folds; AP unchanged",
    "max_seconds": 3600,
}


def build_nested_splits(pieces, seed=PLAN["fold_seed"], folds=PLAN["folds"]):
    pieces = np.asarray(sorted(pieces), dtype=object)
    shuffled = pieces[np.random.default_rng(seed).permutation(len(pieces))].tolist()
    groups = [shuffled[index::folds] for index in range(folds)]
    result = []
    for fold in range(folds):
        outer = groups[fold]
        inner = groups[(fold + 1) % folds]
        train = [piece for index, group in enumerate(groups) if index not in {fold, (fold + 1) % folds} for piece in group]
        if set(train) & set(inner) or set(train) & set(outer) or set(inner) & set(outer):
            raise AssertionError("Nested split overlap")
        if set(train) | set(inner) | set(outer) != set(pieces):
            raise AssertionError("Nested split does not cover all development works")
        result.append({"fold": fold, "train": train, "inner": inner, "outer": outer})
    outer_once = [piece for split in result for piece in split["outer"]]
    if sorted(outer_once) != sorted(pieces):
        raise AssertionError("Each development work must be outer exactly once")
    return result


def train_fold(split, fold, config, guard, device, began):
    fold_out = OUT / f"fold{fold}"
    fold_out.mkdir(parents=True, exist_ok=True)
    goal5.OUT = fold_out
    labels = {}
    train_samples = goal5.load_subset("train", split, guard, config, labels)
    inner_samples = goal5.load_subset("inner", split, guard, config, labels)
    mean, scale, varying = normalization(train_samples)
    train = prepare(train_samples, mean, scale, varying)
    inner = prepare(inner_samples, mean, scale, varying)
    positive = sum(float(labels[piece][0].sum()) for piece in split["train"])
    negative = sum(float(labels[piece][1].sum() - labels[piece][0].sum()) for piece in split["train"])
    pos_weight = min(negative / max(positive, 1), 50)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="none")

    model = goal5.make_model(PLAN["model_seed"], device, "scratch")
    optimizer = torch.optim.Adam(model.parameters(), lr=PLAN["lr"])
    order = np.random.default_rng(PLAN["model_seed"] + fold)
    available, best, best_path, exposure, history = [], -1.0, None, 0, []
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
            metrics = goal5.score_records(predict(model, inner, labels, device), 0.5)[0]
            history.append({"fold": fold, "step": step, "train_batch_loss": float(loss.detach()), **metrics})
            print(f"fold{fold} step{step} inner_AP={metrics['AP']:.4f}", flush=True)
            if metrics["AP"] > best:
                best = metrics["AP"]
                best_path = CHECKPOINTS / f"fold{fold}_seed{PLAN['model_seed']}.pt"
                torch.save({"model": copy.deepcopy(model.state_dict()), "step": step, "plan": PLAN,
                            "split": split, "mean": mean, "scale": scale, "varying": varying,
                            "pos_weight": pos_weight}, best_path)

    state = torch.load(best_path, weights_only=False, map_location=device)
    model.load_state_dict(state["model"])
    inner_records = predict(model, inner, labels, device)
    scans = [(threshold, goal5.score_records(inner_records, threshold)[0]["F1"]) for threshold in PLAN["thresholds"]]
    threshold, inner_f1 = max(scans, key=lambda pair: (pair[1], pair[0]))
    prior = fit_prior(labels, split["train"])
    frozen = {"fold": fold, "best_step": state["step"], "inner_AP": best, "inner_F1": inner_f1,
              "threshold": threshold, "pos_weight": pos_weight, "effective_beat_exposure": exposure,
              "checkpoint": best_path.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256(best_path),
              "train_prior_works": prior["works"], "train_prior_intervals": prior["intervals"],
              "outer_loaded_after_freeze": True}

    # The outer fold is deliberately loaded only after every choice above is frozen.
    outer_samples = goal5.load_subset("outer", split, guard, config, labels)
    outer = prepare(outer_samples, mean, scale, varying)
    outer_records = predict(model, outer, labels, device)
    local, local_rows = score(outer_records, threshold, prior, "local")
    interval, interval_rows = score(outer_records, threshold, prior, "interval")
    if not math.isclose(local["AP"], interval["AP"], abs_tol=1e-15):
        raise AssertionError("Decoder changed probability AP")
    performance = pd.concat([local_rows, interval_rows], ignore_index=True)
    performance.insert(0, "fold", fold)
    performance.to_csv(fold_out / "outer_performance_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(fold_out / "learning_curve.csv", index=False)
    pd.DataFrame(scans, columns=["threshold", "inner_F1"]).to_csv(fold_out / "inner_threshold_scan.csv", index=False)
    (fold_out / "train_interval_prior.json").write_text(json.dumps(prior, indent=2), encoding="utf-8")
    return frozen, local, interval, performance, history


def run():
    began = time.monotonic()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    previous = json.loads((ROOT / "reports/goal4/registered_protocol.json").read_text(encoding="utf-8"))
    development = previous["split"]["train"] + previous["split"]["validation"]
    reserved = previous["split"]["reserved_not_loaded"]
    folds = build_nested_splits(development)
    if set(development) & set(reserved):
        raise AssertionError("Reserved works overlap development works")
    contract = {"plan": PLAN, "folds": folds, "reserved_not_loaded": reserved,
                "input": "Goal6-consistent 25-D content; columns 25:58 and metadata masked",
                "model": "71029-parameter scratch BoundaryTransformer",
                "status": "registered_before_training", "source_sha256": sha256(Path(__file__).resolve())}
    (OUT / "registered_protocol.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    config = tomllib.loads((ROOT / "configs/data_paths.toml").read_text(encoding="utf-8"))
    guard = goal5.DownstreamGuard(Path(config["raw"]["dcml_chopin"]), Path(config["raw"]["mazurkabl"]),
                                  Path(config["legacy_read_only"]["experiment_root"]))
    guard.install()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries, frozen_rows, performances, histories = [], [], [], []
    for split in folds:
        fold = split["fold"]
        frozen, local, interval, performance, history = train_fold(split, fold, config, guard, device, began)
        frozen_rows.append(frozen)
        performances.append(performance)
        histories.extend(history)
        for decoder, metrics in (("local", local), ("interval", interval)):
            summaries.append({"fold": fold, "decoder": decoder, **metrics})
        print(f"fold{fold} outer F1 local={local['F1']:.4f} interval={interval['F1']:.4f}", flush=True)
        del performance
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = pd.DataFrame(summaries)
    oof = pd.concat(performances, ignore_index=True)
    metrics = ["P", "R", "F1", "Exact_F1", "AP"]
    overall = oof.groupby(["decoder", "piece_id"])[metrics].mean().groupby("decoder").mean()
    delta = (overall.loc["interval"] - overall.loc["local"]).to_dict()
    fold_wide = summary.pivot(index="fold", columns="decoder", values=metrics)
    fold_deltas = pd.DataFrame({metric: fold_wide[(metric, "interval")] - fold_wide[(metric, "local")] for metric in metrics})
    fold_deltas.index.name = "fold"
    promoted = bool(delta["F1"] >= 0.02 and int((fold_deltas.F1 > 0).sum()) >= 4 and
                    math.isclose(delta["AP"], 0, abs_tol=1e-15))
    result = {"status": "completed", "scientific_status": "preliminary_nested_oof_decoder_evidence",
              "test_loaded": False, "development_works": len(development), "outer_predictions_per_work": 1,
              "overall": {decoder: {metric: float(overall.loc[decoder, metric]) for metric in metrics}
                          for decoder in ("local", "interval")},
              "interval_minus_local": {key: float(value) for key, value in delta.items()},
              "positive_fold_counts": {metric: int((fold_deltas[metric] > 0).sum()) for metric in metrics},
              "promoted_for_decoding_only": promoted, "seconds": time.monotonic() - began,
              "limitation": "Nested OOF on the 38 development works, not a new corpus or terminal test; model seed fixed to 43."}
    summary.to_csv(OUT / "fold_summary.csv", index=False)
    fold_deltas.to_csv(OUT / "fold_deltas.csv")
    oof.to_csv(OUT / "oof_performance_metrics.csv", index=False)
    pd.DataFrame(frozen_rows).to_csv(OUT / "frozen_choices.csv", index=False)
    pd.DataFrame(histories).to_csv(OUT / "learning_curves.csv", index=False)
    guard.access_frame().to_csv(OUT / "source_access.csv", index=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
