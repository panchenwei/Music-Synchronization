"""Replay frozen Goal 6 probabilities with a train-only interval prior."""
from __future__ import annotations

import json
import math
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

import src.goal5_downstream as goal5
from src.goal1_unlabeled import sha256
from src.goal2_smoke import ROOT
from src.goal4_pretrain import normalization, prepare
from src.goal6_input_consistency import predict


OUT = ROOT / "reports/goal9"
STRENGTH = 0.5


def fit_prior(labels, pieces):
    histogram = np.zeros(256, float)
    works = intervals = 0
    for piece in pieces:
        y, coverage = labels[piece]
        starts = np.flatnonzero((y > 0.5) & coverage)
        gaps = [int(right - left) for left, right in zip(starts[:-1], starts[1:]) if coverage[left:right + 1].all()]
        if not gaps:
            continue
        if max(gaps) > 256:
            raise ValueError("Training interval exceeds registered support")
        histogram += np.bincount(np.asarray(gaps) - 1, minlength=256) / len(gaps)
        works += 1
        intervals += len(gaps)
    if not works:
        raise ValueError("No complete training intervals")
    histogram /= works
    mean = float(histogram @ np.arange(1, 257))
    kernel = np.exp(-0.5 * np.arange(-3, 4, dtype=float) ** 2)
    kernel /= kernel.sum()
    histogram = np.convolve(histogram, kernel, "same")
    histogram /= histogram.sum()
    return {"histogram": histogram.tolist(), "mean": mean, "works": works, "intervals": intervals,
            "support": 256, "geometric_mixture": 0.5, "smoothing_sigma": 1.0}


def prior_log_ratio(prior, distance):
    distance = np.asarray(distance, dtype=int)
    if (distance < 1).any():
        raise ValueError("Distances must be positive")
    rate = 1 / max(float(prior["mean"]), 1.00001)
    geometric = np.log(rate) + (distance - 1) * np.log1p(-rate)
    histogram = np.asarray(prior["histogram"])
    local = np.where(distance <= len(histogram), histogram[np.minimum(distance, len(histogram)) - 1], 0)
    log_local = np.full(distance.shape, -np.inf)
    np.log(local, out=log_local, where=local > 0)
    return np.logaddexp(geometric + np.log(0.5), log_local + np.log(0.5)) - geometric


def local_maxima(raw, coverage):
    candidates = []
    for index, value in enumerate(raw):
        if not coverage[index]:
            continue
        left = raw[index - 1] if index and coverage[index - 1] else -np.inf
        right = raw[index + 1] if index + 1 < len(raw) and coverage[index + 1] else -np.inf
        if value > left and value >= right:
            candidates.append(index)
    return np.asarray(candidates, dtype=int)


def decode(raw, coverage, threshold, prior, strength=STRENGTH):
    candidates = local_maxima(np.asarray(raw, float), np.asarray(coverage, bool))
    if not len(candidates):
        return candidates
    probability = np.clip(np.asarray(raw)[candidates], 1e-7, 1 - 1e-7)
    emission = np.log(probability) - np.log1p(-probability) - np.log(threshold / (1 - threshold))
    values = np.empty(len(candidates))
    parent = np.full(len(candidates), -1, dtype=int)
    for index, beat in enumerate(candidates):
        values[index] = emission[index]
        if index:
            score = values[:index] + strength * prior_log_ratio(prior, beat - candidates[:index])
            predecessor = int(np.argmax(score))
            if score[predecessor] > 0:
                values[index] += score[predecessor]
                parent[index] = predecessor
    index = int(np.argmax(values))
    if values[index] <= 0:
        return np.array([], dtype=int)
    path = []
    while index >= 0:
        path.append(int(candidates[index]))
        index = int(parent[index])
    return np.asarray(path[::-1], dtype=int)


def score(records, threshold, prior, decoder):
    rows = []
    for record in records:
        probability, y, coverage = (record[key] for key in ("probability", "labels", "coverage"))
        actual = np.flatnonzero((y > 0.5) & coverage)
        predicted = (goal5.decode(probability, coverage, threshold) if decoder == "local"
                     else decode(probability, coverage, threshold, prior))
        precision, recall, f1 = goal5.match_f1(predicted, actual, 1)
        exact = goal5.match_f1(predicted, actual, 0)[2]
        ap = average_precision_score(y[coverage], probability[coverage])
        rows.append({"piece_id": record["piece_id"], "performance_id": record["performance_id"],
                     "decoder": decoder, "P": precision, "R": recall, "F1": f1,
                     "Exact_F1": exact, "AP": ap, "predictions": len(predicted), "truth": len(actual)})
    frame = pd.DataFrame(rows)
    metrics = frame.groupby("piece_id")[["P", "R", "F1", "Exact_F1", "AP"]].mean().mean().to_dict()
    return metrics, frame


def run():
    began = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / "reports/goal6/registered_protocol.json").read_text(encoding="utf-8"))["split"]
    choices = pd.read_csv(ROOT / "reports/goal6/validation_summary.csv")
    choices = choices[choices.condition == "scratch"].set_index("seed")
    contract = {"split": split, "strength": STRENGTH, "thresholds": "frozen Goal6 scratch thresholds",
                "checkpoint_source_sha256": sha256(ROOT / "reports/goal6/validation_summary.csv"),
                "test_policy": "not loaded", "status": "registered_before_replay",
                "source_sha256": sha256(Path(__file__).resolve())}
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
    validation = prepare(validation_samples, mean, scale, varying)
    prior = fit_prior(labels, split["train"])
    (OUT / "train_interval_prior.json").write_text(json.dumps(prior, indent=2), encoding="utf-8")

    summaries, deltas = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed, choice in choices.iterrows():
        model = goal5.make_model(int(seed), device, "scratch")
        checkpoint = ROOT / choice.checkpoint
        state = torch.load(checkpoint, weights_only=False, map_location=device)
        model.load_state_dict(state["model"])
        records = predict(model, validation, labels, device)
        local, local_rows = score(records, float(choice.threshold), prior, "local")
        interval, interval_rows = score(records, float(choice.threshold), prior, "interval")
        pd.concat([local_rows, interval_rows]).to_csv(OUT / f"seed{int(seed)}_performance_metrics.csv", index=False)
        for name, metrics in (("local", local), ("interval", interval)):
            summaries.append({"seed": int(seed), "decoder": name, "threshold": float(choice.threshold), **metrics})
        delta = {metric: float(interval[metric] - local[metric]) for metric in ("P", "R", "F1", "Exact_F1", "AP")}
        if not math.isclose(delta["AP"], 0, abs_tol=1e-15):
            raise AssertionError("Decoder changed probability AP")
        deltas.append({"seed": int(seed), **delta})
        for metric in ("F1", "Exact_F1", "AP"):
            expected = float(choice.validation_F1 if metric == "F1" else choice.Exact_F1 if metric == "Exact_F1" else choice.validation_AP)
            if not math.isclose(local[metric], expected, abs_tol=1e-10):
                raise AssertionError(f"Goal6 local replay mismatch: {seed} {metric}")

    delta = pd.DataFrame(deltas)
    promoted = bool(delta.F1.mean() >= 0.02 and (delta.F1 > 0).sum() >= 2)
    result = {"status": "completed", "scientific_status": "preliminary_decoder_diagnostic",
              "new_training_runs": 0, "test_loaded": False, "prior": {k: v for k, v in prior.items() if k != "histogram"},
              "deltas": deltas, "mean_delta": delta.mean(numeric_only=True).to_dict(),
              "promoted_for_decoding_only": promoted, "seconds": time.monotonic() - began,
              "limitation": "Validation replay only; any F1 gain is decoding, not improved representation or AP."}
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    delta.to_csv(OUT / "paired_seed_deltas.csv", index=False)
    guard.access_frame().to_csv(OUT / "source_access.csv", index=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
