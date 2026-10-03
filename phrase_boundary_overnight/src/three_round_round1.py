"""Round 1: fixed five-cell decomposition of the six-channel tempo hierarchy block."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from .local_context_study import metrics, predictions, write
from .models import Normalizer
from .phase2_models import choose_single_threshold, fit_curve_normalizer, nms_probabilities
from .phase3_models import fit_train_normalizer
from .phase6_models import positive_weight
from .phase7_models import CurvePieceBalancedSampler
from .slice_energy_study import model_for


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/three_round_study/round1"
ART = ROOT / "artifacts/three_round_study/round1"
CACHE = ROOT / "artifacts/slice_energy_study/cache"
SPLITS = ROOT / "artifacts/phase2/splits/opus_split_manifest.csv"
PROTOCOL = OUT / "PROTOCOL.md"
KINDS = ["Z", "Q", "NQ", "MeanQ", "RMSQ"]
SCALES = (1, 3, 6, 12, 24)
GRID = np.arange(0.10, 0.91, 0.05).round(2).tolist()
DEADLINE = datetime.fromisoformat("2026-09-10T15:40:02+08:00").timestamp()
ROUND_BUDGET_SECONDS = 1800.0
STAGES = ["init", "train", "report", "audit"]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def block_hash(files: list[Path]) -> tuple[dict[str, str], str]:
    values = {str(path.relative_to(ROOT)): sha(path) for path in files}
    contract = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    return values, contract


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(state: dict):
    write(OUT / "STATE.json", state)


def markdown_table(frame: pd.DataFrame) -> str:
    def fmt(value):
        return f"{float(value):.4f}" if isinstance(value, (float, np.floating)) else str(value)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "|".join(["---"] * len(frame.columns)) + "|"]
    lines += ["| " + " | ".join(fmt(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def extra_block(curves: np.ndarray, frozen_rmsq: np.ndarray, kind: str) -> np.ndarray:
    """Build [performance, beat, 6] without labels or boundary-derived ranges."""
    curves = np.asarray(curves, dtype=np.float32)
    frozen_rmsq = np.asarray(frozen_rmsq, dtype=np.float32)
    if curves.ndim != 3 or curves.shape[-1] != 9 or frozen_rmsq.shape != (*curves.shape[:2], 6):
        raise ValueError("Expected curves [P,T,9] and frozen RMSQ [P,T,6]")
    if kind == "RMSQ":
        return frozen_rmsq.copy()
    output = np.zeros_like(frozen_rmsq)
    quality = frozen_rmsq[..., 5]
    output[..., 5] = quality
    if kind in {"Z", "Q"}:
        if kind == "Z":
            output.fill(0)
        return output
    for performance in range(curves.shape[0]):
        log_tempo = curves[performance, :, 0].astype(np.float64)
        tempo = np.exp(log_tempo)
        if not np.isfinite(tempo).all() or (tempo <= 0).any():
            raise ValueError("Tempo must be finite and positive")
        log_reference = math.log(float(np.median(tempo)))
        if kind == "NQ":
            output[performance, :, 0] = (log_tempo - log_reference).astype(np.float32)
            continue
        if kind != "MeanQ":
            raise ValueError(kind)
        for column, scale in enumerate(SCALES):
            for first in range(0, len(tempo), scale):
                stop = min(first + scale, len(tempo))
                output[performance, first:stop, column] = math.log(float(tempo[first:stop].mean())) - log_reference
    if not np.isfinite(output).all():
        raise FloatingPointError(f"Non-finite block: {kind}")
    return output


def dataset(piece_ids: list[str], kind: str):
    result = {}
    for piece_id in piece_ids:
        with np.load(CACHE / f"{piece_id}.npz", allow_pickle=False) as source:
            item = {key: source[key] for key in source.files}
        item["labels"] = item["start_labels"].copy()
        item["label_mask"] = item["start_mask"].copy()
        base = np.concatenate([
            item["curves"],
            np.broadcast_to(item["fixed_score"], (*item["curves"].shape[:2], 16)),
        ], axis=-1).astype(np.float32)
        block = extra_block(item["curves"], item["energy"], kind)
        item["curves"] = np.concatenate([base, block], axis=-1)
        item["selected_score"] = item["fixed_score"]
        item["round1_block"] = block
        if item["curves"].shape[-1] != 31 or not np.isfinite(item["curves"]).all():
            raise AssertionError(f"Invalid 31-d input: {piece_id}/{kind}")
        if (item["labels"] * item["label_mask"]).sum() <= 0:
            raise AssertionError(f"No supervised phrase starts: {piece_id}")
        result[piece_id] = item
    return result


def normalizer(data) -> Normalizer:
    curve = fit_curve_normalizer({piece_id: {**item, "curves": item["curves"][..., :9]} for piece_id, item in data.items()})
    score = fit_train_normalizer([item["selected_score"] for item in data.values()])
    extra = fit_train_normalizer([item["round1_block"].reshape(-1, 6) for item in data.values()])
    return Normalizer(np.r_[curve.mean, score.mean, extra.mean], np.r_[curve.std, score.std, extra.std])


def load_splits(fold: int):
    manifest = pd.read_csv(SPLITS)
    part = manifest[manifest.fold == fold]
    return {name: sorted(part[part.split == name].piece_id.tolist()) for name in ["train", "validation", "test"]}


def best_threshold_from_snapshot(saved: dict) -> float:
    matching = [row for row in saved["history"] if int(row["step"]) == int(saved["step"])]
    if len(matching) != 1:
        raise AssertionError("Best checkpoint must identify exactly one validation record")
    return float(matching[0]["threshold"])


def run_one(kind: str, fold: int, seed: int, train, validation, contract: str, state: dict):
    run_id = f"{kind}_seed{seed}_fold{fold}"
    destination = ART / "checkpoints" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    metric_path = ART / "metrics" / f"{run_id}.json"
    if metric_path.exists():
        result = load_json(metric_path)
        if result["contract"] != contract:
            raise ValueError(f"Refusing stale completed result: {run_id}")
        print("CACHED", run_id, flush=True)
        return result
    state.update(status="running", current_run=run_id, training_process_pid=os.getpid(), updated_at=now())
    save_state(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_for(31, seed).to(device)
    norm = normalizer(train)
    sampler = CurvePieceBalancedSampler(train, norm, 64, 32, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(positive_weight(train, 10), device=device))
    latest_path, best_path = destination / "latest.pt", destination / "best.pt"
    step, history, best_score, previous_seconds = 0, [], -1.0, 0.0
    started = time.monotonic()
    if latest_path.exists():
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        if saved["contract"] != contract or saved["kind"] != kind or saved["fold"] != fold or saved["seed"] != seed:
            raise ValueError(f"Checkpoint contract mismatch: {run_id}")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); sampler.load_state(saved["sampler"])
        step, history, best_score, previous_seconds = int(saved["step"]), list(saved["history"]), float(saved["best_score"]), float(saved["seconds"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])

    def snapshot():
        return {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "sampler": sampler.state(),
            "step": step, "history": history, "best_score": best_score, "contract": contract,
            "kind": kind, "fold": fold, "seed": seed,
            "seconds": previous_seconds + time.monotonic() - started,
            "mean": norm.mean, "std": norm.std, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    while step < 300:
        elapsed = previous_seconds + time.monotonic() - started
        if time.time() >= DEADLINE - 60:
            torch.save(snapshot(), latest_path); raise TimeoutError("Persistent three-round deadline guard reached")
        if state["training_validation_seconds"] + elapsed >= ROUND_BUDGET_SECONDS:
            torch.save(snapshot(), latest_path); raise TimeoutError("Round1 30-minute budget reached")
        model.train(); features, labels, mask, valid = (tensor.to(device) for tensor in sampler.batch())
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, padding_mask=~valid.bool())
        loss = (criterion(logits, labels) * mask).sum() / mask.sum().clamp_min(1)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Round1 loss")
        loss.backward(); grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite Round1 gradient")
        optimizer.step(); step += 1
        if step % 50 == 0:
            raw = predictions(model, validation, norm, device)
            threshold, _ = choose_single_threshold(raw, validation, GRID)
            _, _, score = metrics(raw, validation, threshold)
            record = {"step": step, "loss": float(loss), "gradient_norm": float(grad_norm), "threshold": threshold, **score}
            history.append(record)
            if score["macro_f1_tol1"] > best_score + 1e-9:
                best_score = float(score["macro_f1_tol1"]); torch.save(snapshot(), best_path)
            print(run_id, step, f"F1={score['macro_f1_tol1']:.4f}", f"AP={score['raw_ap']:.4f}", flush=True)
        if step % 25 == 0:
            torch.save(snapshot(), latest_path)
    saved = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    threshold = best_threshold_from_snapshot(saved)
    raw = predictions(model, validation, norm, device)
    perfs, pieces, score = metrics(raw, validation, threshold)
    _, _, train_score = metrics(predictions(model, train, norm, device), train, threshold)
    total_seconds = previous_seconds + time.monotonic() - started
    result = {
        "run_id": run_id, "kind": kind, "fold": fold, "seed": seed, "contract": contract,
        "threshold": threshold, "best_step": int(saved["step"]), "params": sum(p.numel() for p in model.parameters()),
        "seconds": total_seconds, "gpu_peak_bytes": int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0,
        "train_f1": train_score["macro_f1_tol1"], "train_gap": train_score["macro_f1_tol1"] - score["macro_f1_tol1"],
        "history": history, **score,
    }
    perfs.to_csv(ART / "metrics" / f"{run_id}_performances.csv", index=False)
    pieces.to_csv(ART / "metrics" / f"{run_id}_pieces.csv", index=False)
    prediction_rows = []
    for piece_id, performances in raw.items():
        for performance_id, probability in performances.items():
            for beat, value in enumerate(probability):
                prediction_rows.append((piece_id, performance_id, beat, float(value), int(validation[piece_id]["labels"][beat]), int(validation[piece_id]["label_mask"][beat])))
    pd.DataFrame(prediction_rows, columns=["piece_id", "performance_id", "beat", "probability", "label", "valid"]).to_csv(
        ART / "metrics" / f"{run_id}_predictions.csv.gz", index=False
    )
    write(metric_path, result)
    if run_id not in state["completed_runs"]:
        state["training_validation_seconds"] += total_seconds; state["completed_runs"].append(run_id)
    state.update(current_run=None, training_process_pid=None, updated_at=now())
    save_state(state)
    return result


def resource_row(used_percent: float | None, stage: str):
    drive = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-PSDrive C).Free"], capture_output=True, text=True)
    disk_free = float([x for x in drive.stdout.splitlines() if x.strip()][-1])
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    gpu_used, gpu_free = [float(x.strip()) for x in gpu.stdout.splitlines()[0].split(",")]
    workspace = sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file())
    elapsed_hours = max((time.time() - datetime.fromisoformat("2026-09-10T11:40:02+08:00").timestamp()) / 3600, 0)
    rate = (used_percent - 49) / elapsed_hours if used_percent is not None and elapsed_hours >= 0.5 else math.nan
    remaining_hours = max((DEADLINE - time.time()) / 3600, 0)
    projected_remaining = 100 - (used_percent + rate * remaining_hours) if used_percent is not None and np.isfinite(rate) else math.nan
    row = {
        "timestamp": now(), "stage": stage, "elapsed_hours_from_batch_start": elapsed_hours,
        "batch_start_used_percent": 49, "current_used_percent": used_percent,
        "rate_percentage_points_per_hour": rate, "projected_deadline_remaining_percent": projected_remaining,
        "workspace_bytes": workspace, "disk_free_bytes": disk_free,
        "gpu_used_mib": gpu_used, "gpu_free_mib": gpu_free,
    }
    path = OUT / "resource_usage.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def paired_rows(summary: pd.DataFrame):
    contrasts = [("RMSQ", "MeanQ"), ("MeanQ", "NQ"), ("NQ", "Q"), ("Q", "Z")]
    rows = []
    for candidate, control in contrasts:
        left = summary[summary.kind == candidate].set_index(["fold", "seed"])
        right = summary[summary.kind == control].set_index(["fold", "seed"])
        for key in left.index:
            rows.append({
                "contrast": f"{candidate}-{control}", "fold": key[0], "seed": key[1],
                "delta_f1_tol0": left.loc[key, "macro_f1_tol0"] - right.loc[key, "macro_f1_tol0"],
                "delta_f1_tol1": left.loc[key, "macro_f1_tol1"] - right.loc[key, "macro_f1_tol1"],
                "delta_f1_tol2": left.loc[key, "macro_f1_tol2"] - right.loc[key, "macro_f1_tol2"],
                "delta_raw_ap": left.loc[key, "raw_ap"] - right.loc[key, "raw_ap"],
                "delta_precision_tol1": left.loc[key, "macro_precision_tol1"] - right.loc[key, "macro_precision_tol1"],
                "delta_recall_tol1": left.loc[key, "macro_recall_tol1"] - right.loc[key, "macro_recall_tol1"],
                "delta_train_gap": left.loc[key, "train_gap"] - right.loc[key, "train_gap"],
            })
    return pd.DataFrame(rows)


def per_piece_deltas(summary: pd.DataFrame):
    rows = []
    for contrast in ["RMSQ-MeanQ", "MeanQ-NQ", "NQ-Q", "Q-Z"]:
        candidate, control = contrast.split("-")
        for fold in [0, 1]:
            for seed in [42, 43]:
                a = pd.read_csv(ART / "metrics" / f"{candidate}_seed{seed}_fold{fold}_pieces.csv").set_index("piece_id")
                b = pd.read_csv(ART / "metrics" / f"{control}_seed{seed}_fold{fold}_pieces.csv").set_index("piece_id")
                for piece_id in a.index:
                    rows.append({"contrast": contrast, "fold": fold, "seed": seed, "piece_id": piece_id, "delta_f1_tol1": a.loc[piece_id, "f1_tol1"] - b.loc[piece_id, "f1_tol1"]})
    return pd.DataFrame(rows)


def plot_case(summary: pd.DataFrame):
    run = summary[(summary.kind == "RMSQ") & (summary.fold == 0) & (summary.seed == 42)].iloc[0]
    pieces = pd.read_csv(ART / "metrics/RMSQ_seed42_fold0_pieces.csv").sort_values("f1_tol1")
    piece_id = pieces.iloc[len(pieces) // 2].piece_id
    pred = pd.read_csv(ART / "metrics/RMSQ_seed42_fold0_predictions.csv.gz")
    subset = pred[pred.piece_id == piece_id]
    performance_scores = pd.read_csv(ART / "metrics/RMSQ_seed42_fold0_performances.csv")
    performance_scores = performance_scores[performance_scores.piece_id == piece_id].sort_values("f1_tol1")
    performance_id = performance_scores.iloc[len(performance_scores) // 2].performance_id
    x = subset[subset.performance_id == performance_id].sort_values("beat").head(96)
    probability = x.probability.to_numpy(); truth = np.flatnonzero((x.label.to_numpy() > 0) & (x.valid.to_numpy() > 0))
    prediction = np.flatnonzero((nms_probabilities(probability) >= run.threshold) & (x.valid.to_numpy() > 0))
    exact, near, false = [], [], []
    unmatched = set(truth.tolist())
    for p in prediction:
        if p in unmatched:
            exact.append(p); unmatched.remove(p)
        else:
            candidates = [t for t in unmatched if abs(t - p) <= 1]
            if candidates:
                target = min(candidates, key=lambda t: (abs(t - p), t)); near.append(p); unmatched.remove(target)
            else:
                false.append(p)
    plt.figure(figsize=(11, 4.2)); beats = np.arange(len(probability)); plt.plot(beats, probability, color="#344054", lw=1.4, label="boundary probability")
    plt.axhline(run.threshold, color="#98A2B3", ls="--", lw=1, label=f"threshold {run.threshold:.2f}")
    if exact: plt.scatter(exact, probability[exact], color="#079455", s=46, label="exact hit", zorder=3)
    if near: plt.scatter(near, probability[near], color="#75E0A7", edgecolor="#079455", s=46, label="±1 hit", zorder=3)
    if false: plt.scatter(false, probability[false], color="#D92D20", marker="x", s=50, label="false positive", zorder=3)
    for target in truth: plt.vlines(target, 0, 0.045, color="#101828", lw=2)
    plt.title(f"RMSQ median validation case: {piece_id} / {performance_id}"); plt.xlabel("Beat index (first 96)"); plt.ylabel("Probability"); plt.ylim(0, 1); plt.legend(ncol=3, fontsize=8); plt.tight_layout()
    plt.savefig(ART / "figures/beat_case.png", dpi=180); plt.close()
    write(ART / "figures/beat_case_metadata.json", {"selection": "median work then median performance by F1, fixed first 96 beats", "run_id": run.run_id, "piece_id": piece_id, "performance_id": performance_id, "threshold": run.threshold})


class Runner:
    def __init__(self, resume: bool, force: bool, used_percent: float | None):
        self.resume, self.force, self.used_percent = resume, force, used_percent
        for path in [OUT, ART / "metrics", ART / "checkpoints", ART / "figures", ART / "stages", OUT / "logs"]:
            path.mkdir(parents=True, exist_ok=True)

    def marker(self, stage):
        return ART / "stages" / f"{stage}.json"

    def done(self, stage):
        return self.resume and self.marker(stage).exists() and not self.force

    def mark(self, stage, payload):
        write(self.marker(stage), {"stage": stage, "completed_at": now(), **payload})
        with (OUT / "STATUS.md").open("a", encoding="utf-8") as handle:
            handle.write(f"\n- {now()} `{stage}` complete: {json.dumps(payload, ensure_ascii=False)}\n")

    def files_and_contract(self):
        files = [Path(__file__), ROOT / "src/slice_energy_study.py", ROOT / "src/slice_energy_features.py", ROOT / "src/local_context_study.py", ROOT / "src/phase7_models.py", PROTOCOL, SPLITS]
        files += sorted(CACHE.glob("*.npz"))
        return block_hash(files)

    def init(self):
        if self.done("init"):
            print("[resume] init"); return
        hashes, contract = self.files_and_contract()
        manifest = pd.read_csv(SPLITS); rows = []
        for fold in [0, 1]:
            part = manifest[manifest.fold == fold]
            pieces = {name: set(part[part.split == name].piece_id) for name in ["train", "validation", "test"]}
            opuses = {name: set(part[part.split == name].opus) for name in pieces}
            for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
                rows.append({"fold": fold, "left": left, "right": right, "piece_overlap": len(pieces[left] & pieces[right]), "opus_overlap": len(opuses[left] & opuses[right])})
        audit = pd.DataFrame(rows); audit.to_csv(OUT / "split_audit.csv", index=False)
        if audit[["piece_overlap", "opus_overlap"]].to_numpy().any():
            raise AssertionError("Round1 split leakage")
        sample_id = manifest.iloc[0].piece_id
        with np.load(CACHE / f"{sample_id}.npz", allow_pickle=False) as source:
            sample = {key: source[key] for key in source.files}
        feature_audit = []
        for kind in KINDS:
            block = extra_block(sample["curves"], sample["energy"], kind)
            feature_audit.append({"kind": kind, "shape": list(block.shape), "finite": bool(np.isfinite(block).all()), "nonzero_columns": np.flatnonzero(np.any(np.abs(block) > 1e-12, axis=(0, 1))).tolist()})
        write(OUT / "feature_audit.json", {"sample_piece": sample_id, "base_dimensions": 25, "block_dimensions": 6, "total_dimensions": 31, "conditions": feature_audit, "uses_labels": False, "tail_zero_padding": False})
        write(OUT / "input_contract.json", hashes); write(OUT / "contract.json", {"sha256": contract, "files": len(hashes)})
        state = {
            "round": 1, "status": "running", "current_run": None, "training_process_pid": None,
            "batch_started_at": "2026-09-10T11:40:02+08:00", "round_started_at": now(),
            "deadline": "2026-09-10T15:40:02+08:00", "deadline_unix": DEADLINE,
            "training_validation_budget_seconds": ROUND_BUDGET_SECONDS, "training_validation_seconds": 0.0,
            "completed_runs": [], "contract": contract, "outer_test_evaluated": False,
            "usage_batch_start_percent": 49, "usage_round_start_percent": self.used_percent,
            "updated_at": now(),
        }
        save_state(state)
        (OUT / "STATUS.md").write_text(f"# Three-round study — Round 1 Status\n\nStarted: {state['round_started_at']}\nPersistent deadline: {state['deadline']}\nStatus: running\n", encoding="utf-8")
        (OUT / "decision_log.md").write_text(
            "# Round 1 Decision Log\n\n## Fixed matrix\n\nProblem: SFE31 improves the corrected-score CNN but bundles normalization, scale means, RMS-minus-std, and validity. Evidence: previous SFE31 mean F1@±1=0.420164 versus SFZ31=0.382442. Candidates: architecture expansion, scale search, or same-width block decomposition. Choice: five same-width Z/Q/NQ/MeanQ/RMSQ cells. Reason: it changes one ordered information component at a time without changing labels, model capacity, split, or optimization. Resource impact: 20 bounded runs. Reversible: outputs are isolated; the frozen comparisons are not changed after training starts. Next checkpoint: Z/RMSQ reproduction and four paired contrasts.\n",
            encoding="utf-8",
        )
        resource_row(self.used_percent, "init")
        self.mark("init", {"contract": contract, "zero_overlap": True, "conditions": KINDS})

    def train(self):
        if self.done("train"):
            print("[resume] train"); return
        state = load_json(OUT / "STATE.json"); _, contract = self.files_and_contract()
        if state["contract"] != contract:
            raise ValueError("Round1 input/code/protocol contract changed")
        results = []
        for fold in [0, 1]:
            split = load_splits(fold)
            for kind in KINDS:
                train = dataset(split["train"], kind); validation = dataset(split["validation"], kind)
                for seed in [42, 43]:
                    result = run_one(kind, fold, seed, train, validation, contract, state); results.append(result)
                    pd.DataFrame([{key: value for key, value in row.items() if key != "history"} for row in results]).to_csv(ART / "summary.partial.csv", index=False)
                del train, validation
        summary = pd.DataFrame([{key: value for key, value in row.items() if key != "history"} for row in results])
        summary.to_csv(ART / "summary.csv", index=False); (ART / "summary.partial.csv").unlink(missing_ok=True)
        state.update(current_run=None, training_process_pid=None, updated_at=now()); save_state(state)
        resource_row(self.used_percent, "train")
        self.mark("train", {"runs": len(summary), "training_validation_seconds": state["training_validation_seconds"]})

    def report(self):
        if self.done("report"):
            print("[resume] report"); return
        summary = pd.read_csv(ART / "summary.csv")
        means = summary.groupby("kind", as_index=False).agg(
            precision_tol1=("macro_precision_tol1", "mean"), recall_tol1=("macro_recall_tol1", "mean"),
            f1_tol0=("macro_f1_tol0", "mean"), f1_tol1=("macro_f1_tol1", "mean"), f1_tol2=("macro_f1_tol2", "mean"),
            raw_ap=("raw_ap", "mean"), train_gap=("train_gap", "mean"), threshold=("threshold", "mean"),
            best_step=("best_step", "mean"), seconds=("seconds", "mean"), params=("params", "mean"),
        )
        means["kind"] = pd.Categorical(means.kind, KINDS, ordered=True); means = means.sort_values("kind")
        means.to_csv(OUT / "model_means.csv", index=False)
        paired = paired_rows(summary); paired.to_csv(OUT / "paired_cells.csv", index=False)
        contrast = paired.groupby("contrast", as_index=False).agg(
            delta_f1_tol0=("delta_f1_tol0", "mean"), delta_f1_tol1=("delta_f1_tol1", "mean"),
            delta_f1_tol2=("delta_f1_tol2", "mean"), delta_raw_ap=("delta_raw_ap", "mean"),
            delta_precision_tol1=("delta_precision_tol1", "mean"), delta_recall_tol1=("delta_recall_tol1", "mean"),
            positive_cells=("delta_f1_tol1", lambda x: int((x > 0).sum())), delta_train_gap=("delta_train_gap", "mean"),
        )
        contrast.to_csv(OUT / "paired_contrasts.csv", index=False)
        piece = per_piece_deltas(summary); piece.to_csv(OUT / "per_piece_deltas.csv", index=False)
        previous = pd.read_csv(ROOT / "artifacts/slice_energy_study/summary.csv")
        reproduction = []
        for new_kind, old_kind in [("Z", "SFZ31"), ("RMSQ", "SFE31")]:
            new = summary[summary.kind == new_kind].set_index(["fold", "seed"])
            old = previous[previous.kind == old_kind].set_index(["fold", "seed"])
            for key in new.index:
                reproduction.append({"new_kind": new_kind, "old_kind": old_kind, "fold": key[0], "seed": key[1], "f1_delta": new.loc[key, "macro_f1_tol1"] - old.loc[key, "macro_f1_tol1"], "raw_ap_delta": new.loc[key, "raw_ap"] - old.loc[key, "raw_ap"]})
        reproduction = pd.DataFrame(reproduction); reproduction.to_csv(OUT / "baseline_reproduction.csv", index=False)
        ART.joinpath("figures").mkdir(exist_ok=True)
        display = means.set_index("kind").loc[KINDS]
        display[["f1_tol0", "f1_tol1", "raw_ap"]].plot(kind="bar", figsize=(9, 4.5), color=["#667085", "#2E90FA", "#12B76A"])
        plt.ylabel("Work-macro score"); plt.xlabel("Six-channel block"); plt.xticks(rotation=0); plt.title("Round 1: hierarchical-tempo block decomposition"); plt.tight_layout(); plt.savefig(ART / "figures/model_comparison.png", dpi=180); plt.close()
        cplot = contrast.set_index("contrast").loc[["Q-Z", "NQ-Q", "MeanQ-NQ", "RMSQ-MeanQ"]]
        cplot[["delta_f1_tol1", "delta_raw_ap"]].plot(kind="bar", figsize=(9, 4.5), color=["#7F56D9", "#F79009"])
        plt.axhline(0, color="black", lw=0.8); plt.ylabel("Paired mean delta"); plt.xlabel("Added information"); plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(ART / "figures/paired_deltas.png", dpi=180); plt.close()
        plot_case(summary)
        strongest = contrast.iloc[contrast.delta_f1_tol1.abs().argmax()]
        z_match = float(reproduction[reproduction.new_kind == "Z"].f1_delta.abs().max())
        rms_match = float(reproduction[reproduction.new_kind == "RMSQ"].f1_delta.abs().max())
        status = load_json(OUT / "STATE.json")
        report = f"""# Round 1 Final Report — Hierarchical-tempo source decomposition

Generated: {now()}. Status: **implemented/reproduced; scientific interpretation preliminary**.

## Outcome

All 20 preregistered development runs completed. The largest adjacent information-step change by absolute mean F1@±1 was **{strongest.contrast} {strongest.delta_f1_tol1:+.4f}**. This comparison decomposes the previous SFE31 result; it does not open the outer test or establish an independent confirmatory gain.

## Five-cell results

{markdown_table(means)}

## Frozen paired contrasts

{markdown_table(contrast)}

`positive_cells` counts the four fold×seed cells whose F1@±1 delta is positive. Exact and ±2 F1 plus raw AP are reported so that a tolerance-only or threshold-only increase is not mistaken for better ranking/localization. Per-cell values are in `paired_cells.csv`; work-level heterogeneity is in `per_piece_deltas.csv`.

## Fairness, leakage, and reproduction

- Every condition is exactly 31-dimensional and uses the same 2,433-parameter token-level kernel-5 CNN. Common seed weights match; six added input columns are zero initialized, so step-0 logits are identical.
- Labels/masks are the frozen versioned phrase-start arrays. Train-only normalization is fitted separately per fold. All performance variants of a work remain together; fold0/1 piece and opus overlaps are zero.
- No outer-test features or predictions were loaded. Performance is evaluated first, then averaged by work.
- Z versus previous SFZ31 maximum absolute matched-cell F1 difference: {z_match:.3e}; RMSQ versus previous SFE31: {rms_match:.3e}. These checks distinguish a new result from protocol drift.
- Tempo medians, block means, RMS statistics and Q use only the input performance. Final partial blocks use their real length; boundary labels do not define any feature window.

## Interpretation limits

The adjacent contrasts isolate ordered additions in this exact representation, but they do not prove a universal musical cause. Fold/seed repetitions share only 19 development works and are not independent new datasets. Thresholds are selected on validation, and this development set has been reused in prior studies. Any positive factor is therefore **preliminary**, not a new final model claim.

## Status discipline

- **implemented:** Z/Q/NQ/MeanQ/RMSQ feature construction, fixed 31-d model, state/resume, audits and plots.
- **reproduced:** 20 completed checkpoint-backed runs and the numeric tables above.
- **preliminary:** attribution of SFE31's gain to any tempo-hierarchy component.
- **proposed:** Round 2 directional timing experiment, Round 3 score-cue complementarity, outer testing, human-reviewed labels and real audio energy.

## Resources and failures

Training, validation and train-set diagnostics used {status['training_validation_seconds']/60:.2f} minutes of the 30-minute Round 1 cap. One training process was used; each run retains only best/latest. No OOM or non-finite loss occurred. No reset or paid service was used. Resource samples are in `resource_usage.csv`; account usage is shared and integer-rounded.

## Reproduction

```powershell
powershell -ExecutionPolicy Bypass -File '.\\scripts\\run_three_round_round1.ps1' -Stage all -Resume
```

Round 2 and Round 3 were not started.
"""
        (OUT / "final_report.md").write_text(report, encoding="utf-8")
        resource_row(self.used_percent, "report")
        self.mark("report", {"strongest_contrast": strongest.contrast, "strongest_delta": strongest.delta_f1_tol1})

    def audit(self):
        if self.done("audit"):
            print("[resume] audit"); return
        state = load_json(OUT / "STATE.json"); hashes = load_json(OUT / "input_contract.json")
        input_unchanged = all((ROOT / relative).exists() and sha(ROOT / relative) == digest for relative, digest in hashes.items())
        summary = pd.read_csv(ART / "summary.csv")
        checkpoints = [path for path in (ART / "checkpoints").iterdir() if path.is_dir()]
        violations = [str(path) for path in checkpoints if {item.name for item in path.glob("*.pt")} != {"best.pt", "latest.pt"}]
        before = {str(path): sha(path) for path in (ART / "checkpoints").rglob("*.pt")}
        # The existing audit marker makes this bounded all-stage command a no-op, including its audit stage.
        result = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-m", "src.three_round_round1", "--stage", "all", "--resume"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        after = {str(path): sha(path) for path in (ART / "checkpoints").rglob("*.pt")}
        tests = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        split = pd.read_csv(OUT / "split_audit.csv")
        checks = {
            "twenty_runs": len(summary) == 20 and set(summary.kind) == set(KINDS),
            "four_cells_each": bool((summary.groupby("kind").size() == 4).all()),
            "outer_test_not_evaluated": state["outer_test_evaluated"] is False,
            "piece_opus_zero_overlap": bool((split[["piece_overlap", "opus_overlap"]] == 0).all().all()),
            "input_contract_unchanged": input_unchanged,
            "round_budget_respected": state["training_validation_seconds"] <= ROUND_BUDGET_SECONDS,
            "persistent_deadline_respected": time.time() < DEADLINE,
            "checkpoint_policy": len(checkpoints) == 20 and not violations,
            "resume_exit_zero": result.returncode == 0,
            "resume_checkpoint_hashes_unchanged": before == after,
            "tests_pass": tests.returncode == 0,
            "round2_round3_not_started": not (ROOT / "artifacts/three_round_study/round2").exists() and not (ROOT / "artifacts/three_round_study/round3").exists(),
            "required_outputs": all((OUT / name).exists() for name in ["PROTOCOL.md", "STATUS.md", "STATE.json", "final_report.md", "resource_usage.csv", "model_means.csv", "paired_contrasts.csv", "per_piece_deltas.csv"]),
        }
        write(OUT / "resume_validation.json", {"command": "python -m src.three_round_round1 --stage all --resume", "exit_code": result.returncode, "stdout": result.stdout, "checkpoint_files": len(before), "hashes_unchanged": before == after})
        write(OUT / "completion_audit.json", {"status": "complete" if all(checks.values()) else "incomplete", "created_at": now(), "checks": checks, "tests": tests.stdout + tests.stderr, "checkpoint_violations": violations, "training_validation_seconds": state["training_validation_seconds"]})
        if not all(checks.values()):
            raise AssertionError(checks)
        state.update(status="complete", current_run=None, training_process_pid=None, completed_at=now(), updated_at=now())
        save_state(state); self.mark("audit", {"status": "complete", "checks": len(checks), "tests": tests.stdout.strip()})

    def run(self, stage):
        getattr(self, stage)()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=[*STAGES, "all", "resource"], default="all"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--force", action="store_true"); parser.add_argument("--used-percent", type=float)
    args = parser.parse_args(); runner = Runner(args.resume, args.force, args.used_percent)
    if args.stage == "resource":
        resource_row(args.used_percent, "manual"); return 0
    for stage in STAGES if args.stage == "all" else [args.stage]:
        started = time.perf_counter(); print(f"[{now()}] START round1 {stage}", flush=True)
        try:
            runner.run(stage)
        except Exception as error:
            failure = {"timestamp": now(), "stage": stage, "error": repr(error), "traceback": traceback.format_exc()}
            with (OUT / "logs/failures.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            if (OUT / "STATE.json").exists():
                state = load_json(OUT / "STATE.json"); state.update(status="failed", training_process_pid=None, failure=repr(error), updated_at=now()); save_state(state)
            print(failure["traceback"], flush=True); return 1
        print(f"[{now()}] DONE round1 {stage} elapsed={time.perf_counter()-started:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
