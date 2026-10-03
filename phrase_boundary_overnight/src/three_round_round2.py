"""Round 2: frozen RMSQ timing interventions, isolated from all earlier artifacts."""
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
OUT = ROOT / "reports/three_round_study/round2"
ART = ROOT / "artifacts/three_round_study/round2"
CACHE = ROOT / "artifacts/slice_energy_study/cache"
SPLITS = ROOT / "artifacts/phase2/splits/opus_split_manifest.csv"
PROTOCOL = OUT / "PROTOCOL.md"
DISPATCH = ROOT / "reports/supervision/round2_dispatch.md"
KINDS = ["B", "L", "D", "LD"]
GRID = np.arange(0.10, 0.91, 0.05).round(2).tolist()
DEADLINE = datetime.fromisoformat("2026-09-10T15:40:02+08:00").timestamp()
ROUND_BUDGET_SECONDS = 1800.0
BATCH_BUDGET_SECONDS = 5400.0
ROUND1_SECONDS = 978.705
STAGES = ["init", "train", "report", "audit"]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hashes_and_contract(files):
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}
    return hashes, hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_state(state):
    write(OUT / "STATE.json", state)


def markdown_table(frame):
    def fmt(value):
        return f"{float(value):.4f}" if isinstance(value, (float, np.floating)) else str(value)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "|".join(["---"] * len(frame.columns)) + "|"]
    lines += ["| " + " | ".join(fmt(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def interval_observed(time_mask: np.ndarray):
    mask = np.asarray(time_mask) > 0.5
    observed = np.zeros(len(mask), dtype=np.float32)
    if len(mask) > 1:
        observed[:-1] = mask[:-1] & mask[1:]
    return observed


def timing_features(curves: np.ndarray, rmsq: np.ndarray):
    """Return original/lag hierarchy, flags and direction from inputs only."""
    curves = np.asarray(curves, np.float32); rmsq = np.asarray(rmsq, np.float32)
    if curves.ndim != 3 or curves.shape[-1] != 9 or rmsq.shape != (*curves.shape[:2], 6):
        raise ValueError("Expected curves [P,T,9] and RMSQ [P,T,6]")
    original = rmsq.copy(); lagged = np.zeros_like(rmsq)
    lag_available = np.zeros(curves.shape[:2], np.float32)
    direction = np.zeros(curves.shape[:2], np.float32)
    direction_available = np.zeros(curves.shape[:2], np.float32)
    for performance in range(curves.shape[0]):
        tempo = np.exp(curves[performance, :, 0].astype(np.float64))
        observed = interval_observed(curves[performance, :, 7])
        lagged[performance, 1:] = original[performance, :-1]
        lag_available[performance, 1:] = observed[:-1]
        for beat in range(1, len(tempo)):
            before_indices = np.arange(max(0, beat - 3), beat)
            after_indices = np.arange(beat, min(len(tempo), beat + 3))
            before = tempo[before_indices]; after = tempo[after_indices]
            if len(before) and len(after):
                direction[performance, beat] = math.log(float(after.mean())) - math.log(float(before.mean()))
                indices = np.r_[before_indices, after_indices]
                direction_available[performance, beat] = float(observed[indices].mean())
    arrays = [original, lagged, lag_available, direction, direction_available]
    if not all(np.isfinite(value).all() for value in arrays):
        raise FloatingPointError("Non-finite Round2 timing feature")
    return original, lagged, lag_available, direction, direction_available


def assemble_extra(curves: np.ndarray, rmsq: np.ndarray, kind: str):
    original, lagged, lag_available, direction, direction_available = timing_features(curves, rmsq)
    hierarchy = lagged if kind in {"L", "LD"} else original
    directional = direction if kind in {"D", "LD"} else np.zeros_like(direction)
    extra = np.concatenate([hierarchy, lag_available[..., None], directional[..., None], direction_available[..., None]], axis=-1).astype(np.float32)
    if extra.shape != (*curves.shape[:2], 9):
        raise AssertionError("Round2 extra block must have nine columns")
    return extra


def dataset(piece_ids, kind):
    result = {}
    for piece_id in piece_ids:
        with np.load(CACHE / f"{piece_id}.npz", allow_pickle=False) as source:
            item = {key: source[key] for key in source.files}
        item["labels"] = item["start_labels"].copy(); item["label_mask"] = item["start_mask"].copy()
        base = np.concatenate([item["curves"], np.broadcast_to(item["fixed_score"], (*item["curves"].shape[:2], 16))], axis=-1).astype(np.float32)
        extra = assemble_extra(item["curves"], item["energy"], kind)
        item["curves"] = np.concatenate([base, extra], axis=-1)
        item["selected_score"] = item["fixed_score"]; item["round2_extra"] = extra
        if item["curves"].shape[-1] != 34 or not np.isfinite(item["curves"]).all():
            raise AssertionError(f"Invalid input {piece_id}/{kind}")
        result[piece_id] = item
    return result


def normalizer(data):
    curves = fit_curve_normalizer({piece_id: {**item, "curves": item["curves"][..., :9]} for piece_id, item in data.items()})
    score = fit_train_normalizer([item["selected_score"] for item in data.values()])
    extra = fit_train_normalizer([item["round2_extra"].reshape(-1, 9) for item in data.values()])
    return Normalizer(np.r_[curves.mean, score.mean, extra.mean], np.r_[curves.std, score.std, extra.std])


def make_model34(seed):
    model = model_for(25, seed); old = model.input_projection
    with torch.random.fork_rng(devices=[]):
        expanded = nn.Linear(34, 32)
    with torch.no_grad():
        expanded.weight.zero_(); expanded.weight[:, :25].copy_(old.weight); expanded.bias.copy_(old.bias)
    model.input_projection = expanded
    return model


def split_ids(fold):
    frame = pd.read_csv(SPLITS); part = frame[frame.fold == fold]
    return {name: sorted(part[part.split == name].piece_id.tolist()) for name in ["train", "validation", "test"]}


def checkpoint_threshold(saved):
    row = [item for item in saved["history"] if int(item["step"]) == int(saved["step"])]
    if len(row) != 1:
        raise AssertionError("Missing best validation record")
    return float(row[0]["threshold"])


def run_one(kind, fold, seed, train, validation, contract, state):
    run_id = f"{kind}_seed{seed}_fold{fold}"; destination = ART / "checkpoints" / run_id; destination.mkdir(parents=True, exist_ok=True)
    result_path = ART / "metrics" / f"{run_id}.json"
    if result_path.exists():
        result = load_json(result_path)
        if result["contract"] != contract: raise ValueError("Completed contract changed")
        print("CACHED", run_id, flush=True); return result
    state.update(status="running", current_run=run_id, training_process_pid=os.getpid(), updated_at=now()); save_state(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model34(seed).to(device); norm = normalizer(train); sampler = CurvePieceBalancedSampler(train, norm, 64, 32, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(positive_weight(train, 10), device=device))
    best_path, latest_path = destination / "best.pt", destination / "latest.pt"
    step, history, best_score, prior_seconds = 0, [], -1.0, 0.0; started = time.monotonic()
    if latest_path.exists():
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        if (saved["contract"], saved["kind"], saved["fold"], saved["seed"]) != (contract, kind, fold, seed): raise ValueError("Resume contract mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); sampler.load_state(saved["sampler"])
        step, history, best_score, prior_seconds = int(saved["step"]), list(saved["history"]), float(saved["best_score"]), float(saved["seconds"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])

    def snapshot():
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "sampler": sampler.state(), "step": step, "history": history, "best_score": best_score, "contract": contract, "kind": kind, "fold": fold, "seed": seed, "seconds": prior_seconds + time.monotonic() - started, "mean": norm.mean, "std": norm.std, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
    while step < 300:
        elapsed = prior_seconds + time.monotonic() - started
        if time.time() >= DEADLINE - 60: torch.save(snapshot(), latest_path); raise TimeoutError("Persistent deadline guard")
        if state["training_validation_seconds"] + elapsed >= ROUND_BUDGET_SECONDS: torch.save(snapshot(), latest_path); raise TimeoutError("Round2 budget guard")
        if ROUND1_SECONDS + state["training_validation_seconds"] + elapsed >= BATCH_BUDGET_SECONDS: torch.save(snapshot(), latest_path); raise TimeoutError("Three-round budget guard")
        model.train(); features, labels, mask, valid = (value.to(device) for value in sampler.batch())
        optimizer.zero_grad(set_to_none=True); logits = model(features, padding_mask=~valid.bool())
        loss = (criterion(logits, labels) * mask).sum() / mask.sum().clamp_min(1)
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite loss")
        loss.backward(); grad = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad): raise FloatingPointError("Non-finite gradient")
        optimizer.step(); step += 1
        if step % 50 == 0:
            raw = predictions(model, validation, norm, device); threshold, _ = choose_single_threshold(raw, validation, GRID); _, _, score = metrics(raw, validation, threshold)
            record = {"step": step, "train_loss": float(loss.detach()), "gradient_norm": float(grad.detach()), "threshold": threshold, **score}; history.append(record)
            if score["macro_f1_tol1"] > best_score + 1e-9:
                best_score = float(score["macro_f1_tol1"]); torch.save(snapshot(), best_path)
            print(run_id, step, f"exact={score['macro_f1_tol0']:.4f}", f"F1={score['macro_f1_tol1']:.4f}", f"AP={score['raw_ap']:.4f}", flush=True)
        if step % 25 == 0: torch.save(snapshot(), latest_path)
    saved = torch.load(best_path, map_location=device, weights_only=False); model.load_state_dict(saved["model"]); threshold = checkpoint_threshold(saved)
    raw = predictions(model, validation, norm, device); perfs, pieces, score = metrics(raw, validation, threshold); _, _, train_score = metrics(predictions(model, train, norm, device), train, threshold)
    seconds = prior_seconds + time.monotonic() - started
    result = {"run_id": run_id, "kind": kind, "fold": fold, "seed": seed, "contract": contract, "threshold": threshold, "best_step": int(saved["step"]), "params": sum(p.numel() for p in model.parameters()), "seconds": seconds, "gpu_peak_bytes": int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0, "train_f1": train_score["macro_f1_tol1"], "train_gap": train_score["macro_f1_tol1"] - score["macro_f1_tol1"], "history": history, **score}
    perfs.to_csv(ART / "metrics" / f"{run_id}_performances.csv", index=False); pieces.to_csv(ART / "metrics" / f"{run_id}_pieces.csv", index=False)
    rows = []
    for piece_id, performances in raw.items():
        for performance_id, probability in performances.items():
            for beat, value in enumerate(probability): rows.append((piece_id, performance_id, beat, float(value), int(validation[piece_id]["labels"][beat]), int(validation[piece_id]["label_mask"][beat])))
    pd.DataFrame(rows, columns=["piece_id", "performance_id", "beat", "probability", "label", "valid"]).to_csv(ART / "metrics" / f"{run_id}_predictions.csv.gz", index=False)
    write(result_path, result)
    if run_id not in state["completed_runs"]: state["training_validation_seconds"] += seconds; state["completed_runs"].append(run_id)
    state.update(current_run=None, training_process_pid=None, updated_at=now()); save_state(state); return result


def resource_row(used_percent, stage):
    disk = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-PSDrive C).Free"], capture_output=True, text=True); disk_free = float([x for x in disk.stdout.splitlines() if x.strip()][-1])
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True); gpu_used, gpu_free = [float(x.strip()) for x in gpu.stdout.splitlines()[0].split(",")]
    workspace = sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file()); elapsed = max((time.time() - datetime.fromisoformat("2026-09-10T11:40:02+08:00").timestamp()) / 3600, 0)
    rate = (used_percent - 49) / elapsed if used_percent is not None and elapsed >= 0.5 else math.nan; remaining_h = max((DEADLINE - time.time()) / 3600, 0); projected = 100 - (used_percent + rate * remaining_h) if used_percent is not None and np.isfinite(rate) else math.nan
    path = OUT / "resource_usage.csv"; pd.DataFrame([{"timestamp": now(), "stage": stage, "elapsed_hours": elapsed, "start_used_percent": 49, "current_used_percent": used_percent, "rate_percent_per_hour": rate, "projected_deadline_remaining_percent": projected, "workspace_bytes": workspace, "disk_free_bytes": disk_free, "gpu_used_mib": gpu_used, "gpu_free_mib": gpu_free}]).to_csv(path, mode="a", header=not path.exists(), index=False)


def paired(summary):
    rows = []
    for candidate, control in [("L", "B"), ("D", "B"), ("LD", "B"), ("LD", "L"), ("LD", "D")]:
        a = summary[summary.kind == candidate].set_index(["fold", "seed"]); b = summary[summary.kind == control].set_index(["fold", "seed"])
        for key in a.index:
            rows.append({"contrast": f"{candidate}-{control}", "fold": key[0], "seed": key[1], "delta_exact": a.loc[key, "macro_f1_tol0"] - b.loc[key, "macro_f1_tol0"], "delta_f1_tol1": a.loc[key, "macro_f1_tol1"] - b.loc[key, "macro_f1_tol1"], "delta_f1_tol2": a.loc[key, "macro_f1_tol2"] - b.loc[key, "macro_f1_tol2"], "delta_raw_ap": a.loc[key, "raw_ap"] - b.loc[key, "raw_ap"], "delta_precision": a.loc[key, "macro_precision_tol1"] - b.loc[key, "macro_precision_tol1"], "delta_recall": a.loc[key, "macro_recall_tol1"] - b.loc[key, "macro_recall_tol1"], "delta_train_gap": a.loc[key, "train_gap"] - b.loc[key, "train_gap"]})
    return pd.DataFrame(rows)


def work_deltas(summary):
    rows = []
    for candidate in ["L", "D", "LD"]:
        for fold in [0, 1]:
            for seed in [42, 43]:
                a = pd.read_csv(ART / "metrics" / f"{candidate}_seed{seed}_fold{fold}_pieces.csv").set_index("piece_id"); b = pd.read_csv(ART / "metrics" / f"B_seed{seed}_fold{fold}_pieces.csv").set_index("piece_id")
                for piece_id in a.index: rows.append({"contrast": f"{candidate}-B", "fold": fold, "seed": seed, "piece_id": piece_id, "delta_exact": a.loc[piece_id, "f1_tol0"] - b.loc[piece_id, "f1_tol0"], "delta_f1_tol1": a.loc[piece_id, "f1_tol1"] - b.loc[piece_id, "f1_tol1"]})
    return pd.DataFrame(rows)


def offset_rows(summary):
    rows = []
    for run in summary.itertuples(index=False):
        frame = pd.read_csv(ART / "metrics" / f"{run.run_id}_predictions.csv.gz")
        counts = {offset: 0 for offset in range(-2, 3)}; matched = 0
        for (_, _), group in frame.groupby(["piece_id", "performance_id"]):
            group = group.sort_values("beat"); probability = group.probability.to_numpy(); valid = group.valid.to_numpy().astype(bool); truth = np.flatnonzero((group.label.to_numpy() > 0) & valid); pred = np.flatnonzero((nms_probabilities(probability) >= run.threshold) & valid)
            candidates = sorted((abs(int(p)-int(t)), int(p), int(t)) for p in pred for t in truth if abs(int(p)-int(t)) <= 2); used_p, used_t = set(), set()
            for _, p, t in candidates:
                if p not in used_p and t not in used_t: used_p.add(p); used_t.add(t); counts[p-t] += 1; matched += 1
        rows.append({"run_id": run.run_id, "kind": run.kind, "fold": run.fold, "seed": run.seed, "matched_events": matched, **{f"offset_{i:+d}": counts[i] for i in range(-2, 3)}, "mean_signed_offset": sum(i*counts[i] for i in counts)/max(matched,1), "mean_absolute_offset": sum(abs(i)*counts[i] for i in counts)/max(matched,1), "weighting": "event-weighted across validation performances; descriptive"})
    return pd.DataFrame(rows)


class Runner:
    def __init__(self, resume, force, used_percent):
        self.resume, self.force, self.used_percent = resume, force, used_percent
        for path in [OUT, OUT / "logs", ART / "metrics", ART / "checkpoints", ART / "figures", ART / "stages"]: path.mkdir(parents=True, exist_ok=True)

    def marker(self, stage): return ART / "stages" / f"{stage}.json"
    def done(self, stage): return self.resume and self.marker(stage).exists() and not self.force
    def mark(self, stage, payload):
        write(self.marker(stage), {"stage": stage, "completed_at": now(), **payload})
        with (OUT / "STATUS.md").open("a", encoding="utf-8") as handle: handle.write(f"\n- {now()} `{stage}` complete: {json.dumps(payload, ensure_ascii=False)}\n")

    def contract(self):
        files = [Path(__file__), ROOT / "src/slice_energy_study.py", ROOT / "src/slice_energy_features.py", ROOT / "src/local_context_study.py", ROOT / "src/phase7_models.py", PROTOCOL, DISPATCH, SPLITS] + sorted(CACHE.glob("*.npz"))
        return hashes_and_contract(files)

    def init(self):
        if self.done("init"): print("[resume] init"); return
        hashes, contract = self.contract(); manifest = pd.read_csv(SPLITS); audit = []
        for fold in [0, 1]:
            part = manifest[manifest.fold == fold]; pieces = {name: set(part[part.split == name].piece_id) for name in ["train", "validation", "test"]}; opuses = {name: set(part[part.split == name].opus) for name in pieces}
            for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]: audit.append({"fold": fold, "left": left, "right": right, "piece_overlap": len(pieces[left]&pieces[right]), "opus_overlap": len(opuses[left]&opuses[right])})
        audit = pd.DataFrame(audit); audit.to_csv(OUT / "split_audit.csv", index=False)
        if audit[["piece_overlap", "opus_overlap"]].to_numpy().any(): raise AssertionError("Round2 split leakage")
        sample_id = manifest.iloc[0].piece_id
        with np.load(CACHE / f"{sample_id}.npz", allow_pickle=False) as source: sample = {key: source[key] for key in source.files}
        blocks = {kind: assemble_extra(sample["curves"], sample["energy"], kind) for kind in KINDS}
        flag_equal = all(np.array_equal(blocks["B"][..., [6,8]], blocks[kind][..., [6,8]]) for kind in KINDS)
        labels_equal = all(np.array_equal(dataset([sample_id], "B")[sample_id][key], dataset([sample_id], kind)[sample_id][key]) for kind in KINDS for key in ["labels", "label_mask"])
        write(OUT / "feature_audit.json", {"sample_piece": sample_id, "shape_each": list(blocks["B"].shape), "flags_identical": flag_equal, "labels_masks_identical": labels_equal, "lag_no_wrap": bool(np.all(blocks["L"][:,0,:6] == 0)), "b0_flags_zero": bool(np.all(blocks["B"][:,0,[6,8]] == 0)), "direction_b0_zero": bool(np.all(blocks["D"][:,0,7] == 0)), "last_tempo_semantics": "estimated by build_curve_features: previous positive interval, else median positive fallback; never a directly observed outgoing interval", "uses_targets": False})
        if not flag_equal or not labels_equal: raise AssertionError("Round2 fairness invariant failed")
        write(OUT / "input_contract.json", hashes); write(OUT / "contract.json", {"sha256": contract, "files": len(hashes)})
        state = {"round": 2, "status": "running", "current_run": None, "training_process_pid": None, "batch_started_at": "2026-09-10T11:40:02+08:00", "round_started_at": now(), "deadline": "2026-09-10T15:40:02+08:00", "deadline_unix": DEADLINE, "round_budget_seconds": ROUND_BUDGET_SECONDS, "training_validation_seconds": 0.0, "prior_round_seconds": ROUND1_SECONDS, "batch_budget_seconds": BATCH_BUDGET_SECONDS, "completed_runs": [], "contract": contract, "outer_test_evaluated": False, "usage_batch_start_percent": 49, "usage_round_start_percent": self.used_percent, "updated_at": now()}; save_state(state)
        (OUT / "STATUS.md").write_text(f"# Three-round study — Round 2 Status\n\nStarted: {state['round_started_at']}\nPersistent deadline: {state['deadline']}\nStatus: running\n", encoding="utf-8")
        (OUT / "decision_log.md").write_text("# Round 2 Decision Log\n\n## Frozen timing design\n\nProblem: RMSQ has the best tolerant F1 but one selected cell shifts early and loses exact/AP. Evidence: Round1 and supervisor diagnostics. Candidates: replace hierarchy, search offsets, or preserve hierarchy and add fixed timing controls. Choice: 34-d B/L/D/LD with identical flags and zero-initialized added weights. Reason: isolates lag and direction without deleting context or altering architecture/selection. Resource impact: 16 bounded runs. Reversibility: isolated outputs; frozen protocol cannot be changed after training. Next checkpoint: exact and raw AP paired breadth.\n", encoding="utf-8")
        resource_row(self.used_percent, "init"); self.mark("init", {"contract": contract, "zero_overlap": True, "flags_identical": flag_equal, "conditions": KINDS})

    def train(self):
        if self.done("train"): print("[resume] train"); return
        state = load_json(OUT / "STATE.json"); _, contract = self.contract()
        if state["contract"] != contract: raise ValueError("Round2 contract changed")
        results = []
        for fold in [0, 1]:
            ids = split_ids(fold)
            for kind in KINDS:
                train = dataset(ids["train"], kind); validation = dataset(ids["validation"], kind)
                for seed in [42, 43]:
                    results.append(run_one(kind, fold, seed, train, validation, contract, state)); pd.DataFrame([{k:v for k,v in row.items() if k != "history"} for row in results]).to_csv(ART / "summary.partial.csv", index=False)
                del train, validation
        summary = pd.DataFrame([{k:v for k,v in row.items() if k != "history"} for row in results]); summary.to_csv(ART / "summary.csv", index=False); (ART / "summary.partial.csv").unlink(missing_ok=True)
        state.update(current_run=None, training_process_pid=None, updated_at=now()); save_state(state); resource_row(self.used_percent, "train"); self.mark("train", {"runs": len(summary), "seconds": state["training_validation_seconds"]})

    def report(self):
        if self.done("report"): print("[resume] report"); return
        summary = pd.read_csv(ART / "summary.csv"); means = summary.groupby("kind", as_index=False).agg(precision=("macro_precision_tol1","mean"), recall=("macro_recall_tol1","mean"), exact=("macro_f1_tol0","mean"), f1_tol1=("macro_f1_tol1","mean"), f1_tol2=("macro_f1_tol2","mean"), raw_ap=("raw_ap","mean"), train_gap=("train_gap","mean"), threshold=("threshold","mean"), best_step=("best_step","mean"), seconds=("seconds","mean"), params=("params","mean")); means["kind"] = pd.Categorical(means.kind, KINDS, ordered=True); means = means.sort_values("kind"); means.to_csv(OUT / "model_means.csv", index=False)
        cells = paired(summary); cells.to_csv(OUT / "paired_cells.csv", index=False); comparisons = cells.groupby("contrast", as_index=False).agg(delta_exact=("delta_exact","mean"), delta_f1_tol1=("delta_f1_tol1","mean"), delta_f1_tol2=("delta_f1_tol2","mean"), delta_raw_ap=("delta_raw_ap","mean"), delta_precision=("delta_precision","mean"), delta_recall=("delta_recall","mean"), delta_train_gap=("delta_train_gap","mean"), positive_exact_cells=("delta_exact",lambda x:int((x>0).sum())), positive_ap_cells=("delta_raw_ap",lambda x:int((x>0).sum())), positive_tol1_cells=("delta_f1_tol1",lambda x:int((x>0).sum()))); comparisons.to_csv(OUT / "paired_comparisons.csv", index=False)
        works = work_deltas(summary); works.to_csv(OUT / "per_work_deltas.csv", index=False); offsets = offset_rows(summary); offsets.to_csv(OUT / "offset_diagnostics.csv", index=False)
        direct = comparisons[comparisons.contrast.isin(["L-B","D-B","LD-B"])].copy(); direct["timing_supported"] = (direct.delta_exact > 0) & (direct.delta_raw_ap > 0) & (direct.positive_exact_cells >= 3) & (direct.positive_ap_cells >= 3); supported = direct[direct.timing_supported]
        selection = "B" if supported.empty else supported.sort_values(["delta_exact","delta_raw_ap"], ascending=False).iloc[0].contrast.split("-")[0]
        write(OUT / "frozen_round2_conclusion.json", {"selected_for_interpretation": selection, "timing_supported": not supported.empty, "rule": "mean exact and raw AP positive with >=3/4 positive cells each", "outer_test_used": False, "comparisons": direct.to_dict("records")})
        history = []
        for path in sorted((ART / "metrics").glob("[BLD]*_seed*_fold*.json")):
            result = load_json(path)
            for row in result["history"]: history.append({"run_id": result["run_id"], "kind": result["kind"], "fold": result["fold"], "seed": result["seed"], **row})
        pd.DataFrame(history).to_csv(OUT / "validation_step_diagnostics.csv", index=False)
        ART.joinpath("figures").mkdir(exist_ok=True); plot = means.set_index("kind").loc[KINDS]; plot[["exact","f1_tol1","raw_ap"]].plot(kind="bar", figsize=(8.5,4.5), color=["#667085","#2E90FA","#12B76A"]); plt.ylabel("Work-macro score"); plt.xlabel("Timing condition"); plt.xticks(rotation=0); plt.title("Round 2 timing interventions"); plt.tight_layout(); plt.savefig(ART / "figures/model_comparison.png", dpi=180); plt.close()
        cp = direct.set_index("contrast"); cp[["delta_exact","delta_f1_tol1","delta_raw_ap"]].plot(kind="bar", figsize=(8.5,4.5)); plt.axhline(0,color="black",lw=.8); plt.ylabel("Paired mean delta vs B"); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(ART / "figures/paired_deltas.png", dpi=180); plt.close()
        state = load_json(OUT / "STATE.json"); b = means[means.kind == "B"].iloc[0]
        report = f"""# Round 2 Final Report — Phrase-start timing

Generated: {now()}. Status: **implemented/reproduced; scientific interpretation preliminary**.

## Outcome

All 16 frozen development runs completed. The decision rule {'found timing support for '+selection if not supported.empty else 'did not find a timing intervention whose exact F1 and raw AP improved broadly together; B is retained'}. No outer test was used and Round 3 was not started.

## Matched 34-d results

{markdown_table(means)}

## Paired comparisons

{markdown_table(comparisons)}

The main test is each intervention versus recomputed 34-d B, not the 31-d Round1 RMSQ number. `positive_*_cells` counts four fold×seed development cells. A tolerant-F1-only rise is not treated as more accurate localization.

## Time, mask and leakage semantics

- All models have {int(b.params)} parameters, identical seed initialization and nine zero-initialized added input weights. Step-zero logits match across all conditions.
- RMSQ is retained in every condition. L/LD shifts only its six values by one beat with no wrap; D/LD adds the one fixed directional value. Both availability columns are identical across B/L/D/LD.
- tempo[i] denotes i→i+1. Direction is log mean of actual available b:b+3 intervals minus log mean of actual available max(0,b−3):b intervals; b=0 is zero/unavailable. The final tempo is an estimate copied from the prior positive interval or median fallback, and observation-derived availability exposes this.
- Phrase-start labels/masks are byte-identical across conditions. Normalization fits train only. Work/opus overlap is zero. No test outcome or prediction participates in training, threshold/checkpoint selection or evaluation.

## Error and selection diagnostics

Every validation opportunity's exact/±1/±2 F1, raw AP, threshold, train loss and gradient norm is in `validation_step_diagnostics.csv`. Selected-checkpoint offset counts are in `offset_diagnostics.csv` and explicitly event-weight performances; they are descriptive, not the work-macro primary metric. Work-level paired results are in `per_work_deltas.csv`.

## Status discipline

- **implemented:** B/L/D/LD feature assembly, observed/estimated interval flags, matched model, resume and audits.
- **reproduced:** 16 checkpoint-backed development runs and numeric tables.
- **preliminary:** any timing mechanism interpretation on the reused 19-work development set.
- **proposed:** Round3 score-cue complementarity, outer testing, human-reviewed labels and real-audio energy.

## Resources

Round2 training/validation/diagnostics used {state['training_validation_seconds']/60:.2f}/30 minutes; Round1+2 cumulative is {(ROUND1_SECONDS+state['training_validation_seconds'])/60:.2f}/90 minutes. One training process, best/latest only, no reset or paid service. Shared-account usage is recorded separately and is not an exact task meter.

## Reproduction

```powershell
powershell -ExecutionPolicy Bypass -File '.\\scripts\\run_three_round_round2.ps1' -Stage all -Resume
```
"""
        (OUT / "final_report.md").write_text(report, encoding="utf-8"); resource_row(self.used_percent, "report"); self.mark("report", {"selection": selection, "timing_supported": not supported.empty})

    def audit(self):
        if self.done("audit"): print("[resume] audit"); return
        # Sentinel prevents the self-check's all-stage command from recursively entering audit.
        write(self.marker("audit"), {"stage":"audit","status":"resume_self_check_sentinel","created_at":now()})
        state = load_json(OUT / "STATE.json"); hashes = load_json(OUT / "input_contract.json"); unchanged = all((ROOT / key).exists() and sha(ROOT / key) == value for key,value in hashes.items()); summary = pd.read_csv(ART / "summary.csv"); checkpoints = [p for p in (ART / "checkpoints").iterdir() if p.is_dir()]; violations = [str(p) for p in checkpoints if {x.name for x in p.glob("*.pt")} != {"best.pt","latest.pt"}]; before = {str(p):sha(p) for p in (ART / "checkpoints").rglob("*.pt")}
        resume = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"),"-m","src.three_round_round2","--stage","all","--resume"],cwd=ROOT,capture_output=True,text=True,encoding="utf-8",errors="replace"); after = {str(p):sha(p) for p in (ART / "checkpoints").rglob("*.pt")}; tests = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"),"-m","pytest","-q"],cwd=ROOT,capture_output=True,text=True,encoding="utf-8",errors="replace"); split = pd.read_csv(OUT / "split_audit.csv")
        checks = {"sixteen_runs":len(summary)==16 and set(summary.kind)==set(KINDS),"four_cells_each":bool((summary.groupby('kind').size()==4).all()),"outer_test_not_evaluated":state['outer_test_evaluated'] is False,"piece_opus_zero_overlap":bool((split[['piece_overlap','opus_overlap']]==0).all().all()),"input_contract_unchanged":unchanged,"round_budget":state['training_validation_seconds']<=ROUND_BUDGET_SECONDS,"batch_budget":ROUND1_SECONDS+state['training_validation_seconds']<=BATCH_BUDGET_SECONDS,"deadline":time.time()<DEADLINE,"checkpoint_policy":len(checkpoints)==16 and not violations,"resume_exit_zero":resume.returncode==0,"resume_hashes_unchanged":before==after,"tests_pass":tests.returncode==0,"round3_not_started":not (ROOT/'artifacts/three_round_study/round3').exists(),"required_outputs":all((OUT/name).exists() for name in ['PROTOCOL.md','STATUS.md','STATE.json','final_report.md','resource_usage.csv','model_means.csv','paired_comparisons.csv','per_work_deltas.csv','offset_diagnostics.csv','validation_step_diagnostics.csv'])}
        write(OUT/'resume_validation.json',{"exit_code":resume.returncode,"stdout":resume.stdout,"checkpoint_files":len(before),"hashes_unchanged":before==after}); write(OUT/'completion_audit.json',{"status":"complete" if all(checks.values()) else "incomplete","created_at":now(),"checks":checks,"tests":tests.stdout+tests.stderr,"checkpoint_violations":violations,"training_validation_seconds":state['training_validation_seconds']})
        if not all(checks.values()): raise AssertionError(checks)
        state.update(status="complete",current_run=None,training_process_pid=None,completed_at=now(),updated_at=now()); save_state(state); self.mark("audit",{"status":"complete","checks":len(checks),"tests":tests.stdout.strip()})

    def run(self, stage): getattr(self,stage)()


def main():
    if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=[*STAGES,'all','resource'],default='all');parser.add_argument('--resume',action='store_true');parser.add_argument('--force',action='store_true');parser.add_argument('--used-percent',type=float);args=parser.parse_args();runner=Runner(args.resume,args.force,args.used_percent)
    if args.stage=='resource':resource_row(args.used_percent,'manual');return 0
    for stage in STAGES if args.stage=='all' else [args.stage]:
        started=time.perf_counter();print(f'[{now()}] START round2 {stage}',flush=True)
        try:runner.run(stage)
        except Exception as error:
            failure={"timestamp":now(),"stage":stage,"error":repr(error),"traceback":traceback.format_exc()}
            with (OUT/'logs/failures.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(failure,ensure_ascii=False)+'\n')
            if (OUT/'STATE.json').exists():state=load_json(OUT/'STATE.json');state.update(status='failed',training_process_pid=None,failure=repr(error),updated_at=now());save_state(state)
            print(failure['traceback'],flush=True);return 1
        print(f'[{now()}] DONE round2 {stage} elapsed={time.perf_counter()-started:.2f}s',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
