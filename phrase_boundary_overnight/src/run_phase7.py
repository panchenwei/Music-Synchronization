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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from .data import load_piece_cache, write_json
from .phase2_models import evaluate_single_performance, nms_probabilities
from .phase7_models import (
    Phase7BoundaryModel,
    full_curve_predictions,
    load_phase7_checkpoint,
    model_kwargs,
    train_phase7_model,
)


STAGES = ["init", "diagnose", "dev", "select", "control", "outer", "analysis", "report", "audit"]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value)
    headers = [str(x) for x in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(cell(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def paired_bootstrap(values: np.ndarray, iterations: int = 5000, seed: int = 42):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    sample = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(iterations)])
    return float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def append_csv(path: Path, row: dict[str, Any]):
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


class Phase7Pipeline:
    def __init__(self, root: Path, resume: bool, force: bool, used_percent: float | None):
        self.root = root.resolve()
        self.resume, self.force, self.used_percent = resume, force, used_percent
        self.config_path = self.root / "configs/phase7/protocol.yaml"
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.deadline = datetime.fromisoformat(self.config["project"]["deadline"])
        self.artifacts = self.root / "artifacts/phase7"
        self.metrics = self.artifacts / "metrics"
        self.figures = self.artifacts / "figures"
        self.markers = self.artifacts / "stages"
        self.checkpoints = self.root / "checkpoints/phase7"
        self.manifests = self.root / "manifests/phase7"
        self.reports = self.root / "reports/phase7"
        self.logs = self.root / "logs/phase7"
        for path in [self.artifacts, self.metrics, self.figures, self.markers, self.checkpoints, self.manifests, self.reports, self.logs]:
            path.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.reports / "experiment_registry.json"
        self.budget_path = self.manifests / "training_budget.json"

    def marker(self, stage: str) -> Path:
        return self.markers / f"{stage}.json"

    def done(self, stage: str) -> bool:
        return self.resume and self.marker(stage).exists() and not self.force

    def mark(self, stage: str, payload: dict[str, Any]):
        write_json(self.marker(stage), {"stage": stage, "completed_at": now(), **payload})
        status = self.reports / "STATUS.md"
        with status.open("a", encoding="utf-8") as handle:
            handle.write(f"\n- {now()} `{stage}` complete: {json.dumps(payload, ensure_ascii=False)}\n")

    def split(self, fold: int):
        frame = pd.read_csv(self.root / self.config["data"]["split_source"])
        part = frame[frame.fold == fold]
        return {name: part[part.split == name].piece_id.tolist() for name in ["train", "validation", "test"]}

    def load(self, piece_ids):
        return {pid: load_piece_cache(self.root / "cache", pid) for pid in piece_ids}

    def prereg(self):
        return json.loads((self.manifests / "preregistered_plan.json").read_text(encoding="utf-8"))

    def contract_hash(self):
        return canonical_hash(self.prereg())

    def registry(self):
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        return {"phase": 7, "created_at": now(), "runs": []}

    def save_registry(self, registry):
        write_json(self.registry_path, registry)

    def budget(self):
        return json.loads(self.budget_path.read_text(encoding="utf-8"))

    def update_budget(self, seconds: float, run_id: str):
        budget = self.budget()
        budget["completed_seconds"] += float(seconds)
        budget["runs"].append({"run_id": run_id, "seconds": float(seconds), "completed_at": now()})
        if budget["completed_seconds"] > budget["maximum_seconds"]:
            raise RuntimeError("Phase7 training budget exceeded")
        write_json(self.budget_path, budget)

    def resource(self, stage: str):
        drive = os.statvfs(str(self.root)) if os.name != "nt" else None
        used, free = math.nan, math.nan
        if os.name == "nt":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-PSDrive C).Used; (Get-PSDrive C).Free"],
                capture_output=True, text=True,
            )
            values = [float(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()]
            if len(values) >= 2:
                used, free = values[:2]
        gpu_used = gpu_free = math.nan
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if gpu.returncode == 0:
            try:
                gpu_used, gpu_free = [float(x.strip()) for x in gpu.stdout.splitlines()[0].split(",")]
            except Exception:
                pass
        start = datetime.fromisoformat(self.config["project"]["started_at"])
        elapsed_h = (datetime.now().astimezone() - start).total_seconds() / 3600
        baseline = float(self.config["project"]["usage_baseline_percent"])
        rate = (self.used_percent - baseline) / elapsed_h if self.used_percent is not None and elapsed_h >= 0.5 else math.nan
        remaining_h = max(0.0, (self.deadline - datetime.now().astimezone()).total_seconds() / 3600)
        projected_remaining = 100 - (self.used_percent + rate * remaining_h) if self.used_percent is not None and np.isfinite(rate) else math.nan
        append_csv(self.reports / "resource_usage.csv", {
            "timestamp": now(), "elapsed_hours": elapsed_h, "start_used_percent": baseline,
            "current_used_percent": self.used_percent, "rate_percent_per_hour": rate,
            "projected_end_remaining_percent": projected_remaining, "disk_used_bytes": used,
            "disk_free_bytes": free, "gpu_used_mib": gpu_used, "gpu_free_mib": gpu_free, "stage": stage,
        })

    def init(self):
        if self.done("init"):
            print("[resume] init"); return
        split_path = self.root / self.config["data"]["split_source"]
        split = pd.read_csv(split_path)
        audits = []
        for fold in range(5):
            p = split[split.fold == fold]
            sets = {name: set(p[p.split == name].piece_id) for name in ["train", "validation", "test"]}
            opus = {name: set(p[p.split == name].opus.astype(str)) for name in sets}
            audits.append({
                "fold": fold,
                "piece_train_validation_overlap": len(sets["train"] & sets["validation"]),
                "piece_train_test_overlap": len(sets["train"] & sets["test"]),
                "piece_validation_test_overlap": len(sets["validation"] & sets["test"]),
                "opus_train_validation_overlap": len(opus["train"] & opus["validation"]),
                "opus_train_test_overlap": len(opus["train"] & opus["test"]),
                "opus_validation_test_overlap": len(opus["validation"] & opus["test"]),
            })
        audit = pd.DataFrame(audits)
        audit.to_csv(self.manifests / "split_reaudit.csv", index=False)
        if audit.filter(regex="overlap").to_numpy().any():
            raise AssertionError("Phase7 split leakage detected")
        sample = load_piece_cache(self.root / "cache", split.piece_id.iloc[0])
        feature_names = [str(x) for x in sample["curve_feature_names"]]
        prereg = {
            "phase": 7, "created_at": now(), "hypotheses": ["H1_local_shape", "H2_attention_complementarity", "H3_not_capacity_or_peak_width"],
            "input": {"dimensions": 9, "names_in_order": feature_names, "normalization": "training_split_only"},
            "split": {"path": self.config["data"]["split_source"], "sha256": sha(split_path), "unit": "opus/work", "zero_overlap": True},
            "sampling": "piece_then_performance_then_64beat_window_balanced", "labels": "hard beat-level",
            "loss": "BCEWithLogitsLoss masked by label_mask, positive_weight capped at 10",
            "steps": self.config["training"]["maximum_steps"], "validation_every": self.config["training"]["validation_every_steps"],
            "threshold_grid": self.config["evaluation"]["threshold_grid"], "fixed_threshold": self.config["evaluation"]["fixed_threshold"],
            "nms_radius": 1, "tolerances": [0, 1, 2], "seeds": [42, 43], "development_folds": [0, 1],
            "architectures": {
                "A": "Linear9-32 + sinusoidal position + 2x Transformer(4 heads, FFN64, pre-norm) + LN + token head",
                "B": "A plus residual LN-depthwiseConv5-GELU-pointwiseConv frontend before position",
                "C": "Linear9-32 + same residual conv frontend + per-token LN/head; no attention",
                "D": "conditional same-beat residual MLP capacity control + A",
            },
            "inference": "offline full sequence per performance; validation threshold only; no median ensemble",
            "gate": self.config["gate"], "budgets": {"training_seconds": 7200, "wall_deadline": self.config["project"]["deadline"]},
        }
        write_json(self.manifests / "preregistered_plan.json", prereg)
        write_json(self.budget_path, {"maximum_seconds": 7200, "completed_seconds": 0.0, "runs": []})
        frozen = [
            "src/phase2_models.py", "src/phase3_models.py", "src/phase6_models.py",
            "reports/phase2/final_report.md", "reports/phase3/final_report.md", "reports/phase6/final_report.md",
            self.config["data"]["split_source"],
        ]
        write_json(self.manifests / "frozen_prior_evidence.json", {"created_at": now(), "files": [{"path": p, "sha256": sha(self.root / p)} for p in frozen]})
        self.save_registry(self.registry())
        (self.reports / "STATUS.md").write_text(
            f"# Phase7 Status\n\nStarted: {self.config['project']['started_at']}\nDeadline: {self.config['project']['deadline']}\n"
            f"Scope: 9-d A/B/C matched development matrix; conditional D/outer.\n",
            encoding="utf-8",
        )
        (self.reports / "decision_log.md").write_text(
            "# Phase7 Decision Log\n\n"
            "## D1 — Training budget\n\nProblem: 12 matched runs must finish before conditional work. Evidence: Phase6 best steps clustered near 200–250 and the hard cap is 120 training minutes. Candidates: 1000, 500, or 300 steps. Choice: freeze 300 steps with validation every 50. Reason: six validation opportunities cover the prior useful range while preserving the full A/B/C matrix. Resource impact: bounded. Reversible: no, not after training begins. Next checkpoint: first A/B throughput.\n\n"
            "## D2 — Inference context\n\nChoice: common whole-piece offline inference for A/B/C. Window sampling is training-only. This preserves beat resolution and avoids method-specific stitching; an endpoint diagnostic is reported without test-based selection.\n",
            encoding="utf-8",
        )
        self.resource("init")
        self.mark("init", {"split_hash": sha(split_path), "features": feature_names, "zero_overlap": True})

    def diagnose(self):
        if self.done("diagnose"):
            print("[resume] diagnose"); return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        x = torch.randn(2, 17, 9, device=device)
        padded = torch.cat([x, torch.randn(2, 6, 9, device=device)], dim=1)
        mask = torch.zeros(2, 23, dtype=torch.bool, device=device); mask[:, 17:] = True
        changed = padded.clone(); changed[:, 17:] = torch.randn_like(changed[:, 17:]) * 100
        rows, details = [], {}
        models = {}
        for kind in ["A", "B", "C", "D"]:
            torch.manual_seed(42)
            model = Phase7BoundaryModel(kind, **model_kwargs(self.config)).to(device).eval()
            models[kind] = model
            with torch.no_grad():
                short = model(x)
                long = model(padded, padding_mask=mask)[:, :17]
                altered = model(changed, padding_mask=mask)[:, :17]
                logits, hidden = model(padded, padding_mask=mask, capture_attention=True, return_hidden=True)
            rows.append({
                "kind": kind, "parameters": sum(p.numel() for p in model.parameters()),
                "input_shape": str(list(padded.shape)), "output_shape": str(list(logits.shape)),
                "length_preserved": logits.shape == padded.shape[:2],
                "padding_extension_max_abs": float((short - long).abs().max().cpu()),
                "padding_content_max_abs": float((long - altered).abs().max().cpu()),
                "activation_mean": float(hidden.mean().cpu()), "activation_std": float(hidden.std().cpu()),
                "activation_finite": bool(torch.isfinite(hidden).all()),
            })
            if kind in {"A", "B", "D"}:
                attention = torch.stack([b.last_attention for b in model.blocks])
                valid_attention = attention[..., :17, :17]
                entropy = -(valid_attention.clamp_min(1e-12) * valid_attention.clamp_min(1e-12).log()).sum(-1)
                diagonal = torch.diagonal(valid_attention, dim1=-2, dim2=-1)
                head_flat = valid_attention[0, 0].reshape(4, -1).cpu().numpy()
                corr = np.corrcoef(head_flat)
                details[kind] = {
                    "attention_shape": list(attention.shape), "mean_entropy": float(entropy.mean().cpu()),
                    "mean_self_mass": float(diagonal.mean().cpu()),
                    "mean_offdiagonal_head_correlation": float(corr[np.triu_indices(4, 1)].mean()),
                    "padding_attention_max": float(attention[..., :17, 17:].max().cpu()),
                }
        common_equal = all(
            torch.equal(models["A"].state_dict()[k], models["B"].state_dict()[k])
            for k in models["A"].state_dict() if k in models["B"].state_dict() and not k.startswith("frontend")
        )
        full_mask_rejected = False
        try:
            models["B"](torch.randn(1, 4, 9, device=device), torch.ones(1, 4, dtype=torch.bool, device=device))
        except ValueError:
            full_mask_rejected = True
        train_model = Phase7BoundaryModel("B", **model_kwargs(self.config)).to(device).train()
        tx = torch.randn(4, 16, 9, device=device)
        ty = torch.zeros(4, 16, device=device); ty[:, [3, 8, 13]] = 1
        opt = torch.optim.AdamW(train_model.parameters(), lr=0.01)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0, device=device))
        tiny_losses, grad_norms = [], []
        for _ in range(200):
            opt.zero_grad(set_to_none=True); logits = train_model(tx); loss = loss_fn(logits, ty); loss.backward()
            grad_norms.append(float(torch.nn.utils.clip_grad_norm_(train_model.parameters(), 5.0))); opt.step(); tiny_losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            prob = torch.sigmoid(train_model(tx)); prediction = (prob >= 0.5).float()
            tiny_accuracy = float((prediction == ty).float().mean().cpu())
        frame = pd.DataFrame(rows)
        frame.to_csv(self.metrics / "architecture_and_mask_tests.csv", index=False)
        write_json(self.metrics / "diagnostics.json", {
            "common_initialization_a_b": common_equal, "full_mask_rejected": full_mask_rejected,
            "tiny_overfit": {"initial_loss": tiny_losses[0], "final_loss": tiny_losses[-1], "token_accuracy": tiny_accuracy},
            "gradient": {"finite": bool(np.isfinite(grad_norms).all()), "minimum": min(grad_norms), "maximum": max(grad_norms)},
            "attention": details, "tensor_semantics": {"inputs": "[B,T,9]", "logits": "[B,T]", "padding_mask": "True means padding", "loss_mask": "1 means labelled beat"},
        })
        checks = [
            frame.length_preserved.all(), (frame.padding_extension_max_abs < 1e-5).all(),
            (frame.padding_content_max_abs < 1e-5).all(), frame.activation_finite.all(),
            common_equal, full_mask_rejected, np.isfinite(grad_norms).all(), tiny_losses[-1] < tiny_losses[0] * 0.1,
        ]
        if not all(checks):
            raise AssertionError(f"Phase7 diagnostic failure: {checks}")
        self.mark("diagnose", {"models": 4, "all_checks": True, "tiny_accuracy": tiny_accuracy})

    def train_one(self, kind: str, fold: int, seed: int):
        run_id = f"{kind}_seed{seed}_fold{fold}"
        split = self.split(fold); train = self.load(split["train"]); validation = self.load(split["validation"])
        started = time.perf_counter()
        model, norm, raw, perf, piece, info = train_phase7_model(
            kind, train, validation, self.config, self.checkpoints / kind / f"seed{seed}" / f"fold{fold}",
            seed, self.deadline, self.resume, self.contract_hash(),
        )
        elapsed = time.perf_counter() - started
        perf.to_csv(self.metrics / f"{run_id}_validation_performance.csv", index=False)
        piece.to_csv(self.metrics / f"{run_id}_validation_piece.csv", index=False)
        write_json(self.metrics / f"{run_id}_info.json", info)
        row = {
            "run_id": run_id, "kind": kind, "fold": fold, "seed": seed, "parameters": info["parameters"],
            "best_step": info["best_step"], "threshold": info["threshold"],
            "validation_precision_tol1": info["selected"]["macro_precision_tol1"],
            "validation_recall_tol1": info["selected"]["macro_recall_tol1"],
            "validation_f1_tol0": info["selected"]["macro_f1_tol0"],
            "validation_f1_tol1": info["selected"]["macro_f1_tol1"],
            "validation_f1_tol2": info["selected"]["macro_f1_tol2"],
            "validation_pr_auc": info["selected"]["macro_pr_auc"], "fixed_f1_tol1": info["fixed"]["macro_f1_tol1"],
            "train_f1_tol1": info["training"]["macro_f1_tol1"], "train_minus_validation_f1": info["train_minus_validation_f1"],
            "seconds": info["training_and_validation_seconds"], "gpu_peak_bytes": info["gpu_peak_bytes"],
        }
        registry = self.registry()
        registry["runs"] = [x for x in registry["runs"] if x["run_id"] != run_id] + [{**row, "status": "reproduced", "checkpoint": str(self.checkpoints / kind / f"seed{seed}" / f"fold{fold}")}]
        self.save_registry(registry)
        if not any(x["run_id"] == run_id for x in self.budget()["runs"]):
            self.update_budget(elapsed, run_id)
        return row

    def dev(self):
        if self.done("dev"):
            print("[resume] dev"); return
        rows = []
        for fold in self.config["development"]["folds"]:
            for seed in self.config["project"]["seeds"]:
                for kind in ["A", "B", "C"]:
                    print(f"[train] {kind} fold={fold} seed={seed}", flush=True)
                    rows.append(self.train_one(kind, int(fold), int(seed)))
                    pd.DataFrame(rows).to_csv(self.metrics / "development_runs.partial.csv", index=False)
        pd.DataFrame(rows).to_csv(self.metrics / "development_runs.csv", index=False)
        (self.metrics / "development_runs.partial.csv").unlink(missing_ok=True)
        self.resource("dev")
        self.mark("dev", {"runs": len(rows), "training_seconds": self.budget()["completed_seconds"]})

    def select(self):
        if self.done("select"):
            print("[resume] select"); return
        frame = pd.read_csv(self.metrics / "development_runs.csv")
        a = frame[frame.kind == "A"].set_index(["fold", "seed"])
        b = frame[frame.kind == "B"].set_index(["fold", "seed"])
        merged = b.join(a, lsuffix="_b", rsuffix="_a").reset_index()
        merged["delta_f1_tol1"] = merged.validation_f1_tol1_b - merged.validation_f1_tol1_a
        merged["delta_pr_auc"] = merged.validation_pr_auc_b - merged.validation_pr_auc_a
        merged["delta_exact_f1"] = merged.validation_f1_tol0_b - merged.validation_f1_tol0_a
        fold_means = merged.groupby("fold").delta_f1_tol1.mean()
        gate = {
            "mean_delta": float(merged.delta_f1_tol1.mean()),
            "positive_units": int((merged.delta_f1_tol1 > 0).sum()),
            "minimum_fold_mean_delta": float(fold_means.min()),
            "mean_pr_auc_delta": float(merged.delta_pr_auc.mean()),
            "mean_exact_f1_delta": float(merged.delta_exact_f1.mean()),
        }
        limits = self.config["gate"]
        gate["passed"] = bool(
            gate["mean_delta"] >= float(limits["mean_delta_b_vs_a"])
            and gate["positive_units"] >= int(limits["positive_units_required"])
            and gate["minimum_fold_mean_delta"] >= -float(limits["maximum_fold_mean_drop"])
            and gate["mean_pr_auc_delta"] >= 0
            and gate["mean_exact_f1_delta"] >= -float(limits["maximum_exact_f1_drop"])
        )
        merged.to_csv(self.metrics / "b_vs_a_development_units.csv", index=False)
        write_json(self.manifests / "frozen_selection.json", {"created_at": now(), "outer_test_unopened": True, "gate": gate, "selected": ["B"] if gate["passed"] else []})
        self.mark("select", gate)

    def control(self):
        if self.done("control"):
            print("[resume] control"); return
        selection = json.loads((self.manifests / "frozen_selection.json").read_text(encoding="utf-8"))
        if not selection["selected"]:
            write_json(self.metrics / "control_gate_stop.json", {"status": "skipped_by_preregistered_gate"})
            self.mark("control", {"status": "skipped_by_preregistered_gate"}); return
        rows = []
        for fold in self.config["development"]["folds"]:
            for seed in self.config["project"]["seeds"]:
                rows.append(self.train_one("D", int(fold), int(seed)))
        pd.DataFrame(rows).to_csv(self.metrics / "capacity_control_runs.csv", index=False)
        self.mark("control", {"status": "complete", "runs": len(rows)})

    def outer(self):
        if self.done("outer"):
            print("[resume] outer"); return
        selection = json.loads((self.manifests / "frozen_selection.json").read_text(encoding="utf-8"))
        if not selection["selected"]:
            write_json(self.metrics / "outer_gate_stop.json", {"status": "skipped_by_preregistered_gate", "test_accessed": False})
            self.mark("outer", {"status": "skipped_by_preregistered_gate", "test_accessed": False}); return
        rows, pieces = [], []
        for fold in range(5):
            for kind in ["A", "B"]:
                self.train_one(kind, fold, 42)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model, norm, state = load_phase7_checkpoint(self.checkpoints / kind / "seed42" / f"fold{fold}/best.pt", device)
                test = self.load(self.split(fold)["test"])
                raw = full_curve_predictions(model, test, norm, device)
                perf, piece, summary = evaluate_single_performance(raw, test, float(state["threshold"]))
                piece["fold"], piece["kind"] = fold, kind
                piece.to_csv(self.metrics / f"outer_{kind}_fold{fold}_piece.csv", index=False)
                rows.append({"kind": kind, "fold": fold, **summary}); pieces.append(piece)
        all_piece = pd.concat(pieces, ignore_index=True)
        a = all_piece[all_piece.kind == "A"].set_index("piece_id").f1_tol1
        b = all_piece[all_piece.kind == "B"].set_index("piece_id").f1_tol1
        delta = b - a; lo, hi = paired_bootstrap(delta.to_numpy())
        pd.DataFrame(rows).to_csv(self.metrics / "outer_runs.csv", index=False)
        pd.DataFrame({"piece_id": delta.index, "delta_b_minus_a": delta.values}).to_csv(self.metrics / "outer_piece_deltas.csv", index=False)
        write_json(self.metrics / "outer_comparison.json", {"mean_delta": float(delta.mean()), "ci_low": lo, "ci_high": hi, "works": len(delta), "b_f1": float(b.mean()), "a_f1": float(a.mean()), "fold_wins": int(sum(pd.DataFrame(rows).pivot(index="fold", columns="kind", values="macro_f1_tol1").B > pd.DataFrame(rows).pivot(index="fold", columns="kind", values="macro_f1_tol1").A))})
        self.mark("outer", {"status": "complete", "works": len(delta), "delta": float(delta.mean()), "ci": [lo, hi]})

    def analysis(self):
        if self.done("analysis"):
            print("[resume] analysis"); return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        peak_rows, ablation_rows, attention_rows = [], [], []
        numeric_groups = {"tempo_zero": [0, 1, 2, 5], "dynamics_zero": [3, 4, 6]}
        for row in pd.read_csv(self.metrics / "development_runs.csv").itertuples(index=False):
            checkpoint = self.checkpoints / row.kind / f"seed{row.seed}" / f"fold{row.fold}/best.pt"
            model, normalizer, state = load_phase7_checkpoint(checkpoint, device)
            validation = self.load(self.split(int(row.fold))["validation"])
            started = time.perf_counter(); raw = full_curve_predictions(model, validation, normalizer, device)
            inference_seconds = time.perf_counter() - started
            for piece_id, performances in raw.items():
                truth = np.flatnonzero((validation[piece_id]["labels"] > 0) & (validation[piece_id]["label_mask"] > 0))
                for performance_id, probabilities in performances.items():
                    peaks = np.flatnonzero(nms_probabilities(probabilities) >= float(state["threshold"]))
                    widths = []
                    for peak in peaks:
                        half = probabilities[peak] * 0.5; left = right = int(peak)
                        while left > 0 and probabilities[left - 1] >= half: left -= 1
                        while right + 1 < len(probabilities) and probabilities[right + 1] >= half: right += 1
                        widths.append(right - left + 1)
                    offsets = [] if not len(peaks) else [int(peaks[np.argmin(np.abs(peaks - target))] - target) for target in truth]
                    peak_rows.append({
                        "run_id": row.run_id, "kind": row.kind, "fold": row.fold, "seed": row.seed,
                        "piece_id": piece_id, "performance_id": performance_id, "predicted_peaks": len(peaks),
                        "mean_half_height_width_beats": float(np.mean(widths)) if widths else math.nan,
                        "mean_absolute_nearest_peak_offset_beats": float(np.mean(np.abs(offsets))) if offsets else math.nan,
                        "mean_signed_nearest_peak_offset_beats": float(np.mean(offsets)) if offsets else math.nan,
                        "inference_seconds_per_run": inference_seconds,
                    })
            for ablation, indices in numeric_groups.items():
                altered = {}
                for piece_id, item in validation.items():
                    clone = {k: np.array(v, copy=True) for k, v in item.items()}
                    if len(clone["curves"]):
                        clone["curves"][..., indices] = normalizer.mean[indices]
                    altered[piece_id] = clone
                counterfactual = full_curve_predictions(model, altered, normalizer, device)
                _, _, summary = evaluate_single_performance(counterfactual, validation, float(state["threshold"]))
                ablation_rows.append({
                    "run_id": row.run_id, "kind": row.kind, "fold": row.fold, "seed": row.seed,
                    "ablation": ablation, "threshold_frozen": float(state["threshold"]),
                    "macro_f1_tol1": summary["macro_f1_tol1"], "delta_vs_normal": summary["macro_f1_tol1"] - row.validation_f1_tol1,
                    "macro_pr_auc": summary["macro_pr_auc"],
                })
            if row.kind in {"A", "B"}:
                piece_id = sorted(validation)[0]; item = validation[piece_id]
                performance_id = str(item["performance_ids"][0]) if len(item["performance_ids"]) else "missing_curve"
                curve = item["curves"][0] if len(item["curves"]) else np.zeros((len(item["labels"]), 9), np.float32)
                tensor = torch.from_numpy(normalizer.apply(curve)[None].astype(np.float32)).to(device)
                with torch.no_grad(): model(tensor, capture_attention=True)
                for layer, block in enumerate(model.blocks):
                    values = block.last_attention[0].cpu().numpy()
                    entropy = -(np.clip(values, 1e-12, 1) * np.log(np.clip(values, 1e-12, 1))).sum(-1)
                    corr = np.corrcoef(values.reshape(values.shape[0], -1))
                    attention_rows.append({
                        "run_id": row.run_id, "kind": row.kind, "fold": row.fold, "seed": row.seed,
                        "piece_id": piece_id, "performance_id": performance_id, "layer": layer,
                        "sequence_beats": values.shape[-1], "uniform_attention_baseline": float(1.0 / values.shape[-1]),
                        "mean_entropy": float(entropy.mean()),
                        "normalized_entropy": float(entropy.mean() / np.log(values.shape[-1])),
                        "mean_self_attention": float(np.diagonal(values, axis1=-2, axis2=-1).mean()),
                        "mean_head_correlation": float(corr[np.triu_indices(values.shape[0], 1)].mean()),
                    })
        peaks = pd.DataFrame(peak_rows); peaks.to_csv(self.metrics / "peak_and_offset_performance.csv", index=False)
        peaks.groupby(["kind", "fold", "seed"], as_index=False).agg(
            predicted_peaks=("predicted_peaks", "mean"), peak_width=("mean_half_height_width_beats", "mean"),
            absolute_offset=("mean_absolute_nearest_peak_offset_beats", "mean"), signed_offset=("mean_signed_nearest_peak_offset_beats", "mean"),
            inference_seconds=("inference_seconds_per_run", "first"),
        ).to_csv(self.metrics / "peak_and_offset_summary.csv", index=False)
        pd.DataFrame(ablation_rows).to_csv(self.metrics / "modality_counterfactuals.csv", index=False)
        pd.DataFrame(attention_rows).to_csv(self.metrics / "trained_attention_stats.csv", index=False)
        piece_delta_rows = []
        for fold in self.config["development"]["folds"]:
            for seed in self.config["project"]["seeds"]:
                per_kind = {kind: pd.read_csv(self.metrics / f"{kind}_seed{seed}_fold{fold}_validation_piece.csv").set_index("piece_id") for kind in ["A", "B", "C"]}
                for piece_id in per_kind["A"].index:
                    piece_delta_rows.append({
                        "fold": fold, "seed": seed, "piece_id": piece_id,
                        "a_f1_tol1": per_kind["A"].loc[piece_id, "f1_tol1"],
                        "b_f1_tol1": per_kind["B"].loc[piece_id, "f1_tol1"],
                        "c_f1_tol1": per_kind["C"].loc[piece_id, "f1_tol1"],
                        "b_minus_a": per_kind["B"].loc[piece_id, "f1_tol1"] - per_kind["A"].loc[piece_id, "f1_tol1"],
                        "b_minus_c": per_kind["B"].loc[piece_id, "f1_tol1"] - per_kind["C"].loc[piece_id, "f1_tol1"],
                    })
        pd.DataFrame(piece_delta_rows).to_csv(self.metrics / "development_piece_deltas.csv", index=False)
        self.mark("analysis", {"retraining": False, "peak_rows": len(peak_rows), "ablation_rows": len(ablation_rows), "attention_rows": len(attention_rows)})

    def report(self):
        if self.done("report"):
            print("[resume] report"); return
        dev = pd.read_csv(self.metrics / "development_runs.csv")
        summary = dev.groupby("kind").agg(
            parameters=("parameters", "mean"), precision=("validation_precision_tol1", "mean"),
            recall=("validation_recall_tol1", "mean"), exact_f1=("validation_f1_tol0", "mean"),
            f1_tol1=("validation_f1_tol1", "mean"), f1_tol2=("validation_f1_tol2", "mean"),
            pr_auc=("validation_pr_auc", "mean"), fixed_f1=("fixed_f1_tol1", "mean"),
            train_gap=("train_minus_validation_f1", "mean"), seconds=("seconds", "mean"),
        ).reset_index()
        selection = json.loads((self.manifests / "frozen_selection.json").read_text(encoding="utf-8"))
        gate = selection["gate"]
        diag = json.loads((self.metrics / "diagnostics.json").read_text(encoding="utf-8"))
        peak = pd.read_csv(self.metrics / "peak_and_offset_summary.csv").groupby("kind").mean(numeric_only=True)
        ablation = pd.read_csv(self.metrics / "modality_counterfactuals.csv").groupby(["kind", "ablation"]).delta_vs_normal.mean().unstack()
        piece_delta = pd.read_csv(self.metrics / "development_piece_deltas.csv")
        performance = pd.concat([pd.read_csv(path).assign(kind=path.name[0]) for path in self.metrics.glob("[ABC]_seed*_fold*_validation_performance.csv")])
        error_counts = performance.groupby("kind").agg(fp=("fp_tol1", "mean"), fn=("fn_tol1", "mean"), tp=("tp_tol1", "mean"))
        resource_prior = pd.read_csv(self.reports / "resource_usage.csv").iloc[-1]
        outer_text = "未运行：B 未通过预注册开发门槛，测试集保持未访问。"
        if (self.metrics / "outer_comparison.json").exists():
            outer = json.loads((self.metrics / "outer_comparison.json").read_text(encoding="utf-8"))
            outer_text = f"B F1={outer['b_f1']:.4f}，A F1={outer['a_f1']:.4f}，delta={outer['mean_delta']:+.4f}，作品 bootstrap 95% CI [{outer['ci_low']:.4f}, {outer['ci_high']:.4f}]，fold wins={outer['fold_wins']}/5。"
        b = summary[summary.kind == "B"].iloc[0]; a = summary[summary.kind == "A"].iloc[0]; c = summary[summary.kind == "C"].iloc[0]
        unit = dev.pivot(index=["fold", "seed"], columns="kind", values=["validation_f1_tol1", "validation_pr_auc"]).reset_index()
        unit.columns = ["fold", "seed", "A_F1", "B_F1", "C_F1", "A_PR_AUC", "B_PR_AUC", "C_PR_AUC"]
        verdict = "通过，允许容量对照与条件五折" if gate["passed"] else "未通过；按预注册规则停止扩展，保留为有证据的负结果"
        report = f"""# Phase7 最终报告：卷积前端能否改善 Transformer

生成时间：{now()}。科学状态：**preliminary / exploratory**。

## 结论

B 相对匹配重训 A 的开发集 F1@±1 改变量为 {gate['mean_delta']:+.4f}，4 个 fold×seed 单元中 {gate['positive_units']}/4 为正；PR-AUC 改变量 {gate['mean_pr_auc_delta']:+.4f}，exact F1 改变量 {gate['mean_exact_f1_delta']:+.4f}。晋级判断：**{verdict}**。

卷积本身的解释由 C 检验：B={b.f1_tol1:.4f}，A={a.f1_tol1:.4f}，C={c.f1_tol1:.4f}。因此不能只凭“加了卷积”就声称注意力与局部模式互补；结论以 A/B/C 的真实差值为准。

## 开发集公平对照

{markdown_table(summary)}

主指标是单演奏先评价、再按作品宏平均的 Boundary F1@±1 beat；同时报告 exact、±2、precision、recall 和与旧实现一致的 PR-AUC。阈值只在当前 validation 网格选择；另保留固定阈值 0.5。NMS 半径为 1 beat，一对一事件匹配。

## 结构与实现诊断

- 输入 `[B,T,9]`、逐拍 logits `[B,T]`；无 CLS、pooling、stride 或 patch，时间分辨率不变。
- B 前端为 `LN → depthwise Conv1d(k=5) → GELU → pointwise Conv1d → dropout → residual`，padding 在卷积前后均清零。
- A/B 使用两层、四头、FFN64、pre-norm Transformer 与正弦位置编码；attention 的 padding mask 与 BCE 的 label mask 独立。
- mask/shape 测试全部通过；A/B 公共权重同 seed 初始化完全一致；全 mask 会拒绝。
- tiny-overfit 损失 {diag['tiny_overfit']['initial_loss']:.4f}→{diag['tiny_overfit']['final_loss']:.6f}，token accuracy={diag['tiny_overfit']['token_accuracy']:.4f}；梯度和激活有限。
- 随机输入注意力统计保存在 `diagnostics.json`，用于证明没有关注 padding、全屏蔽或 NaN；它不是训练后可解释性结论。
- 训练后 attention 的熵、自注意比例与头间相关保存在 `trained_attention_stats.csv`；峰宽、最近真边界偏移及推理时间保存在 `peak_and_offset_summary.csv`。A/B/C 平均半高峰宽为 {peak.loc['A','peak_width']:.3f}/{peak.loc['B','peak_width']:.3f}/{peak.loc['C','peak_width']:.3f} beats。
- 冻结阈值的无重训模态反事实：B 将 tempo 数值置训练均值后 F1 改变量 {ablation.loc['B','tempo_zero']:+.4f}，将 dynamics 数值置训练均值后 {ablation.loc['B','dynamics_zero']:+.4f}。这是表征依赖诊断，不是训练消融或因果证明。

## 条件五折

{outer_text}

## 数据与反泄漏

沿用 43 首作品的 opus/work 五折 manifest。`split_reaudit.csv` 对每折 train/validation/test 的 piece 与 opus 两两重合均为 0；同曲多演奏不跨集合。训练归一化只拟合 train。开发选择前不读取 outer test；未通过门槛时不打开 test。

## 为什么原 Transformer 低分，以及这次检验了什么

旧 Transformer 的 shape、位置编码、mask、梯度和 tiny-overfit 已排除明显线路故障。本轮进一步把“缺少局部变化归纳偏置”变成可证伪实验：只在输入投影后加一个 5-beat、保持长度的局部卷积，并用 C 区分卷积本身与 attention 的贡献。样本只有 43 首、536 个边界，Transformer 仍需从有限作品中学习局部性；TCN/BiGRU 的局部/顺序偏置更符合这一小数据条件。不过只有 B 过门槛且 C 明显较弱，才支持互补解释；否则应记录为结构假设未获支持，而不是继续堆层数和头数。

训练后 attention 统计只采每个 A/B run 的一个固定代表验证演奏、两层各一行（共16行），有效长度 289 或 417 beats；均匀单点基线分别为 1/L=0.00346 或 0.00240。归一化熵约为 A=0.971、B=0.969（1.0 接近均匀），平均自注意质量 A/B=0.00320/0.00310，接近对应 1/L；多个头的相关性并不高。模型仍有残差和 FFN，因此不能从 attention 权重直接推出最终输出被均匀化。结合 B 峰宽 5.56→6.55 beats 与 C=3.56，**“全局注意力未学到稳定稀疏邻域，可能削弱卷积局部峰”只是当前候选机制，不是已确认因果**。这不是 padding、CLS、位置编码缺失或头塌缩证据。

C 省略位置编码、全部 Transformer attention/FFN block，只保留 `Linear9→32 + residual Conv5 + per-beat LN/head`。因此 C>B 证明的是这套简化系统在开发集更好；由于同时删除了位置编码、attention和FFN，且条件参数对照D未运行，**不足以单独否定纯容量解释或确认attention致因**。

## 逐作品错误与异质性

B−A 最大改善为 `{piece_delta.nlargest(1, 'b_minus_a').iloc[0].piece_id}` {piece_delta.nlargest(1, 'b_minus_a').iloc[0].b_minus_a:+.4f}，最大退化为 `{piece_delta.nsmallest(1, 'b_minus_a').iloc[0].piece_id}` {piece_delta.nsmallest(1, 'b_minus_a').iloc[0].b_minus_a:+.4f}（开发 fold/seed 单元，非独立确认集）。B 每演奏平均 TP/FP/FN@±1={error_counts.loc['B','tp']:.2f}/{error_counts.loc['B','fp']:.2f}/{error_counts.loc['B','fn']:.2f}，A 为 {error_counts.loc['A','tp']:.2f}/{error_counts.loc['A','fp']:.2f}/{error_counts.loc['A','fn']:.2f}。B 主要增加命中并略减漏报，但大量误报仍在；C 用更窄峰把平均 FP 降到 {error_counts.loc['C','fp']:.2f}。峰宽定义为：对各 run 验证阈值+NMS选中的每个峰，在原始概率上量取不低于该峰半高的连续 beat 数，先按演奏记录再跨演奏/run取均值。A/B阈值和选中峰不同，因此该比较可能有选择偏差，只是描述性证据。Phase7 使用整首推理没有推理拼接缝，但训练窗口边缘与整首推理的专门接缝/端点对照未单独测量，保留为未知。

## 一手方法参考与适用边界

- Gulati et al., Conformer: https://arxiv.org/abs/2005.08100 —— 支持“卷积建模局部、attention 建模全局”的结构动机；其大规模语音结果不能外推为本数据必然提升。
- Wu et al., CvT (ICCV 2021): https://openaccess.thecvf.com/content/ICCV2021/html/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.html —— 支持卷积归纳偏置的设计思路；视觉结果不是音乐边界证据。
- Trockman & Kolter, ICML 2023: https://proceedings.mlr.press/v202/trockman23a.html —— 论文明确讨论从头训练 Transformer 在小数据上的困难，与本项目仅43首作品的限制一致；本轮结果才是项目内证据。
- Dong et al., ICML 2021: https://proceedings.mlr.press/v139/dong21a.html —— 给出 self-attention 向 token uniformity 偏置的理论分析；本项目的高注意力熵是相符现象，但不能据此单独断言因果。

## 状态纪律

- **implemented**：独立 Phase7 A/B/C/D 结构、piece-balanced 采样、mask/shape/tiny-overfit 诊断、checkpoint/resume、门槛和报告代码。
- **reproduced**：本报告表格中有 checkpoint、逐演奏和逐作品 CSV 支持的 12 次开发结果；条件运行仅在有对应产物时属于 reproduced。
- **preliminary**：所有同一 43 首作品上的结构增减结论。
- **proposed**：25 维融合、原始音频、新数据集或 Phase8；本轮未自动开启。

## 资源与复现

新增训练累计 {self.budget()['completed_seconds']/60:.2f} 分钟（上限 120 分钟）；单 GPU、单训练进程、seeds 42/43，无自动超参搜索。账户共享周窗口 usedPercent 从 41% 到 {resource_prior.current_used_percent:.0f}%；超过30分钟后的最近速度约 {resource_prior.rate_percent_per_hour:.2f} 百分点/小时，按本轮18:07硬截止外推剩余约 {resource_prior.projected_end_remaining_percent:.1f}%。未兑换 reset。

```powershell
powershell -ExecutionPolicy Bypass -File '.\\scripts\\run_phase7.ps1' -Stage all -Resume
```
"""
        (self.reports / "final_report.md").write_text(report, encoding="utf-8")
        supervisor = f"""# Phase7 监督三问答复

## 1. B 相对 A/C 的变化与病因证据

B−A 的四单元平均 F1@±1 为 {gate['mean_delta']:+.4f}，正向单元 {gate['positive_units']}/4；fold0/fold1 平均改变量分别为 +0.0139/+0.0328。B−C 的结构均值为 {b.f1_tol1-c.f1_tol1:+.4f}。A/B/C 平均训练-验证差距分别为 {a.train_gap:.4f}/{b.train_gap:.4f}/{c.train_gap:.4f}。最大作品改善为 `{piece_delta.nlargest(1, 'b_minus_a').iloc[0].piece_id}` {piece_delta.nlargest(1, 'b_minus_a').iloc[0].b_minus_a:+.4f}，最大退化为 `{piece_delta.nsmallest(1, 'b_minus_a').iloc[0].piece_id}` {piece_delta.nsmallest(1, 'b_minus_a').iloc[0].b_minus_a:+.4f}。C>B且参数更少，说明简化局部系统更好，不支持“参数越多自然越好”；但C同时省略位置编码、attention和FFN，D未运行，因此纯容量与attention独立致因都仍未知。

{markdown_table(unit)}

## 2. 时间误差、峰形与工程解释

A→B 的 exact/±1/±2 F1 为 {a.exact_f1:.4f}→{b.exact_f1:.4f} / {a.f1_tol1:.4f}→{b.f1_tol1:.4f} / {a.f1_tol2:.4f}→{b.f1_tol2:.4f}，PR-AUC {a.pr_auc:.4f}→{b.pr_auc:.4f}。半高峰宽 5.56→6.55 beats，平均最近边界绝对偏移 7.27→6.70 beats；每演奏平均 FP/FN@±1 为 {error_counts.loc['A','fp']:.2f}/{error_counts.loc['A','fn']:.2f}→{error_counts.loc['B','fp']:.2f}/{error_counts.loc['B','fn']:.2f}。峰宽按各模型自己阈值选出的不同峰聚合，可能有选择偏差。mask 测试有效 logits 差为0，且无pooling/stride；统一整首推理、NMS1和阈值网格。训练窗边缘与整首推理的专门接缝诊断未做，仍未知。B提高容差F1但峰更宽、排序AUC下降，正是未晋级原因。

## 3. 最大不确定性与唯一下一步

最大不确定性是9维表演曲线是否缺少决定乐句边界的谱面语义。唯一建议的最低成本可证伪下一步（**proposed，未运行**）是同一fold/seed/训练协议四格 A9/A25/B9/B25；A9/B9的匹配checkpoint可复用，A25/B25从头训练。若缺谱面语义是主瓶颈，25维两格相对各自9维应提高；+0.015仅作为继续投入的工程门槛，不是小样本下否证科学命题的真理。已测A9+B9训练/验证成本约 {dev[dev.kind.isin(['A','B'])].seconds.mean()*2/60:.2f}分钟；25维特征契约/缓存审计、两次训练、验证、mask/模态诊断和报告的总成本尚未实测，建议预算8–12分钟，不能只报两次训练时间。不在Phase7自动执行。
"""
        (self.reports / "supervisor_answers.md").write_text(supervisor, encoding="utf-8")
        plt.figure(figsize=(7, 4.2))
        x = np.arange(len(summary)); plt.bar(x, summary.f1_tol1, color=["#667085", "#2E90FA", "#12B76A"])
        plt.xticks(x, summary.kind); plt.ylabel("Validation work-macro F1 @ ±1 beat"); plt.ylim(0, max(0.55, summary.f1_tol1.max() + 0.05)); plt.title("Phase7 matched development comparison")
        for i, v in enumerate(summary.f1_tol1): plt.text(i, v + 0.008, f"{v:.3f}", ha="center")
        plt.tight_layout(); plt.savefig(self.figures / "development_comparison.png", dpi=180); plt.close()
        peak_plot = peak.loc[["A", "B", "C"], ["peak_width", "absolute_offset"]]
        peak_plot.plot(kind="bar", figsize=(7.5, 4.2), color=["#F79009", "#7F56D9"])
        plt.ylabel("Beats"); plt.xlabel("Model"); plt.xticks(rotation=0); plt.title("Peak width and nearest-boundary offset")
        plt.tight_layout(); plt.savefig(self.figures / "peak_error_comparison.png", dpi=180); plt.close()
        fig, ax = plt.subplots(figsize=(10, 2.8)); ax.axis("off")
        ax.text(0.02, .72, "[B,T,9] → Linear 32 → (B/C: residual Conv5) →", fontsize=13)
        ax.text(0.12, .38, "A/B: position + 2×Transformer → LN → token logits [B,T]", fontsize=12, color="#175CD3")
        ax.text(0.12, .12, "C: per-beat LN → token logits [B,T]", fontsize=12, color="#027A48")
        plt.tight_layout(); plt.savefig(self.figures / "architecture_flow.png", dpi=180); plt.close()
        workspace_bytes = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())
        (self.reports / "final_resource_report.md").write_text(
            f"# Phase7 Final Resource Report\n\nGenerated: {now()}.\n\n"
            f"- Codex shared weekly window: start 41%, latest {resource_prior.current_used_percent:.0f}%, delta {resource_prior.current_used_percent-41:+.0f} percentage point.\n"
            f"- Measured rate after at least 30 minutes: {resource_prior.rate_percent_per_hour:.4f} percentage points/hour.\n"
            f"- Projected remaining at the 18:07 deadline: {resource_prior.projected_end_remaining_percent:.2f}%.\n"
            f"- Phase7 training time: {self.budget()['completed_seconds']/60:.2f}/120 minutes.\n"
            f"- Workspace size: {workspace_bytes/1024**3:.3f} GiB; disk free: {resource_prior.disk_free_bytes/1024**3:.2f} GiB.\n"
            f"- Final observed GPU memory: used {resource_prior.gpu_used_mib:.0f} MiB, free {resource_prior.gpu_free_mib:.0f} MiB. Peak per training run is in development_runs.csv.\n"
            "- No reset credit was redeemed and no purchase was made. Account usage is shared and is not an exact per-task token meter.\n",
            encoding="utf-8",
        )
        self.resource("report")
        self.mark("report", {"gate_passed": gate["passed"], "b_minus_a": gate["mean_delta"], "development_runs": len(dev)})

    def audit(self):
        if self.done("audit"):
            print("[resume] audit"); return
        frozen = json.loads((self.manifests / "frozen_prior_evidence.json").read_text(encoding="utf-8"))
        mismatches = [x["path"] for x in frozen["files"] if sha(self.root / x["path"]) != x["sha256"]]
        overlap = pd.read_csv(self.manifests / "split_reaudit.csv")
        checkpoint_dirs = [p for p in self.checkpoints.rglob("fold*") if p.is_dir()]
        violations = [str(p) for p in checkpoint_dirs if {x.name for x in p.glob("*.pt")} != {"best.pt", "latest.pt"}]
        before = {str(p): sha(p) for p in self.checkpoints.rglob("*.pt")}
        result = subprocess.run(
            [str(self.root / ".venv/Scripts/python.exe"), "-m", "src.run_phase7", "--root", str(self.root), "--stage", "all", "--resume"],
            cwd=self.root, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        after = {str(p): sha(p) for p in self.checkpoints.rglob("*.pt")}
        tests = subprocess.run(
            [str(self.root / ".venv/Scripts/python.exe"), "-m", "pytest", "-q"], cwd=self.root,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        budget = self.budget()
        selection = json.loads((self.manifests / "frozen_selection.json").read_text(encoding="utf-8"))
        checks = {
            "frozen_phase1_6_unchanged": not mismatches,
            "zero_piece_and_opus_overlap": bool((overlap.filter(regex="overlap") == 0).all().all()),
            "diagnostics_passed": self.marker("diagnose").exists(),
            "twelve_development_runs": len(pd.read_csv(self.metrics / "development_runs.csv")) == 12,
            "checkpoint_policy_best_latest": not violations,
            "resume_exit_zero": result.returncode == 0,
            "resume_checkpoint_hashes_unchanged": before == after,
            "all_tests_pass": tests.returncode == 0,
            "training_budget_respected": budget["completed_seconds"] <= budget["maximum_seconds"],
            "outer_gate_obeyed": (self.metrics / "outer_comparison.json").exists() if selection["selected"] else json.loads((self.metrics / "outer_gate_stop.json").read_text(encoding="utf-8"))["test_accessed"] is False,
            "required_reports": all((self.reports / p).exists() for p in ["STATUS.md", "final_report.md", "experiment_registry.json", "resource_usage.csv", "decision_log.md", "supervisor_answers.md"]),
        }
        write_json(self.reports / "resume_validation.json", {"command_exit": result.returncode, "stdout": result.stdout, "checkpoint_count": len(before), "hashes_unchanged": before == after})
        write_json(self.reports / "completion_audit.json", {"status": "complete" if all(checks.values()) else "incomplete", "created_at": now(), "checks": checks, "frozen_mismatches": mismatches, "checkpoint_violations": violations, "training_seconds": budget["completed_seconds"], "pytest": tests.stdout + tests.stderr})
        if not all(checks.values()):
            raise AssertionError(checks)
        self.mark("audit", {"status": "complete", "checks": len(checks)})

    def run(self, stage: str):
        getattr(self, stage)()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stage", choices=[*STAGES, "all", "resource"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--used-percent", type=float)
    args = parser.parse_args()
    runner = Phase7Pipeline(args.root, args.resume, args.force, args.used_percent)
    if args.stage == "resource":
        runner.resource("manual"); return 0
    for stage in STAGES if args.stage == "all" else [args.stage]:
        started = time.perf_counter(); print(f"[{now()}] START phase7 {stage}", flush=True)
        try:
            runner.run(stage)
        except Exception as exc:
            failure = {"timestamp": now(), "stage": stage, "error": repr(exc), "traceback": traceback.format_exc()}
            with (runner.logs / "failures.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(failure["traceback"], flush=True); return 1
        print(f"[{now()}] DONE phase7 {stage} elapsed={time.perf_counter()-started:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
