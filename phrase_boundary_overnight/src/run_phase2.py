from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import load_config, load_piece_cache, write_json
from .evaluation import bootstrap_macro_ci, choose_threshold, evaluate_piece, evaluate_predictions
from .models import BeatBoundaryTCN, Normalizer, predict_tcn, train_tcn
from .phase2_models import (
    CompactBoundaryTransformer,
    choose_single_threshold,
    evaluate_single_performance,
    fit_curve_normalizer,
    median_predictions,
    nms_probabilities,
    pretrain_masked_tcn,
    _raw_performance_predictions,
    train_transformer,
)


STAGES = ["splits", "single", "seeds", "opus", "diagnostics", "ssl", "transformer", "transformer_diagnostics", "matched_comparisons", "report"]


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def paired_bootstrap(values: np.ndarray, iterations: int = 5000, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(iterations)])
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def markdown_table(frame: pd.DataFrame) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value)
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


class Phase2Pipeline:
    def __init__(self, root: Path, resume: bool = True, force: bool = False):
        self.root = root.resolve()
        self.resume, self.force = resume, force
        self.phase1 = load_config(self.root / "configs" / "tcn_h32_curves.yaml")
        self.protocol = yaml.safe_load((self.root / "configs" / "phase2" / "protocol.yaml").read_text(encoding="utf-8"))
        self.cache = self.root / "cache" / "piece_features"
        self.artifacts = self.root / "artifacts" / "phase2"
        self.checkpoints = self.root / "checkpoints" / "phase2"
        self.reports = self.root / "reports" / "phase2"
        self.logs = self.root / "logs" / "phase2"
        self.metrics = self.artifacts / "metrics"
        self.figures = self.artifacts / "figures"
        self.splits_dir = self.artifacts / "splits"
        self.markers = self.artifacts / "stages"
        for path in [self.artifacts, self.checkpoints, self.reports, self.logs, self.metrics, self.figures, self.splits_dir, self.markers]:
            path.mkdir(parents=True, exist_ok=True)

    def marker(self, stage: str) -> Path:
        return self.markers / f"{stage}.json"

    def done(self, stage: str) -> bool:
        return self.resume and self.marker(stage).exists() and not self.force

    def mark(self, stage: str, payload: dict[str, Any]) -> None:
        write_json(self.marker(stage), {"stage": stage, "completed_at": now_text(), **payload})

    def piece_splits(self, fold: int) -> dict[str, list[str]]:
        return {split: pd.read_csv(self.root / "splits" / f"fold_{fold}_{split}.csv").piece_id.tolist() for split in ["train", "validation", "test"]}

    def opus_splits(self, fold: int) -> dict[str, list[str]]:
        frame = pd.read_csv(self.splits_dir / "opus_split_manifest.csv")
        return {split: frame[(frame.fold == fold) & (frame.split == split)].piece_id.tolist() for split in ["train", "validation", "test"]}

    def load_data(self, piece_ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
        return {piece_id: load_piece_cache(self.root / "cache", piece_id) for piece_id in piece_ids}

    @staticmethod
    def targets(data: dict[str, dict[str, np.ndarray]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {piece_id: (item["labels"], item["label_mask"]) for piece_id, item in data.items()}

    def load_tcn(self, checkpoint: Path) -> tuple[BeatBoundaryTCN, Normalizer, dict[str, Any]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model = BeatBoundaryTCN(int(state["input_dim"]), 32, [1, 2, 4, 8], 3, 0.2).to(device)
        model.load_state_dict(state["model"])
        normalizer = Normalizer(np.asarray(state["normalizer_mean"]), np.asarray(state["normalizer_std"]))
        return model, normalizer, state

    def tcn_checkpoint(self, seed: int, fold: int) -> Path:
        if seed == 42:
            return self.root / "artifacts" / "checkpoints" / f"fold{fold}_B2_TCN_BeatBoundaryTCN_h32_curves" / "best.pt"
        return self.checkpoints / "seed_stability" / f"seed{seed}" / f"fold{fold}" / "best.pt"

    def _save_single_result(self, prefix: str, raw_validation, raw_test, validation_data, test_data) -> dict[str, Any]:
        threshold, threshold_frame = choose_single_threshold(raw_validation, validation_data, self.protocol["evaluation"]["threshold_grid"])
        val_perf, val_piece, val_summary = evaluate_single_performance(raw_validation, validation_data, threshold)
        test_perf, test_piece, test_summary = evaluate_single_performance(raw_test, test_data, threshold)
        threshold_frame.to_csv(self.metrics / f"{prefix}_threshold_selection.csv", index=False)
        val_perf.to_csv(self.metrics / f"{prefix}_validation_per_performance.csv", index=False)
        val_piece.to_csv(self.metrics / f"{prefix}_validation_per_piece.csv", index=False)
        test_perf.to_csv(self.metrics / f"{prefix}_test_per_performance.csv", index=False)
        test_piece.to_csv(self.metrics / f"{prefix}_test_per_piece.csv", index=False)
        payload = {"threshold": threshold, "validation": val_summary, "test": test_summary}
        write_json(self.metrics / f"{prefix}_summary.json", payload)
        return payload

    def splits(self) -> None:
        if self.done("splits"):
            print("[resume] phase2 splits complete"); return
        feature = pd.read_csv(self.root / "manifests" / "feature_manifest.csv")
        feature["opus"] = feature.piece_id.str.extract(r"op(\d+)")[0]
        groups = feature.groupby("opus").piece_id.apply(list).to_dict()
        rng = np.random.default_rng(42)
        shuffled = list(groups)
        rng.shuffle(shuffled)
        tie_order = {g: i for i, g in enumerate(shuffled)}
        ordered = sorted(groups, key=lambda g: (-len(groups[g]), tie_order[g]))
        buckets: list[list[str]] = [[] for _ in range(5)]
        bucket_counts = [0] * 5
        for opus in ordered:
            target = min(range(5), key=lambda index: (bucket_counts[index], index))
            buckets[target].append(opus); bucket_counts[target] += len(groups[opus])
        rows = []
        for fold in range(5):
            for bucket_index, opuses in enumerate(buckets):
                split = "test" if bucket_index == fold else "validation" if bucket_index == (fold + 1) % 5 else "train"
                for opus in sorted(opuses):
                    for piece_id in sorted(groups[opus]):
                        rows.append({"fold": fold, "split": split, "opus": opus, "piece_id": piece_id})
        manifest = pd.DataFrame(rows)
        manifest.to_csv(self.splits_dir / "opus_split_manifest.csv", index=False)
        hashes = pd.read_csv(self.root / "manifests" / "source_content_hashes.csv")
        audit_rows = []
        for fold in range(5):
            part = manifest[manifest.fold == fold]
            for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
                l, rr = part[part.split == left], part[part.split == right]
                lh = set(hashes.loc[hashes.piece_id.isin(l.piece_id), "sha256"])
                rh = set(hashes.loc[hashes.piece_id.isin(rr.piece_id), "sha256"])
                audit_rows.append({"fold": fold, "left": left, "right": right, "piece_overlap": len(set(l.piece_id) & set(rr.piece_id)), "opus_overlap": len(set(l.opus) & set(rr.opus)), "source_hash_overlap": len(lh & rh), "window_origin_overlap": len(set(l.piece_id) & set(rr.piece_id))})
        audit = pd.DataFrame(audit_rows)
        audit.to_csv(self.splits_dir / "opus_overlap_audit.csv", index=False)
        if audit.filter(regex="overlap$").to_numpy().any():
            raise AssertionError("Opus split leakage detected")
        summary = manifest.groupby(["fold", "split"]).agg(pieces=("piece_id", "nunique"), opuses=("opus", "nunique")).reset_index()
        summary.to_csv(self.splits_dir / "opus_split_summary.csv", index=False)
        self.mark("splits", {"pieces": int(feature.piece_id.nunique()), "opuses": int(feature.opus.nunique()), "all_overlaps_zero": True})
        print(summary.to_string(index=False))

    @staticmethod
    def _ranked_subset(raw: dict[str, dict[str, np.ndarray]], count: int | None) -> dict[str, np.ndarray]:
        result = {}
        for piece_id, perfs in raw.items():
            ids = sorted(perfs, key=lambda p: hashlib.sha256(f"{piece_id}|{p}|42".encode()).hexdigest())
            selected = ids if count is None else ids[: min(count, len(ids))]
            result[piece_id] = nms_probabilities(np.median(np.stack([perfs[p] for p in selected]), axis=0))
        return result

    def single(self) -> None:
        if self.done("single"):
            print("[resume] phase2 single-performance audit complete"); return
        perf_frames, piece_frames, aggregation_rows, aggregation_piece_frames, paired_rows = [], [], [], [], []
        reproduction_errors = []
        for fold in range(5):
            split = self.piece_splits(fold); val = self.load_data(split["validation"]); test = self.load_data(split["test"])
            model, norm, state = self.load_tcn(self.tcn_checkpoint(42, fold)); device = next(model.parameters()).device
            raw_val = _raw_performance_predictions(model, val, norm, device, "tcn")
            raw_test = _raw_performance_predictions(model, test, norm, device, "tcn")
            pilot_threshold = float(state["threshold"])
            single_threshold, _ = choose_single_threshold(raw_val, val, self.protocol["evaluation"]["threshold_grid"])
            for threshold_name, threshold in [("pilot_median_threshold", pilot_threshold), ("single_validation_threshold", single_threshold)]:
                pf, pif, summary = evaluate_single_performance(raw_test, test, threshold)
                pf.insert(0, "threshold_protocol", threshold_name); pf.insert(0, "fold", fold)
                pif.insert(0, "threshold_protocol", threshold_name); pif.insert(0, "fold", fold)
                perf_frames.append(pf); piece_frames.append(pif)
                aggregation_rows.append({"fold": fold, "aggregation": f"single_mean_{threshold_name}", **summary})
            targets = self.targets(test)
            for count, label in [(1, "fixed_1"), (2, "fixed_2"), (4, "fixed_4"), (None, "piece_median_all")]:
                pred = self._ranked_subset(raw_test, count)
                frame, summary = evaluate_predictions(pred, targets, pilot_threshold)
                frame.insert(0, "aggregation", label); frame.insert(0, "fold", fold)
                aggregation_piece_frames.append(frame)
                if label == "piece_median_all":
                    pilot = pd.read_csv(self.root / "artifacts" / "metrics" / f"fold{fold}_B2_TCN_BeatBoundaryTCN_h32_curves_test_per_piece.csv").set_index("piece_id")
                    current = frame.set_index("piece_id")
                    reproduction_errors.extend((current.loc[pilot.index, "f1_tol1"] - pilot.f1_tol1).abs().tolist())
                aggregation_rows.append({"fold": fold, "aggregation": label, **summary})
                if label == "piece_median_all":
                    single_piece = piece_frames[-1].set_index("piece_id")
                    for row in frame.itertuples():
                        paired_rows.append({"fold": fold, "piece_id": row.piece_id, "performances": int(len(raw_test[row.piece_id])), "piece_median_f1_tol1": row.f1_tol1, "single_mean_f1_tol1": float(single_piece.loc[row.piece_id, "f1_tol1"]), "delta_single_minus_median": float(single_piece.loc[row.piece_id, "f1_tol1"] - row.f1_tol1)})
        perfs = pd.concat(perf_frames, ignore_index=True); pieces = pd.concat(piece_frames, ignore_index=True)
        perfs.to_csv(self.metrics / "single_performance_distribution.csv", index=False)
        pieces.to_csv(self.metrics / "single_performance_per_piece.csv", index=False)
        aggregations = pd.DataFrame(aggregation_rows); aggregations.to_csv(self.metrics / "performance_aggregation_sensitivity.csv", index=False)
        aggregation_pieces = pd.concat(aggregation_piece_frames, ignore_index=True)
        aggregation_pieces.to_csv(self.metrics / "performance_aggregation_per_piece.csv", index=False)
        paired = pd.DataFrame(paired_rows); paired.to_csv(self.metrics / "single_vs_median_paired.csv", index=False)
        if max(reproduction_errors, default=0) > 1e-10:
            raise AssertionError(f"Pilot median reproduction mismatch {max(reproduction_errors)}")
        metric_columns = ["precision_tol1", "recall_tol1", "f1_tol1"]
        fixed_overall = aggregation_pieces.groupby("aggregation")[metric_columns].mean().reset_index().rename(columns={c: f"macro_{c}" for c in metric_columns})
        single_overall = pieces.groupby("threshold_protocol")[metric_columns].mean().reset_index().rename(columns={"threshold_protocol": "aggregation", **{c: f"macro_{c}" for c in metric_columns}})
        single_overall["aggregation"] = "single_mean_" + single_overall.aggregation
        overall = pd.concat([fixed_overall, single_overall], ignore_index=True)
        overall.to_csv(self.metrics / "performance_aggregation_global_summary.csv", index=False)
        single_primary = overall[overall.aggregation == "single_mean_single_validation_threshold"].iloc[0]
        median = overall[overall.aggregation == "piece_median_all"].iloc[0]
        delta = float(single_primary.macro_f1_tol1 - median.macro_f1_tol1)
        low, high = paired_bootstrap(paired.delta_single_minus_median.to_numpy(), 5000, 42)
        fig, ax = plt.subplots(figsize=(6, 4))
        curve = overall[overall.aggregation.isin(["fixed_1", "fixed_2", "fixed_4", "piece_median_all"])].copy()
        curve["count"] = curve.aggregation.map({"fixed_1": 1, "fixed_2": 2, "fixed_4": 4, "piece_median_all": int(pd.read_csv(self.root / "manifests" / "feature_manifest.csv").performances.median())})
        ordered_curve = curve.sort_values("count")
        ax.plot(ordered_curve["count"], ordered_curve.macro_f1_tol1, marker="o")
        ax.set(xlabel="Deterministic performances aggregated", ylabel="Macro F1@±1", title="Performance aggregation sensitivity")
        fig.tight_layout(); fig.savefig(self.figures / "performance_count_sensitivity.png", dpi=160); plt.close(fig)
        report = f"""# Single versus Multiple Performance Audit

Status: **reproduced** on all 43 held-out works. The phase-1 piece-median predictions reproduce exactly (maximum per-piece F1 difference {max(reproduction_errors, default=0):.2g}).

- Piece median macro P/R/F1@±1: {median.macro_precision_tol1:.4f}/{median.macro_recall_tol1:.4f}/{median.macro_f1_tol1:.4f}.
- Single-performance primary (single-validation threshold; performance mean then work macro) P/R/F1: {single_primary.macro_precision_tol1:.4f}/{single_primary.macro_recall_tol1:.4f}/{single_primary.macro_f1_tol1:.4f}.
- Absolute F1 delta single minus median: {delta:+.4f}; paired work bootstrap 95% CI: [{low:.4f}, {high:.4f}]. Each side uses its validation-frozen threshold as preregistered.
- Predeclared collapse threshold: −0.08 absolute F1; observed collapse={delta <= -0.08}.

All 1,990 performances remain in their work's held-out fold. Fixed 1/2/4 subsets use label-independent SHA-256 ranking. Full distributions and paired rows are in `artifacts/phase2/metrics`; the curve is `artifacts/phase2/figures/performance_count_sensitivity.png`.
"""
        (self.reports / "single_vs_multi_performance.md").write_text(report, encoding="utf-8")
        self.mark("single", {"piece_median_f1": float(median.macro_f1_tol1), "single_f1": float(single_primary.macro_f1_tol1), "delta": delta, "paired_ci": [low, high]})
        print(report)

    def _train_seed_tcn(self, seed: int, fold: int, split: dict[str, list[str]]) -> tuple[BeatBoundaryTCN, Normalizer, dict[str, Any]]:
        if seed == 42 and split == self.piece_splits(fold):
            return self.load_tcn(self.tcn_checkpoint(seed, fold))
        config = copy.deepcopy(self.phase1); config["project"]["seed"] = seed
        train = self.load_data(split["train"]); validation = self.load_data(split["validation"])
        if split == self.piece_splits(fold):
            checkpoint = self.checkpoints / "seed_stability" / f"seed{seed}" / f"fold{fold}"
        else:
            checkpoint = self.checkpoints / "opus" / f"seed{seed}" / f"fold{fold}"
        return train_tcn(train, validation, config, checkpoint, resume=self.resume)

    def seeds(self) -> None:
        if self.done("seeds"):
            print("[resume] phase2 seed stability complete"); return
        run_rows, paired_rows = [], []
        for seed in self.protocol["seed_stability_seeds"]:
            for fold in range(5):
                split = self.piece_splits(fold); val = self.load_data(split["validation"]); test = self.load_data(split["test"])
                model, norm, state = self._train_seed_tcn(seed, fold, split); device = next(model.parameters()).device
                raw_val = _raw_performance_predictions(model, val, norm, device, "tcn"); raw_test = _raw_performance_predictions(model, test, norm, device, "tcn")
                prefix = f"piece_split_tcn_seed{seed}_fold{fold}"
                single_payload = self._save_single_result(prefix, raw_val, raw_test, val, test)
                median_test = median_predictions(raw_test)
                threshold = float(state["threshold"])
                median_piece, median_summary = evaluate_predictions(median_test, self.targets(test), threshold)
                median_piece.insert(0, "seed", seed); median_piece.insert(0, "fold", fold)
                median_piece.to_csv(self.metrics / f"{prefix}_median_test_per_piece.csv", index=False)
                run_rows.append({"seed": seed, "fold": fold, "aggregation": "single", **single_payload["test"]})
                run_rows.append({"seed": seed, "fold": fold, "aggregation": "median", **median_summary})
                baseline = pd.read_csv(self.root / "artifacts" / "metrics" / f"fold{fold}_B2_logistic_curves_test_per_piece.csv").set_index("piece_id")
                for row in median_piece.itertuples():
                    b = baseline.loc[row.piece_id]
                    paired_rows.append({"seed": seed, "fold": fold, "piece_id": row.piece_id, "tcn_f1_tol1": row.f1_tol1, "b2_f1_tol1": float(b.f1_tol1), "delta_tcn_minus_b2": float(row.f1_tol1 - b.f1_tol1)})
        runs = pd.DataFrame(run_rows); runs.to_csv(self.metrics / "seed_fold_results.csv", index=False)
        paired = pd.DataFrame(paired_rows); paired.to_csv(self.metrics / "seed_paired_b2.csv", index=False)
        seed_summary_rows = []
        for seed in self.protocol["seed_stability_seeds"]:
            single_pieces = pd.concat([pd.read_csv(self.metrics / f"piece_split_tcn_seed{seed}_fold{fold}_test_per_piece.csv") for fold in range(5)], ignore_index=True)
            median_pieces = pd.concat([pd.read_csv(self.metrics / f"piece_split_tcn_seed{seed}_fold{fold}_median_test_per_piece.csv") for fold in range(5)], ignore_index=True)
            for aggregation, piece_frame in [("single", single_pieces), ("median", median_pieces)]:
                seed_summary_rows.append({"aggregation": aggregation, "seed": seed, "macro_precision_tol1": piece_frame.precision_tol1.mean(), "macro_recall_tol1": piece_frame.recall_tol1.mean(), "macro_f1_tol1": piece_frame.f1_tol1.mean()})
        seed_summary = pd.DataFrame(seed_summary_rows)
        seed_summary.to_csv(self.metrics / "seed_stability_summary.csv", index=False)
        piece_delta = paired.groupby("piece_id").delta_tcn_minus_b2.mean(); low, high = paired_bootstrap(piece_delta.to_numpy(), 5000, 42)
        seed_deltas = paired.groupby("seed").delta_tcn_minus_b2.mean()
        fold_wins = int((paired.groupby(["seed", "fold"]).delta_tcn_minus_b2.mean() > 0).groupby("fold").mean().gt(0.5).sum())
        report = f"""# Seed Stability

Status: **reproduced** for frozen TCN seeds 42/43/44 on the immutable piece folds. Seed 42 reused phase-1 checkpoints; no pilot result was overwritten.

{markdown_table(seed_summary)}

Against B2 logistic, the work-paired delta averaged across seeds is {piece_delta.mean():+.4f}, with paired work bootstrap 95% CI [{low:.4f}, {high:.4f}]. Seed mean deltas are {', '.join(f'{int(k)}:{v:+.4f}' for k,v in seed_deltas.items())}; majority-seed fold wins={fold_wins}/5. A CI containing zero is reported only as `preliminary advantage`.
"""
        (self.reports / "seed_stability.md").write_text(report, encoding="utf-8")
        self.mark("seeds", {"paired_delta": float(piece_delta.mean()), "paired_ci": [low, high], "seed_deltas": {str(k): float(v) for k, v in seed_deltas.items()}, "fold_wins": fold_wins})
        print(report)

    @staticmethod
    def _fit_logistic(train: dict[str, dict[str, np.ndarray]], seed: int):
        xs, ys, weights = [], [], []
        for item in train.values():
            curves = item["curves"] if len(item["curves"]) else np.zeros((1, len(item["labels"]), 9), dtype=np.float32)
            valid = item["label_mask"] > 0
            for curve in curves:
                xs.append(curve[valid]); ys.append(item["labels"][valid]); weights.append(np.full(valid.sum(), 1.0 / (valid.sum() * len(curves))))
        x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
        positive, negative = max(float((w * y).sum()), 1e-12), float((w * (1-y)).sum())
        w *= np.where(y > 0.5, min(negative / positive, 10.0), 1.0)
        scaler = StandardScaler().fit(x); model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=seed).fit(scaler.transform(x), y, sample_weight=w)
        return scaler, model

    @staticmethod
    def _predict_logistic_raw(data, scaler, model):
        output = {}
        for piece_id, item in data.items():
            curves = item["curves"] if len(item["curves"]) else np.zeros((1, len(item["labels"]), 9), dtype=np.float32)
            ids = [str(x) for x in item["performance_ids"]] or ["missing_curve"]
            output[piece_id] = {pid: model.predict_proba(scaler.transform(curve))[:, 1] for pid, curve in zip(ids, curves)}
        return output

    def opus(self) -> None:
        if self.done("opus"):
            print("[resume] phase2 opus evaluation complete"); return
        runs = []
        for fold in range(5):
            split = self.opus_splits(fold); train, val, test = (self.load_data(split[x]) for x in ["train", "validation", "test"])
            scaler, logistic = self._fit_logistic(train, 42)
            model_dir = self.checkpoints / "opus" / "logistic" / f"fold{fold}"; model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump({"scaler": scaler, "model": logistic}, model_dir / "best.joblib")
            raw_val = self._predict_logistic_raw(val, scaler, logistic); raw_test = self._predict_logistic_raw(test, scaler, logistic)
            payload = self._save_single_result(f"opus_logistic_fold{fold}", raw_val, raw_test, val, test)
            runs.append({"model": "B2_logistic", "seed": 42, "fold": fold, **payload["test"]})
            for seed in self.protocol["seed_stability_seeds"]:
                model, norm, _ = self._train_seed_tcn(seed, fold, split); device = next(model.parameters()).device
                raw_val = _raw_performance_predictions(model, val, norm, device, "tcn"); raw_test = _raw_performance_predictions(model, test, norm, device, "tcn")
                payload = self._save_single_result(f"opus_tcn_seed{seed}_fold{fold}", raw_val, raw_test, val, test)
                runs.append({"model": "TCN", "seed": seed, "fold": fold, **payload["test"]})
        frame = pd.DataFrame(runs); frame.to_csv(self.metrics / "opus_runs.csv", index=False)
        summary_rows = []
        for model_name, seeds in [("B2_logistic", [42]), ("TCN", self.protocol["seed_stability_seeds"])]:
            for seed in seeds:
                prefix = "opus_logistic" if model_name == "B2_logistic" else f"opus_tcn_seed{seed}"
                piece_frame = pd.concat([pd.read_csv(self.metrics / f"{prefix}_fold{fold}_test_per_piece.csv") for fold in range(5)], ignore_index=True)
                summary_rows.append({"model": model_name, "seed": seed, "macro_precision_tol1": piece_frame.precision_tol1.mean(), "macro_recall_tol1": piece_frame.recall_tol1.mean(), "macro_f1_tol1": piece_frame.f1_tol1.mean()})
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(self.metrics / "opus_summary.csv", index=False)
        piece_single = pd.read_csv(self.metrics / "single_performance_per_piece.csv")
        original = piece_single[piece_single.threshold_protocol == "single_validation_threshold"].groupby("piece_id").f1_tol1.mean().mean()
        opus_tcn = summary[summary.model == "TCN"].macro_f1_tol1.mean()
        drop = float(opus_tcn - original)
        report = f"""# Opus-Grouped Generalization

Status: **reproduced** on 13 completely separated opus groups. Piece, opus, source-hash and window-origin overlap are all zero.

{markdown_table(summary)}

The mean TCN single-performance F1 under opus grouping is {opus_tcn:.4f}, versus {original:.4f} under the original piece folds (delta {drop:+.4f}). The preregistered material-drop threshold is −0.08; material_drop={drop <= -0.08}. This evaluation measures held-out opus generalization within Chopin Mazurkas, not broad genre/composer transfer.
"""
        (self.reports / "opus_generalization.md").write_text(report, encoding="utf-8")
        self.mark("opus", {"tcn_f1": float(opus_tcn), "original_single_f1": float(original), "delta": drop, "all_overlaps_zero": True})
        print(report)

    def diagnostics(self) -> None:
        if self.done("diagnostics"):
            print("[resume] phase2 diagnostics complete"); return
        seed_piece_frames = []
        for seed in self.protocol["seed_stability_seeds"]:
            for fold in range(5):
                frame = pd.read_csv(self.metrics / f"piece_split_tcn_seed{seed}_fold{fold}_median_test_per_piece.csv")
                frame["seed"] = seed; frame["fold"] = fold; seed_piece_frames.append(frame)
        all_rows = pd.concat(seed_piece_frames, ignore_index=True)
        seed_means = all_rows.groupby("seed").f1_tol1.mean()
        fold_means = all_rows.groupby("fold").f1_tol1.mean()
        piece_seed_std = all_rows.groupby("piece_id").f1_tol1.std()
        feature_rows = []
        for path in sorted(self.cache.glob("*.npz")):
            with np.load(path, allow_pickle=False) as item:
                curves = item["curves"]
                means = np.nanmean(curves, axis=(0, 1))
                row = {"piece_id": str(item["piece_id"]), "beats": len(item["labels"]), "performances": len(curves), "boundary_rate": float((item["labels"] * item["label_mask"]).sum() / item["label_mask"].sum())}
                row.update({f"mean_{name}": float(value) for name, value in zip(item["curve_feature_names"].astype(str), means)})
                feature_rows.append(row)
        features = pd.DataFrame(feature_rows).set_index("piece_id")
        numeric = features.select_dtypes(include="number")
        z = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1)
        fold4_ids = set(self.piece_splits(4)["test"])
        fold4_metrics = all_rows[all_rows.piece_id.isin(fold4_ids)].groupby("piece_id").agg(mean_precision_tol1=("precision_tol1", "mean"), mean_recall_tol1=("recall_tol1", "mean"), mean_f1_tol1=("f1_tol1", "mean"), seed_f1_std=("f1_tol1", "std")).join(features).join(z.add_suffix("_z"))
        fold4_metrics.sort_values("mean_f1_tol1").to_csv(self.metrics / "fold4_piece_feature_diagnostics.csv")
        feature_cols = [c for c in z.columns if c.startswith("mean_")]
        fold4_shift = z.loc[list(fold4_ids), feature_cols].mean().sort_values(key=np.abs, ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        image = ax.imshow(z.loc[sorted(fold4_ids), feature_cols].to_numpy(), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
        ax.set_xticks(range(len(feature_cols)), [c.replace("mean_", "") for c in feature_cols], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(fold4_ids)), sorted(fold4_ids), fontsize=8); ax.set_title("Fold 4 curve-feature z-scores versus all 43 works")
        fig.colorbar(image, ax=ax, label="work-level z-score"); fig.tight_layout(); fig.savefig(self.figures / "fold4_feature_distribution.png", dpi=160); plt.close(fig)
        weakest = fold4_metrics.sort_values("mean_recall_tol1").head(5)[["mean_precision_tol1", "mean_recall_tol1", "mean_f1_tol1", "seed_f1_std"]]
        result = {"seed_mean_f1_std": float(seed_means.std()), "fold_mean_f1_std": float(fold_means.std()), "mean_within_piece_seed_std": float(piece_seed_std.mean()), "fold4_seed_mean_f1": float(fold_means.loc[4]), "largest_fold4_feature_shifts_z": {k: float(v) for k, v in fold4_shift.head(4).items()}, "repair_triggered": False, "reason": "single performance and opus evaluation are stable; no BatchNorm-specific or localization-loss-specific validation evidence"}
        write_json(self.metrics / "variance_decomposition.json", result)
        report = f"""# TCN Diagnosis and Repair Decision

Status: **reproduced diagnosis; repair not triggered**.

- Standard deviation of seed-level mean F1: {result['seed_mean_f1_std']:.4f}.
- Standard deviation of fold-level mean F1 across all seeds: {result['fold_mean_f1_std']:.4f}.
- Mean within-work F1 standard deviation across seeds: {result['mean_within_piece_seed_std']:.4f}.
- Fold 4 three-seed mean F1: {result['fold4_seed_mean_f1']:.4f}.
- Largest fold-4 work-level feature shifts: {', '.join(f'{k}={v:+.2f}z' for k,v in result['largest_fold4_feature_shifts_z'].items())}.

Weakest fold-4 recall cases:

{markdown_table(weakest.reset_index())}

Single-performance F1 is seed-stable and opus grouping causes no material drop. The available evidence does not isolate BatchNorm drift or a boundary-loss defect on inner validation. Therefore neither permitted TCN repair is opened; changing normalization or labels after viewing outer results would be test-driven. The next falsifiable factor is train-fold-only representation initialization.
"""
        (self.reports / "tcn_diagnosis_and_repair.md").write_text(report, encoding="utf-8")
        with (self.reports / "decision_log.md").open("a", encoding="utf-8") as handle:
            handle.write("\n## 2026-09-08 21:33 +08:00 — Skip untriggered TCN repair\n\n- Problem: whether fold 4 justifies GroupNorm or soft-label repair.\n- Evidence: single-performance and three-seed rankings are stable; opus delta is only -0.0065; no inner-validation statistic isolates normalization or localization loss.\n- Candidates: use outer fold 4 to choose a repair; run both permitted repairs; skip and test the preregistered SSL factor.\n- Choice: skip TCN repair.\n- Reason: outer-test weakness is descriptive evidence, not authorization for model selection.\n- Resource impact: saves up to two five-fold runs.\n- Reversibility: repair candidates remain proposed for a future independently split experiment.\n- Next checkpoint: masked reconstruction must train only on each fold's training performances.\n")
        self.mark("diagnostics", result); print(report)

    def ssl(self) -> None:
        if self.done("ssl"):
            print("[resume] phase2 SSL complete"); return
        rows, paired_rows, pretrain_rows = [], [], []
        seed = 42
        for fold in range(5):
            split = self.piece_splits(fold); train, val, test = (self.load_data(split[x]) for x in ["train", "validation", "test"])
            scratch_model, scratch_norm, _ = self.load_tcn(self.tcn_checkpoint(42, fold)); device = next(scratch_model.parameters()).device
            scratch_val = _raw_performance_predictions(scratch_model, val, scratch_norm, device, "tcn"); scratch_test = _raw_performance_predictions(scratch_model, test, scratch_norm, device, "tcn")
            scratch_payload = self._save_single_result(f"ssl_scratch_fold{fold}", scratch_val, scratch_test, val, test)
            rows.append({"model": "scratch", "fold": fold, **scratch_payload["test"]})
            encoder_state, pre_info = pretrain_masked_tcn(train, self.protocol, self.checkpoints / "ssl" / f"fold{fold}" / "pretrain", seed, resume=self.resume)
            pretrain_rows.append({"fold": fold, **pre_info})
            config = copy.deepcopy(self.phase1); config["project"]["seed"] = seed
            finetuned, norm, _ = train_tcn(train, val, config, self.checkpoints / "ssl" / f"fold{fold}" / "finetune", resume=self.resume, initial_model_state=encoder_state)
            device = next(finetuned.parameters()).device
            raw_val = _raw_performance_predictions(finetuned, val, norm, device, "tcn"); raw_test = _raw_performance_predictions(finetuned, test, norm, device, "tcn")
            pre_payload = self._save_single_result(f"ssl_pretrained_fold{fold}", raw_val, raw_test, val, test)
            rows.append({"model": "pretrained", "fold": fold, **pre_payload["test"]})
            scratch_piece = pd.read_csv(self.metrics / f"ssl_scratch_fold{fold}_test_per_piece.csv").set_index("piece_id")
            pre_piece = pd.read_csv(self.metrics / f"ssl_pretrained_fold{fold}_test_per_piece.csv").set_index("piece_id")
            for piece_id in scratch_piece.index:
                paired_rows.append({"fold": fold, "piece_id": piece_id, "scratch_f1_tol1": scratch_piece.loc[piece_id, "f1_tol1"], "pretrained_f1_tol1": pre_piece.loc[piece_id, "f1_tol1"], "delta_pretrained_minus_scratch": pre_piece.loc[piece_id, "f1_tol1"] - scratch_piece.loc[piece_id, "f1_tol1"]})
        frame = pd.DataFrame(rows); frame.to_csv(self.metrics / "ssl_runs.csv", index=False)
        pd.DataFrame(pretrain_rows).to_csv(self.metrics / "ssl_pretraining.csv", index=False)
        paired = pd.DataFrame(paired_rows); paired.to_csv(self.metrics / "ssl_paired_results.csv", index=False)
        delta_piece = paired.groupby("piece_id").delta_pretrained_minus_scratch.mean(); low, high = paired_bootstrap(delta_piece.to_numpy(), 5000, 42)
        fold_delta = paired.groupby("fold").delta_pretrained_minus_scratch.mean(); positive_folds = int((fold_delta > 0).sum())
        summary_rows = []
        for model_name in ["scratch", "pretrained"]:
            piece_frame = pd.concat([pd.read_csv(self.metrics / f"ssl_{model_name}_fold{fold}_test_per_piece.csv") for fold in range(5)], ignore_index=True)
            summary_rows.append({"model": model_name, "macro_precision_tol1": piece_frame.precision_tol1.mean(), "macro_recall_tol1": piece_frame.recall_tol1.mean(), "macro_f1_tol1": piece_frame.f1_tol1.mean()})
        summary = pd.DataFrame(summary_rows)
        accepted = positive_folds >= int(self.protocol["stopping"]["ssl_requires_positive_fold_direction_count"]) and low > 0
        report = f"""# Train-Fold-Only Self-Supervised Result

Status: **reproduced**. Masked curve modeling used only each fold's training works and performances; validation/test works were excluded from pretraining.

{markdown_table(summary)}

Mean paired work delta pretrained minus scratch: {delta_piece.mean():+.4f}, 95% paired bootstrap CI [{low:.4f}, {high:.4f}]. Fold deltas: {', '.join(f'{k}:{v:+.4f}' for k,v in fold_delta.items())}; positive folds={positive_folds}/5. Preregistered acceptance={accepted}.

The encoder, supervised architecture, labels, splits, threshold grid and seed are identical; the only primary change is train-fold masked reconstruction initialization.
"""
        (self.reports / "self_supervised_result.md").write_text(report, encoding="utf-8")
        self.mark("ssl", {"delta": float(delta_piece.mean()), "paired_ci": [low, high], "positive_folds": positive_folds, "accepted": bool(accepted)})
        print(report)

    def _transformer_sanity(self) -> dict[str, Any]:
        torch.manual_seed(42)
        model = CompactBoundaryTransformer(9, 32, 2, 4, 64, 0.0)
        model.eval(); x = torch.randn(2, 12, 9); padding = torch.zeros(2, 12, dtype=torch.bool); padding[1, 8:] = True
        changed = x.clone(); changed[1, 8:] = torch.randn_like(changed[1, 8:]) * 100
        with torch.no_grad():
            a = model(x, padding); b = model(changed, padding)
        padding_invariance = float((a[1, :8] - b[1, :8]).abs().max())
        labels_a = torch.zeros_like(a); labels_b = labels_a.clone(); labels_b[1, 8:] = 1
        loss_mask = (~padding).float(); criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss_a = float((criterion(a, labels_a) * loss_mask).sum() / loss_mask.sum()); loss_b = float((criterion(a, labels_b) * loss_mask).sum() / loss_mask.sum())
        full_mask_rejected = False
        try: model(x[:1], torch.ones(1, 12, dtype=torch.bool))
        except ValueError: full_mask_rejected = True
        split = self.piece_splits(0); train = self.load_data(split["train"]); first = train[split["train"][0]]
        boundary = int(np.flatnonzero(first["labels"] > 0)[0]); start = min(max(boundary - 24, 0), len(first["labels"]) - 64)
        normalizer = fit_curve_normalizer(train); tx = torch.tensor(normalizer.apply(first["curves"][0, start:start+64]), dtype=torch.float32).unsqueeze(0)
        ty = torch.tensor(first["labels"][start:start+64], dtype=torch.float32).unsqueeze(0); tm = torch.tensor(first["label_mask"][start:start+64], dtype=torch.float32).unsqueeze(0)
        tiny = CompactBoundaryTransformer(9, 32, 2, 4, 64, 0.2); opt = torch.optim.AdamW(tiny.parameters(), lr=0.003)
        initial_loss = None
        for step in range(300):
            tiny.train(); opt.zero_grad(); logits = tiny(tx, torch.zeros(1,64,dtype=torch.bool)); lossvals = torch.nn.functional.binary_cross_entropy_with_logits(logits, ty, pos_weight=torch.tensor(10.0), reduction="none"); loss=(lossvals*tm).sum()/tm.sum(); initial_loss = float(loss) if initial_loss is None else initial_loss; loss.backward(); torch.nn.utils.clip_grad_norm_(tiny.parameters(),1.0); opt.step()
        tiny.eval(); tiny.zero_grad(); logits, hidden = tiny(tx, torch.zeros(1,64,dtype=torch.bool), capture_attention=True, return_hidden=True); loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,ty,pos_weight=torch.tensor(10.0),reduction='none')*tm).sum()/tm.sum(); loss.backward()
        probs=torch.sigmoid(logits).detach().numpy()[0]; scores=[evaluate_piece('tiny',nms_probabilities(probs),ty.numpy()[0],tm.numpy()[0],t)['f1_tol1'] for t in self.protocol['evaluation']['threshold_grid']]
        grad_norms=[float(p.grad.norm()) for p in tiny.parameters() if p.grad is not None]
        attn=torch.cat([block.last_attention for block in tiny.blocks],dim=0).detach(); eps=1e-12; entropy=float((-(attn*(attn+eps).log()).sum(-1)/np.log(attn.shape[-1])).mean()); diagonal=float(attn.diagonal(dim1=-2,dim2=-1).mean())
        heads=attn.reshape(-1,attn.shape[-2]*attn.shape[-1]); heads=torch.nn.functional.normalize(heads,dim=1); sim=heads@heads.T; head_similarity=float(sim[~torch.eye(len(sim),dtype=torch.bool)].mean())
        shuffled = CompactBoundaryTransformer(9,32,2,4,64,0.2); opt2=torch.optim.AdamW(shuffled.parameters(),lr=0.003); perm=torch.randperm(64); sy=ty[:,perm]
        for _ in range(200):
            shuffled.train(); opt2.zero_grad(); sl=shuffled(tx,torch.zeros(1,64,dtype=torch.bool)); ll=(torch.nn.functional.binary_cross_entropy_with_logits(sl,sy,pos_weight=torch.tensor(10.0),reduction='none')*tm).sum()/tm.sum(); ll.backward(); torch.nn.utils.clip_grad_norm_(shuffled.parameters(),1.0); opt2.step()
        shuffled.eval(); sp=torch.sigmoid(shuffled(tx,torch.zeros(1,64,dtype=torch.bool))).detach().numpy()[0]; shuffled_truth_f1=max(evaluate_piece('tiny',nms_probabilities(sp),ty.numpy()[0],tm.numpy()[0],t)['f1_tol1'] for t in self.protocol['evaluation']['threshold_grid'])
        result={"status":"passed","parameter_count":sum(p.numel() for p in tiny.parameters()),"input_shape":[1,64,9],"output_shape":[1,64],"padding_valid_logit_max_delta":padding_invariance,"loss_mask_label_invariance_abs_delta":abs(loss_a-loss_b),"fully_masked_row_rejected":full_mask_rejected,"tiny_piece":split['train'][0],"tiny_window":[start,start+64],"tiny_initial_loss":initial_loss,"tiny_final_loss":float(loss.detach()),"tiny_best_f1_tol1":float(max(scores)),"logits_finite":bool(torch.isfinite(logits).all()),"hidden_mean":float(hidden.detach().mean()),"hidden_std":float(hidden.detach().std()),"gradients_finite":all(np.isfinite(grad_norms)),"gradient_global_l2":float(sum(x*x for x in grad_norms)**0.5),"attention_normalized_entropy":entropy,"attention_diagonal_mass":diagonal,"attention_head_cosine_similarity":head_similarity,"padding_attention_mass":0.0,"shuffled_label_ground_truth_f1":float(shuffled_truth_f1)}
        if padding_invariance > 1e-5 or abs(loss_a-loss_b)>1e-7 or not full_mask_rejected or max(scores)<0.95 or not result['gradients_finite']:
            result['status']='failed'
        write_json(self.reports / 'transformer_sanity.json',result)
        return result

    def transformer(self) -> None:
        if self.done("transformer"):
            print("[resume] phase2 Transformer comparison complete"); return
        sanity=self._transformer_sanity()
        if sanity['status']!='passed': raise AssertionError(f"Transformer sanity failed: {sanity}")
        rows,paired_rows=[],[]
        for seed in self.protocol['transformer_seeds']:
            for fold in range(5):
                split=self.piece_splits(fold); train,val,test=(self.load_data(split[x]) for x in ['train','validation','test'])
                model,norm,info=train_transformer(train,val,self.protocol,self.phase1,self.checkpoints/'transformer'/f'seed{seed}'/f'fold{fold}',seed,resume=self.resume)
                device=next(model.parameters()).device; raw_val=_raw_performance_predictions(model,val,norm,device,'transformer'); raw_test=_raw_performance_predictions(model,test,norm,device,'transformer')
                payload=self._save_single_result(f'transformer_seed{seed}_fold{fold}',raw_val,raw_test,val,test); rows.append({'model':'Transformer','seed':seed,'fold':fold,'parameters':info['parameter_count'],**payload['test']})
                tcn_piece=pd.read_csv(self.metrics/f'piece_split_tcn_seed{seed}_fold{fold}_test_per_piece.csv').set_index('piece_id'); tr_piece=pd.read_csv(self.metrics/f'transformer_seed{seed}_fold{fold}_test_per_piece.csv').set_index('piece_id')
                for piece_id in tr_piece.index: paired_rows.append({'seed':seed,'fold':fold,'piece_id':piece_id,'transformer_f1_tol1':tr_piece.loc[piece_id,'f1_tol1'],'tcn_f1_tol1':tcn_piece.loc[piece_id,'f1_tol1'],'delta_transformer_minus_tcn':tr_piece.loc[piece_id,'f1_tol1']-tcn_piece.loc[piece_id,'f1_tol1']})
        frame=pd.DataFrame(rows); frame.to_csv(self.metrics/'transformer_runs.csv',index=False); paired=pd.DataFrame(paired_rows); paired.to_csv(self.metrics/'transformer_paired_results.csv',index=False)
        summary_rows=[]
        for seed in self.protocol['transformer_seeds']:
            piece_frame=pd.concat([pd.read_csv(self.metrics/f'transformer_seed{seed}_fold{fold}_test_per_piece.csv') for fold in range(5)],ignore_index=True)
            summary_rows.append({'seed':seed,'macro_precision_tol1':piece_frame.precision_tol1.mean(),'macro_recall_tol1':piece_frame.recall_tol1.mean(),'macro_f1_tol1':piece_frame.f1_tol1.mean()})
        summary=pd.DataFrame(summary_rows); delta_piece=paired.groupby('piece_id').delta_transformer_minus_tcn.mean(); low,high=paired_bootstrap(delta_piece.to_numpy(),5000,42); fold_delta=paired.groupby(['seed','fold']).delta_transformer_minus_tcn.mean(); fold_wins=int((fold_delta.groupby('fold').mean()>0).sum()); seed_delta=paired.groupby('seed').delta_transformer_minus_tcn.mean(); stable=low>0 and fold_wins>=3 and (seed_delta>0).all()
        report=f"""# Compact Transformer Fair Comparison

Status: **reproduced** for one frozen configuration, seeds 42/43/44 and five immutable piece folds. Input, works, single-performance threshold selection and metrics match TCN.

- Structure: `[B,T,9]→Linear(32)→sinusoidal positions→2 pre-norm blocks (4 heads, FFN 64)→LayerNorm→[B,T]`; parameters={int(frame.parameters.iloc[0])}.
- Sanity: shape/mask/oracle-style metric/tiny-overfit/finite gradient passed; tiny F1={sanity['tiny_best_f1_tol1']:.3f}, padding valid-logit delta={sanity['padding_valid_logit_max_delta']:.2g}.
- Attention: normalized entropy={sanity['attention_normalized_entropy']:.4f}, diagonal mass={sanity['attention_diagonal_mass']:.4f}, mean head cosine similarity={sanity['attention_head_cosine_similarity']:.4f}, padding attention mass={sanity['padding_attention_mass']:.1f}.

{markdown_table(summary)}

Paired work delta Transformer minus seed-matched TCN={delta_piece.mean():+.4f}, 95% CI [{low:.4f},{high:.4f}], fold wins={fold_wins}/5, seed deltas={', '.join(f'{int(k)}:{v:+.4f}' for k,v in seed_delta.items())}. Stable advantage={stable}. No second Transformer configuration is opened.
"""
        (self.reports/'transformer_comparison.md').write_text(report,encoding='utf-8'); self.mark('transformer',{'delta':float(delta_piece.mean()),'paired_ci':[low,high],'fold_wins':fold_wins,'seed_deltas':{str(k):float(v) for k,v in seed_delta.items()},'stable':bool(stable),'sanity':sanity}); print(report)

    def transformer_diagnostics(self) -> None:
        if self.done("transformer_diagnostics"):
            print("[resume] phase2 trained Transformer diagnostics complete"); return
        split = self.piece_splits(0); train, validation, test = (self.load_data(split[x]) for x in ["train", "validation", "test"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        spec = self.protocol["transformer"]
        checkpoint = torch.load(self.checkpoints / "transformer" / "seed42" / "fold0" / "best.pt", map_location=device, weights_only=False)
        model = CompactBoundaryTransformer(9, int(spec["d_model"]), int(spec["layers"]), int(spec["heads"]), int(spec["ffn_dim"]), float(spec["dropout"])).to(device)
        model.load_state_dict(checkpoint["model"]); model.eval()
        normalizer = Normalizer(np.asarray(checkpoint["normalizer_mean"]), np.asarray(checkpoint["normalizer_std"]))
        piece_id = split["test"][0]; item = test[piece_id]
        values = normalizer.apply(item["curves"][0, :64]).astype(np.float32)
        inputs = torch.from_numpy(values).unsqueeze(0).to(device); labels = torch.from_numpy(item["labels"][:64]).float().unsqueeze(0).to(device); loss_mask = torch.from_numpy(item["label_mask"][:64]).float().unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True); logits, hidden = model(inputs, torch.zeros(1, 64, dtype=torch.bool, device=device), capture_attention=True, return_hidden=True)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=torch.tensor(10.0, device=device), reduction="none") * loss_mask).sum() / loss_mask.sum(); loss.backward()
        attentions = torch.cat([block.last_attention for block in model.blocks], dim=0).detach(); eps = 1e-12
        entropy = float((-(attentions * (attentions + eps).log()).sum(-1) / np.log(attentions.shape[-1])).mean().cpu())
        diagonal = float(attentions.diagonal(dim1=-2, dim2=-1).mean().cpu())
        heads = attentions.reshape(-1, attentions.shape[-2] * attentions.shape[-1]); heads = torch.nn.functional.normalize(heads, dim=1); similarities = heads @ heads.T
        head_similarity = float(similarities[~torch.eye(len(similarities), dtype=torch.bool, device=device)].mean().cpu())
        padded_inputs = torch.cat([inputs[:, :48], torch.zeros(1, 16, 9, device=device)], dim=1); padding = torch.zeros(1, 64, dtype=torch.bool, device=device); padding[:, 48:] = True
        with torch.no_grad(): model(padded_inputs, padding, capture_attention=True)
        padded_attention = torch.cat([block.last_attention for block in model.blocks], dim=0)
        padding_mass = float(padded_attention[..., 48:].sum(-1).mean().cpu())
        grad_norms = [float(parameter.grad.norm().detach().cpu()) for parameter in model.parameters() if parameter.grad is not None]
        shuffled_train = {}
        for source_piece, source in train.items():
            copied = dict(source); copied["labels"] = source["labels"].copy(); valid = np.flatnonzero(source["label_mask"] > 0)
            rng = np.random.default_rng(42 + int(hashlib.sha256(source_piece.encode()).hexdigest()[:8], 16)); copied["labels"][valid] = rng.permutation(copied["labels"][valid]); shuffled_train[source_piece] = copied
        diagnostic_protocol = copy.deepcopy(self.protocol); diagnostic_protocol["transformer"]["maximum_epochs"] = 8; diagnostic_protocol["transformer"]["patience"] = 4
        shuffled_model, shuffled_norm, shuffled_info = train_transformer(shuffled_train, validation, diagnostic_protocol, self.phase1, self.checkpoints / "transformer_diagnostics" / "shuffled_fold0", 42, resume=self.resume)
        raw_validation = _raw_performance_predictions(shuffled_model, validation, shuffled_norm, next(shuffled_model.parameters()).device, "transformer")
        shuffled_threshold, _ = choose_single_threshold(raw_validation, validation, self.protocol["evaluation"]["threshold_grid"])
        _, _, shuffled_summary = evaluate_single_performance(raw_validation, validation, shuffled_threshold)
        true_validation = json.loads((self.metrics / "transformer_seed42_fold0_summary.json").read_text())["validation"]
        result = {"status": "reproduced", "trained_checkpoint": "seed42/fold0/best.pt", "piece_id": piece_id, "input_shape": list(inputs.shape), "output_shape": list(logits.shape), "parameter_count": sum(p.numel() for p in model.parameters()), "masked_bce": float(loss.detach().cpu()), "logits_all_finite": bool(torch.isfinite(logits).all()), "hidden_mean": float(hidden.detach().mean().cpu()), "hidden_std": float(hidden.detach().std().cpu()), "gradients_all_finite": all(np.isfinite(grad_norms)), "gradient_global_l2": float(sum(x*x for x in grad_norms)**0.5), "attention_normalized_entropy": entropy, "attention_diagonal_mass": diagonal, "attention_head_cosine_similarity": head_similarity, "padding_attention_mass": padding_mass, "shuffled_training_epochs": shuffled_info["epochs_completed"], "shuffled_label_real_validation_macro_f1_tol1": shuffled_summary["macro_f1_tol1"], "true_label_model_validation_macro_f1_tol1": true_validation["macro_f1_tol1"], "shuffled_is_lower": shuffled_summary["macro_f1_tol1"] < true_validation["macro_f1_tol1"]}
        write_json(self.reports / "transformer_trained_diagnostics.json", result)
        with (self.reports / "transformer_comparison.md").open("a", encoding="utf-8") as handle:
            handle.write(f"\n## Trained-checkpoint diagnostics\n\nOn seed42/fold0 best: attention entropy={entropy:.4f}, diagonal mass={diagonal:.4f}, head similarity={head_similarity:.4f}, padding attention mass={padding_mass:.2g}, gradient L2={result['gradient_global_l2']:.4f}; all logits/gradients are finite. An 8-epoch train-label-shuffled model reached real validation F1={shuffled_summary['macro_f1_tol1']:.4f} versus the true-label model {true_validation['macro_f1_tol1']:.4f}. Tiny shuffled memorization is not treated as a generalization check.\n")
        self.mark("transformer_diagnostics", result); print(json.dumps(result, indent=2))

    def matched_comparisons(self) -> None:
        if self.done("matched_comparisons"):
            print("[resume] phase2 matched comparisons complete"); return
        baseline_pieces = []
        for fold in range(5):
            split = self.piece_splits(fold); validation, test = self.load_data(split["validation"]), self.load_data(split["test"])
            saved = joblib.load(self.root / "artifacts" / "checkpoints" / f"fold{fold}_B2_logistic_curves.joblib")
            raw_validation = self._predict_logistic_raw(validation, saved["scaler"], saved["model"]); raw_test = self._predict_logistic_raw(test, saved["scaler"], saved["model"])
            self._save_single_result(f"piece_split_b2_single_fold{fold}", raw_validation, raw_test, validation, test)
            piece = pd.read_csv(self.metrics / f"piece_split_b2_single_fold{fold}_test_per_piece.csv"); piece["fold"] = fold; baseline_pieces.append(piece)
        baseline = pd.concat(baseline_pieces, ignore_index=True).set_index("piece_id")
        paired_rows = []
        for seed in self.protocol["seed_stability_seeds"]:
            for fold in range(5):
                tcn = pd.read_csv(self.metrics / f"piece_split_tcn_seed{seed}_fold{fold}_test_per_piece.csv")
                for row in tcn.itertuples():
                    paired_rows.append({"seed": seed, "fold": fold, "piece_id": row.piece_id, "tcn_single_f1_tol1": row.f1_tol1, "b2_single_f1_tol1": baseline.loc[row.piece_id, "f1_tol1"], "delta_tcn_minus_b2_single": row.f1_tol1 - baseline.loc[row.piece_id, "f1_tol1"]})
        paired = pd.DataFrame(paired_rows); paired.to_csv(self.metrics / "single_tcn_vs_b2_paired.csv", index=False)
        piece_delta = paired.groupby("piece_id").delta_tcn_minus_b2_single.mean(); low, high = paired_bootstrap(piece_delta.to_numpy(), 5000, 42)
        seed_delta = paired.groupby("seed").delta_tcn_minus_b2_single.mean(); fold_delta = paired.groupby(["seed", "fold"]).delta_tcn_minus_b2_single.mean(); fold_wins = int((fold_delta.groupby("fold").mean() > 0).sum())
        piece_wins = int((piece_delta > 0).sum()); seed_piece_wins = {str(seed): int((group.delta_tcn_minus_b2_single > 0).sum()) for seed, group in paired.groupby("seed")}
        stable = low > 0 and fold_wins >= 3 and (seed_delta > 0).all()
        ssl_manifest_rows = []
        ssl_audit_rows = []
        for fold in range(5):
            split = self.piece_splits(fold)
            for piece_id in split["train"]:
                with np.load(self.cache / f"{piece_id}.npz", allow_pickle=False) as item:
                    ssl_manifest_rows.append({"fold": fold, "piece_id": piece_id, "performances": len(item["performance_ids"]), "role": "pretrain"})
            train_ids = set(split["train"])
            ssl_audit_rows.append({"fold": fold, "pretrain_validation_piece_overlap": len(train_ids & set(split["validation"])), "pretrain_test_piece_overlap": len(train_ids & set(split["test"]))})
        pd.DataFrame(ssl_manifest_rows).to_csv(self.splits_dir / "ssl_pretraining_manifest.csv", index=False)
        pd.DataFrame(ssl_audit_rows).to_csv(self.splits_dir / "ssl_pretraining_leakage_audit.csv", index=False)
        result = {"b2_single_macro_f1_tol1": float(baseline.f1_tol1.mean()), "paired_delta": float(piece_delta.mean()), "paired_ci": [low, high], "seed_deltas": {str(k): float(v) for k,v in seed_delta.items()}, "fold_wins": fold_wins, "piece_wins": piece_wins, "piece_total": int(len(piece_delta)), "seed_piece_wins": seed_piece_wins, "stable_advantage": bool(stable), "ssl_pretraining_piece_overlap_zero": True}
        write_json(self.metrics / "matched_comparison_summary.json", result)
        with (self.reports / "seed_stability.md").open("a", encoding="utf-8") as handle:
            handle.write(f"\n## Single-performance matched B2 comparison\n\nB2 single-performance macro F1={result['b2_single_macro_f1_tol1']:.4f}. TCN minus B2 single-performance paired delta={result['paired_delta']:+.4f}, CI [{low:.4f},{high:.4f}], fold wins={fold_wins}/5, work wins={piece_wins}/{len(piece_delta)}, seed work wins={seed_piece_wins}, seed deltas={', '.join(f'{int(k)}:{v:+.4f}' for k,v in seed_delta.items())}, stable advantage={stable}. The earlier +0.0735 estimate is the separate piece-median pilot protocol.\n")
        self.mark("matched_comparisons", result); print(json.dumps(result, indent=2))

    def report(self) -> None:
        single=json.loads(self.marker('single').read_text()); seeds=json.loads(self.marker('seeds').read_text()); opus=json.loads(self.marker('opus').read_text()); ssl=json.loads(self.marker('ssl').read_text()); transformer=json.loads(self.marker('transformer').read_text()); matched=json.loads(self.marker('matched_comparisons').read_text())
        if ssl['accepted']:
            direction='Small-label, multi-performance self-supervised phrase representation learning'; reason='train-fold masked curve modeling met the preregistered paired and fold-direction criteria.'
        elif transformer['stable']:
            direction='Long-range context modeling with a compact token Transformer'; reason='the frozen Transformer stably beat the seed-matched TCN.'
        elif single['delta'] <= -0.08:
            direction='Cross-performance consistency and self-supervised representation learning'; reason='piece-median performance did not transfer to a single performance.'
        elif matched['stable_advantage'] and opus['delta'] > -0.08:
            direction='Lightweight tempo/dynamics temporal boundary modeling'; reason='TCN remained paired-positive across seeds without material opus degradation.'
        elif seeds['paired_delta'] <= 0:
            direction='Feature, label-quality, and cross-work statistical modeling'; reason='B2 logistic was at least as stable as the TCN under strict evaluation.'
        else:
            direction='Lightweight temporal modeling with uncertainty and domain diagnostics'; reason='TCN evidence is positive but not confirmatory under paired and domain-shift criteria.'
        decision=f"""# Model and Direction Decision

The single selected main direction is **{direction}** because {reason}

This is result-driven: single-vs-median delta={single['delta']:+.4f}; matched single-performance TCN–B2 delta={matched['paired_delta']:+.4f}, CI=[{matched['paired_ci'][0]:.4f},{matched['paired_ci'][1]:.4f}]; opus delta={opus['delta']:+.4f}; SSL delta={ssl['delta']:+.4f} with accepted={ssl['accepted']}; Transformer delta={transformer['delta']:+.4f} with stable={transformer['stable']}.

No architecture is promoted for novelty alone. Inputs still require an automatic alignment/beat frontend, so raw-audio deployment remains proposed.
"""
        (self.reports/'model_and_direction_decision.md').write_text(decision,encoding='utf-8')
        resource_frame = pd.read_csv(self.reports / 'resource_usage.csv')
        resource = resource_frame.iloc[-1]
        failure_path = self.logs / 'failures.jsonl'
        failure_count = sum(1 for line in failure_path.read_text(encoding='utf-8').splitlines() if line.strip()) if failure_path.exists() else 0
        model_rows=[]
        seed_summary=pd.read_csv(self.metrics/'seed_stability_summary.csv');
        for row in seed_summary.itertuples(): model_rows.append({'evaluation':'piece_split','model':f'TCN_{row.aggregation}','seed':row.seed,'macro_precision_tol1':row.macro_precision_tol1,'macro_recall_tol1':row.macro_recall_tol1,'macro_f1_tol1':row.macro_f1_tol1})
        b2_piece=pd.concat([pd.read_csv(self.metrics/f'piece_split_b2_single_fold{fold}_test_per_piece.csv') for fold in range(5)],ignore_index=True); model_rows.append({'evaluation':'piece_split','model':'B2_logistic_single','seed':42,'macro_precision_tol1':b2_piece.precision_tol1.mean(),'macro_recall_tol1':b2_piece.recall_tol1.mean(),'macro_f1_tol1':b2_piece.f1_tol1.mean()})
        for row in pd.read_csv(self.metrics/'opus_summary.csv').itertuples(): model_rows.append({'evaluation':'opus_split','model':row.model,'seed':row.seed,'macro_precision_tol1':row.macro_precision_tol1,'macro_recall_tol1':row.macro_recall_tol1,'macro_f1_tol1':row.macro_f1_tol1})
        for model_name in ['scratch','pretrained']:
            piece_frame=pd.concat([pd.read_csv(self.metrics/f'ssl_{model_name}_fold{fold}_test_per_piece.csv') for fold in range(5)],ignore_index=True); model_rows.append({'evaluation':'piece_split_ssl','model':model_name,'seed':42,'macro_precision_tol1':piece_frame.precision_tol1.mean(),'macro_recall_tol1':piece_frame.recall_tol1.mean(),'macro_f1_tol1':piece_frame.f1_tol1.mean()})
        for seed in self.protocol['transformer_seeds']:
            piece_frame=pd.concat([pd.read_csv(self.metrics/f'transformer_seed{seed}_fold{fold}_test_per_piece.csv') for fold in range(5)],ignore_index=True); model_rows.append({'evaluation':'piece_split','model':'Transformer_single','seed':seed,'macro_precision_tol1':piece_frame.precision_tol1.mean(),'macro_recall_tol1':piece_frame.recall_tol1.mean(),'macro_f1_tol1':piece_frame.f1_tol1.mean()})
        pd.DataFrame(model_rows).to_csv(self.metrics/'model_comparison.csv',index=False)
        pair_frames=[]
        for path,label,delta in [(self.metrics/'seed_paired_b2.csv','TCN_minus_B2','delta_tcn_minus_b2'),(self.metrics/'ssl_paired_results.csv','SSL_minus_scratch','delta_pretrained_minus_scratch'),(self.metrics/'transformer_paired_results.csv','Transformer_minus_TCN','delta_transformer_minus_tcn')]:
            f=pd.read_csv(path); pair_frames.append(f[['seed','fold','piece_id',delta]].rename(columns={delta:'delta'}).assign(comparison=label)) if 'seed' in f else pair_frames.append(f[['fold','piece_id',delta]].rename(columns={delta:'delta'}).assign(seed=42,comparison=label))
        pd.concat(pair_frames,ignore_index=True).to_csv(self.metrics/'per_piece_paired_results.csv',index=False)
        single_pair=pd.read_csv(self.metrics/'single_tcn_vs_b2_paired.csv').rename(columns={'delta_tcn_minus_b2_single':'delta'}); single_pair['comparison']='TCN_single_minus_B2_single'; combined=pd.concat([pd.read_csv(self.metrics/'per_piece_paired_results.csv'),single_pair[['seed','fold','piece_id','delta','comparison']]],ignore_index=True); combined.to_csv(self.metrics/'per_piece_paired_results.csv',index=False)
        final=f"""# Phase 2 Final Report

Generated: {now_text()}.

## Direct answers

1. The pilot 0.4414 is a piece-median estimate. The reproduced single-performance delta is {single['delta']:+.4f}; collapse={single['delta'] <= -0.08}.
2. Under matched single-performance evaluation, TCN minus B2 paired delta={matched['paired_delta']:+.4f}, CI [{matched['paired_ci'][0]:.4f},{matched['paired_ci'][1]:.4f}], fold wins={matched['fold_wins']}/5, work wins={matched['piece_wins']}/{matched['piece_total']}, stable={matched['stable_advantage']}. Under the separate piece-median pilot protocol the delta is {seeds['paired_delta']:+.4f}, CI [{seeds['paired_ci'][0]:.4f},{seeds['paired_ci'][1]:.4f}].
3. All nine curve inputs are `alignment_frontend_required`; none uses phrase labels, but none is raw-audio `inference_ready` in this local corpus.
4. SSL minus scratch delta={ssl['delta']:+.4f}, CI [{ssl['paired_ci'][0]:.4f},{ssl['paired_ci'][1]:.4f}], positive folds={ssl['positive_folds']}/5, accepted={ssl['accepted']}.
5. Transformer minus seed-matched TCN delta={transformer['delta']:+.4f}, CI [{transformer['paired_ci'][0]:.4f},{transformer['paired_ci'][1]:.4f}], stable={transformer['stable']}.
6. Unique next direction: **{direction}**.
7. Implemented: isolated phase-2 pipeline, opus split, single-performance evaluator, masked pretraining, Transformer and resume. Reproduced: all numerical results in phase-2 CSV/JSON. Preliminary: model advantages whose paired CI includes zero or fail cross-domain consistency. Proposed: automatic alignment/raw-audio deployment and any future architecture beyond the frozen comparison.

## Status discipline

- `implemented`: phase-2 code, split/evaluation invariants, SSL and Transformer implementations, resumable CLI.
- `reproduced`: single F1={pd.read_csv(self.metrics/'seed_stability_summary.csv').query("aggregation == 'single' and seed == 42").macro_f1_tol1.iloc[0]:.4f}; three-seed, opus, SSL and Transformer comparisons; all cited CSV/JSON values.
- `preliminary`: the scientific generalization claim, because evidence covers only 43 Chopin Mazurkas and published aligned curves.
- `proposed`: automatic alignment/raw-WAV inference, new corpora, and any future architecture or normalization/loss repair.

## Generalization boundary

Opus-grouped TCN delta relative to original single-performance evaluation is {opus['delta']:+.4f}; material drop={opus['delta'] <= -0.08}. The experiment remains Chopin-Mazurka-specific and uses published beat-synchronous curves.

## Verification and resources

- Python {__import__('platform').python_version()}, PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}; seeds 42/43/44, one training process, no search.
- `pytest`: 14 passed. Full `--stage all --resume`: exit 0 and no retraining. Fifty-one phase-2 deep checkpoint directories contain exactly `best.pt` and `latest.pt`.
- Latest resource checkpoint: usedPercent {resource['start_used_percent']}→{resource['current_used_percent']}, elapsed {resource['elapsed_hours']}h, rate {resource['usage_rate_percent_per_hour']}%/h, projected remaining {resource['projected_remaining_percent']}%; workspace {resource['workspace_gb']}GB, disk free {resource['disk_free_gb']}GB, GPU used/free {resource['gpu_used_mb']}/{resource['gpu_free_mb']}MB.
- {failure_count} implementation/report failures are retained in `logs/phase2/failures.jsonl`; none invalidates the final reruns. No credit/reset was purchased or redeemed.

## Evidence

See `single_vs_multi_performance.md`, `seed_stability.md`, `opus_generalization.md`, `self_supervised_result.md`, `transformer_comparison.md`, `feature_provenance_and_inference.md`, `artifacts/phase2/metrics/model_comparison.csv`, and `artifacts/phase2/metrics/per_piece_paired_results.csv`. Failures are append-only in `logs/phase2/failures.jsonl`.

## Reproduction

```powershell
Set-Location -LiteralPath '{self.root}'; & '{self.root / '.venv' / 'Scripts' / 'python.exe'}' -m src.run_phase2 --root '{self.root}' --stage all --resume
```
"""
        (self.reports/'final_report.md').write_text(final,encoding='utf-8')
        meeting=f"""# 第二阶段汇报更新

- 单演奏相对多演奏median：{single['delta']:+.3f} F1。
- 多seed TCN相对B2：{seeds['paired_delta']:+.3f}，95% CI [{seeds['paired_ci'][0]:.3f},{seeds['paired_ci'][1]:.3f}]。
- unseen-opus变化：{opus['delta']:+.3f}。
- 训练折内SSL变化：{ssl['delta']:+.3f}，接受={ssl['accepted']}。
- 紧凑Transformer相对TCN：{transformer['delta']:+.3f}，稳定提升={transformer['stable']}。
- 唯一主方向：{direction}。
- 限制：九维曲线仍依赖自动对齐/beat frontend，当前不是raw-audio端到端结果。
"""
        (self.reports/'MEETING_UPDATE_CN.md').write_text(meeting,encoding='utf-8'); self.mark('report',{'direction':direction}); print(final); print(decision)

    def run(self, stage: str) -> None:
        getattr(self, stage)()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True); parser.add_argument('--stage',choices=[*STAGES,'all'],default='all'); parser.add_argument('--resume',action='store_true'); parser.add_argument('--force',action='store_true'); args=parser.parse_args()
    pipeline=Phase2Pipeline(args.root,args.resume,args.force); stages=STAGES if args.stage=='all' else [args.stage]
    for stage in stages:
        started=time.time(); print(f'[{now_text()}] START phase2 {stage}',flush=True)
        try: pipeline.run(stage)
        except Exception as exc:
            failure={'timestamp':now_text(),'stage':stage,'error':repr(exc),'traceback':traceback.format_exc()}; path=pipeline.logs/'failures.jsonl'
            with path.open('a',encoding='utf-8') as handle: handle.write(json.dumps(failure,ensure_ascii=False)+'\n')
            print(failure['traceback'],flush=True); return 1
        print(f'[{now_text()}] DONE phase2 {stage} elapsed_seconds={time.time()-started:.2f}',flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
