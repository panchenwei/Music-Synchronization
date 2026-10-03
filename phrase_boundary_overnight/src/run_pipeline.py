from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import (
    DcmlPiece,
    build_piece_cache,
    discover_dcml_pieces,
    discover_mazurka_files,
    load_config,
    load_piece_cache,
    make_piece_splits,
    map_piece_boundaries,
    read_csv_clean,
    write_json,
)
from .evaluation import bootstrap_macro_ci, choose_threshold, evaluate_predictions, non_maximum_suppression
from .models import Normalizer, predict_tcn, train_tcn


STAGES = ["audit", "mapping", "split", "features", "baselines", "tcn", "debug", "audio-audit", "report"]


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Pipeline:
    def __init__(self, config_path: Path, resume: bool = False, force: bool = False, smoke_test: bool = False, fold: int | None = None, validation_only: bool = False):
        self.config_path = config_path.resolve()
        self.config = load_config(self.config_path)
        self.root = Path(self.config["project"]["workspace"]).resolve()
        self.resume = resume
        self.force = force
        self.smoke_test = smoke_test
        self.fold = fold
        self.validation_only = validation_only
        self.manifests = self.root / "manifests"
        self.splits = self.root / "splits"
        self.cache = self.root / "cache"
        self.artifacts = self.root / "artifacts"
        self.reports = self.root / "reports"
        self.logs = self.root / "logs"
        self.markers = self.artifacts / "stages"
        for path in [self.manifests, self.splits, self.cache / "piece_features", self.artifacts / "metrics", self.artifacts / "predictions", self.artifacts / "checkpoints", self.artifacts / "figures", self.reports, self.logs, self.markers]:
            path.mkdir(parents=True, exist_ok=True)
        self.dcml_root = Path(self.config["data"]["dcml_chopin_root"])
        self.mazurka_root = Path(self.config["data"]["mazurkabl_root"])
        self.asap_root = Path(self.config["data"]["asap_root"])
        self.maestro_root = Path(self.config["data"]["maestro_root"])

    def marker_path(self, stage: str, suffix: str = "") -> Path:
        return self.markers / f"{stage}{suffix}.json"

    def mark(self, stage: str, payload: dict[str, Any], suffix: str = "") -> None:
        write_json(self.marker_path(stage, suffix), {"stage": stage, "completed_at": now_text(), "config": str(self.config_path), **payload})

    def audit(self) -> None:
        marker = self.marker_path("audit")
        if self.resume and marker.exists() and not self.force:
            print("[resume] audit already complete")
            return
        dcml = discover_dcml_pieces(self.dcml_root)
        mazurka = discover_mazurka_files(self.mazurka_root)
        metadata = pd.read_csv(self.dcml_root / "metadata.tsv", sep="\t")
        phrase_rows = 0
        phrase_values: dict[str, int] = {}
        for path in sorted((self.dcml_root / "harmonies").glob("*.tsv")):
            frame = pd.read_csv(path, sep="\t", usecols=["phraseend"])
            values = frame.phraseend.fillna("").astype(str)
            phrase_rows += int(values.str.contains("}", regex=False).sum())
            for value, count in values[values != ""].value_counts().items():
                phrase_values[str(value)] = phrase_values.get(str(value), 0) + int(count)
        meta_path = Path(r"C:\Users\pa1018\Desktop\learn\柴柴\TEST1\data\all\meta.csv")
        processed = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
        asap_meta = pd.read_csv(self.asap_root / "metadata.csv")
        maestro_wav = list(self.maestro_root.rglob("*.wav"))
        maestro_midi = list(self.maestro_root.rglob("*.mid")) + list(self.maestro_root.rglob("*.midi"))
        alignment_path = Path(self.config["data"]["current_alignment_csv"])
        inventory = {
            "timestamp": now_text(),
            "dcml_metadata_rows": int(len(metadata)),
            "dcml_pieces_with_required_files": len(dcml),
            "dcml_harmony_files": len(list((self.dcml_root / "harmonies").glob("*.tsv"))),
            "dcml_notes_files": len(list((self.dcml_root / "notes").glob("*.tsv"))),
            "dcml_phrase_rows_folded": phrase_rows,
            "dcml_phraseend_value_counts": phrase_values,
            "mazurkabl_beat_time_files": len(mazurka),
            "mazurkabl_beat_dyn_files": len(list((self.mazurka_root / "beat_dyn").glob("*.csv"))),
            "mazurkabl_xml_files": len(list((self.mazurka_root / "xml_scores").glob("*.xml"))),
            "dcml_mazurka_piece_overlap": len(set(dcml) & set(mazurka)),
            "dcml_only_pieces": sorted(set(dcml) - set(mazurka)),
            "mazurka_only_pieces": sorted(set(mazurka) - set(dcml)),
            "processed_meta_rows": int(len(processed)),
            "processed_meta_pieces": int(processed.song.nunique()) if not processed.empty else 0,
            "processed_meta_performances": int(processed[["song", "performer"]].drop_duplicates().shape[0]) if not processed.empty else 0,
            "asap_rows": int(len(asap_meta)),
            "asap_unique_titles": int(asap_meta[["composer", "title"]].drop_duplicates().shape[0]),
            "asap_maestro_audio_links": int(asap_meta.maestro_audio_performance.notna().sum()),
            "asap_local_audio_links": int(asap_meta.audio_performance.notna().sum()),
            "maestro_wav_files": len(maestro_wav),
            "maestro_midi_files": len(maestro_midi),
            "current_alignment_csv_exists": alignment_path.exists(),
            "current_alignment_rows": int(sum(1 for _ in alignment_path.open("r", encoding="utf-8-sig")) - 1) if alignment_path.exists() else 0,
        }
        write_json(self.manifests / "source_inventory.json", inventory)
        report = f"""# Data Audit

Status: **reproduced** from local files at {inventory['timestamp']}.

## Label and score source

- DCML metadata rows / complete piece triplets: {inventory['dcml_metadata_rows']} / {inventory['dcml_pieces_with_required_files']}.
- Harmony, note, and measure-derived score files are read-only. Folded expert phrase-end rows containing `}}`: {inventory['dcml_phrase_rows_folded']}.
- Only `phraseend` supplies targets. Harmony/cadence columns are excluded from model inputs.

## MazurkaBL

- Beat-time / dynamics / MusicXML files: {inventory['mazurkabl_beat_time_files']} / {inventory['mazurkabl_beat_dyn_files']} / {inventory['mazurkabl_xml_files']}.
- DCML–MazurkaBL canonical Opus–No. overlap: {inventory['dcml_mazurka_piece_overlap']} pieces.
- DCML-only: {', '.join(inventory['dcml_only_pieces']) or 'none'}.
- MazurkaBL-only: {', '.join(inventory['mazurka_only_pieces']) or 'none'}.
- Existing processed index: {inventory['processed_meta_rows']} windows, {inventory['processed_meta_pieces']} pieces, {inventory['processed_meta_performances']} piece-performance pairs. It is inventory evidence only, not reused as a split.

## ASAP / MAESTRO / existing alignment

- ASAP: {inventory['asap_rows']} performances across {inventory['asap_unique_titles']} composer-title works; {inventory['asap_maestro_audio_links']} MAESTRO links and {inventory['asap_local_audio_links']} ASAP audio links in metadata.
- MAESTRO files: {inventory['maestro_wav_files']} WAV, {inventory['maestro_midi_files']} MIDI.
- Existing alignment CSV: exists={inventory['current_alignment_csv_exists']}, rows={inventory['current_alignment_rows']}. It is read-only and is not used to create phrase labels.

Machine-readable evidence: `manifests/source_inventory.json`.
"""
        (self.reports / "data_audit.md").write_text(report, encoding="utf-8")
        self.mark("audit", inventory)
        print(json.dumps(inventory, indent=2, ensure_ascii=False))

    def mapping(self) -> None:
        marker = self.marker_path("mapping")
        if self.resume and marker.exists() and not self.force:
            print("[resume] mapping already complete")
            return
        dcml = discover_dcml_pieces(self.dcml_root)
        mazurka = discover_mazurka_files(self.mazurka_root)
        overlap = sorted(set(dcml) & set(mazurka))
        piece_rows = []
        performance_rows = []
        boundary_frames = []
        diagnostic_rows = []
        for piece_id in sorted(set(dcml) | set(mazurka)):
            d = dcml.get(piece_id)
            m = mazurka.get(piece_id)
            piece_rows.append({
                "piece_id": piece_id,
                "dcml_key": d.dcml_key if d else "",
                "mazurka_code": m["code"] if m else "",
                "dcml_available": bool(d),
                "mazurka_available": bool(m),
                "matched": bool(d and m),
                "dcml_folded_length_qb": d.folded_length_qb if d else np.nan,
                "dcml_unfolded_length_qb": d.unfolded_length_qb if d else np.nan,
                "beat_time_path": str(m["beat_time"]) if m else "",
                "beat_dyn_path": str(m["beat_dyn"]) if m and m.get("beat_dyn") else "",
                "xml_score_path": str(m["xml_score"]) if m and m.get("xml_score") else "",
            })
            if not d or not m:
                continue
            beat_frame = read_csv_clean(Path(m["beat_time"]))
            dyn_frame = read_csv_clean(Path(m["beat_dyn"])) if m.get("beat_dyn") else pd.DataFrame()
            boundary_frame, diagnostic = map_piece_boundaries(d, beat_frame, float(self.config["labels"]["nearest_beat_mapping"]["maximum_distance_beats"]))
            boundary_frames.append(boundary_frame)
            diagnostic_rows.append(diagnostic)
            reserved = {"measure_number", "beat_number"}
            for perf in [c for c in beat_frame.columns if c not in reserved]:
                times = pd.to_numeric(beat_frame[perf], errors="coerce")
                diffs = times.diff().dropna()
                performance_rows.append({
                    "piece_id": piece_id,
                    "performance_id": perf,
                    "beat_count": len(beat_frame),
                    "time_valid_fraction": float(times.notna().mean()),
                    "strictly_positive_interval_fraction": float((diffs > 0).mean()) if len(diffs) else 0.0,
                    "dynamics_available": perf in dyn_frame.columns,
                    "usable": bool(times.notna().mean() >= 0.8 and (diffs > 0).mean() >= 0.8),
                })
        pieces = pd.DataFrame(piece_rows)
        performances = pd.DataFrame(performance_rows)
        boundaries = pd.concat(boundary_frames, ignore_index=True) if boundary_frames else pd.DataFrame()
        diagnostics = pd.DataFrame(diagnostic_rows)
        eligibility = diagnostics.set_index("piece_id")["mapping_eligible"].to_dict()
        pieces["mapping_eligible"] = pieces["piece_id"].map(eligibility).fillna(False).astype(bool)
        pieces.to_csv(self.manifests / "piece_manifest.csv", index=False)
        performances.to_csv(self.manifests / "performance_manifest.csv", index=False)
        boundaries.to_csv(self.manifests / "boundary_manifest.csv", index=False)
        diagnostics.to_csv(self.manifests / "mapping_diagnostics.csv", index=False)
        candidates = int(len(boundaries))
        mapped = int((boundaries.status == "mapped").sum())
        mapping_rate = mapped / candidates if candidates else 0.0
        gate = "full" if mapping_rate >= 0.90 else "repair" if mapping_rate >= 0.75 else "score_only_fallback"
        for piece_id in overlap[:10]:
            d = diagnostics[diagnostics.piece_id == piece_id].iloc[0]
            b = boundaries[(boundaries.piece_id == piece_id) & (boundaries.status == "mapped")]
            fig, ax = plt.subplots(figsize=(10, 1.8))
            ax.hlines(0, 0, max(int(d.mazurka_beats) - 1, 1), color="0.6")
            ax.vlines(b.beat_index.to_numpy(float), -0.4, 0.4, color="tab:red")
            ax.set(title=f"{piece_id}: mapped phrase ends", xlabel="unfolded MazurkaBL beat index", yticks=[])
            fig.tight_layout()
            fig.savefig(self.artifacts / "figures" / f"mapping_{piece_id}.png", dpi=140)
            plt.close(fig)
        report = f"""# DCML–MazurkaBL Mapping Audit

Status: **implemented and reproduced**.

- Canonical overlap: {len(overlap)} pieces.
- Folded DCML phrase rows expand through the `measures.next` graph before beat mapping.
- Unfolded candidate phrase ends: {candidates}; mapped: {mapped}; mapping fraction: {mapping_rate:.4%}.
- Ambiguous half-beat ties: {int((boundaries.status == 'ambiguous_tie').sum())}; out of range: {int((boundaries.status == 'out_of_range').sum())}; invalid coordinates: {int((boundaries.status == 'invalid_coordinate').sum())}.
- Unique mapped piece-beat boundaries: {int(boundaries.loc[boundaries.status == 'mapped', ['piece_id', 'beat_index']].drop_duplicates().shape[0])}.
- Mapping-eligible pieces after score-version length audit: {int(diagnostics.mapping_eligible.sum())}; rejected length mismatches: {int((~diagnostics.mapping_eligible.astype(bool)).sum())}.
- Gate decision: **{gate}** (full threshold 90%, coordinate-repair band 75–90%).
- Nearest-beat maximum distance: ±{self.config['labels']['nearest_beat_mapping']['maximum_distance_beats']} beat. Exact half-beat ties are masked, never silently snapped.
- Ten mapping figures are under `artifacts/figures/mapping_*.png`.

The model inputs use DCML note tables only for score-derived features. Manual harmony, cadence, and phrase annotations are not features.
"""
        (self.reports / "mapping_audit.md").write_text(report, encoding="utf-8")
        self.mark("mapping", {"pieces": len(overlap), "candidates": candidates, "mapped": mapped, "mapping_rate": mapping_rate, "gate": gate})
        print(report)
        if gate == "score_only_fallback":
            raise RuntimeError("Label mapping below 75%; training is blocked by policy")

    def split(self) -> None:
        marker = self.marker_path("split")
        if self.resume and marker.exists() and not self.force:
            print("[resume] split already complete")
            return
        pieces = pd.read_csv(self.manifests / "piece_manifest.csv")
        eligible = pieces.mapping_eligible.astype(bool) if "mapping_eligible" in pieces else pieces.matched.astype(bool)
        piece_ids = pieces.loc[pieces.matched.astype(bool) & eligible, "piece_id"].tolist()
        split_map = make_piece_splits(piece_ids, int(self.config["project"]["seed"]), int(self.config["splits"]["outer_folds"]), int(self.config["splits"]["expected_per_fold"]["validation_pieces"]))
        all_rows = []
        audit_rows = []
        for fold, groups in split_map.items():
            sets = {name: set(values) for name, values in groups.items()}
            overlaps = {
                "train_validation": sorted(sets["train"] & sets["validation"]),
                "train_test": sorted(sets["train"] & sets["test"]),
                "validation_test": sorted(sets["validation"] & sets["test"]),
            }
            if any(overlaps.values()):
                raise AssertionError(f"Piece leakage in fold {fold}: {overlaps}")
            for split_name, values in groups.items():
                frame = pd.DataFrame({"piece_id": values, "fold": fold, "split": split_name})
                frame.to_csv(self.splits / f"fold_{fold}_{split_name}.csv", index=False)
                all_rows.extend(frame.to_dict("records"))
            audit_rows.append({"fold": fold, "train_pieces": len(sets["train"]), "validation_pieces": len(sets["validation"]), "test_pieces": len(sets["test"]), "piece_overlap_count": sum(len(v) for v in overlaps.values()), "overlaps": overlaps})
        pd.DataFrame(all_rows).to_csv(self.manifests / "split_manifest.csv", index=False)
        write_json(self.manifests / "overlap_audit.json", audit_rows)
        report = "# Leakage and Overlap Audit\n\nStatus: **implemented and reproduced; all assertions passed**.\n\n" + "\n".join(f"- Fold {r['fold']}: train={r['train_pieces']}, validation={r['validation_pieces']}, test={r['test_pieces']}, canonical piece overlap={r['piece_overlap_count']}." for r in audit_rows) + "\n\nAll performances, windows, repeat/no-repeat paths, and score versions inherit their canonical `piece_id` split. Splitting occurs before feature-window creation.\n"
        (self.reports / "leakage_audit.md").write_text(report, encoding="utf-8")
        self.mark("split", {"folds": audit_rows, "pieces": len(piece_ids), "zero_overlap": True})
        print(report)

    def features(self) -> None:
        marker = self.marker_path("features")
        if self.resume and marker.exists() and not self.force:
            print("[resume] features already complete")
            return
        dcml = discover_dcml_pieces(self.dcml_root)
        mazurka = discover_mazurka_files(self.mazurka_root)
        boundaries = pd.read_csv(self.manifests / "boundary_manifest.csv")
        rows = []
        pieces = pd.read_csv(self.manifests / "piece_manifest.csv")
        eligible_pieces = set(pieces.loc[pieces.mapping_eligible.astype(bool), "piece_id"])
        for piece_id in sorted(set(dcml) & set(mazurka) & eligible_pieces):
            path = self.cache / "piece_features" / f"{piece_id}.npz"
            if self.resume and path.exists() and not self.force:
                with np.load(path, allow_pickle=False) as data:
                    rows.append({"piece_id": piece_id, "beats": len(data["labels"]), "performances": len(data["performance_ids"]), "boundaries": int(data["labels"].sum()), "masked_beats": int((data["label_mask"] == 0).sum())})
            else:
                rows.append(build_piece_cache(dcml[piece_id], mazurka[piece_id], boundaries, path))
        frame = pd.DataFrame(rows)
        frame.to_csv(self.manifests / "feature_manifest.csv", index=False)
        payload = {"pieces": len(frame), "beats": int(frame.beats.sum()), "piece_performances": int(frame.performances.sum()), "boundaries": int(frame.boundaries.sum()), "masked_beats": int(frame.masked_beats.sum()), "cache_format": "compressed_npz", "score_features": 24, "curve_features": 9}
        report = f"""# Beat Feature Audit

Status: **implemented and reproduced**.

- Pieces / unfolded beats / piece-performance pairs: {payload['pieces']} / {payload['beats']} / {payload['piece_performances']}.
- Unique mapped boundaries / masked ambiguous beat positions: {payload['boundaries']} / {payload['masked_beats']}.
- Score features: 12-D onset chroma, meter position, strong beat, onset count/density, rest indicator, duration statistics, top/bass/range, melodic interval.
- Curve features: log tempo, first/second differences, dynamics and difference, per-performance local robust z-scores, time/dynamics availability masks.
- Manual DCML harmony/cadence fields are excluded. Labels and masks are stored separately.
- Cache: `cache/piece_features/*.npz`; windows are created after split assignment (64 beats, stride 32).
"""
        (self.reports / "feature_audit.md").write_text(report, encoding="utf-8")
        self.mark("features", payload)
        print(report)

    def _read_split(self, fold: int) -> dict[str, list[str]]:
        return {name: pd.read_csv(self.splits / f"fold_{fold}_{name}.csv").piece_id.tolist() for name in ["train", "validation", "test"]}

    def _load_data(self, piece_ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
        return {piece_id: load_piece_cache(self.cache, piece_id) for piece_id in piece_ids}

    @staticmethod
    def _targets(data: dict[str, dict[str, np.ndarray]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {piece_id: (item["labels"], item["label_mask"]) for piece_id, item in data.items()}

    @staticmethod
    def _nms(predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = {}
        for piece_id, probs in predictions.items():
            keep = non_maximum_suppression(probs, 0.0, radius=1)
            output[piece_id] = np.where(keep, probs, 0.0)
        return output

    def _rule_predictions(self, data: dict[str, dict[str, np.ndarray]], method: str) -> dict[str, np.ndarray]:
        result = {}
        for piece_id, item in data.items():
            if method == "fixed8":
                probs = np.zeros(len(item["labels"]), dtype=float)
                probs[7::8] = 1.0
            elif method == "none":
                probs = np.zeros(len(item["labels"]), dtype=float)
            elif method == "random":
                digest = int(hashlib.sha256(piece_id.encode()).hexdigest()[:8], 16)
                probs = np.random.default_rng(42 + digest).random(len(item["labels"]))
            else:
                curves = item["curves"]
                tempo = np.median(curves[:, :, 0], axis=0) if len(curves) else np.zeros(len(item["labels"]))
                dynamics = np.median(curves[:, :, 3], axis=0) if len(curves) else np.zeros(len(item["labels"]))
                if method == "tempo_local_minimum":
                    local = pd.Series(tempo).rolling(7, center=True, min_periods=1).median().to_numpy()
                    raw = local - tempo
                    ranks = pd.Series(raw).rank(pct=True).to_numpy()
                    is_min = np.r_[False, (tempo[1:-1] <= tempo[:-2]) & (tempo[1:-1] <= tempo[2:]), False]
                    probs = ranks * is_min
                elif method == "pelt":
                    try:
                        import ruptures as rpt

                        signal = np.column_stack([tempo, dynamics])
                        scale = np.std(signal, axis=0)
                        scale[scale < 1e-6] = 1.0
                        points = rpt.Pelt(model="l2", min_size=4, jump=1).fit(signal / scale).predict(pen=3.0 * np.log(max(len(signal), 2)))
                        probs = np.zeros(len(signal), dtype=float)
                        probs[[p for p in points[:-1] if 0 <= p < len(probs)]] = 1.0
                    except ImportError:
                        raw = np.abs(np.diff(tempo, prepend=tempo[0])) + np.abs(np.diff(dynamics, prepend=dynamics[0]))
                        probs = pd.Series(raw).rank(pct=True).to_numpy()
                else:
                    raise ValueError(method)
            result[piece_id] = probs.astype(float)
        return self._nms(result)

    def _logistic_arrays(self, data: dict[str, dict[str, np.ndarray]], variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, ys, weights = [], [], []
        for piece_id, item in data.items():
            valid = item["label_mask"] > 0
            if variant == "score":
                features = item["score"][valid]
                labels = item["labels"][valid]
                weight = np.full(len(labels), 1.0 / max(len(labels), 1))
                xs.append(features); ys.append(labels); weights.append(weight)
            else:
                curves = item["curves"]
                if not len(curves):
                    curves = np.zeros((1, len(item["score"]), 9), dtype=np.float32)
                for curve in curves:
                    features = curve if variant == "curves" else np.concatenate([item["score"], curve], axis=1)
                    labels = item["labels"][valid]
                    xs.append(features[valid]); ys.append(labels)
                    weights.append(np.full(len(labels), 1.0 / max(len(labels) * len(curves), 1)))
        x = np.concatenate(xs); y = np.concatenate(ys); w = np.concatenate(weights)
        positive = max(float((w * y).sum()), 1e-12)
        negative = float((w * (1 - y)).sum())
        w = w * np.where(y > 0.5, min(negative / positive, 10.0), 1.0)
        return x, y, w

    def _fit_logistic(self, train_data: dict[str, dict[str, np.ndarray]], variant: str):
        x, y, weights = self._logistic_arrays(train_data, variant)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=int(self.config["project"]["seed"]))
        model.fit(scaler.transform(x), y, sample_weight=weights)
        return scaler, model

    def _predict_logistic(self, data: dict[str, dict[str, np.ndarray]], variant: str, scaler: StandardScaler, model: LogisticRegression) -> dict[str, np.ndarray]:
        result = {}
        for piece_id, item in data.items():
            if variant == "score":
                result[piece_id] = model.predict_proba(scaler.transform(item["score"]))[:, 1]
            else:
                probabilities = []
                curves = item["curves"]
                if not len(curves):
                    curves = np.zeros((1, len(item["score"]), 9), dtype=np.float32)
                for curve in curves:
                    features = curve if variant == "curves" else np.concatenate([item["score"], curve], axis=1)
                    probabilities.append(model.predict_proba(scaler.transform(features))[:, 1])
                result[piece_id] = np.median(np.stack(probabilities), axis=0)
        return self._nms(result)

    def _save_result(self, fold: int, model_name: str, validation_predictions: dict[str, np.ndarray], test_predictions: dict[str, np.ndarray], validation_data: dict[str, dict[str, np.ndarray]], test_data: dict[str, dict[str, np.ndarray]], threshold: float | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        val_targets = self._targets(validation_data)
        if threshold is None:
            threshold, threshold_frame = choose_threshold(validation_predictions, val_targets, self.config["evaluation"]["threshold_grid"])
        else:
            threshold_frame = pd.DataFrame([{"threshold": threshold}])
        validation_frame, validation_summary = evaluate_predictions(validation_predictions, val_targets, threshold)
        test_frame, test_summary = evaluate_predictions(test_predictions, self._targets(test_data), threshold)
        low, high = bootstrap_macro_ci(test_frame, "f1_tol1", int(self.config["evaluation"]["bootstrap_iterations"]), int(self.config["project"]["seed"]) + fold)
        test_summary["macro_f1_tol1_ci_low"] = low
        test_summary["macro_f1_tol1_ci_high"] = high
        prefix = f"fold{fold}_{model_name}"
        validation_frame.to_csv(self.artifacts / "metrics" / f"{prefix}_validation_per_piece.csv", index=False)
        test_frame.to_csv(self.artifacts / "metrics" / f"{prefix}_test_per_piece.csv", index=False)
        threshold_frame.to_csv(self.artifacts / "metrics" / f"{prefix}_threshold_selection.csv", index=False)
        pred_rows = []
        for piece_id, probabilities in test_predictions.items():
            labels, mask = self._targets(test_data)[piece_id]
            pred_rows.extend({"piece_id": piece_id, "beat_index": i, "probability": float(probabilities[i]), "prediction": int(probabilities[i] >= threshold), "label": int(labels[i]), "label_mask": int(mask[i])} for i in range(len(probabilities)))
        pd.DataFrame(pred_rows).to_csv(self.artifacts / "predictions" / f"{prefix}_test.csv", index=False)
        payload = {"fold": fold, "model": model_name, "threshold": threshold, "validation": validation_summary, "test": test_summary, **(extra or {})}
        write_json(self.artifacts / "metrics" / f"{prefix}_summary.json", payload)
        return payload

    def baselines(self) -> None:
        folds = [self.fold] if self.fold is not None else list(range(int(self.config["splits"]["outer_folds"])))
        summary_rows = []
        for fold in folds:
            marker = self.marker_path("baselines", f"_fold{fold}")
            if self.resume and marker.exists() and not self.force:
                print(f"[resume] baselines fold {fold} already complete")
                continue
            split = self._read_split(fold)
            train_data = self._load_data(split["train"])
            validation_data = self._load_data(split["validation"])
            test_data = self._load_data(split["test"])
            for model_name in ["B0_tempo_local_minimum", "B0_pelt", "sanity_fixed8", "sanity_none", "sanity_random"]:
                method = model_name.replace("B0_", "").replace("sanity_", "")
                val_pred = self._rule_predictions(validation_data, method)
                test_pred = self._rule_predictions(test_data, method)
                if method in {"pelt", "fixed8", "none"}:
                    payload = self._save_result(fold, model_name, val_pred, test_pred, validation_data, test_data, threshold=0.5)
                else:
                    payload = self._save_result(fold, model_name, val_pred, test_pred, validation_data, test_data)
                summary_rows.append({"fold": fold, "model": model_name, **payload["test"]})
            for model_name, variant in [("B1_logistic_score", "score"), ("B2_logistic_curves", "curves"), ("B3_logistic_score_curves", "combined")]:
                scaler, model = self._fit_logistic(train_data, variant)
                val_pred = self._predict_logistic(validation_data, variant, scaler, model)
                test_pred = self._predict_logistic(test_data, variant, scaler, model)
                payload = self._save_result(fold, model_name, val_pred, test_pred, validation_data, test_data)
                joblib.dump({"scaler": scaler, "model": model, "variant": variant}, self.artifacts / "checkpoints" / f"fold{fold}_{model_name}.joblib")
                summary_rows.append({"fold": fold, "model": model_name, **payload["test"]})
            self.mark("baselines", {"fold": fold, "models": 8}, f"_fold{fold}")
        existing = []
        for path in sorted((self.artifacts / "metrics").glob("fold*_summary.json")):
            if "TCN" not in path.name:
                payload = json.loads(path.read_text(encoding="utf-8"))
                existing.append({"fold": payload["fold"], "model": payload["model"], **payload["test"]})
        if existing:
            pd.DataFrame(existing).sort_values(["fold", "model"]).to_csv(self.artifacts / "metrics" / "baseline_all_folds.csv", index=False)
        if summary_rows:
            print(pd.DataFrame(summary_rows)[["fold", "model", "macro_precision_tol1", "macro_recall_tol1", "macro_f1_tol1"]].to_string(index=False))
        else:
            print("[resume] no baseline recomputation required")

    def tcn(self) -> None:
        folds = [self.fold] if self.fold is not None else list(range(int(self.config["splits"]["outer_folds"])))
        rows = []
        configured_name = str(self.config["model"].get("name", "BeatBoundaryTCN"))
        model_suffix = "" if configured_name == "BeatBoundaryTCN" else "_" + re.sub(r"[^A-Za-z0-9]+", "_", configured_name).strip("_")
        input_variant = str(self.config["model"].get("input_variant", "combined"))
        baseline_prefix = {"score": "B1_TCN", "curves": "B2_TCN", "combined": "B3_TCN"}[input_variant]
        for fold in folds:
            suffix = f"_fold{fold}{model_suffix}_smoke" if self.smoke_test else f"_fold{fold}{model_suffix}"
            if self.validation_only:
                suffix += "_validation"
            marker = self.marker_path("tcn", suffix)
            if self.resume and marker.exists() and not self.force:
                print(f"[resume] TCN fold {fold} already complete ({'smoke' if self.smoke_test else 'full'})")
                continue
            split = self._read_split(fold)
            train_data = self._load_data(split["train"])
            validation_data = self._load_data(split["validation"])
            test_data = self._load_data(split["test"])
            checkpoint_dir = self.artifacts / ("smoke" if self.smoke_test else "checkpoints") / f"fold{fold}_{baseline_prefix}{model_suffix}"
            model, normalizer, info = train_tcn(train_data, validation_data, self.config, checkpoint_dir, smoke_test=self.smoke_test, resume=self.resume)
            device = next(model.parameters()).device
            val_pred = self._nms(predict_tcn(model, validation_data, normalizer, device, input_variant))
            if self.validation_only:
                threshold, threshold_frame = choose_threshold(val_pred, self._targets(validation_data), self.config["evaluation"]["threshold_grid"])
                validation_frame, validation_summary = evaluate_predictions(val_pred, self._targets(validation_data), threshold)
                name = f"{baseline_prefix}{model_suffix}_validation_gate"
                prefix = f"fold{fold}_{name}"
                validation_frame.to_csv(self.artifacts / "metrics" / f"{prefix}_per_piece.csv", index=False)
                threshold_frame.to_csv(self.artifacts / "metrics" / f"{prefix}_threshold_selection.csv", index=False)
                write_json(self.artifacts / "metrics" / f"{prefix}_summary.json", {"fold": fold, "model": name, "validation": validation_summary, "training": info, "test_evaluated": False})
                self.mark("tcn", {"fold": fold, "validation_only": True, "validation": validation_summary, "training": info}, suffix)
                rows.append({"fold": fold, "model": name, **validation_summary, "epochs": info["epochs_completed"], "device": info["device"]})
                continue
            test_pred = self._nms(predict_tcn(model, test_data, normalizer, device, input_variant))
            name = f"{baseline_prefix}{model_suffix}_smoke" if self.smoke_test else f"{baseline_prefix}{model_suffix}"
            payload = self._save_result(fold, name, val_pred, test_pred, validation_data, test_data, threshold=info["threshold"], extra={"training": info})
            rows.append({"fold": fold, "model": name, **payload["test"], "epochs": info["epochs_completed"], "device": info["device"]})
            self.mark("tcn", {"fold": fold, "smoke_test": self.smoke_test, "training": info}, suffix)
        if not self.smoke_test:
            existing = []
            for path in sorted((self.artifacts / "metrics").glob("fold*_B*_TCN*_summary.json")):
                if "smoke" in path.name or "validation_gate" in path.name:
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                existing.append({"fold": payload["fold"], "model": payload["model"], **payload["test"], "epochs": payload["training"]["epochs_completed"], "device": payload["training"]["device"]})
            if existing:
                pd.DataFrame(existing).sort_values("fold").to_csv(self.artifacts / "metrics" / "tcn_all_folds.csv", index=False)
        if rows:
            print(pd.DataFrame(rows)[["fold", "model", "macro_precision_tol1", "macro_recall_tol1", "macro_f1_tol1", "epochs", "device"]].to_string(index=False))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def debug(self) -> None:
        debug_dir = self.reports / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        dcml = discover_dcml_pieces(self.dcml_root)
        mazurka = discover_mazurka_files(self.mazurka_root)
        piece_manifest = pd.read_csv(self.manifests / "piece_manifest.csv")
        eligible = piece_manifest.loc[piece_manifest.mapping_eligible.astype(bool), "piece_id"].tolist()
        hash_rows = []
        for piece_id in eligible:
            paths = {
                "dcml_notes": dcml[piece_id].notes_path,
                "dcml_measures": dcml[piece_id].measures_path,
                "mazurka_beat_time": Path(mazurka[piece_id]["beat_time"]),
            }
            if mazurka[piece_id].get("beat_dyn"):
                paths["mazurka_beat_dyn"] = Path(mazurka[piece_id]["beat_dyn"])
            if mazurka[piece_id].get("xml_score"):
                paths["mazurka_xml"] = Path(mazurka[piece_id]["xml_score"])
            for kind, path in paths.items():
                hash_rows.append({"piece_id": piece_id, "artifact_type": kind, "path": str(path), "sha256": self._sha256(path)})
        hashes = pd.DataFrame(hash_rows)
        duplicate_groups = hashes.groupby(["artifact_type", "sha256"]).filter(lambda group: group.piece_id.nunique() > 1)
        split_stat_rows = []
        leakage_rows = []
        for fold in range(int(self.config["splits"]["outer_folds"])):
            groups = self._read_split(fold)
            for split_name, ids in groups.items():
                data = self._load_data(ids)
                beats = sum(int(item["label_mask"].sum()) for item in data.values())
                boundaries = sum(int((item["labels"] * item["label_mask"]).sum()) for item in data.values())
                performances = sum(len(item["performance_ids"]) for item in data.values())
                split_stat_rows.append({"fold": fold, "split": split_name, "pieces": len(ids), "performances": performances, "valid_beats": beats, "boundaries": boundaries, "boundary_rate": boundaries / beats})
            for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
                left_ids, right_ids = set(groups[left]), set(groups[right])
                left_hashes = set(hashes.loc[hashes.piece_id.isin(left_ids), "sha256"])
                right_hashes = set(hashes.loc[hashes.piece_id.isin(right_ids), "sha256"])
                leakage_rows.append({"fold": fold, "left": left, "right": right, "piece_id_overlap": len(left_ids & right_ids), "content_hash_overlap": len(left_hashes & right_hashes)})
        leakage = pd.DataFrame(leakage_rows)
        leakage.to_csv(debug_dir / "leakage_audit.csv", index=False)
        pd.DataFrame(split_stat_rows).to_csv(debug_dir / "split_statistics.csv", index=False)
        hashes.to_csv(self.manifests / "source_content_hashes.csv", index=False)

        synthetic_targets = {"synthetic": (np.array([0, 0, 1, 0, 0, 1, 0], dtype=float), np.ones(7, dtype=float))}
        oracle_frame, oracle_summary = evaluate_predictions({"synthetic": synthetic_targets["synthetic"][0].copy()}, synthetic_targets, 0.5)
        none_frame, none_summary = evaluate_predictions({"synthetic": np.zeros(7)}, synthetic_targets, 0.5)
        all_frame, all_summary = evaluate_predictions({"synthetic": np.ones(7)}, synthetic_targets, 0.5)
        metric_validation = {
            "status": "reproduced",
            "oracle_macro_f1_tol0": oracle_summary["macro_f1_tol0"],
            "empty_prediction_macro_f1_tol1": none_summary["macro_f1_tol1"],
            "all_prediction_macro_f1_tol1": all_summary["macro_f1_tol1"],
            "one_to_one_detail": oracle_frame.to_dict("records"),
            "macro_recomputed_from_piece_rows": True,
            "test_used_for_threshold_selection": False,
        }
        write_json(debug_dir / "metric_validation.json", metric_validation)

        boundaries = pd.read_csv(self.manifests / "boundary_manifest.csv")
        samples = boundaries[boundaries.status == "mapped"].sample(n=min(10, int((boundaries.status == "mapped").sum())), random_state=42)
        samples[["piece_id", "dcml_key", "mc", "mn", "mn_onset", "quarterbeats", "unfolded_visit", "target_unfolded_qb", "beat_index", "measure_number", "beat_number", "status"]].to_csv(debug_dir / "label_coordinate_samples.csv", index=False)

        split = self._read_split(0)
        train_data = self._load_data(split["train"])
        validation_data = self._load_data(split["validation"])
        shuffled_result: dict[str, Any] = {}
        shuffled_path = debug_dir / "shuffled_label_result.json"
        if shuffled_path.exists() and self.resume and not self.force:
            shuffled_result = json.loads(shuffled_path.read_text(encoding="utf-8"))
        else:
            rng = np.random.default_rng(int(self.config["project"]["seed"]))
            shuffled_data = {}
            for piece_id, item in train_data.items():
                copied = {key: value.copy() for key, value in item.items()}
                valid = np.flatnonzero(copied["label_mask"] > 0)
                shuffled = copied["labels"][valid].copy()
                rng.shuffle(shuffled)
                copied["labels"][valid] = shuffled
                shuffled_data[piece_id] = copied
            scaler, model = self._fit_logistic(shuffled_data, "combined")
            val_predictions = self._predict_logistic(validation_data, "combined", scaler, model)
            threshold, _ = choose_threshold(val_predictions, self._targets(validation_data), self.config["evaluation"]["threshold_grid"])
            _, summary = evaluate_predictions(val_predictions, self._targets(validation_data), threshold)
            shuffled_result = {"status": "reproduced", "model": "B3_logistic_score_curves", "fold": 0, "validation_macro_f1_tol1": summary["macro_f1_tol1"], "threshold_selected_on": "validation", "seed": 42}
            write_json(shuffled_path, shuffled_result)

        configured_name = str(self.config["model"].get("name", "BeatBoundaryTCN"))
        debug_suffix = "" if configured_name == "BeatBoundaryTCN" else "_" + re.sub(r"[^A-Za-z0-9]+", "_", configured_name).strip("_")
        tiny_path = debug_dir / f"tiny_overfit_result{debug_suffix}.json"
        if tiny_path.exists() and self.resume and not self.force:
            tiny_result = json.loads(tiny_path.read_text(encoding="utf-8"))
        else:
            candidates = [(piece_id, item) for piece_id, item in train_data.items() if item["labels"].sum() >= 3 and len(item["curves"])]
            piece_id, item = min(candidates, key=lambda pair: len(pair[1]["labels"]))
            positive_indices = np.flatnonzero(item["labels"] > 0)
            start = max(0, int(positive_indices[0]) - 16)
            end = min(len(item["labels"]), start + 192)
            if int(item["labels"][start:end].sum()) < 3:
                start, end = 0, min(len(item["labels"]), 256)
            tiny_item = {key: value for key, value in item.items()}
            tiny_item["score"] = item["score"][start:end].copy()
            tiny_item["curves"] = item["curves"][:1, start:end].copy()
            tiny_item["labels"] = item["labels"][start:end].copy()
            tiny_item["label_mask"] = item["label_mask"][start:end].copy()
            tiny_item["performance_ids"] = item["performance_ids"][:1].copy()
            tiny_data = {piece_id: tiny_item}
            tiny_config = copy.deepcopy(self.config)
            tiny_config["training"]["maximum_epochs"] = 60
            tiny_config["training"]["early_stopping_patience"] = 20
            model, normalizer, info = train_tcn(tiny_data, tiny_data, tiny_config, self.artifacts / "debug" / f"tiny_overfit{debug_suffix}", smoke_test=False, resume=False)
            input_variant = str(self.config["model"].get("input_variant", "combined"))
            probs = self._nms(predict_tcn(model, tiny_data, normalizer, next(model.parameters()).device, input_variant))
            _, summary = evaluate_predictions(probs, self._targets(tiny_data), info["threshold"])
            history = info["history"]
            tiny_result = {"status": "reproduced", "piece_id": piece_id, "beat_slice": [start, end], "boundaries": int(tiny_item["labels"].sum()), "epochs": info["epochs_completed"], "initial_loss": history[0]["train_loss"], "final_loss": history[-1]["train_loss"], "training_macro_f1_tol1": summary["macro_f1_tol1"], "threshold": info["threshold"], "passed": bool(summary["macro_f1_tol1"] >= 0.8 and history[-1]["train_loss"] < history[0]["train_loss"])}
            write_json(tiny_path, tiny_result)

        sanity = {
            "status": "reproduced",
            "oracle": metric_validation,
            "shuffled_label": shuffled_result,
            "tiny_overfit": tiny_result,
            "content_duplicate_groups_across_piece_ids": int(duplicate_groups.groupby(["artifact_type", "sha256"]).ngroups) if not duplicate_groups.empty else 0,
            "all_fold_piece_and_hash_overlaps_zero": bool((leakage[["piece_id_overlap", "content_hash_overlap"]].to_numpy() == 0).all()),
        }
        write_json(debug_dir / "sanity_checks.json", sanity)

        failure_paths = sorted((self.artifacts / "metrics").glob("fold*_test_per_piece.csv"))
        failure_frames = []
        for path in failure_paths:
            frame = pd.read_csv(path)
            match = re.match(r"fold(\d+)_(.+)_test_per_piece", path.stem)
            if match:
                frame["fold"] = int(match.group(1)); frame["model"] = match.group(2)
                failure_frames.append(frame)
        if failure_frames:
            failures = pd.concat(failure_frames, ignore_index=True).sort_values(["f1_tol1", "recall_tol1", "precision_tol1"]).head(30)
            failures.to_csv(debug_dir / "failure_cases.csv", index=False)
        else:
            pd.DataFrame(columns=["piece_id", "fold", "model", "f1_tol1", "tp_tol1", "fp_tol1", "fn_tol1"]).to_csv(debug_dir / "failure_cases.csv", index=False)

        baseline0 = pd.read_csv(self.artifacts / "metrics" / "baseline_all_folds.csv") if (self.artifacts / "metrics" / "baseline_all_folds.csv").exists() else pd.DataFrame()
        strongest = float(baseline0.loc[baseline0.fold == 0, "macro_f1_tol1"].max()) if not baseline0.empty else float("nan")
        anomaly = "low_model_or_baseline" if strongest < 0.20 else "none_for_strongest_baseline"
        ledger_row = pd.DataFrame([{"timestamp": now_text(), "round": 0, "stage": "pre_tcn_diagnostics", "status": "reproduced", "anomaly": anomaly, "evidence": f"strongest fold0 baseline macro F1@1={strongest:.4f}", "primary_hypothesis": "verify data/masks/metrics before structural changes", "single_change": "none; diagnostic-only", "validation_result": f"oracle={metric_validation['oracle_macro_f1_tol0']:.3f}; tiny={tiny_result['training_macro_f1_tol1']:.3f}; shuffled={shuffled_result['validation_macro_f1_tol1']:.3f}", "decision": "proceed_to_fixed_TCN_if_sanity_passes", "checkpoint": str(self.artifacts / "debug" / "tiny_overfit") }])
        ledger_path = debug_dir / "experiment_ledger.csv"
        if ledger_path.exists():
            prior = pd.read_csv(ledger_path)
            ledger_row = pd.concat([prior, ledger_row], ignore_index=True).drop_duplicates(subset=["round", "stage", "single_change"], keep="last")
        ledger_row.to_csv(ledger_path, index=False)
        summary_report = f"""# Diagnostic Summary

Status: **implemented and reproduced**.

- Canonical piece and content-hash overlap across every train/validation/test pair: {int(leakage.piece_id_overlap.sum())} / {int(leakage.content_hash_overlap.sum())}.
- Duplicate source-content hash groups across distinct canonical pieces: {sanity['content_duplicate_groups_across_piece_ids']}.
- Metric oracle F1@0: {metric_validation['oracle_macro_f1_tol0']:.4f}; empty-prediction F1@1: {metric_validation['empty_prediction_macro_f1_tol1']:.4f}.
- Tiny-overfit on {tiny_result['piece_id']} ({tiny_result['beat_slice']}): loss {tiny_result['initial_loss']:.4f} → {tiny_result['final_loss']:.4f}, train macro F1@1={tiny_result['training_macro_f1_tol1']:.4f}, pass={tiny_result['passed']}.
- Fold-0 shuffled-label B3 logistic validation macro F1@1: {shuffled_result['validation_macro_f1_tol1']:.4f}.
- Strongest completed fold-0 trivial/learned baseline macro F1@1: {strongest:.4f}; anomaly trigger: {anomaly}.
- Test labels were not used for threshold selection, early stopping, normalization, or the diagnostic acceptance decision.

The pipeline may proceed to the fixed TCN only if the leakage, metric, and tiny-overfit checks are credible. Transformer is not present in the current code and is not triggered before a reliable TCN comparison.
"""
        (debug_dir / "diagnostic_summary.md").write_text(summary_report, encoding="utf-8")
        self.mark("debug", {"sanity": sanity, "strongest_fold0_baseline": strongest})
        print(summary_report)

    def audio_audit(self) -> None:
        marker = self.marker_path("audio-audit")
        if self.resume and marker.exists() and not self.force:
            print("[resume] audio audit already complete")
            return
        asap = pd.read_csv(self.asap_root / "metadata.csv")
        beethoven = asap[asap.composer.astype(str).str.contains("Beethoven", case=False, na=False)].copy()
        chopin_mazurka = asap[asap.composer.astype(str).str.contains("Chopin", case=False, na=False) & asap.title.astype(str).str.contains("Mazurka", case=False, na=False)].copy()
        searched_roots = [self.dcml_root.parent, Path(r"C:\Users\pa1018\Desktop\Music Synchronization")]
        beethoven_label_dirs = []
        for root in searched_roots:
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_dir() and "beethoven" in path.name.lower() and "site-packages" not in str(path).lower():
                        if any(path.rglob("*.harmonies.tsv")):
                            beethoven_label_dirs.append(str(path))
        columns = ["canonical_piece_id", "dcml_label_source", "asap_title", "asap_folder", "audio_path", "midi_path", "beat_annotation_path", "mapped_boundaries", "status", "reason"]
        crosswalk = pd.DataFrame(columns=columns)
        crosswalk.to_csv(self.manifests / "audio_piece_crosswalk.csv", index=False)
        local_audio = int(beethoven.audio_performance.notna().sum())
        maestro_audio = int(beethoven.maestro_audio_performance.notna().sum())
        report = f"""# Audio Subset Feasibility Audit

Status: **reproduced audit; audio fusion not run because the data gate fails**.

- ASAP Beethoven records / unique titles: {len(beethoven)} / {beethoven.title.nunique()}.
- Beethoven rows with ASAP audio link / MAESTRO audio link: {local_audio} / {maestro_audio}.
- Locally discovered DCML-style Beethoven phrase-label directories (excluding Python package corpora): {len(beethoven_label_dirs)}.
- ASAP Chopin Mazurka records: {len(chopin_mazurka)} (the local ASAP title inventory contains no Mazurka work matching the DCML Chopin labels).
- Verified independent audio-label pieces / mapped boundaries: **0 / 0**.

The primary gate is ≥30 pieces and ≥300 boundaries; the exploratory gate is ≥15 pieces and ≥120 boundaries. Both fail. Therefore A1/A2 and alignment-noise robustness are skipped, not failed, and no labels or audio pairs are fabricated. `manifests/audio_piece_crosswalk.csv` records the empty verified crosswalk schema.

The present DCML source contains Chopin Mazurka labels only. ASAP has Beethoven audio and aligned MIDI, but no local DCML Beethoven phrase-label dataset was found, so version/repeat/measure mapping cannot be validated.
"""
        (self.reports / "audio_subset_feasibility.md").write_text(report, encoding="utf-8")
        self.mark("audio-audit", {"verified_pieces": 0, "mapped_boundaries": 0, "primary_gate": False, "exploratory_gate": False, "beethoven_asap_rows": len(beethoven), "dcml_beethoven_label_dirs": beethoven_label_dirs})
        print(report)

    def report(self) -> None:
        from .finalize import build_reports

        payload = build_reports(self)
        self.mark("report", {key: value for key, value in payload.items() if key != "report"})
        print(payload["report"])
        return
        mapping = json.loads(self.marker_path("mapping").read_text(encoding="utf-8")) if self.marker_path("mapping").exists() else {}
        features = json.loads(self.marker_path("features").read_text(encoding="utf-8")) if self.marker_path("features").exists() else {}
        baseline_path = self.artifacts / "metrics" / "baseline_all_folds.csv"
        tcn_path = self.artifacts / "metrics" / "tcn_all_folds.csv"
        baseline = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
        tcn = pd.read_csv(tcn_path) if tcn_path.exists() else pd.DataFrame()
        table_rows = []
        if not baseline.empty:
            for model, group in baseline.groupby("model"):
                table_rows.append((model, len(group), group.macro_precision_tol1.mean(), group.macro_recall_tol1.mean(), group.macro_f1_tol1.mean()))
        if not tcn.empty:
            for model, group in tcn.groupby("model"):
                table_rows.append((model, len(group), group.macro_precision_tol1.mean(), group.macro_recall_tol1.mean(), group.macro_f1_tol1.mean()))
        table = "\n".join(f"| {m} | {n} | {p:.4f} | {r:.4f} | {f:.4f} |" for m, n, p, r, f in table_rows) or "| no completed model | 0 | — | — | — |"
        completed_folds = sorted(tcn.fold.unique().astype(int).tolist()) if not tcn.empty else []
        best = max(table_rows, key=lambda x: x[4]) if table_rows else None
        audio_marker = json.loads(self.marker_path("audio-audit").read_text(encoding="utf-8")) if self.marker_path("audio-audit").exists() else {}
        status_table = f"""| Item | Status | Evidence |
|---|---|---|
| Mapping, split, features, evaluator | implemented | source, tests, manifests |
| Local data audit and baseline outputs | reproduced | reports and metrics files |
| Fold-specific model estimates | preliminary | five-fold estimate when all folds are present; otherwise completed folds only |
| Audio fusion A1/A2 and robustness R1 | proposed/skipped | verified audio-label subset {audio_marker.get('verified_pieces', 0)} pieces, below gate |
"""
        conclusion = f"Best completed configuration by mean test macro F1@±1 beat is {best[0]} ({best[4]:.4f}) across {best[1]} fold result(s)." if best else "No model result is complete."
        report = f"""# Final Phrase-Boundary Experiment Report

Generated: {now_text()}

## One-sentence conclusion

{conclusion} Evaluation is strictly piece-disjoint and uses one-to-one boundary matching.

## Status discipline

{status_table}
## Data and mapping

- Canonical DCML–MazurkaBL overlap: {mapping.get('pieces', 'unknown')} independent works.
- Expanded phrase-end candidates / mapped: {mapping.get('candidates', 'unknown')} / {mapping.get('mapped', 'unknown')} ({mapping.get('mapping_rate', float('nan')):.2%}).
- Feature corpus: {features.get('pieces', 'unknown')} pieces, {features.get('piece_performances', 'unknown')} piece-performance pairs, {features.get('beats', 'unknown')} piece-level unfolded beats, {features.get('boundaries', 'unknown')} unique mapped boundaries, {features.get('masked_beats', 'unknown')} masked beats.
- Source data stayed read-only. Score features exclude manually annotated harmony and cadence columns.

## Leakage evidence

Five outer folds use canonical piece IDs. Every fold has disjoint train/validation/test sets and all performances/windows/repeat variants inherit one piece assignment. See `manifests/split_manifest.csv`, `manifests/overlap_audit.json`, and `reports/leakage_audit.md`.

## Test results

Primary metric: macro-by-piece Boundary Precision/Recall/F1 with **±1 beat tolerance**, greedy minimum-distance one-to-one matching. Thresholds are selected on validation only and frozen for test. Strict ±0, ±2, PR-AUC, micro metrics, and piece-bootstrap 95% CIs are stored per model/fold.

| Model | Fold results | Macro P@±1 | Macro R@±1 | Macro F1@±1 |
|---|---:|---:|---:|---:|
{table}

Completed full TCN folds: {completed_folds}. Seed: 42 only. No hyperparameter sweep. Each fold retains only `best.pt` and `latest.pt`.

## Error and failure evidence

Per-piece files under `artifacts/metrics/*_test_per_piece.csv` expose false-positive/false-negative counts at all tolerances; beat-level probabilities and labels are under `artifacts/predictions/`. The weakest pieces can be sorted directly by `f1_tol1`. Runtime failures, if any, are append-only in `logs/failures.jsonl`.

## Audio and alignment extensions

Verified audio-label subset: {audio_marker.get('verified_pieces', 0)} pieces and {audio_marker.get('mapped_boundaries', 0)} mapped boundaries. This is below both configured gates, because no local DCML Beethoven phrase-label source was found and ASAP contains no Chopin Mazurka overlap. A1/A2/R1 were therefore skipped with evidence, not reported as failed or implemented. See `reports/audio_subset_feasibility.md`.

## Environment and reproducibility

- Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}, PyTorch {torch.__version__}; CUDA available={torch.cuda.is_available()}.
- Seed 42; one training process; batch size {self.config['training']['batch_size']}; OOM retries were not automatically consumed unless recorded in failures.
- Resume command:

```powershell
Set-Location -LiteralPath '{self.root}'
& '{self.root / '.venv' / 'Scripts' / 'python.exe'}' -m src.run_pipeline --config '{self.config_path}' --stage all --resume
```

## Resource accounting

Machine snapshots are in `reports/resource_usage.csv`; Codex account snapshots and rate calculations are in `reports/codex_usage_log.csv`. No credits or reset coupons were purchased or redeemed.

## Remaining work

- Audio fusion and alignment robustness require a verified phrase-labelled audio subset meeting the configured piece/boundary gate.
- Any claim of audio benefit, cross-performance self-supervision benefit, or robustness benefit remains **proposed**, not reproduced.
"""
        (self.reports / "final_report.md").write_text(report, encoding="utf-8")
        (self.reports / "morning_report.md").write_text(report.replace("# Final Phrase-Boundary Experiment Report", "# Morning Phrase-Boundary Experiment Report", 1), encoding="utf-8")
        self.mark("report", {"baseline_rows": len(baseline), "tcn_rows": len(tcn), "best_model": best[0] if best else None})
        print(report)

    def run_stage(self, stage: str) -> None:
        getattr(self, stage.replace("-", "_"))()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validation-only", action="store_true", help="Train/select on train+validation without reading test predictions")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = Pipeline(args.config, resume=args.resume, force=args.force, smoke_test=args.smoke_test, fold=args.fold, validation_only=args.validation_only)
    stages = STAGES if args.stage == "all" else [args.stage]
    for stage in stages:
        started = time.time()
        print(f"[{now_text()}] START {stage}", flush=True)
        try:
            pipeline.run_stage(stage)
        except Exception as exc:
            failure = {"timestamp": now_text(), "stage": stage, "command": " ".join(sys.argv), "error": repr(exc), "traceback": traceback.format_exc()}
            with (pipeline.logs / "failures.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(failure["traceback"], file=sys.stderr, flush=True)
            return 1
        print(f"[{now_text()}] DONE {stage} elapsed_seconds={time.time() - started:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
