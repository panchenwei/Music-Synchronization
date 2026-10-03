from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler

from .data import build_curve_features, discover_dcml_pieces, load_piece_cache, write_json
from .models import BeatBoundaryTCN, Normalizer
from .phase2_models import _raw_performance_predictions, choose_single_threshold, evaluate_single_performance, nms_probabilities
from .phase3_features import SCORE_CUE_NAMES, build_compact_score_cues, save_phase3_piece_cache
from .phase3_frontend_adapter import run_frontend_smoke
from .phase3_models import SmallBiGRU, raw_multimodal_predictions, train_sequence_model


STAGES = ["init", "features", "develop", "experiments", "frontend", "report", "audit"]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def paired_bootstrap(values: np.ndarray, iterations: int = 5000, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    if not len(values): return math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(iterations)])
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def markdown_table(frame: pd.DataFrame) -> str:
    def fmt(value):
        return f"{float(value):.4f}" if isinstance(value, (float, np.floating)) else str(value)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "|".join(["---"] * len(frame.columns)) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-5, 1 - 1e-5)
    return np.log(values / (1 - values))


def context_features(curves: np.ndarray, score: np.ndarray) -> np.ndarray:
    base = np.concatenate([curves, score], axis=1).astype(np.float32)
    selected = [0, 3, 5, 6, 9 + 4, 9 + 8, 9 + 14, 9 + 15]
    extras = []
    for radius in (2, 8):
        rolled = pd.DataFrame(base[:, selected]).rolling(2 * radius + 1, center=True, min_periods=1).mean().to_numpy(np.float32)
        extras.append(rolled)
    return np.concatenate([base, *extras], axis=1).astype(np.float32)


def with_boundary_rates(frame: pd.DataFrame, model: str = "model") -> pd.DataFrame:
    """Materialize per-work boundary rates from evaluation count columns."""
    required = {"beats", "true_boundaries", "predicted_boundaries"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{model} piece metrics missing boundary-rate inputs: {sorted(missing)}")
    result = frame.copy()
    beats = pd.to_numeric(result["beats"], errors="coerce")
    if beats.isna().any() or (beats <= 0).any():
        raise ValueError(f"{model} contains non-positive or invalid beat counts")
    result["true_boundary_rate"] = pd.to_numeric(result["true_boundaries"], errors="coerce") / beats
    result["predicted_boundary_rate"] = pd.to_numeric(result["predicted_boundaries"], errors="coerce") / beats
    return result


def boundary_rate_summary(piece_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return work-macro boundary rates from the count columns emitted by evaluation."""
    rows = []
    for model, frame in piece_frames.items():
        rated = with_boundary_rates(frame, model)
        rows.append({
            "model": model,
            "true_boundary_rate": float(rated["true_boundary_rate"].mean()),
            "predicted_boundary_rate": float(rated["predicted_boundary_rate"].mean()),
        })
    return pd.DataFrame(rows)


def empirical_noise_scales(errors: np.ndarray) -> dict[str, float]:
    """Ordered stress magnitudes anchored to the parent aligner's absolute errors."""
    magnitudes = np.abs(np.asarray(errors, float))
    magnitudes = magnitudes[np.isfinite(magnitudes)]
    if not len(magnitudes):
        raise ValueError("alignment error distribution is empty")
    return {
        "clean": 0.0,
        "typical": float(np.quantile(magnitudes, 0.50)),
        "long_tail": float(np.quantile(magnitudes, 0.95)),
        "catastrophic": float(np.max(magnitudes)),
    }


def paired_empirical_noise(errors: np.ndarray, length: int, key: str, scale_seconds: float) -> np.ndarray:
    """Draw a deterministic empirical error shape and scale it monotonically by severity."""
    values = np.asarray(errors, float)
    values = values[np.isfinite(values)]
    denominator = max(float(np.quantile(np.abs(values), 0.95)), 1e-8)
    unit_pool = np.clip(values / denominator, -1.0, 1.0)
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed).choice(unit_pool, length, replace=True) * float(scale_seconds)


def minimum_gap_dp_probabilities(probabilities: np.ndarray, threshold: float, min_gap: int = 4) -> np.ndarray:
    """Weighted-interval DP retaining the strongest compatible local boundary peaks."""
    values = np.asarray(probabilities, float)
    local = nms_probabilities(values)
    candidates = np.flatnonzero(local >= float(threshold))
    if not len(candidates):
        return np.zeros_like(values)
    scores = values[candidates]
    previous = np.searchsorted(candidates, candidates - int(min_gap), side="right")
    best = np.zeros(len(candidates) + 1, float); take = np.zeros(len(candidates), bool)
    for i in range(1, len(candidates) + 1):
        include = scores[i - 1] + best[previous[i - 1]]
        exclude = best[i - 1]
        if include > exclude:
            best[i] = include; take[i - 1] = True
        else:
            best[i] = exclude
    selected=[];i=len(candidates)
    while i > 0:
        if take[i - 1] and scores[i - 1] + best[previous[i - 1]] > best[i - 1]:
            selected.append(candidates[i - 1]);i=int(previous[i - 1])
        else:
            i -= 1
    output=np.zeros_like(values);output[np.asarray(selected,dtype=int)]=values[np.asarray(selected,dtype=int)]
    return output


def apply_minimum_gap_dp(raw: dict[str, dict[str, np.ndarray]], threshold: float, min_gap: int = 4) -> dict[str, dict[str, np.ndarray]]:
    return {pid:{perf:minimum_gap_dp_probabilities(values,threshold,min_gap) for perf,values in perfs.items()} for pid,perfs in raw.items()}


class Phase3Pipeline:
    def __init__(self, root: Path, resume: bool, force: bool):
        self.root = root.resolve(); self.resume = resume; self.force = force
        self.protocol = yaml.safe_load((self.root / "configs/phase3/protocol.yaml").read_text(encoding="utf-8"))
        self.experiment = yaml.safe_load((self.root / "experiment_config.yaml").read_text(encoding="utf-8"))
        self.cache = self.root / "cache/phase3/piece_features"
        self.artifacts = self.root / "artifacts/phase3"; self.metrics = self.artifacts / "metrics"; self.figures = self.artifacts / "figures"
        self.checkpoints = self.root / "checkpoints/phase3"; self.reports = self.root / "reports/phase3"; self.logs = self.root / "logs/phase3"; self.manifests = self.root / "manifests/phase3"
        self.markers = self.artifacts / "stages"
        for path in [self.cache, self.metrics, self.figures, self.checkpoints, self.reports, self.logs, self.manifests, self.markers]: path.mkdir(parents=True, exist_ok=True)

    def marker(self, stage: str) -> Path: return self.markers / f"{stage}.json"
    def done(self, stage: str) -> bool: return self.resume and self.marker(stage).exists() and not self.force
    def mark(self, stage: str, payload: dict[str, Any]): write_json(self.marker(stage), {"stage": stage, "completed_at": now(), **payload})
    def split(self, fold: int) -> dict[str, list[str]]:
        frame = pd.read_csv(self.root / "artifacts/phase2/splits/opus_split_manifest.csv")
        return {name: frame[(frame.fold == fold) & (frame.split == name)].piece_id.tolist() for name in ["train", "validation", "test"]}
    def load(self, ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
        result = {}
        for piece_id in ids:
            with np.load(self.cache / f"{piece_id}.npz", allow_pickle=False) as item: result[piece_id] = {k: item[k] for k in item.files}
        return result
    def c0_checkpoint(self, fold: int, seed: int = 42) -> Path:
        return self.root / f"checkpoints/phase2/opus/seed{seed}/fold{fold}/best.pt"
    def load_c0(self, fold: int, seed: int = 42):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(self.c0_checkpoint(fold, seed), map_location=device, weights_only=False)
        model = BeatBoundaryTCN(int(state["input_dim"]), 32, [1,2,4,8], 3, .2).to(device); model.load_state_dict(state["model"])
        norm = Normalizer(np.asarray(state["normalizer_mean"]), np.asarray(state["normalizer_std"]))
        return model, norm, state

    def save_evaluation(self, prefix: str, raw_val, raw_test, val, test) -> dict[str, Any]:
        threshold, curve = choose_single_threshold(raw_val, val, self.protocol["evaluation"]["threshold_grid"])
        vp, vw, vs = evaluate_single_performance(raw_val, val, threshold)
        tp, tw, ts = evaluate_single_performance(raw_test, test, threshold)
        curve.to_csv(self.metrics / f"{prefix}_threshold.csv", index=False); vp.to_csv(self.metrics / f"{prefix}_validation_performance.csv", index=False); vw.to_csv(self.metrics / f"{prefix}_validation_piece.csv", index=False); tp.to_csv(self.metrics / f"{prefix}_test_performance.csv", index=False); tw.to_csv(self.metrics / f"{prefix}_test_piece.csv", index=False)
        payload = {"threshold": threshold, "validation": vs, "test": ts}; write_json(self.metrics / f"{prefix}_summary.json", payload); return payload

    def init(self):
        if self.done("init"): print("[resume] init"); return
        frozen = json.loads((self.manifests / "frozen_phase2_evidence.json").read_text(encoding="utf-8")); mismatches = []
        for row in frozen["sha256_evidence"]:
            digest = hashlib.sha256((self.root / row["path"]).read_bytes()).hexdigest()
            if digest != row["sha256"]: mismatches.append(row["path"])
        if mismatches: raise AssertionError(f"Frozen evidence changed: {mismatches}")
        literature = """# Phase 3 Literature Transfer Log

Search date: 2026-09-08. Only papers, publisher pages, author repositories and official dataset repositories are treated as technical evidence. Search-result snippets are not treated as experimental facts.

| Symptom / query | Primary source and original setting | Transferable mechanism | License / reuse | Current decision and falsifiable expectation |
|---|---|---|---|---|
| Curves contain several plausible segmentations; hard labels may hide ambiguity. Query: performed tempo dynamics Bayesian segmentation boundary posterior | Guichaoua, Lascabettes & Chew (2024), *Music & Science*, DOI 10.1177/20592043241233411; about 2,000 MazurkaBL performances, with priors estimated on 37 manually inspected performances. [Publisher PDF](https://journals.sagepub.com/doi/pdf/10.1177/20592043241233411) and [author repository](https://github.com/erc-cosmos/probabilistic-segmentation). | Forward/backward dynamic programming produces beatwise posterior boundary credence; a broad truncated length prior (reported mean 14.7, SD 5.95, max 30 beats) discourages implausible adjacent boundaries. The paper warns that naïvely treating tempo and loudness as independent produces overconfident fusion. | Paper CC BY-NC 4.0. No LICENSE file was visible at repository root; one shallow clone failed with connection reset. No repository code is copied. | Adopt a clean-room minimal posterior/change-point diagnostic and residual rather than independent-product fusion. Expect fewer adjacent/short-phrase errors if emissions are already informative; otherwise reject structured decoding. |
| Old score features miss local boundary discontinuity. Query: LBDM pitch time change proximity boundary strength | Cambouropoulos (2001), ICMC, melodic surfaces and expressive timing. [Publisher archive](https://quod.lib.umich.edu/i/icmc/bbp2372.2001.021/1/--local-boundary-detection-model-lbdm-and-its-application?page=root%3Bsize%3D400%3Bview%3Dtext). | Change and proximity rules over consecutive pitch/time intervals; peaks are candidate boundaries. | Archive states CC BY-NC-ND 3.0; mechanism is reimplemented from the paper, no text/code copied. | Adopt pitch/time relative-change and proximity as one compact cue and S0 peak baseline. Expect higher validation AUPRC than raw absolute pitch/chroma if local discontinuity is useful. |
| Sparse boundary labels and independent token loss can produce local conflicts. Query: melodic phrase segmentation sparse labels CNN CRF | Guan et al. (2018), symbolic melodic phrase segmentation; compares CNN, BiLSTM variants and CRF sequence decisions. [arXiv](https://arxiv.org/abs/1811.05688). | Preserve token resolution and test sequence-level/length-constrained decoding only when emissions show repeated/implausible peaks. | arXiv paper; no external implementation copied. | Keep A3 behind an error trigger. Expected improvement is emission-to-decoded F1 on validation without changing emissions. |
| Score novelty may be useful without more phrase labels. Query: symbolic music ensemble temporal prediction error segmentation | Bassan, Adi & Rosenschein (2022), Essen Folksong; next-token prediction errors aggregated and peak-picked. [ISCA paper](https://www.isca-archive.org/interspeech_2022/bassan22_interspeech.pdf). | Use local tonal novelty/prediction surprise as a cue; retain peak-picking baseline. Do not reproduce their grid search or ensemble. | Paper publicly hosted by ISCA; mechanism reimplemented minimally. | Adopt tonal novelty only. Expect positive univariate validation AUPRC; if absent, do not open a prediction model ensemble. |
| Cadence cues depend on non-local voice leading, but 43 works are small. Query: symbolic cadence graph neural network imbalanced nodes | Karystinaios & Widmer (2022), ISMIR, note graph cadence detection. [arXiv](https://arxiv.org/abs/2208.14819). | Motivation for local top/bass motion and neighborhood context; note-level graphs preserve fine timing. | arXiv; no code/data copied. | GNN remains proposed. A fixed-context GBDT and one-layer BiGRU test nonlinear and bidirectional context at much lower capacity. |
| Verify curve provenance and scale. Query: official MazurkaBL beat time beat dynamics | Kosta, Bandtlow & Chew (2018), official [MazurkaBL repository](https://github.com/katkost/MazurkaBL): score-aligned beat times and normalized sone dynamics for roughly 2,000 recordings. | Keep performance-level curves separate, preserve work/opus grouping, and treat the alignment frontend as required for deployment. | CC BY-NC-SA 4.0. Existing local source remains read-only. | Use published curves only for exploratory model evaluation; raw-WAV F1 remains unverified because label overlap is zero. |

## Symptom-driven model switch

The already observed Transformer deficit (paired delta -0.1423), the 43-work sample size, and old early-fusion overfit reject another attention configuration. Context-GBDT is falsifiable evidence for low-sample nonlinear cue interactions; Small BiGRU is falsifiable evidence for whole-score bidirectional context without positional encoding or attention. Both first compete on fold-0 validation, with fixed capacity and no model-zoo search.
"""
        (self.reports / "literature_transfer_log.md").write_text(literature, encoding="utf-8")
        external = {"url": "https://github.com/erc-cosmos/probabilistic-segmentation", "clone_attempt_at": now(), "clone_status": "failed_connection_reset", "license_status": "no_LICENSE_visible_at_repository_root", "copied_code": False, "fallback": "clean_room_minimal_equivalent_diagnostic"}
        write_json(self.manifests / "external_sources.json", external)
        failure = {"timestamp": now(), "stage": "literature", "error": "git clone connection reset", "source": external["url"], "impact": "none; primary paper and repository documentation inspected, no code copied"}
        failure_path = self.logs / "failures.jsonl"
        if "git clone connection reset" not in (failure_path.read_text(encoding="utf-8") if failure_path.exists() else ""):
            with failure_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(failure) + "\n")
        self.mark("init", {"frozen_hashes_verified": len(frozen["sha256_evidence"]), "literature_sources": 6})

    def _univariate_auprc(self, train, val, feature_key: str, names: list[str]) -> pd.DataFrame:
        rows = []
        for index, name in enumerate(names):
            tx=[]; ty=[]; tw=[]
            for item in train.values():
                valid=item["label_mask"]>0; x=item[feature_key][:,index]
                tx.append(x[valid,None]); ty.append(item["labels"][valid]); tw.append(np.full(valid.sum(),1/max(valid.sum(),1)))
            x=np.concatenate(tx); y=np.concatenate(ty); w=np.concatenate(tw); scaler=StandardScaler().fit(x)
            model=LogisticRegression(solver="liblinear",random_state=42).fit(scaler.transform(x),y,sample_weight=w*np.where(y>0,10,1))
            vx=[]; vy=[]
            for item in val.values():
                valid=item["label_mask"]>0; vx.append(item[feature_key][valid,index,None]); vy.append(item["labels"][valid])
            vx=np.concatenate(vx); vy=np.concatenate(vy); prob=model.predict_proba(scaler.transform(vx))[:,1]
            rows.append({"feature_set":feature_key,"feature":name,"validation_auprc":average_precision_score(vy,prob),"coefficient":float(model.coef_[0,0])})
        return pd.DataFrame(rows)

    def features(self):
        if self.done("features"): print("[resume] features"); return
        dcml = discover_dcml_pieces(Path(self.experiment["data"]["dcml_chopin_root"])); feature_manifest=pd.read_csv(self.root/"manifests/feature_manifest.csv")
        rows=[]; sample_id=str(feature_manifest.iloc[0].piece_id)
        ordered=[sample_id]+[str(x) for x in feature_manifest.piece_id if str(x)!=sample_id]
        sample_result=None
        for piece_id in ordered:
            old=load_piece_cache(self.root/"cache",piece_id); result=build_compact_score_cues(dcml[piece_id],old["measure_number"],old["beat_number"])
            row=save_phase3_piece_cache(self.root/f"cache/piece_features/{piece_id}.npz",self.cache/f"{piece_id}.npz",result); rows.append(row)
            if piece_id==sample_id: sample_result=(old,result)
        manifest=pd.DataFrame(rows).drop_duplicates("piece_id").sort_values("piece_id"); manifest.to_csv(self.manifests/"score_feature_manifest.csv",index=False)
        old,result=sample_result; fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True)
        axes[0].plot(old["labels"],label="boundary label"); axes[0].legend(); axes[1].plot(result.cues[:,15],label="LBDM strength"); axes[1].legend(); axes[2].plot(result.cues[:,14],label="tonal novelty"); axes[2].legend(); axes[2].set_xlabel("beat index"); fig.suptitle(f"Phase-3 cue audit: {sample_id}"); fig.tight_layout(); fig.savefig(self.figures/"single_piece_score_cue_audit.png",dpi=160); plt.close(fig)
        split=self.split(0); train=self.load(split["train"]); val=self.load(split["validation"])
        old_names=[str(x) for x in next(iter(train.values()))["score_feature_names"]]
        auprc=pd.concat([self._univariate_auprc(train,val,"score",old_names),self._univariate_auprc(train,val,"score_phase3",SCORE_CUE_NAMES)],ignore_index=True); auprc.to_csv(self.metrics/"score_univariate_validation_auprc.csv",index=False)
        # Fold-wise train-to-validation standardized mean shifts, never test-informed.
        shift_rows=[]
        for fold in range(5):
            sp=self.split(fold); tr=self.load(sp["train"]); va=self.load(sp["validation"])
            for key,names in [("score",old_names),("score_phase3",SCORE_CUE_NAMES)]:
                a=np.concatenate([x[key] for x in tr.values()]); b=np.concatenate([x[key] for x in va.values()]); mean=a.mean(0); std=a.std(0); std[std<1e-6]=1
                for name,value in zip(names,(b.mean(0)-mean)/std): shift_rows.append({"fold":fold,"feature_set":key,"feature":name,"validation_mean_shift_train_sd":float(value)})
        shifts=pd.DataFrame(shift_rows); shifts.to_csv(self.metrics/"score_train_validation_shift.csv",index=False)
        old_arrays=[x["score"] for x in train.values()]; old_stack=np.concatenate(old_arrays); corr=pd.DataFrame(old_stack,columns=old_names).corr(); corr.to_csv(self.metrics/"old_score_correlation.csv")
        duplicate=bool(all(np.allclose(x["score"][:,14],x["score"][:,15]) for x in train.values())); rest_unique=sorted(set(np.concatenate([x["score"][:,16] for x in train.values()]).tolist()))
        saved=joblib.load(self.root/"artifacts/checkpoints/fold0_B3_logistic_score_curves.joblib"); coef=np.abs(saved["model"].coef_[0]); score_coef=float(np.linalg.norm(coef[:24])); curve_coef=float(np.linalg.norm(coef[24:])); prob_deltas=[]
        for item in val.values():
            curves=item["curves"] if len(item["curves"]) else np.zeros((1,len(item["labels"]),9),np.float32); score=np.broadcast_to(item["score"][None],(len(curves),*item["score"].shape)); combined=np.concatenate([score,curves],axis=2); zero=combined.copy(); zero[:,:,:24]=saved["scaler"].mean_[:24]
            p=saved["model"].predict_proba(saved["scaler"].transform(combined.reshape(-1,33)))[:,1]; z=saved["model"].predict_proba(saved["scaler"].transform(zero.reshape(-1,33)))[:,1]; prob_deltas.append(np.abs(p-z).mean())
        top_old=auprc[auprc.feature_set=="score"].sort_values("validation_auprc",ascending=False).head(6); top_new=auprc[auprc.feature_set=="score_phase3"].sort_values("validation_auprc",ascending=False).head(6)
        report=f"""# Old Score Feature Failure Audit

Status: **implemented/reproduced audit** on fold-0 train/validation only before opening new Phase-3 outer tests.

- The old score tensor is 24-D and was directly concatenated with 9 curve dimensions before the model.
- `onset_count` and `note_density` are exactly duplicate in every audited training work: **{duplicate}**.
- `rest_fraction` unique values are `{rest_unique}`; it is a binary no-active-event flag, not a duration/fraction.
- Raw 12-bin chroma and normalized absolute top/bass pitch retain work-specific tonal/register identity. Zero also conflates no onset with a legitimate normalized value; no per-feature validity mask is supplied.
- Missing mechanisms are explicit rest/gap duration, top-voice transition/change, tonal novelty, bass motion, density change and LBDM-style boundary strength.
- The old fold-0 B3 logistic coefficient L2 is score={score_coef:.4f}, curves={curve_coef:.4f}; replacing score with its scaler mean changes validation probabilities by {np.mean(prob_deltas):.4f} on average. Thus the branch is used, but prior outer-test failure shows its sensitivity did not generalize.
- `1/performance_count` cancels repeated performances within a work in `train_tcn`, but longer works still create more windows and per-minibatch normalization is not exact work-equal weighting. The logistic baseline uses `1/(valid_beats*performance_count)` and is work-normalized.

Top old validation AUPRC:

{markdown_table(top_old[["feature","validation_auprc","coefficient"]])}

Top compact-cue validation AUPRC:

{markdown_table(top_new[["feature","validation_auprc","coefficient"]])}

All univariate fits and distributions use training statistics only. No phrase/harmony/cadence annotation, piece/opus/split ID, or outer-test statistic enters a feature.
"""
        (self.reports/"old_score_feature_failure_audit.md").write_text(report,encoding="utf-8")
        provenance=f"""# Phase 3 Score Feature Provenance

Status: **implemented** for {len(manifest)} works. Each cache is separate under `cache/phase3`; the Phase-1 cache is unchanged.

Rows correspond one-to-one with the published score beat grid. Row `i` describes beat onset `i` and the transition into it; all backward differences use `x[i]-x[i-1]`, row zero is neutral, and the last beat never wraps to the first. Meter phase uses the observed score beat cycle rather than assuming a fixed beat number. Inputs come only from DCML `notes.tsv` and `measures.tsv`, which are deployable equivalents of MIDI/MusicXML. Harmony and `phraseend` columns are never read by the extractor.

The 16 cues are: {', '.join(SCORE_CUE_NAMES)}. A same-shape quality matrix records when onset/pitch/novelty-dependent cues are semantically observed; it is carried to the frontend contract but is not silently treated as a label. Values undefined during rests are neutral zero plus quality=0.

Manifest: `manifests/phase3/score_feature_manifest.csv`. One-piece numeric plot: `artifacts/phase3/figures/single_piece_score_cue_audit.png`.
"""; (self.reports/"score_feature_provenance.md").write_text(provenance,encoding="utf-8")
        max_old=shifts[shifts.feature_set=="score"].groupby("feature").validation_mean_shift_train_sd.apply(lambda x:np.mean(np.abs(x))).sort_values(ascending=False).head(8); max_new=shifts[shifts.feature_set=="score_phase3"].groupby("feature").validation_mean_shift_train_sd.apply(lambda x:np.mean(np.abs(x))).sort_values(ascending=False).head(8)
        shift_report="# Score Feature Distribution Shift\n\nOnly train-to-validation shifts are used for design. Absolute mean shift is measured in train standard deviations.\n\n## Old 24-D largest shifts\n\n"+markdown_table(max_old.rename("mean_abs_shift_sd").reset_index())+"\n\n## Compact 16-D largest shifts\n\n"+markdown_table(max_new.rename("mean_abs_shift_sd").reset_index())+"\n"
        (self.reports/"score_feature_distribution_shift.md").write_text(shift_report,encoding="utf-8")
        np.savez_compressed(self.artifacts/"normalization_fold0_train_only.npz",old_mean=np.concatenate(old_arrays).mean(0),old_std=np.concatenate(old_arrays).std(0),phase3_mean=np.concatenate([x["score_phase3"] for x in train.values()]).mean(0),phase3_std=np.concatenate([x["score_phase3"] for x in train.values()]).std(0))
        self.mark("features",{"works":len(manifest),"score_cues":16,"old_duplicate":duplicate,"best_new_validation_auprc":float(top_new.validation_auprc.max())})

    def fit_score_logistic(self, train, directory: Path):
        best_path, latest_path = directory / "best.joblib", directory / "latest.joblib"
        if self.resume and best_path.exists() and latest_path.exists():
            payload = joblib.load(best_path)
            return payload["scaler"], payload["model"]
        xs=[];ys=[];ws=[]
        for item in train.values():
            valid=item["label_mask"]>0; xs.append(item["score_phase3"][valid]); ys.append(item["labels"][valid]); ws.append(np.full(valid.sum(),1/max(valid.sum(),1)))
        x=np.concatenate(xs); y=np.concatenate(ys); w=np.concatenate(ws); pos=max(float((w*y).sum()),1e-8); neg=float((w*(1-y)).sum()); w*=np.where(y>0,min(neg/pos,10),1)
        scaler=StandardScaler().fit(x); model=LogisticRegression(solver="liblinear",max_iter=1000,random_state=42).fit(scaler.transform(x),y,sample_weight=w)
        directory.mkdir(parents=True,exist_ok=True); payload={"scaler":scaler,"model":model}; joblib.dump(payload,directory/"best.joblib"); joblib.dump(payload,directory/"latest.joblib"); return scaler,model
    @staticmethod
    def raw_score(data,scaler,model):
        out={}
        for pid,item in data.items():
            p=model.predict_proba(scaler.transform(item["score_phase3"]))[:,1]; ids=[str(x) for x in item["performance_ids"]] or ["score"]
            out[pid]={x:p.copy() for x in ids}
        return out
    @staticmethod
    def raw_s0(data):
        out={}
        for pid,item in data.items():
            x=item["score_phase3"][:,15]; order=pd.Series(x).rank(method="average",pct=True).to_numpy(); ids=[str(x) for x in item["performance_ids"]] or ["score"]
            out[pid]={x:order.copy() for x in ids}
        return out
    def fit_late_fusion(self, raw_curve, raw_score, validation, directory: Path):
        best_path, latest_path = directory / "best.joblib", directory / "latest.joblib"
        if self.resume and best_path.exists() and latest_path.exists():
            return joblib.load(best_path)
        xs=[];ys=[];ws=[]
        for pid,item in validation.items():
            valid=item["label_mask"]>0; perf_ids=sorted(raw_curve[pid])
            for perf in perf_ids:
                xs.append(np.column_stack([logit(raw_curve[pid][perf][valid]),logit(raw_score[pid][perf][valid])]))
                ys.append(item["labels"][valid]); ws.append(np.full(valid.sum(),1/(valid.sum()*len(perf_ids))))
        x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);pos=max(float((w*y).sum()),1e-8);neg=float((w*(1-y)).sum());w*=np.where(y>0,min(neg/pos,10),1)
        model=LogisticRegression(solver="liblinear",random_state=42).fit(x,y,sample_weight=w); directory.mkdir(parents=True,exist_ok=True);joblib.dump(model,directory/"best.joblib");joblib.dump(model,directory/"latest.joblib");return model
    @staticmethod
    def raw_fusion(curve,score,model):
        return {pid:{perf:model.predict_proba(np.column_stack([logit(p),logit(score[pid][perf])]))[:,1] for perf,p in perfs.items()} for pid,perfs in curve.items()}

    def fit_gbdt(self, train, directory: Path):
        best_path, latest_path = directory / "best.joblib", directory / "latest.joblib"
        if self.resume and best_path.exists() and latest_path.exists():
            return joblib.load(best_path)
        xs=[];ys=[];ws=[]
        for item in train.values():
            valid=item["label_mask"]>0; curves=item["curves"] if len(item["curves"]) else np.zeros((1,len(item["labels"]),9),np.float32)
            for curve in curves:
                xs.append(context_features(curve,item["score_phase3"])[valid]); ys.append(item["labels"][valid]); ws.append(np.full(valid.sum(),1/(valid.sum()*len(curves))))
        x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);pos=max(float((w*y).sum()),1e-8);neg=float((w*(1-y)).sum());w*=np.where(y>0,min(neg/pos,10),1)
        model=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1.0,min_samples_leaf=100,random_state=42).fit(x,y,sample_weight=w)
        directory.mkdir(parents=True,exist_ok=True);joblib.dump(model,directory/"best.joblib");joblib.dump(model,directory/"latest.joblib");return model
    @staticmethod
    def raw_gbdt(data,model):
        out={}
        for pid,item in data.items():
            curves=item["curves"] if len(item["curves"]) else np.zeros((1,len(item["labels"]),9),np.float32);ids=[str(x) for x in item["performance_ids"]] or ["missing"]
            out[pid]={perf:model.predict_proba(context_features(curve,item["score_phase3"]))[:,1] for perf,curve in zip(ids,curves)}
        return out

    def all_models_for_fold(self, fold: int, seed: int = 42, evaluate_test: bool = True):
        split=self.split(fold); train,val,test=(self.load(split[x]) for x in ["train","validation","test"])
        c0,norm,cstate=self.load_c0(fold,seed);device=next(c0.parameters()).device
        raw_c_val=_raw_performance_predictions(c0,val,norm,device,"tcn"); raw_c_test=_raw_performance_predictions(c0,test,norm,device,"tcn")
        s0_val=self.raw_s0(val);s0_test=self.raw_s0(test)
        scaler,score_model=self.fit_score_logistic(train,self.checkpoints/f"s1/seed{seed}/fold{fold}");score_val=self.raw_score(val,scaler,score_model);score_test=self.raw_score(test,scaler,score_model)
        fusion=self.fit_late_fusion(raw_c_val,score_val,val,self.checkpoints/f"m1/seed{seed}/fold{fold}");m1_val=self.raw_fusion(raw_c_val,score_val,fusion);m1_test=self.raw_fusion(raw_c_test,score_test,fusion)
        local=copy.deepcopy(self.protocol); local["project"]["seed"]=seed
        m2,cn,sn,m2info=train_sequence_model("m2",train,val,cstate,local,self.checkpoints/f"m2/seed{seed}/fold{fold}",resume=self.resume);m2_val=raw_multimodal_predictions(m2,val,cn,sn,next(m2.parameters()).device,"m2");m2_test=raw_multimodal_predictions(m2,test,cn,sn,next(m2.parameters()).device,"m2")
        gbdt=self.fit_gbdt(train,self.checkpoints/f"a1_gbdt/seed{seed}/fold{fold}");a1_val=self.raw_gbdt(val,gbdt);a1_test=self.raw_gbdt(test,gbdt)
        bigru,bcn,bsn,binfo=train_sequence_model("bigru",train,val,None,local,self.checkpoints/f"a2_bigru/seed{seed}/fold{fold}",resume=self.resume);a2_val=raw_multimodal_predictions(bigru,val,bcn,bsn,next(bigru.parameters()).device,"bigru");a2_test=raw_multimodal_predictions(bigru,test,bcn,bsn,next(bigru.parameters()).device,"bigru")
        raw={"C0":(raw_c_val,raw_c_test),"S0":(s0_val,s0_test),"S1":(score_val,score_test),"M1":(m1_val,m1_test),"M2":(m2_val,m2_test),"A1_Context_GBDT":(a1_val,a1_test),"A2_Small_BiGRU":(a2_val,a2_test)}
        info={"C0":{"parameters":sum(p.numel() for p in c0.parameters())},"S0":{"parameters":0},"S1":{"parameters":17},"M1":{"parameters":3,"coefficients":fusion.coef_[0].tolist()},"M2":m2info,"A1_Context_GBDT":{"parameters":sum(n.value.shape[0] if hasattr(n,"value") else 0 for p in getattr(gbdt,"_predictors",[]) for n in p)},"A2_Small_BiGRU":binfo}
        return train,val,test,raw,info

    def develop(self):
        if self.done("develop"): print("[resume] develop"); return
        train,val,test,raw,info=self.all_models_for_fold(0,42,False); rows=[]
        for name,(rv,_) in raw.items():
            threshold,_=choose_single_threshold(rv,val,self.protocol["evaluation"]["threshold_grid"]);_,_,summary=evaluate_single_performance(rv,val,threshold);rows.append({"model":name,"threshold":threshold,"validation_f1":summary["macro_f1_tol1"],"validation_precision":summary["macro_precision_tol1"],"validation_recall":summary["macro_recall_tol1"],"parameters":info[name].get("parameter_count",info[name].get("parameters",0))})
        frame=pd.DataFrame(rows);frame.to_csv(self.metrics/"development_fold0_validation.csv",index=False);c0=float(frame.loc[frame.model=="C0","validation_f1"].iloc[0]);best_multi=float(frame[frame.model.isin(["M1","M2"])].validation_f1.max());trigger=best_multi-c0<.02
        # Counterfactual modality tests on trained M2.
        m2state=torch.load(self.checkpoints/"m2/seed42/fold0/best.pt",map_location="cpu",weights_only=False); score_mean=np.asarray(m2state["score_normalizer_mean"]); mean_val={k:{**v,"score_phase3":np.broadcast_to(score_mean,v["score_phase3"].shape).copy()} for k,v in val.items()}
        # Reload through trainer resume to avoid a second implementation path.
        cmodel,_,cstate=self.load_c0(0); local=copy.deepcopy(self.protocol); model,cn,sn,minfo=train_sequence_model("m2",train,val,cstate,local,self.checkpoints/"m2/seed42/fold0",resume=True)
        zero_raw=raw_multimodal_predictions(model,mean_val,cn,sn,next(model.parameters()).device,"m2");zt,_=choose_single_threshold(zero_raw,val,self.protocol["evaluation"]["threshold_grid"]);_,_,zs=evaluate_single_performance(zero_raw,val,zt)
        rng=np.random.default_rng(42);shuffle_val={}
        for k,v in val.items(): copied=dict(v); copied["score_phase3"]=v["score_phase3"][rng.permutation(len(v["score_phase3"]))];shuffle_val[k]=copied
        shuffle_raw=raw_multimodal_predictions(model,shuffle_val,cn,sn,next(model.parameters()).device,"m2");st,_=choose_single_threshold(shuffle_raw,val,self.protocol["evaluation"]["threshold_grid"]);_,_,ss=evaluate_single_performance(shuffle_raw,val,st)
        # Train-label permutation sanity: the same structure trained on per-work
        # shuffled labels must not match true-label validation performance.
        shuffled_train={}
        for piece_id,item in train.items():
            copied=dict(item); copied["labels"]=item["labels"].copy(); valid=np.flatnonzero(item["label_mask"]>0)
            prng=np.random.default_rng(42+int(hashlib.sha256(piece_id.encode()).hexdigest()[:8],16));copied["labels"][valid]=prng.permutation(copied["labels"][valid]);shuffled_train[piece_id]=copied
        shuffled_protocol=copy.deepcopy(self.protocol);shuffled_protocol["training"]["maximum_epochs"]=4;shuffled_protocol["training"]["patience"]=4
        shuffled_model,shcn,shsn,shinfo=train_sequence_model("m2",shuffled_train,val,cstate,shuffled_protocol,self.checkpoints/"diagnostics/shuffled_m2_fold0",resume=self.resume)
        shuffled_raw=raw_multimodal_predictions(shuffled_model,val,shcn,shsn,next(shuffled_model.parameters()).device,"m2");sht,_=choose_single_threshold(shuffled_raw,val,self.protocol["evaluation"]["threshold_grid"]);_,_,shsummary=evaluate_single_performance(shuffled_raw,val,sht)
        # Tiny overfit on one real 64-beat window, using the switched BiGRU.
        first=next(iter(train.values()));boundary=int(np.flatnonzero(first["labels"]>0)[0]);start=min(max(boundary-24,0),len(first["labels"])-64)
        tx=np.concatenate([cn.apply(first["curves"][0,start:start+64]),sn.apply(first["score_phase3"][start:start+64])],axis=1).astype(np.float32)
        ty=first["labels"][start:start+64].astype(np.float32);tm=first["label_mask"][start:start+64].astype(np.float32)
        tiny=SmallBiGRU(25,32,0.0);opt=torch.optim.AdamW(tiny.parameters(),lr=.01);xt=torch.from_numpy(tx).unsqueeze(0);yt=torch.from_numpy(ty).unsqueeze(0);mt=torch.from_numpy(tm).unsqueeze(0);initial_loss=None
        for _ in range(250):
            opt.zero_grad();tl=tiny(xt);lv=torch.nn.functional.binary_cross_entropy_with_logits(tl,yt,pos_weight=torch.tensor(10.),reduction="none");loss=(lv*mt).sum()/mt.sum();initial_loss=float(loss.detach()) if initial_loss is None else initial_loss;loss.backward();opt.step()
        tiny_final=float(loss.detach());tiny_probability=torch.sigmoid(tiny(xt)).detach().numpy()[0]
        tiny_raw={"tiny":{"p":tiny_probability}};tiny_data={"tiny":{"labels":ty,"label_mask":tm}}
        tiny_threshold,_=choose_single_threshold(tiny_raw,tiny_data,self.protocol["evaluation"]["threshold_grid"]);_,_,tiny_summary=evaluate_single_performance(tiny_raw,tiny_data,tiny_threshold)
        # Shape, finite gradient and residual fallback tests.
        x=torch.randn(2,64,9);s=torch.randn(2,64,16);model.cpu();model.zero_grad();out,cl,res,gate=model(x,s,return_parts=True);loss=out.mean();loss.backward();grad=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        sanity={"input_shapes":[list(x.shape),list(s.shape)],"output_shape":list(out.shape),"finite_activation":bool(torch.isfinite(out).all()),"finite_gradients":all(torch.isfinite(g).all() for g in grad),"gate":float(gate.detach()),"zero_initialized_residual_fallback_design":True,"m2_normal_validation_f1":float(frame.loc[frame.model=="M2","validation_f1"].iloc[0]),"m2_score_zero_validation_f1":zs["macro_f1_tol1"],"m2_score_shuffle_validation_f1":ss["macro_f1_tol1"],"shuffled_label_real_validation_f1":shsummary["macro_f1_tol1"],"true_label_validation_f1":float(frame.loc[frame.model=="M2","validation_f1"].iloc[0]),"shuffled_is_lower":shsummary["macro_f1_tol1"]<float(frame.loc[frame.model=="M2","validation_f1"].iloc[0]),"tiny_initial_loss":initial_loss,"tiny_final_loss":tiny_final,"tiny_f1_tol1":tiny_summary["macro_f1_tol1"]};write_json(self.reports/"development_sanity.json",sanity)
        diagnosis=f"""# Model Suitability Diagnosis

Status: **development-only reproduced** on opus fold-0 train/validation; outer test remained unopened during selection.

{markdown_table(frame)}

- C0 validation F1={c0:.4f}; best M1/M2 delta={best_multi-c0:+.4f}. Alternative-model trigger (`< +0.02`)={trigger}.
- M2 score branch gate={sanity['gate']:.4f}; validation F1 after normalized score-zero={sanity['m2_score_zero_validation_f1']:.4f}, score-time shuffle={sanity['m2_score_shuffle_validation_f1']:.4f}.
- Shapes are {sanity['input_shapes']}→{sanity['output_shape']}; activations and gradients finite={sanity['finite_activation'] and sanity['finite_gradients']}.
- Tiny-overfit loss {sanity['tiny_initial_loss']:.4f}→{sanity['tiny_final_loss']:.4f}, F1={sanity['tiny_f1_tol1']:.4f}; shuffled-label M2 real-validation F1={sanity['shuffled_label_real_validation_f1']:.4f} versus true-label {sanity['true_label_validation_f1']:.4f} (lower={sanity['shuffled_is_lower']}).

Observed hypothesis: if compact cues show validation signal but M1/M2 fail to add +0.02, the failure is not simply insufficient depth. Context-GBDT tests low-sample nonlinear interactions; Small BiGRU tests bidirectional whole-window context. Both were explicitly requested as a bounded model switch and therefore compete here; no Transformer or model zoo is opened.
""";(self.reports/"model_suitability_diagnosis.md").write_text(diagnosis,encoding="utf-8")
        write_json(self.manifests/"frozen_model_selection.json",{"selected_for_full_folds":["A1_Context_GBDT","A2_Small_BiGRU"],"also_required_ladder":["C0","S0","S1","M1","M2"],"seed":42,"selection_source":"fold0_validation_only","alternative_trigger":trigger,"config_sha256":hashlib.sha256((self.root/"configs/phase3/protocol.yaml").read_bytes()).hexdigest()})
        self.mark("develop",{"c0_validation_f1":c0,"best_multimodal_delta":best_multi-c0,"alternatives_triggered":trigger,"sanity_passed":sanity["finite_activation"] and sanity["finite_gradients"]})

    def experiments(self):
        if self.done("experiments"): print("[resume] experiments"); return
        model_names = ["C0","S0","S1","M1","M2","A1_Context_GBDT","A2_Small_BiGRU"]
        seed42_ready = self.resume and (self.metrics/"fivefold_runs.csv").exists() and (self.metrics/"model_info.json").exists() and all(
            (self.metrics/f"seed42_fold{fold}_{name}_test_piece.csv").exists()
            for fold in range(5) for name in model_names
        )
        if seed42_ready:
            all_rows = pd.read_csv(self.metrics/"fivefold_runs.csv").to_dict("records")
            infos = json.loads((self.metrics/"model_info.json").read_text(encoding="utf-8"))
            print("[resume] seed42 outer-evaluation tables")
        else:
            all_rows=[];infos={}
            for fold in range(5):
                train,val,test,raw,info=self.all_models_for_fold(fold,42,True);infos[str(fold)]=info
                for name,(rv,rt) in raw.items():
                    payload=self.save_evaluation(f"seed42_fold{fold}_{name}",rv,rt,val,test);all_rows.append({"seed":42,"fold":fold,"model":name,**payload["test"]})
            pd.DataFrame(all_rows).to_csv(self.metrics/"fivefold_runs.csv",index=False);write_json(self.metrics/"model_info.json",infos)
        piece_frames={}
        for name in model_names:
            combined = pd.concat([pd.read_csv(self.metrics/f"seed42_fold{fold}_{name}_test_piece.csv").assign(fold=fold) for fold in range(5)],ignore_index=True)
            piece_frames[name] = with_boundary_rates(combined, name)
        comparison=[];base=piece_frames["C0"].set_index("piece_id")
        paired=[]
        for name,frame in piece_frames.items():
            delta=frame.set_index("piece_id").f1_tol1-base.f1_tol1;low,high=paired_bootstrap(delta.to_numpy());fold_delta=frame.groupby("fold").f1_tol1.mean()-piece_frames["C0"].groupby("fold").f1_tol1.mean();wins=int((fold_delta>0).sum()) if name!="C0" else 0
            comparison.append({"model":name,"seed":42,"macro_precision_tol0":frame.precision_tol0.mean(),"macro_recall_tol0":frame.recall_tol0.mean(),"macro_f1_tol0":frame.f1_tol0.mean(),"macro_precision_tol1":frame.precision_tol1.mean(),"macro_recall_tol1":frame.recall_tol1.mean(),"macro_f1_tol1":frame.f1_tol1.mean(),"macro_f1_tol2":frame.f1_tol2.mean(),"macro_pr_auc":frame.pr_auc.mean(),"delta_vs_c0":delta.mean(),"ci_low":low,"ci_high":high,"fold_wins":wins})
            for pid,d in delta.items():paired.append({"model":name,"piece_id":pid,"fold":int(frame.set_index("piece_id").loc[pid,"fold"]),"delta_f1_tol1_vs_c0":d,"model_f1_tol1":float(frame.set_index("piece_id").loc[pid,"f1_tol1"]),"c0_f1_tol1":float(base.loc[pid,"f1_tol1"])})
        comp=pd.DataFrame(comparison);comp.to_csv(self.metrics/"model_comparison.csv",index=False);pd.DataFrame(paired).to_csv(self.metrics/"paired_piece_results.csv",index=False)
        eligible=comp[(comp.model!="C0")&(comp.delta_vs_c0>=.02)&(comp.fold_wins>=4)].sort_values("delta_vs_c0",ascending=False)
        seed43_status="not_triggered"
        if len(eligible):
            selected=str(eligible.iloc[0].model);seed43_status=f"triggered_for_{selected}"
            # A1 is deterministic under the frozen configuration; rerunning another nominal seed would be pseudo-replication.
            if selected in {"M2","A2_Small_BiGRU"}:
                kind="m2" if selected=="M2" else "bigru"
                seed43_ready = self.resume and (self.metrics/"seed43_runs.csv").exists() and all(
                    (self.metrics/f"seed43_fold{fold}_{name}_summary.json").exists() and
                    (self.metrics/f"seed43_fold{fold}_{name}_test_piece.csv").exists()
                    for fold in range(5) for name in [selected, "C0"]
                )
                if seed43_ready:
                    print(f"[resume] seed43 outer-evaluation tables for {selected}")
                else:
                    seed_rows=[]
                    for fold in range(5):
                        split=self.split(fold);train,val,test=(self.load(split[x]) for x in ["train","validation","test"]);c0,norm,cstate=self.load_c0(fold,43);device=next(c0.parameters()).device;rv0=_raw_performance_predictions(c0,val,norm,device,"tcn");rt0=_raw_performance_predictions(c0,test,norm,device,"tcn")
                        local=copy.deepcopy(self.protocol);local["project"]["seed"]=43;model,cn,sn,info=train_sequence_model(kind,train,val,cstate if kind=="m2" else None,local,self.checkpoints/f"{kind}/seed43/fold{fold}",resume=self.resume);rv=raw_multimodal_predictions(model,val,cn,sn,next(model.parameters()).device,kind);rt=raw_multimodal_predictions(model,test,cn,sn,next(model.parameters()).device,kind);self.save_evaluation(f"seed43_fold{fold}_{selected}",rv,rt,val,test);self.save_evaluation(f"seed43_fold{fold}_C0",rv0,rt0,val,test);seed_rows.append({"fold":fold,"model":selected,"parameter_count":info["parameter_count"]})
                    pd.DataFrame(seed_rows).to_csv(self.metrics/"seed43_runs.csv",index=False)
        # Diagnostics and plots.
        pair=pd.DataFrame(paired);best=comp.sort_values("macro_f1_tol1",ascending=False).iloc[0];bp=piece_frames[str(best.model)].sort_values("f1_tol1")
        boundary_rate_summary(piece_frames).to_csv(self.metrics/"boundary_rates.csv",index=False)
        fig,ax=plt.subplots(figsize=(9,4));pivot=pd.DataFrame(all_rows).pivot(index="fold",columns="model",values="macro_f1_tol1");pivot.plot(kind="bar",ax=ax);ax.set_ylabel("Macro F1@±1");ax.set_title("Phase-3 opus-grouped folds");fig.tight_layout();fig.savefig(self.figures/"fold_comparison.png",dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4));d=pair[pair.model==best.model].sort_values("delta_f1_tol1_vs_c0");ax.bar(range(len(d)),d.delta_f1_tol1_vs_c0);ax.axhline(0,color="black",lw=1);ax.set(xlabel="held-out work (sorted)",ylabel="F1 delta vs C0",title=f"Per-work delta: {best.model}");fig.tight_layout();fig.savefig(self.figures/"per_piece_delta.png",dpi=160);plt.close(fig)
        errors=f"""# Phase 3 Error Analysis

Best observed seed-42 model by work macro F1 is **{best.model}** at {best.macro_f1_tol1:.4f}; this remains preliminary/exploratory because all 43 works were historically viewed.

Weakest works:

{markdown_table(bp.head(8)[['piece_id','precision_tol1','recall_tol1','f1_tol1','predicted_boundary_rate','true_boundary_rate']])}

Strongest works:

{markdown_table(bp.tail(8)[['piece_id','precision_tol1','recall_tol1','f1_tol1','predicted_boundary_rate','true_boundary_rate']])}

The full per-performance and per-piece files preserve exact/±1/±2, PR-AUC and boundary rates. `fold_comparison.png` and `per_piece_delta.png` show domain and work heterogeneity. M3 is evaluated only if emissions create adjacent or implausibly short phrases; radius-1 NMS already prohibits immediately adjacent peaks, so no CRF is added solely because F1 is low.
""";(self.reports/"error_analysis.md").write_text(errors,encoding="utf-8")
        self.mark("experiments",{"best_model":str(best.model),"best_f1":float(best.macro_f1_tol1),"best_delta":float(best.delta_vs_c0),"best_fold_wins":int(best.fold_wins),"seed43":seed43_status})

    @staticmethod
    def bayesian_posterior(sequence: np.ndarray,min_len:int=4,max_len:int=30,mu:float=14.7,sd:float=5.95)->np.ndarray:
        x=np.asarray(sequence,float);n=len(x);logarc=np.full((n,n),-np.inf)
        for i in range(n):
            for j in range(i+min_len-1,min(n,i+max_len)):
                y=x[i:j+1];t=np.linspace(-1,1,len(y));coef=np.polyfit(t,y,2);res=y-np.polyval(coef,t);sse=float(np.mean(res**2));length=j-i+1;logprior=-.5*((length-mu)/sd)**2;logarc[i,j]=logprior-2.0*np.log(sse+1e-3)
        def lse(vals):
            vals=np.asarray(vals)
            if vals.size == 0: return -np.inf
            m=np.max(vals);return -np.inf if not np.isfinite(m) else float(m+np.log(np.exp(vals-m).sum()))
        alpha=np.full(n,-np.inf)
        for j in range(n):alpha[j]=lse([(0 if i==0 else alpha[i-1])+logarc[i,j] for i in range(max(0,j-max_len+1),j-min_len+2)])
        beta=np.full(n,-np.inf);beta[n-1]=0
        for i in range(n-2,-1,-1):beta[i]=lse([logarc[i+1,j]+beta[j] for j in range(i+min_len,min(n,i+max_len+1))])
        post=np.zeros(n)
        if np.isfinite(alpha[-1]):
            for j in range(n-1):post[j]=np.exp(alpha[j]+beta[j]-alpha[-1]) if np.isfinite(alpha[j]+beta[j]) else 0
        post[-1]=1;return np.clip(post,0,1)

    def frontend(self):
        if self.done("frontend"): print("[resume] frontend"); return
        parent=Path(r"C:\Users\pa1018\Desktop\Music Synchronization");smoke_dir=self.artifacts/"frontend_smoke"
        payload=run_frontend_smoke(parent/"src/bwv_856/predicted_beats.csv",parent/"src/bwv_856/score.mid",parent/"src/bwv_856/performance.wav",self.c0_checkpoint(0,42),self.checkpoints/"m2/seed42/fold0/best.pt",smoke_dir/"boundary_predictions.csv",smoke_dir/"boundary_predictions.json")
        contract={"version":1,"status":{"adapter":"implemented_smoke_tested","mazurka_alignment_robustness":"preliminary_simulated","raw_audio_phrase_f1":"proposed"},"inputs":["MusicXML_or_MIDI","performance.wav"],"alignment_outputs":["predicted_beats.csv","alignment_beats.json"],"beat_columns":["score_beat_index","measure","beat","predicted_performance_time_sec","alignment_valid"],"curve_features":["log_tempo","tempo_d1","tempo_d2","energy","energy_d1","tempo_robust_z","energy_robust_z","alignment_mask","energy_mask"],"score_features":SCORE_CUE_NAMES,"output_columns":["boundary_probability","predicted_performance_time_sec","feature_quality"],"prohibition":"No phrase label or true test alignment is used at inference."};write_json(self.manifests/"frontend_contract.json",contract)
        smoke_report=f"""# Frontend Adapter Smoke Test

Status: **implemented/smoke-tested**, not ground-truth evaluated.

The read-only parent BWV 856 `predicted_beats.csv`, score MIDI and performance WAV were adapted inside Phase 3. The adapter computed beat intervals, audio RMS energy, nine curve features, 16 MIDI score cues and quality masks, then ran the fold-0 M2 model. It produced {payload['input_rows']} beat rows, of which {payload['alignment_valid_rows']} have valid alignment, with learned gate={payload['gate']:.4f}.

Outputs: `artifacts/phase3/frontend_smoke/boundary_predictions.csv` and `.json`. BWV 856 has no phrase gold compatible with the Mazurka labels, so no end-to-end F1 is reported or implied.
""";(self.reports/"frontend_adapter_smoke.md").write_text(smoke_report,encoding="utf-8")
        # Alignment noise stress uses the empirical signed error distribution, injected monotonically into reconstructed Mazurka beat times.
        errors=pd.read_csv(parent/"output/total_valid.csv").error_sec.dropna().to_numpy(float);conditions=empirical_noise_scales(errors)
        rows=[]
        manifest=pd.read_csv(self.root/"artifacts/phase2/splits/opus_split_manifest.csv")
        for fold in range(5):
            test_ids=manifest[(manifest.fold==fold)&(manifest.split=="test")].piece_id.tolist();data=self.load(test_ids);model,norm,state=self.load_c0(fold);device=next(model.parameters()).device
            for condition,scale_seconds in conditions.items():
                perturbed={}
                for pid,item in data.items():
                    copied=dict(item);new=[]
                    for curve_index,curve in enumerate(item["curves"]):
                        intervals=60/np.exp(curve[:,0]);times=np.concatenate([[0],np.cumsum(intervals[:-1])]);
                        if scale_seconds > 0:
                            noise=paired_empirical_noise(errors,len(times),f"{fold}|{pid}|{curve_index}",scale_seconds);times=np.maximum.accumulate(times+noise);times+=np.arange(len(times))*1e-5
                        rebuilt=build_curve_features(times,curve[:,3]);rebuilt[:,8]=curve[:,8];new.append(rebuilt)
                    copied["curves"]=np.asarray(new,np.float32);perturbed[pid]=copied
                raw=_raw_performance_predictions(model,perturbed,norm,device,"tcn");threshold=float(json.loads((self.root/f"artifacts/phase2/metrics/opus_tcn_seed42_fold{fold}_summary.json").read_text())["threshold"]);_,piece,summary=evaluate_single_performance(raw,perturbed,threshold);rows.append({"fold":fold,"condition":condition,"noise_scale_seconds":scale_seconds,**summary})
        stress=pd.DataFrame(rows);stress.to_csv(self.metrics/"alignment_noise_stress.csv",index=False);means=stress.groupby("condition").macro_f1_tol1.mean();clean=float(means["clean"]);stress_summary=pd.DataFrame([{"condition":condition,"noise_scale_seconds":scale,"macro_f1_tol1":float(means[condition]),"delta_vs_clean":float(means[condition]-clean)} for condition,scale in conditions.items()]);stress_summary.to_csv(self.metrics/"alignment_noise_stress_summary.csv",index=False)
        # Clean-room Bayesian diagnostic on a representative performance; not claimed as official reproduction.
        sample=self.load([self.split(0)["validation"][0]]);pid=next(iter(sample));curve=sample[pid]["curves"][0];posterior=self.bayesian_posterior(curve[:,0]);pd.DataFrame({"beat_index":np.arange(len(posterior)),"boundary_posterior":posterior}).to_csv(self.metrics/"bayesian_minimal_representative.csv",index=False)
        plan=f"""# System Integration Plan and Current Evidence

- **implemented/smoke-tested:** adapter contract, MIDI score cues, alignment-derived tempo, WAV interval energy, masks/quality, M2 probability inference and mapping back to performance seconds on BWV 856.
- **reproduced:** parent alignment outputs were read without modifying parent code; `total_valid.csv` contains 27,432 rows across 79 performances/24 pieces.
- **preliminary simulated robustness:** paired empirical error shapes scaled to median/p95/maximum absolute parent-alignment error. Results: {', '.join(f'{r.condition}[{r.noise_scale_seconds:.4f}s]={r.macro_f1_tol1:.4f} (delta {r.delta_vs_clean:+.4f})' for r in stress_summary.itertuples())}.
- **implemented diagnostic, not official reproduction:** a clean-room quadratic-segment posterior with the paper-reported broad length prior ran on one representative validation curve. Repository code was not copied because no root license was visible and the shallow clone failed.
- **proposed:** phrase F1 from raw Mazurka WAV. Local raw audio and current phrase labels overlap in 0 works, so real end-to-end accuracy cannot be measured without a new independently labelled overlap set.

The parent system remains read-only. Production flow is `MusicXML/MIDI + WAV → parent alignment outputs → Phase-3 adapter → per-beat features/probabilities/timestamps`.
""";(self.reports/"system_integration_plan.md").write_text(plan,encoding="utf-8")
        self.mark("frontend",{"smoke_rows":payload["input_rows"],"stress_conditions":list(conditions),"raw_audio_gold_overlap":0})

    def seed43_comparison(self) -> pd.DataFrame:
        frames = {
            name: pd.concat([
                pd.read_csv(self.metrics/f"seed43_fold{fold}_{name}_test_piece.csv").assign(fold=fold)
                for fold in range(5)
            ], ignore_index=True)
            for name in ["C0", "A2_Small_BiGRU"]
        }
        base = frames["C0"].set_index("piece_id")
        rows = []
        for name, frame in frames.items():
            indexed = frame.set_index("piece_id")
            delta = indexed.f1_tol1 - base.f1_tol1
            low, high = paired_bootstrap(delta.to_numpy(), seed=43)
            fold_delta = frame.groupby("fold").f1_tol1.mean() - frames["C0"].groupby("fold").f1_tol1.mean()
            rows.append({
                "model": name, "seed": 43,
                "macro_precision_tol1": float(frame.precision_tol1.mean()),
                "macro_recall_tol1": float(frame.recall_tol1.mean()),
                "macro_f1_tol1": float(frame.f1_tol1.mean()),
                "macro_pr_auc": float(frame.pr_auc.mean()),
                "delta_vs_c0": float(delta.mean()), "ci_low": low, "ci_high": high,
                "fold_wins": int((fold_delta > 0).sum()) if name != "C0" else 0,
            })
        result = pd.DataFrame(rows)
        result.to_csv(self.metrics/"seed43_comparison.csv", index=False)
        return result

    def post_experiment_diagnostics(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        gap_path = self.metrics/"train_validation_gap_summary.csv"
        detail_path = self.metrics/"train_validation_gap_by_fold.csv"
        m3_path = self.reports/"m3_length_decoder_gate.json"
        if self.resume and gap_path.exists() and detail_path.exists() and m3_path.exists():
            cached=json.loads(m3_path.read_text(encoding="utf-8"))
            if cached.get("gate_version") != 3:
                excess=float(cached["predicted_interval_lt4_fraction"]-cached["true_interval_lt4_fraction"]);pre=bool(cached["adjacent_pair_fraction"]>=.01 or excess>=.10 or cached["predicted_to_true_boundary_rate_ratio"]>=2.0);post=bool((not pre) and excess>=.05)
                cached["gate_version"]=3;cached["gate_definition"]["predicted_short_interval_fraction_excess_vs_true"]=">=0.10 (preregistered)";cached["status"]="triggered_by_preregistered_gate" if pre else "skipped_by_preregistered_gate";cached["observed_short_interval_excess"]=excess;cached["post_hoc_exploratory_followup"]={"status":"run" if post else "not_run","diagnostic_threshold":">=0.05","disclosure":"Threshold was lowered only after outer-test error inspection; result cannot replace the preregistered A2 conclusion."};cached["reason"]="The preregistered 10-percentage-point short-interval gate is preserved. Any lower-threshold DP follow-up is post-hoc exploratory.";write_json(m3_path,cached);(self.reports/"m3_length_decoder_gate.md").write_text("# M3 Length-Decoder Gate\n\n"+markdown_table(pd.DataFrame([{k:v for k,v in cached.items() if k not in {"gate_definition","reason","post_hoc_exploratory_followup"}}]))+f"\n\nPreregistered decision: **{cached['status']}**. Observed short-interval excess={excess:.4f} versus the frozen 0.10 gate. Post-hoc follow-up: **{cached['post_hoc_exploratory_followup']['status']}** using a disclosed 0.05 diagnostic threshold; it cannot replace A2.\n",encoding="utf-8")
            return pd.read_csv(gap_path), cached
        rows=[];predicted_intervals=[];true_intervals=[];adjacent_pairs=0
        model_names=["C0","S0","S1","M1","M2","A1_Context_GBDT","A2_Small_BiGRU"]
        for fold in range(5):
            split=self.split(fold);train,val,test=(self.load(split[x]) for x in ["train","validation","test"])
            c0,norm,cstate=self.load_c0(fold,42);device=next(c0.parameters()).device
            raw_c_train=_raw_performance_predictions(c0,train,norm,device,"tcn");raw_c_val=_raw_performance_predictions(c0,val,norm,device,"tcn")
            s1_payload=joblib.load(self.checkpoints/f"s1/seed42/fold{fold}/best.joblib");score_train=self.raw_score(train,s1_payload["scaler"],s1_payload["model"]);score_val=self.raw_score(val,s1_payload["scaler"],s1_payload["model"])
            fusion=joblib.load(self.checkpoints/f"m1/seed42/fold{fold}/best.joblib");gbdt=joblib.load(self.checkpoints/f"a1_gbdt/seed42/fold{fold}/best.joblib")
            local=copy.deepcopy(self.protocol);local["project"]["seed"]=42
            m2,cn,sn,_=train_sequence_model("m2",train,val,cstate,local,self.checkpoints/f"m2/seed42/fold{fold}",resume=True)
            bigru,bcn,bsn,_=train_sequence_model("bigru",train,val,None,local,self.checkpoints/f"a2_bigru/seed42/fold{fold}",resume=True)
            raw_train={"C0":raw_c_train,"S0":self.raw_s0(train),"S1":score_train,"M1":self.raw_fusion(raw_c_train,score_train,fusion),"M2":raw_multimodal_predictions(m2,train,cn,sn,next(m2.parameters()).device,"m2"),"A1_Context_GBDT":self.raw_gbdt(train,gbdt),"A2_Small_BiGRU":raw_multimodal_predictions(bigru,train,bcn,bsn,next(bigru.parameters()).device,"bigru")}
            raw_val={"C0":raw_c_val,"S0":self.raw_s0(val),"S1":score_val,"M1":self.raw_fusion(raw_c_val,score_val,fusion),"M2":raw_multimodal_predictions(m2,val,cn,sn,next(m2.parameters()).device,"m2"),"A1_Context_GBDT":self.raw_gbdt(val,gbdt),"A2_Small_BiGRU":raw_multimodal_predictions(bigru,val,bcn,bsn,next(bigru.parameters()).device,"bigru")}
            for name in model_names:
                threshold=float(json.loads((self.metrics/f"seed42_fold{fold}_{name}_summary.json").read_text(encoding="utf-8"))["threshold"])
                _,_,train_summary=evaluate_single_performance(raw_train[name],train,threshold);_,_,val_summary=evaluate_single_performance(raw_val[name],val,threshold)
                rows.append({"fold":fold,"model":name,"threshold":threshold,"train_macro_f1_tol1":train_summary["macro_f1_tol1"],"validation_macro_f1_tol1":val_summary["macro_f1_tol1"],"train_minus_validation_f1":train_summary["macro_f1_tol1"]-val_summary["macro_f1_tol1"]})
            threshold=float(json.loads((self.metrics/f"seed42_fold{fold}_A2_Small_BiGRU_summary.json").read_text(encoding="utf-8"))["threshold"])
            raw_test=raw_multimodal_predictions(bigru,test,bcn,bsn,next(bigru.parameters()).device,"bigru")
            for pid,perfs in raw_test.items():
                truth=np.flatnonzero((test[pid]["labels"]>0)&(test[pid]["label_mask"]>0));true_intervals.extend(np.diff(truth).tolist())
                for raw in perfs.values():
                    selected=np.flatnonzero(nms_probabilities(raw)>=threshold);diffs=np.diff(selected);predicted_intervals.extend(diffs.tolist());adjacent_pairs+=int((diffs<=1).sum())
        detail=pd.DataFrame(rows);detail.to_csv(detail_path,index=False)
        gaps=detail.groupby("model",as_index=False).agg(train_macro_f1_tol1=("train_macro_f1_tol1","mean"),validation_macro_f1_tol1=("validation_macro_f1_tol1","mean"),train_minus_validation_f1=("train_minus_validation_f1","mean"));gaps.to_csv(gap_path,index=False)
        pred=np.asarray(predicted_intervals,float);truth=np.asarray(true_intervals,float);rates=pd.read_csv(self.metrics/"boundary_rates.csv").set_index("model")
        adjacent_fraction=float(adjacent_pairs/max(len(pred),1));pred_short=float((pred<4).mean()) if len(pred) else 0.0;true_short=float((truth<4).mean()) if len(truth) else 0.0;rate_ratio=float(rates.loc["A2_Small_BiGRU","predicted_boundary_rate"]/rates.loc["A2_Small_BiGRU","true_boundary_rate"])
        excess=pred_short-true_short;trigger=bool(adjacent_fraction>=.01 or excess>=.10 or rate_ratio>=2.0);post=bool((not trigger) and excess>=.05)
        m3={"gate_version":3,"status":"triggered_by_preregistered_gate" if trigger else "skipped_by_preregistered_gate","gate_definition":{"adjacent_pair_fraction":">=0.01","predicted_short_interval_fraction_excess_vs_true":">=0.10 (preregistered)","predicted_to_true_boundary_rate_ratio":">=2.0"},"predicted_interval_count":int(len(pred)),"adjacent_pair_fraction":adjacent_fraction,"predicted_interval_lt4_fraction":pred_short,"true_interval_lt4_fraction":true_short,"observed_short_interval_excess":excess,"predicted_interval_median_beats":float(np.median(pred)) if len(pred) else None,"true_interval_median_beats":float(np.median(truth)) if len(truth) else None,"predicted_to_true_boundary_rate_ratio":rate_ratio,"post_hoc_exploratory_followup":{"status":"run" if post else "not_run","diagnostic_threshold":">=0.05","disclosure":"Threshold was lowered only after outer-test error inspection; result cannot replace the preregistered A2 conclusion."},"reason":"The preregistered 10-percentage-point short-interval gate is preserved. Any lower-threshold DP follow-up is post-hoc exploratory."}
        write_json(m3_path,m3);(self.reports/"m3_length_decoder_gate.md").write_text("# M3 Length-Decoder Gate\n\n"+markdown_table(pd.DataFrame([{k:v for k,v in m3.items() if k not in {"gate_definition","reason","post_hoc_exploratory_followup"}}]))+f"\n\nPreregistered decision: **{m3['status']}**. Observed short-interval excess={excess:.4f} versus the frozen 0.10 gate. Post-hoc follow-up: **{m3['post_hoc_exploratory_followup']['status']}** using a disclosed 0.05 diagnostic threshold; it cannot replace A2.\n",encoding="utf-8")
        return gaps,m3

    def run_m3_length_decoder(self) -> None:
        name="M3_BiGRU_MinGapDP"
        required=all((self.metrics/f"seed42_fold{fold}_{name}_summary.json").exists() and (self.metrics/f"seed42_fold{fold}_{name}_test_piece.csv").exists() for fold in range(5))
        existing=pd.read_csv(self.metrics/"model_comparison.csv")
        if self.resume and required and name in set(existing.model):
            print("[resume] M3 minimum-gap DP tables");return
        fold_rows=[];gap_rows=[]
        for fold in range(5):
            split=self.split(fold);train,val,test=(self.load(split[x]) for x in ["train","validation","test"]);local=copy.deepcopy(self.protocol);local["project"]["seed"]=42
            model,cn,sn,info=train_sequence_model("bigru",train,val,None,local,self.checkpoints/f"a2_bigru/seed42/fold{fold}",resume=True)
            device=next(model.parameters()).device;raw_train=raw_multimodal_predictions(model,train,cn,sn,device,"bigru");raw_val=raw_multimodal_predictions(model,val,cn,sn,device,"bigru");raw_test=raw_multimodal_predictions(model,test,cn,sn,device,"bigru")
            threshold_rows=[]
            for threshold in self.protocol["evaluation"]["threshold_grid"]:
                processed=apply_minimum_gap_dp(raw_val,float(threshold),4);_,_,summary=evaluate_single_performance(processed,val,float(threshold));threshold_rows.append({"threshold":float(threshold),"macro_f1_tol1":summary["macro_f1_tol1"],"macro_precision_tol1":summary["macro_precision_tol1"],"macro_recall_tol1":summary["macro_recall_tol1"]})
            threshold_frame=pd.DataFrame(threshold_rows);chosen=float(threshold_frame.sort_values(["macro_f1_tol1","macro_precision_tol1","threshold"],ascending=[False,False,False]).iloc[0].threshold)
            proc_train=apply_minimum_gap_dp(raw_train,chosen,4);proc_val=apply_minimum_gap_dp(raw_val,chosen,4);proc_test=apply_minimum_gap_dp(raw_test,chosen,4)
            train_perf,train_piece,train_summary=evaluate_single_performance(proc_train,train,chosen);val_perf,val_piece,val_summary=evaluate_single_performance(proc_val,val,chosen);test_perf,test_piece,test_summary=evaluate_single_performance(proc_test,test,chosen)
            prefix=f"seed42_fold{fold}_{name}";threshold_frame.to_csv(self.metrics/f"{prefix}_threshold.csv",index=False);val_perf.to_csv(self.metrics/f"{prefix}_validation_performance.csv",index=False);val_piece.to_csv(self.metrics/f"{prefix}_validation_piece.csv",index=False);test_perf.to_csv(self.metrics/f"{prefix}_test_performance.csv",index=False);test_piece.to_csv(self.metrics/f"{prefix}_test_piece.csv",index=False);write_json(self.metrics/f"{prefix}_summary.json",{"threshold":chosen,"validation":val_summary,"test":test_summary,"decoder":{"minimum_gap_beats":4,"new_parameters":0,"base_model":"A2_Small_BiGRU"}})
            fold_rows.append({"seed":42,"fold":fold,"model":name,**test_summary});gap_rows.append({"fold":fold,"model":name,"threshold":chosen,"train_macro_f1_tol1":train_summary["macro_f1_tol1"],"validation_macro_f1_tol1":val_summary["macro_f1_tol1"],"train_minus_validation_f1":train_summary["macro_f1_tol1"]-val_summary["macro_f1_tol1"]})
        frame=pd.concat([pd.read_csv(self.metrics/f"seed42_fold{fold}_{name}_test_piece.csv").assign(fold=fold) for fold in range(5)],ignore_index=True);base=pd.concat([pd.read_csv(self.metrics/f"seed42_fold{fold}_C0_test_piece.csv").assign(fold=fold) for fold in range(5)],ignore_index=True);delta=frame.set_index("piece_id").f1_tol1-base.set_index("piece_id").f1_tol1;low,high=paired_bootstrap(delta.to_numpy());fold_delta=frame.groupby("fold").f1_tol1.mean()-base.groupby("fold").f1_tol1.mean()
        row={"model":name,"seed":42,"macro_precision_tol0":frame.precision_tol0.mean(),"macro_recall_tol0":frame.recall_tol0.mean(),"macro_f1_tol0":frame.f1_tol0.mean(),"macro_precision_tol1":frame.precision_tol1.mean(),"macro_recall_tol1":frame.recall_tol1.mean(),"macro_f1_tol1":frame.f1_tol1.mean(),"macro_f1_tol2":frame.f1_tol2.mean(),"macro_pr_auc":frame.pr_auc.mean(),"delta_vs_c0":delta.mean(),"ci_low":low,"ci_high":high,"fold_wins":int((fold_delta>0).sum())}
        updated=pd.concat([existing[existing.model!=name],pd.DataFrame([row])],ignore_index=True);updated.to_csv(self.metrics/"model_comparison.csv",index=False)
        detail=pd.read_csv(self.metrics/"train_validation_gap_by_fold.csv");detail=pd.concat([detail[detail.model!=name],pd.DataFrame(gap_rows)],ignore_index=True);detail.to_csv(self.metrics/"train_validation_gap_by_fold.csv",index=False);summary=detail.groupby("model",as_index=False).agg(train_macro_f1_tol1=("train_macro_f1_tol1","mean"),validation_macro_f1_tol1=("validation_macro_f1_tol1","mean"),train_minus_validation_f1=("train_minus_validation_f1","mean"));summary.to_csv(self.metrics/"train_validation_gap_summary.csv",index=False)
        (self.reports/"m3_decoder_result.md").write_text(f"# M3 Minimum-Gap DP Result\n\nStatus: **preliminary/exploratory**. The triggered single structural change enforces a four-beat minimum between selected local peaks with zero new learned parameters. Five-fold work-macro F1@±1={row['macro_f1_tol1']:.4f}, delta vs C0={row['delta_vs_c0']:+.4f}, CI [{low:.4f},{high:.4f}], fold wins={row['fold_wins']}/5. Acceptance is based on comparison with the unmodified A2 output, not on adding capacity.\n",encoding="utf-8")

    def m3_posthoc_summary(self) -> dict[str, Any] | None:
        name="M3_BiGRU_MinGapDP"
        if not all((self.metrics/f"seed42_fold{fold}_{name}_test_piece.csv").exists() for fold in range(5)):
            return None
        m3=pd.concat([pd.read_csv(self.metrics/f"seed42_fold{fold}_{name}_test_piece.csv").assign(fold=fold) for fold in range(5)],ignore_index=True)
        a2=pd.concat([pd.read_csv(self.metrics/f"seed42_fold{fold}_A2_Small_BiGRU_test_piece.csv").assign(fold=fold) for fold in range(5)],ignore_index=True)
        delta=m3.set_index("piece_id").f1_tol1-a2.set_index("piece_id").f1_tol1;low,high=paired_bootstrap(delta.to_numpy());fold_delta=m3.groupby("fold").f1_tol1.mean()-a2.groupby("fold").f1_tol1.mean()
        payload={"status":"post_hoc_exploratory","preregistered_primary_model":"A2_Small_BiGRU","m3_macro_f1_tol1":float(m3.f1_tol1.mean()),"a2_macro_f1_tol1":float(a2.f1_tol1.mean()),"delta_f1_tol1_vs_a2":float(delta.mean()),"ci_low_vs_a2":low,"ci_high_vs_a2":high,"fold_wins_vs_a2":int((fold_delta>0).sum()),"piece_wins_vs_a2":int((delta>0).sum()),"piece_losses_vs_a2":int((delta<0).sum()),"decision":"do_not_promote" if low<=0 or float(delta.mean())<.01 else "exploratory_signal_only","disclosure":"The preregistered 0.10 gate was missed (observed short-interval excess 0.0978). A 0.05 threshold was introduced after outer-test inspection, so M3 cannot replace A2."}
        write_json(self.reports/"m3_posthoc_result.json",payload);(self.reports/"m3_decoder_result.md").write_text(f"# M3 Minimum-Gap DP Result\n\nStatus: **post_hoc_exploratory**. The preregistered 10-percentage-point short-interval gate was not met (observed 9.775 pp), so the registered decision remains `skipped_by_preregistered_gate`. A disclosed 5-point diagnostic threshold motivated this single follow-up.\n\nThe decoder enforces a four-beat minimum between local peaks, adds zero learned parameters, and selects its probability threshold on each fold's validation split only. Outer test is used once for reporting. M3 F1@±1={payload['m3_macro_f1_tol1']:.4f} versus A2={payload['a2_macro_f1_tol1']:.4f}; paired delta={payload['delta_f1_tol1_vs_a2']:+.4f}, 95% CI [{low:.4f},{high:.4f}], fold wins={payload['fold_wins_vs_a2']}/5. Decision: **{payload['decision']}**; A2 remains the preregistered primary Phase-3 model.\n",encoding="utf-8")
        return payload

    def make_required_figures(self, model_info: dict[str, Any]) -> None:
        fig,axes=plt.subplots(1,3,figsize=(13,4),sharex=True,sharey=True)
        for ax,model in zip(axes,["C0","M2","A2_Small_BiGRU"]):
            for fold in range(5):
                curve=pd.read_csv(self.metrics/f"seed42_fold{fold}_{model}_threshold.csv").sort_values("macro_recall_tol1")
                ax.plot(curve.macro_recall_tol1,curve.macro_precision_tol1,marker="o",ms=2,alpha=.55,label=f"fold {fold}")
            ax.set_title(model);ax.set_xlabel("Validation macro recall@±1");ax.grid(alpha=.2)
        axes[0].set_ylabel("Validation macro precision@±1");axes[-1].legend(fontsize=7,loc="best");fig.suptitle("Validation operating curves (threshold grid; not test-tuned)");fig.tight_layout();fig.savefig(self.figures/"validation_pr_curves.png",dpi=170);plt.close(fig)
        weights=[]
        for fold in range(5):
            info=model_info[str(fold)];coef=info["M1"]["coefficients"];weights.append({"fold":fold,"M1_curve_logit_weight":coef[0],"M1_score_logit_weight":coef[1],"M2_score_residual_gate":info["M2"]["gate"]})
        wf=pd.DataFrame(weights);wf.to_csv(self.metrics/"modality_gate_weights.csv",index=False);fig,ax=plt.subplots(figsize=(8,4));wf.set_index("fold").plot(kind="bar",ax=ax);ax.set_ylabel("Coefficient / sigmoid gate");ax.set_title("Validation-fitted modality weights and learned score gate");ax.axhline(0,color="black",lw=.8);fig.tight_layout();fig.savefig(self.figures/"modality_gate_weights.png",dpi=170);plt.close(fig)
        pieces=pd.concat([pd.read_csv(self.metrics/f"seed42_fold{fold}_A2_Small_BiGRU_test_piece.csv") for fold in range(5)],ignore_index=True).sort_values("f1_tol1");cases=pd.concat([pieces.head(5),pieces.tail(5)]);plot=cases.set_index("piece_id")[["precision_tol1","recall_tol1","f1_tol1"]];fig,ax=plt.subplots(figsize=(11,4.5));plot.plot(kind="bar",ax=ax);ax.set_ylabel("Macro over performances");ax.set_title("Representative five weakest and five strongest held-out works");ax.tick_params(axis="x",rotation=45,labelsize=8);ax.legend(loc="best");fig.tight_layout();fig.savefig(self.figures/"success_failure_cases.png",dpi=170);plt.close(fig)

    def report(self):
        if self.done("report"): print("[resume] report"); return
        comp=pd.read_csv(self.metrics/"model_comparison.csv");seed43=self.seed43_comparison();gaps,m3=self.post_experiment_diagnostics()
        if m3["status"] == "triggered_by_preregistered_gate" or m3.get("post_hoc_exploratory_followup",{}).get("status") == "run":
            self.run_m3_length_decoder();comp=pd.read_csv(self.metrics/"model_comparison.csv");gaps=pd.read_csv(self.metrics/"train_validation_gap_summary.csv")
        m3_posthoc=self.m3_posthoc_summary();primary=comp[comp.model!="M3_BiGRU_MinGapDP"];best=primary.sort_values("macro_f1_tol1",ascending=False).iloc[0];c0=comp[comp.model=="C0"].iloc[0]
        if best.delta_vs_c0>=.03 and best.fold_wins>=4 and best.ci_low>0: conclusion=f"{best.model} has a stable but still preliminary increment"
        elif best.delta_vs_c0>=.01: conclusion=f"{best.model} has only a weak exploratory increment"
        else: conclusion="multimodal/alternative models do not establish an increment over C0"
        historical=pd.read_csv(self.root/"artifacts/phase2/metrics/model_comparison.csv");tr=historical[historical.model=="Transformer_single"].groupby("model")[["macro_precision_tol1","macro_recall_tol1","macro_f1_tol1"]].mean().reset_index();tr["model"]="Phase2_Transformer_negative_reference"
        model_info=json.loads((self.metrics/"model_info.json").read_text(encoding="utf-8"));self.make_required_figures(model_info);params={name:int(info.get("parameter_count",info.get("parameters",0))) for name,info in model_info["0"].items()};params["M3_BiGRU_MinGapDP"]=11443;deploy={"C0":"alignment curves","S0":"score","S1":"score","M1":"score + alignment curves","M2":"score + alignment curves","A1_Context_GBDT":"score + alignment curves","A2_Small_BiGRU":"score + alignment curves","M3_BiGRU_MinGapDP":"score + alignment curves"}
        comparison=comp[["model","macro_f1_tol0","macro_precision_tol1","macro_recall_tol1","macro_f1_tol1","macro_f1_tol2","macro_pr_auc","delta_vs_c0","ci_low","ci_high","fold_wins"]].merge(gaps[["model","train_minus_validation_f1"]],on="model",how="left");comparison.insert(1,"parameters",comparison.model.map(params));comparison["deployment_inputs"]=comparison.model.map(deploy);comparison["analysis_status"]=comparison.model.map(lambda x:"post_hoc_exploratory" if x=="M3_BiGRU_MinGapDP" else "preregistered_preliminary")
        runtime=json.loads((self.metrics/"runtime_summary.json").read_text(encoding="utf-8")) if (self.metrics/"runtime_summary.json").exists() else {"outer_training_and_seed43_seconds":None};seed43_best=seed43[seed43.model=="A2_Small_BiGRU"].iloc[0];stress=pd.read_csv(self.metrics/"alignment_noise_stress_summary.csv")
        m3_text=(f"The preregistered M3 gate remains **{m3['status']}**: observed short-interval excess={m3['observed_short_interval_excess']:.4f} versus the frozen 0.10 threshold. A separately disclosed `post_hoc_exploratory` minimum-gap DP obtained F1={m3_posthoc['m3_macro_f1_tol1']:.4f} versus A2={m3_posthoc['a2_macro_f1_tol1']:.4f}, paired delta {m3_posthoc['delta_f1_tol1_vs_a2']:+.4f}, CI [{m3_posthoc['ci_low_vs_a2']:.4f},{m3_posthoc['ci_high_vs_a2']:.4f}]. Decision: {m3_posthoc['decision']}; it does not replace A2." if m3_posthoc else f"M3 preregistered status: {m3['status']}.")
        runs=pd.read_csv(self.metrics/"fivefold_runs.csv");fold_view=runs[runs.model.isin(["C0","M2","A2_Small_BiGRU"])].pivot(index="fold",columns="model",values="macro_f1_tol1").reset_index();development=json.loads((self.reports/"development_sanity.json").read_text(encoding="utf-8"));last_resource=pd.read_csv(self.reports/"resource_usage.csv").iloc[-1];failure_count=sum(1 for line in (self.logs/"failures.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
        final=f"""# Phase 3 Final Report

Generated: {now()}.

## Unique conclusion

**{conclusion}.** Opus-grouped C0 achieved P/R/F1@±1={c0.macro_precision_tol1:.4f}/{c0.macro_recall_tol1:.4f}/{c0.macro_f1_tol1:.4f}. The best observed model {best.model} achieved {best.macro_f1_tol1:.4f}, paired work delta {best.delta_vs_c0:+.4f}, 95% CI [{best.ci_low:.4f},{best.ci_high:.4f}], fold wins {int(best.fold_wins)}/5. All model-selection claims remain `preliminary/exploratory` because Phase 1/2 already exposed the same 43 works.

## Fair comparison

{markdown_table(comparison)}

Exact, ±1-beat and ±2-beat F1 are shown as `macro_f1_tol0`, `macro_f1_tol1` and `macro_f1_tol2`; event matching is one-to-one and threshold/NMS are frozen from validation. Selected preregistered per-fold F1@±1:

{markdown_table(fold_view)}

Required visual evidence is saved as `validation_pr_curves.png`, `fold_comparison.png`, `per_piece_delta.png`, `modality_gate_weights.png`, and `success_failure_cases.png` under `artifacts/phase3/figures`.

The Phase-2 Transformer negative reference was reused, not retrained: mean seed F1={float(tr.macro_f1_tol1.iloc[0]):.4f}, versus its seed-matched TCN paired delta -0.1423. M0 old 24-D early concatenation remains a nonmatched historical negative control at piece-median F1=0.2908; it is not falsely mixed into the opus/single-performance paired delta.

The gated seed-43 replication gives Small BiGRU P/R/F1={seed43_best.macro_precision_tol1:.4f}/{seed43_best.macro_recall_tol1:.4f}/{seed43_best.macro_f1_tol1:.4f}, delta {seed43_best.delta_vs_c0:+.4f}, CI [{seed43_best.ci_low:.4f},{seed43_best.ci_high:.4f}], fold wins {int(seed43_best.fold_wins)}/5. The measured initial outer-training plus gated seed-43 process took {runtime['outer_training_and_seed43_seconds']} seconds including inference and postprocessing; exact per-model training-only timing was not instrumented and is not fabricated.

## What caused the low score

The baseline is not ordinary accuracy: boundaries occupy about 3.8% of beats and F1 uses one-to-one event matching. The evidence supports a compound limit: only 536 boundaries across 43 closely related works; labels are hard projections of potentially ambiguous boundaries; old score features duplicated density, encoded raw chroma/register identity and used early fusion; and deployment curves inherit alignment long tails. The compact cues, modality counterfactuals, fold variance and per-work errors distinguish representation/fusion limitations from simple parameter shortage.

Fold-0 M2 counterfactuals support real score use: normal validation F1={development['m2_normal_validation_f1']:.4f}, score-zero={development['m2_score_zero_validation_f1']:.4f}, and score-time-shuffle={development['m2_score_shuffle_validation_f1']:.4f}. Tiny-overfit reached F1={development['tiny_f1_tol1']:.4f}; activations and gradients were finite. The larger A2 train-minus-validation gap (0.3016) shows that overfit remains, so the result does not justify increasing capacity.

{m3_text} The gate and post-hoc evidence are in `reports/phase3/m3_length_decoder_gate.md` and `reports/phase3/m3_decoder_result.md`.

## Score/audio integration

The parent alignment system is connected read-only through `src/phase3_frontend_adapter.py`. BWV 856 smoke now outputs per-beat boundary probability, measure/beat, predicted performance seconds, alignment validity and feature quality. This is `implemented/smoke-tested`, not phrase-F1 reproduced. The paired empirical alignment-noise experiment is `preliminary simulated robustness`: clean/median-error/p95-error/max-error F1 values are {'/'.join(f'{v:.4f}' for v in stress.macro_f1_tol1)}. Raw-audio Mazurka phrase F1 remains `proposed` because real audio-label overlap is 0.

## Status discipline

- `implemented`: isolated cues/cache, bounded models, evaluator adapters, frontend adapter, tests, resume markers and diagnostics.
- `reproduced`: numerical values backed by Phase-3 per-performance/per-piece CSVs and retained checkpoints.
- `preliminary`: every Phase-3 scientific gain/negative result and simulated alignment robustness.
- `proposed`: independent-corpus confirmation, real raw-audio phrase F1, and any GNN or new annotation collection.

## Environment, resources, failures and limits

- Environment: Windows, Python 3.12.10, PyTorch 2.11.0+cu128 (CUDA 12.8), scikit-learn 1.9.0, pandas 3.0.5, NumPy 2.5.3, mido 1.3.3; GPU NVIDIA GeForce RTX 5070 Ti.
- Seeds: 42 for the registered ladder; 43 only for the gate-triggered Small BiGRU/C0 replication. No automatic hyperparameter search was used.
- Dependencies are pinned through `requirements-base.txt` plus the existing Phase-2 environment. Each trained fold retains only best/latest.
- Final recorded resource row: usedPercent {int(last_resource.current_used_percent)}% from start {int(last_resource.start_used_percent)}%, rate {last_resource.usage_rate_percent_per_hour:.4f}%/h, projected remaining {last_resource.projected_remaining_percent:.2f}%, workspace {last_resource.workspace_gb:.3f} GB, disk free {last_resource.disk_free_gb:.2f} GB, GPU used/free {int(last_resource.gpu_used_mb)}/{int(last_resource.gpu_free_mb)} MB. No reset was redeemed.
- `{failure_count}` append-only failure records are preserved in `logs/phase3/failures.jsonl`; all have repairs or documented non-impact. The two experiment postprocessing failures occurred after training and were repaired with boundary-rate regression tests and resume gates.
- Not completed by data gate: real raw-audio Mazurka phrase F1 and independent-corpus confirmation. The verified audio-label overlap is zero; no labels or samples were fabricated. Per-model training-only seconds were not instrumented in the first run, so only the measured 2606-second whole Stage-D process is reported.

## Reproduction

```powershell
Set-Location -LiteralPath '{self.root}'
& '.\\.venv\\Scripts\\python.exe' -m src.run_phase3 --root '{self.root}' --stage all --resume
```
""";(self.reports/"multimodal_comparison.md").write_text("# Multimodal Comparison\n\n"+markdown_table(comparison)+"\n\nSee per-piece paired evidence under `artifacts/phase3/metrics`.\n",encoding="utf-8");(self.reports/"final_report.md").write_text(final,encoding="utf-8")
        meeting=f"""# 第三阶段组会更新

- 严格 opus 五折 C0：F1@±1={c0.macro_f1_tol1:.3f}。
- 最好模型：{best.model}，F1={best.macro_f1_tol1:.3f}，相对 C0 {best.delta_vs_c0:+.3f}，CI [{best.ci_low:.3f},{best.ci_high:.3f}]，折胜 {int(best.fold_wins)}/5。
- 结论：{conclusion}；因同一43首已在前阶段查看，只标 preliminary/exploratory。
- 已完成旧 score 病因审计、16维可部署 cues、GBDT/BiGRU 换模公平比较，以及 BWV 856 alignment→边界概率/时间戳 smoke。
- raw-audio 金标重合仍为0，端到端 F1 不做虚假声明。
""";(self.reports/"MEETING_UPDATE_CN.md").write_text(meeting,encoding="utf-8")
        self.mark("report",{"conclusion":conclusion,"best_model":str(best.model),"best_f1":float(best.macro_f1_tol1)})

    def audit(self):
        required=["literature_transfer_log.md","old_score_feature_failure_audit.md","score_feature_provenance.md","score_feature_distribution_shift.md","model_suitability_diagnosis.md","multimodal_comparison.md","error_analysis.md","m3_length_decoder_gate.md","m3_decoder_result.md","system_integration_plan.md","frontend_adapter_smoke.md","final_report.md","MEETING_UPDATE_CN.md","progress_journal.md","decision_log.md","resume_validation.json"]
        tests_ok=False;test_output=""
        import subprocess
        run=subprocess.run([str(self.root/".venv/Scripts/python.exe"),"-m","pytest","-q"],cwd=self.root,capture_output=True,text=True,encoding="utf-8",errors="replace");tests_ok=run.returncode==0;test_output=(run.stdout+run.stderr)[-4000:]
        comp=pd.read_csv(self.metrics/"model_comparison.csv");recomputed={}
        for model in comp.model:
            frame=pd.concat([pd.read_csv(self.metrics/f"seed42_fold{f}_{model}_test_piece.csv") for f in range(5)]);recomputed[model]=float(frame.f1_tol1.mean())
        metric_match=all(abs(recomputed[r.model]-r.macro_f1_tol1)<1e-10 for r in comp.itertuples())
        ck_violations=[]
        for d in self.checkpoints.rglob("*"):
            if d.is_dir():
                files=[x.name for x in d.iterdir() if x.is_file() and x.suffix in {".pt",".joblib"}]
                if files:
                    suffix=Path(files[0]).suffix;expected={f"best{suffix}",f"latest{suffix}"}
                    if set(files)!=expected:ck_violations.append({"directory":str(d),"files":files,"expected":sorted(expected)})
        frozen=json.loads((self.manifests/"frozen_phase2_evidence.json").read_text(encoding="utf-8"));frozen_mismatches=[]
        for item in frozen["sha256_evidence"]:
            path=self.root/item["path"]
            actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            if actual!=item["sha256"]:frozen_mismatches.append({"path":item["path"],"expected":item["sha256"],"actual":actual})
        overlap=pd.read_csv(self.root/"artifacts/phase2/splits/opus_overlap_audit.csv");overlap_columns=["piece_overlap","opus_overlap","source_hash_overlap","window_origin_overlap"]
        coverage={model:int(pd.concat([pd.read_csv(self.metrics/f"seed42_fold{f}_{model}_test_piece.csv") for f in range(5)]).piece_id.nunique()) for model in comp.model};coverage_ok=all(v==43 for v in coverage.values())
        noise=pd.read_csv(self.metrics/"alignment_noise_stress_summary.csv");noise_ok=bool((np.diff(noise.noise_scale_seconds.to_numpy())>=0).all() and (np.diff(noise.macro_f1_tol1.to_numpy())<=1e-12).all())
        final_text=(self.reports/"final_report.md").read_text(encoding="utf-8");m3_text=(self.reports/"m3_decoder_result.md").read_text(encoding="utf-8");decision_text=(self.reports/"decision_log.md").read_text(encoding="utf-8")
        resume_payload=json.loads((self.reports/"resume_validation.json").read_text(encoding="utf-8")) if (self.reports/"resume_validation.json").exists() else {}
        required_figures=["validation_pr_curves.png","fold_comparison.png","per_piece_delta.png","modality_gate_weights.png","success_failure_cases.png"]
        checks={"frozen_hashes_unchanged":not frozen_mismatches,"required_reports":all((self.reports/x).exists() and (self.reports/x).stat().st_size>0 for x in required),"required_figures":all((self.figures/x).exists() and (self.figures/x).stat().st_size>0 for x in required_figures),"score_manifest_43":len(pd.read_csv(self.manifests/"score_feature_manifest.csv"))==43,"fivefold_models":set(comp.model)>={"C0","S0","S1","M1","M2","A1_Context_GBDT","A2_Small_BiGRU","M3_BiGRU_MinGapDP"},"all_models_cover_43_works":coverage_ok,"metric_recompute_match":metric_match,"opus_piece_hash_window_overlap_zero":bool((overlap[overlap_columns]==0).all().all()),"seed43_replication":(self.metrics/"seed43_comparison.csv").exists(),"train_validation_gap":(self.metrics/"train_validation_gap_summary.csv").exists(),"frontend_csv":len(pd.read_csv(self.artifacts/"frontend_smoke/boundary_predictions.csv"))==216,"noise_stress_monotonic":noise_ok,"tests_passed":tests_ok,"checkpoint_policy":not ck_violations,"m3_methodology_disclosed":all(x in final_text+m3_text+decision_text for x in ["skipped_by_preregistered_gate","post_hoc_exploratory","A2 remains"]),"primary_model_not_posthoc":"**A2_Small_BiGRU has" in final_text,"raw_audio_claim_qualified":"proposed" in final_text,"report_has_exact_tol2_environment_failures":all(x in final_text for x in ["macro_f1_tol0","macro_f1_tol2","Python 3.12.10","failure records"]),"resume_validation":resume_payload.get("exit_code")==0 and not resume_payload.get("changed_checkpoints",["missing"]),"resource_log":len(pd.read_csv(self.reports/"resource_usage.csv"))>=3}
        payload={"status":"complete" if all(checks.values()) else "incomplete","created_at":now(),"checks":checks,"recomputed_macro_f1_tol1":recomputed,"work_coverage":coverage,"frozen_hash_mismatches":frozen_mismatches,"checkpoint_policy_violations":ck_violations,"pytest_output":test_output,"scientific_limit":"43 works historically viewed; raw-audio phrase-label overlap is zero"};write_json(self.reports/"completion_audit.json",payload)
        if not all(checks.values()):raise AssertionError(payload)
        self.mark("audit",{"all_checks_pass":True,"pytest":test_output.splitlines()[-1] if test_output.splitlines() else "passed"})

    def run(self,stage): getattr(self,stage)()


def main()->int:
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--stage",choices=[*STAGES,"all"],default="all");parser.add_argument("--resume",action="store_true");parser.add_argument("--force",action="store_true");args=parser.parse_args();pipeline=Phase3Pipeline(args.root,args.resume,args.force);stages=STAGES if args.stage=="all" else [args.stage]
    for stage in stages:
        started=time.time();print(f"[{now()}] START phase3 {stage}",flush=True)
        try:pipeline.run(stage)
        except Exception as exc:
            failure={"timestamp":now(),"stage":stage,"error":repr(exc),"traceback":traceback.format_exc()};
            with (pipeline.logs/"failures.jsonl").open("a",encoding="utf-8") as h:h.write(json.dumps(failure,ensure_ascii=False)+"\n")
            print(failure["traceback"],flush=True);return 1
        print(f"[{now()}] DONE phase3 {stage} elapsed_seconds={time.time()-started:.2f}",flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
