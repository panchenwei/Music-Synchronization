from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .models import BeatBoundaryTCN, Normalizer, seed_everything
from .phase2_models import choose_single_threshold, evaluate_single_performance


def fit_train_normalizer(arrays: list[np.ndarray]) -> Normalizer:
    flat = np.concatenate([x.reshape(-1, x.shape[-1]) for x in arrays], axis=0)
    mean = np.nanmean(flat, axis=0).astype(np.float32)
    std = np.nanstd(flat, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return Normalizer(mean, std)


class MultiModalWindowDataset(Dataset):
    def __init__(self, data: dict[str, dict[str, np.ndarray]], curve_norm: Normalizer, score_norm: Normalizer, window: int, stride: int):
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
        for item in data.values():
            curves = item["curves"] if len(item["curves"]) else np.zeros((1, len(item["labels"]), 9), dtype=np.float32)
            score = score_norm.apply(item["score_phase3"]).astype(np.float32)
            weight = 1.0 / len(curves)
            length = len(item["labels"])
            starts = list(range(0, max(length - window + 1, 1), stride))
            if not starts or starts[-1] != max(0, length - window):
                starts.append(max(0, length - window))
            for curve in curves:
                curve = curve_norm.apply(curve).astype(np.float32)
                for start in sorted(set(starts)):
                    stop = start + window
                    c, s = curve[start:stop], score[start:stop]
                    y = item["labels"][start:stop].astype(np.float32)
                    m = item["label_mask"][start:stop].astype(np.float32)
                    valid = np.ones(len(y), dtype=np.float32)
                    if len(y) < window:
                        pad = window - len(y)
                        c = np.pad(c, ((0, pad), (0, 0)))
                        s = np.pad(s, ((0, pad), (0, 0)))
                        y = np.pad(y, (0, pad)); m = np.pad(m, (0, pad)); valid = np.pad(valid, (0, pad))
                    self.samples.append((c, s, y, m, valid, weight))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class ResidualGatedFusionTCN(nn.Module):
    """Frozen curve TCN plus a low-capacity score residual.

    With ``gate -> 0`` or a zero residual head, output exactly equals the
    established curve logit.  This makes degradation diagnosable and reversible.
    """

    def __init__(self, curve_model: BeatBoundaryTCN, score_dim: int = 16, score_hidden: int = 12):
        super().__init__()
        self.curve = curve_model
        for parameter in self.curve.parameters():
            parameter.requires_grad = False
        self.score_input = nn.Conv1d(score_dim, score_hidden, 1)
        self.score_block = nn.Sequential(
            nn.Conv1d(score_hidden, score_hidden, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(score_hidden, score_hidden, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.score_output = nn.Conv1d(score_hidden, 1, 1)
        nn.init.zeros_(self.score_output.weight)
        nn.init.zeros_(self.score_output.bias)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, curves: torch.Tensor, score: torch.Tensor, return_parts: bool = False):
        # ``model.train()`` must not update the frozen curve BatchNorm buffers.
        self.curve.eval()
        with torch.no_grad():
            curve_logit = self.curve(curves)
        hidden = self.score_input(score.transpose(1, 2))
        residual = self.score_output(hidden + self.score_block(hidden)).squeeze(1)
        gate = torch.sigmoid(self.gate_logit)
        output = curve_logit + gate * residual
        return (output, curve_logit, residual, gate) if return_parts else output


class SmallBiGRU(nn.Module):
    def __init__(self, input_dim: int = 25, hidden_size: int = 32, dropout: float = 0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size * 2, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(self.input_norm(values))
        return self.output(self.dropout(hidden)).squeeze(-1)


def raw_multimodal_predictions(
    model: nn.Module,
    data: dict[str, dict[str, np.ndarray]],
    curve_norm: Normalizer,
    score_norm: Normalizer,
    device: torch.device,
    kind: str,
) -> dict[str, dict[str, np.ndarray]]:
    model.eval(); output: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for piece_id, item in sorted(data.items()):
            curves = item["curves"] if len(item["curves"]) else np.zeros((1, len(item["labels"]), 9), dtype=np.float32)
            ids = [str(x) for x in item["performance_ids"]] or ["missing_curve"]
            c = torch.from_numpy(curve_norm.apply(curves).astype(np.float32)).to(device)
            score = score_norm.apply(item["score_phase3"]).astype(np.float32)
            s = torch.from_numpy(np.broadcast_to(score[None], (len(curves), *score.shape)).copy()).to(device)
            logits = model(c, s) if kind == "m2" else model(torch.cat([c, s], dim=-1))
            probs = torch.sigmoid(logits).cpu().numpy()
            output[piece_id] = {pid: p for pid, p in zip(ids, probs)}
    return output


def _positive_weight(data: dict[str, dict[str, np.ndarray]]) -> float:
    positive = sum(float(x["labels"][x["label_mask"] > 0].sum()) for x in data.values())
    negative = sum(float((x["label_mask"] > 0).sum() - x["labels"][x["label_mask"] > 0].sum()) for x in data.values())
    return float(min(negative / max(positive, 1.0), 10.0))


def train_sequence_model(
    kind: str,
    train_data: dict[str, dict[str, np.ndarray]],
    validation_data: dict[str, dict[str, np.ndarray]],
    curve_checkpoint: dict[str, Any] | None,
    config: dict[str, Any],
    checkpoint_dir: Path,
    resume: bool = True,
    smoke: bool = False,
) -> tuple[nn.Module, Normalizer, Normalizer, dict[str, Any]]:
    seed = int(config["project"]["seed"]); seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curve_arrays = [(x["curves"] if len(x["curves"]) else np.zeros((1, len(x["labels"]), 9), np.float32)) for x in train_data.values()]
    score_arrays = [x["score_phase3"] for x in train_data.values()]
    if kind == "m2":
        if curve_checkpoint is None:
            raise ValueError("M2 requires a frozen curve checkpoint")
        curve_norm = Normalizer(np.asarray(curve_checkpoint["normalizer_mean"]), np.asarray(curve_checkpoint["normalizer_std"]))
    else:
        curve_norm = fit_train_normalizer(curve_arrays)
    score_norm = fit_train_normalizer(score_arrays)
    spec = config["training"]
    dataset = MultiModalWindowDataset(train_data, curve_norm, score_norm, int(config["features"]["window_beats"]), int(config["features"]["stride_beats"]))
    loader = DataLoader(dataset, batch_size=int(spec["batch_size"]), shuffle=True, num_workers=0)
    if kind == "m2":
        curve_model = BeatBoundaryTCN(int(curve_checkpoint["input_dim"]), 32, [1, 2, 4, 8], 3, 0.2)
        curve_model.load_state_dict(curve_checkpoint["model"])
        model: nn.Module = ResidualGatedFusionTCN(curve_model).to(device)
    elif kind == "bigru":
        model = SmallBiGRU(25, 32, 0.2).to(device)
    else:
        raise ValueError(kind)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(_positive_weight(train_data), device=device))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path, latest_path = checkpoint_dir / "best.pt", checkpoint_dir / "latest.pt"
    start, best_score, patience, history = 0, -1.0, 0, []
    if resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start, best_score, patience, history = int(state["epoch"]) + 1, float(state["best_score"]), int(state["patience"]), list(state["history"])
    maximum = 2 if smoke else int(spec["maximum_epochs"])
    if resume and patience >= int(spec["patience"]):
        maximum = start
    for epoch in range(start, maximum):
        model.train(); losses = []
        for curves, score, labels, mask, valid, weight in loader:
            curves, score, labels, mask = curves.to(device), score.to(device), labels.to(device), mask.to(device)
            weight = weight.to(device).view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(curves, score) if kind == "m2" else model(torch.cat([curves, score], dim=-1))
            loss_values = criterion(logits, labels) * mask * weight
            loss = loss_values.sum() / torch.clamp((mask * weight).sum(), min=1.0)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {kind} loss")
            loss.backward(); nn.utils.clip_grad_norm_(trainable, float(spec["gradient_clip_norm"])); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        raw = raw_multimodal_predictions(model, validation_data, curve_norm, score_norm, device, kind)
        threshold, _ = choose_single_threshold(raw, validation_data, config["evaluation"]["threshold_grid"])
        _, _, summary = evaluate_single_performance(raw, validation_data, threshold)
        score_value = float(summary["macro_f1_tol1"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_f1_tol1": score_value, "threshold": threshold})
        if score_value > best_score + 1e-9:
            best_score, patience = score_value, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_score": best_score, "threshold": threshold, "history": history, "curve_normalizer_mean": curve_norm.mean, "curve_normalizer_std": curve_norm.std, "score_normalizer_mean": score_norm.mean, "score_normalizer_std": score_norm.std, "kind": kind}, best_path)
        else:
            patience += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_score": best_score, "patience": patience, "history": history, "kind": kind}, latest_path)
        if not smoke and patience >= int(spec["patience"]):
            break
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    curve_norm = Normalizer(np.asarray(best["curve_normalizer_mean"]), np.asarray(best["curve_normalizer_std"]))
    score_norm = Normalizer(np.asarray(best["score_normalizer_mean"]), np.asarray(best["score_normalizer_std"]))
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = {"kind": kind, "device": str(device), "parameter_count": total_parameters, "trainable_parameter_count": trainable_parameters, "best_epoch": int(best["epoch"]), "validation_macro_f1_tol1": float(best["best_score"]), "threshold": float(best["threshold"]), "history": best["history"]}
    if kind == "m2":
        info["gate"] = float(torch.sigmoid(model.gate_logit).detach().cpu())
    return model, curve_norm, score_norm, info
