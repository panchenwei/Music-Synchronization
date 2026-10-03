from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .evaluation import choose_threshold, evaluate_piece, evaluate_predictions, non_maximum_suppression
from .models import BeatBoundaryTCN, Normalizer, seed_everything


class PerformanceWindowDataset(Dataset):
    """Windows with separate attention-valid and supervised-loss masks."""

    def __init__(self, data: dict[str, dict[str, np.ndarray]], normalizer: Normalizer, window: int, stride: int):
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
        for item in data.values():
            curves = item["curves"]
            if not len(curves):
                curves = np.zeros((1, len(item["labels"]), 9), dtype=np.float32)
            performance_weight = 1.0 / len(curves)
            for curve in curves:
                length = len(item["labels"])
                starts = list(range(0, max(length - window + 1, 1), stride))
                if not starts or starts[-1] != max(0, length - window):
                    starts.append(max(0, length - window))
                for start in sorted(set(starts)):
                    x = normalizer.apply(curve[start : start + window]).astype(np.float32)
                    y = item["labels"][start : start + window].astype(np.float32)
                    loss_mask = item["label_mask"][start : start + window].astype(np.float32)
                    attention_valid = np.ones(len(y), dtype=np.float32)
                    if len(y) < window:
                        pad = window - len(y)
                        x = np.pad(x, ((0, pad), (0, 0)))
                        y = np.pad(y, (0, pad))
                        loss_mask = np.pad(loss_mask, (0, pad))
                        attention_valid = np.pad(attention_valid, (0, pad))
                    self.samples.append((x, y, loss_mask, attention_valid, performance_weight))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def fit_curve_normalizer(data: dict[str, dict[str, np.ndarray]]) -> Normalizer:
    arrays = []
    for item in data.values():
        curves = item["curves"]
        if not len(curves):
            curves = np.zeros((1, len(item["labels"]), 9), dtype=np.float32)
        arrays.append(curves.reshape(-1, 9))
    stacked = np.concatenate(arrays, axis=0)
    mean = np.nanmean(stacked, axis=0).astype(np.float32)
    std = np.nanstd(stacked, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return Normalizer(mean, std)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, maximum_length: int = 2048):
        super().__init__()
        position = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(maximum_length, d_model)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.encoding.shape[1]:
            raise ValueError(f"Sequence length {values.shape[1]} exceeds positional encoding {self.encoding.shape[1]}")
        return values + self.encoding[:, : values.shape[1]].to(values.dtype)


class CompactTransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.dropout2 = nn.Dropout(dropout)
        self.last_attention: torch.Tensor | None = None

    def forward(self, values: torch.Tensor, padding_mask: torch.Tensor | None = None, capture_attention: bool = False) -> torch.Tensor:
        normalized = self.norm1(values)
        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=capture_attention,
            average_attn_weights=False,
        )
        self.last_attention = weights.detach() if capture_attention and weights is not None else None
        values = values + self.dropout1(attended)
        values = values + self.dropout2(self.ffn(self.norm2(values)))
        return values


class CompactBoundaryTransformer(nn.Module):
    def __init__(self, input_dim: int = 9, d_model: int = 32, layers: int = 2, heads: int = 4, ffn_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = SinusoidalPositionEncoding(d_model)
        self.blocks = nn.ModuleList([CompactTransformerBlock(d_model, heads, ffn_dim, dropout) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, 1)

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor | None = None, capture_attention: bool = False, return_hidden: bool = False):
        if padding_mask is not None:
            padding_mask = padding_mask.bool()
            if padding_mask.shape != inputs.shape[:2]:
                raise ValueError("padding_mask must have shape [B,T]")
            if padding_mask.all(dim=1).any():
                raise ValueError("Fully masked attention row is prohibited")
        hidden = self.position(self.input_projection(inputs))
        for block in self.blocks:
            hidden = block(hidden, padding_mask, capture_attention=capture_attention)
        hidden = self.final_norm(hidden)
        logits = self.output(hidden).squeeze(-1)
        return (logits, hidden) if return_hidden else logits


def _raw_performance_predictions(model: nn.Module, data: dict[str, dict[str, np.ndarray]], normalizer: Normalizer, device: torch.device, kind: str) -> dict[str, dict[str, np.ndarray]]:
    model.eval()
    output: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for piece_id, item in sorted(data.items()):
            curves = item["curves"]
            ids = [str(x) for x in item["performance_ids"]]
            if not len(curves):
                curves = np.zeros((1, len(item["labels"]), 9), dtype=np.float32)
                ids = ["missing_curve"]
            batch = torch.from_numpy(normalizer.apply(curves).astype(np.float32)).to(device)
            if kind == "transformer":
                logits = model(batch, padding_mask=torch.zeros(batch.shape[:2], dtype=torch.bool, device=device))
            else:
                logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            output[piece_id] = {performance_id: probabilities for performance_id, probabilities in zip(ids, probs)}
    return output


def nms_probabilities(probabilities: np.ndarray) -> np.ndarray:
    keep = non_maximum_suppression(probabilities, 0.0, radius=1)
    return np.where(keep, probabilities, 0.0)


def median_predictions(performance_predictions: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {piece_id: nms_probabilities(np.median(np.stack(list(perfs.values())), axis=0)) for piece_id, perfs in performance_predictions.items()}


def evaluate_single_performance(
    performance_predictions: dict[str, dict[str, np.ndarray]],
    data: dict[str, dict[str, np.ndarray]],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    rows = []
    for piece_id, perfs in sorted(performance_predictions.items()):
        item = data[piece_id]
        for performance_id, raw in sorted(perfs.items()):
            row = evaluate_piece(piece_id, nms_probabilities(raw), item["labels"], item["label_mask"], threshold)
            row["performance_id"] = performance_id
            rows.append(row)
    performance_frame = pd.DataFrame(rows)
    metric_columns = [c for c in performance_frame.columns if c not in {"piece_id", "performance_id"}]
    piece_frame = performance_frame.groupby("piece_id", as_index=False)[metric_columns].mean(numeric_only=True)
    piece_frame["performances"] = performance_frame.groupby("piece_id").size().to_numpy()
    summary: dict[str, float] = {"pieces": float(len(piece_frame)), "performances": float(len(performance_frame)), "threshold": float(threshold)}
    for metric in ["precision_tol0", "recall_tol0", "f1_tol0", "precision_tol1", "recall_tol1", "f1_tol1", "precision_tol2", "recall_tol2", "f1_tol2", "pr_auc"]:
        summary[f"macro_{metric}"] = float(piece_frame[metric].mean())
    return performance_frame, piece_frame, summary


def choose_single_threshold(performance_predictions: dict[str, dict[str, np.ndarray]], data: dict[str, dict[str, np.ndarray]], grid: list[float]) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in grid:
        _, _, summary = evaluate_single_performance(performance_predictions, data, float(threshold))
        rows.append({"threshold": float(threshold), "macro_f1_tol1": summary["macro_f1_tol1"], "macro_precision_tol1": summary["macro_precision_tol1"], "macro_recall_tol1": summary["macro_recall_tol1"]})
    frame = pd.DataFrame(rows)
    best = frame.sort_values(["macro_f1_tol1", "macro_precision_tol1", "threshold"], ascending=[False, False, False]).iloc[0]
    return float(best.threshold), frame


def train_transformer(
    train_data: dict[str, dict[str, np.ndarray]],
    validation_data: dict[str, dict[str, np.ndarray]],
    phase2_config: dict[str, Any],
    phase1_config: dict[str, Any],
    checkpoint_dir: Path,
    seed: int,
    resume: bool = True,
    smoke_test: bool = False,
) -> tuple[CompactBoundaryTransformer, Normalizer, dict[str, Any]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = phase2_config["transformer"]
    normalizer = fit_curve_normalizer(train_data)
    dataset = PerformanceWindowDataset(train_data, normalizer, int(phase2_config["window_beats"]), int(phase2_config["stride_beats"]))
    loader = DataLoader(dataset, batch_size=int(spec["batch_size"]), shuffle=True, num_workers=0)
    model = CompactBoundaryTransformer(9, int(spec["d_model"]), int(spec["layers"]), int(spec["heads"]), int(spec["ffn_dim"]), float(spec["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))
    positive = sum(float(item["labels"][item["label_mask"] > 0].sum()) for item in train_data.values())
    negative = sum(float((item["label_mask"] > 0).sum() - item["labels"][item["label_mask"] > 0].sum()) for item in train_data.values())
    pos_weight = min(negative / max(positive, 1.0), 10.0)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight, device=device))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path, latest_path = checkpoint_dir / "best.pt", checkpoint_dir / "latest.pt"
    start_epoch, best_score, patience_count, history = 0, -1.0, 0, []
    if resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_score = int(state["epoch"]) + 1, float(state["best_score"])
        patience_count, history = int(state.get("patience_count", 0)), list(state.get("history", []))
    maximum_epochs = 2 if smoke_test else int(spec["maximum_epochs"])
    if resume and patience_count >= int(spec["patience"]):
        maximum_epochs = start_epoch
    targets = {piece_id: (item["labels"], item["label_mask"]) for piece_id, item in validation_data.items()}
    for epoch in range(start_epoch, maximum_epochs):
        model.train(); losses = []
        for features, labels, loss_mask, attention_valid, sequence_weight in loader:
            features, labels = features.to(device), labels.to(device)
            loss_mask, attention_valid = loss_mask.to(device), attention_valid.to(device)
            sequence_weight = sequence_weight.to(device).view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, padding_mask=~attention_valid.bool())
            loss_values = criterion(logits, labels) * loss_mask * sequence_weight
            loss = loss_values.sum() / torch.clamp((loss_mask * sequence_weight).sum(), min=1.0)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Transformer loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        raw_val = _raw_performance_predictions(model, validation_data, normalizer, device, "transformer")
        median_val = {piece_id: np.median(np.stack(list(perfs.values())), axis=0) for piece_id, perfs in raw_val.items()}
        threshold, _ = choose_threshold(median_val, targets, phase1_config["evaluation"]["threshold_grid"])
        _, validation_summary = evaluate_predictions(median_val, targets, threshold)
        score = float(validation_summary["macro_f1_tol1"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_piece_median_macro_f1_tol1": score, "threshold": threshold})
        if score > best_score + 1e-9:
            best_score, patience_count = score, 0
            torch.save({"model": model.state_dict(), "normalizer_mean": normalizer.mean, "normalizer_std": normalizer.std, "epoch": epoch, "best_score": score, "threshold": threshold, "history": history, "seed": seed}, best_path)
        else:
            patience_count += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "normalizer_mean": normalizer.mean, "normalizer_std": normalizer.std, "epoch": epoch, "best_score": best_score, "patience_count": patience_count, "history": history, "seed": seed}, latest_path)
        if not smoke_test and patience_count >= int(spec["patience"]):
            break
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    normalizer = Normalizer(np.asarray(best["normalizer_mean"]), np.asarray(best["normalizer_std"]))
    return model, normalizer, {"device": str(device), "epochs_completed": len(history), "best_epoch": int(best["epoch"]), "validation_piece_median_macro_f1_tol1": float(best["best_score"]), "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad), "history": history}


class TCNMaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn = BeatBoundaryTCN(9, 32, [1, 2, 4, 8], 3, 0.2)
        self.reconstruction = nn.Conv1d(32, 9, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.tcn.blocks(self.tcn.input(inputs.transpose(1, 2)))
        return self.reconstruction(hidden).transpose(1, 2)


def pretrain_masked_tcn(train_data: dict[str, dict[str, np.ndarray]], phase2_config: dict[str, Any], checkpoint_dir: Path, seed: int, resume: bool = True) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = fit_curve_normalizer(train_data)
    dataset = PerformanceWindowDataset(train_data, normalizer, int(phase2_config["window_beats"]), int(phase2_config["stride_beats"]))
    loader = DataLoader(dataset, batch_size=int(phase2_config["tcn"]["batch_size"]), shuffle=True, num_workers=0)
    model = TCNMaskedAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(phase2_config["ssl"]["pretrain_learning_rate"]), weight_decay=float(phase2_config["tcn"]["weight_decay"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path, latest_path = checkpoint_dir / "best.pt", checkpoint_dir / "latest.pt"
    start_epoch, best_loss, history = 0, float("inf"), []
    if resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_loss, history = int(state["epoch"]) + 1, float(state["best_loss"]), list(state["history"])
    for epoch in range(start_epoch, int(phase2_config["ssl"]["pretrain_epochs"])):
        model.train(); losses = []
        for features, _, _, attention_valid, _ in loader:
            features, attention_valid = features.to(device), attention_valid.to(device).bool()
            random_mask = (torch.rand_like(features) < float(phase2_config["ssl"]["mask_probability"])) & attention_valid.unsqueeze(-1)
            if not random_mask.any():
                random_mask[0, 0, 0] = True
            masked = features.masked_fill(random_mask, 0.0)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(masked)
            loss = ((reconstructed - features) ** 2)[random_mask].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite masked-modeling loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)); history.append({"epoch": epoch, "masked_mse": epoch_loss})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_loss": best_loss, "history": history, "seed": seed}, best_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_loss": best_loss, "history": history, "seed": seed}, latest_path)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    state = best["model"]
    encoder_state = {key.removeprefix("tcn."): value for key, value in state.items() if key.startswith("tcn.") and not key.startswith("tcn.output.")}
    return encoder_state, {"epochs": len(history), "initial_masked_mse": history[0]["masked_mse"], "final_masked_mse": history[-1]["masked_mse"], "best_masked_mse": float(best["best_loss"]), "best_epoch": int(best["epoch"]), "training_pieces": len(train_data), "training_performances": sum(len(x["curves"]) for x in train_data.values())}


def raw_performance_predictions(model: nn.Module, data: dict[str, dict[str, np.ndarray]], normalizer: Normalizer, device: torch.device, kind: str) -> dict[str, dict[str, np.ndarray]]:
    return _raw_performance_predictions(model, data, normalizer, device, kind)
