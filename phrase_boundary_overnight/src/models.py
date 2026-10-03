from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .evaluation import choose_threshold, evaluate_predictions


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(inputs + self.net(inputs))


class BeatBoundaryTCN(nn.Module):
    def __init__(self, input_dim: int, hidden_channels: int = 64, dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8), kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input = nn.Conv1d(input_dim, hidden_channels, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels, d, kernel_size, dropout) for d in dilations])
        self.output = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input(inputs.transpose(1, 2))
        return self.output(self.blocks(hidden)).squeeze(1)


class WindowDataset(Dataset):
    def __init__(self, arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]], window: int, stride: int):
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        for features, labels, mask, weight in arrays:
            starts = list(range(0, max(len(labels) - window + 1, 1), stride))
            if not starts or starts[-1] != max(0, len(labels) - window):
                starts.append(max(0, len(labels) - window))
            for start in sorted(set(starts)):
                x = features[start : start + window]
                y = labels[start : start + window]
                m = mask[start : start + window]
                if len(y) < window:
                    pad = window - len(y)
                    x = np.pad(x, ((0, pad), (0, 0)))
                    y = np.pad(y, (0, pad))
                    m = np.pad(m, (0, pad))
                self.samples.append((x.astype(np.float32), y.astype(np.float32), m.astype(np.float32), float(weight)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


def fit_normalizer(arrays: list[np.ndarray]) -> Normalizer:
    stacked = np.concatenate([array.reshape(-1, array.shape[-1]) for array in arrays], axis=0)
    mean = np.nanmean(stacked, axis=0).astype(np.float32)
    std = np.nanstd(stacked, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return Normalizer(mean, std)


def compose_piece_features(data: dict[str, np.ndarray], variant: str = "combined") -> np.ndarray:
    score = data["score"]
    curves = data["curves"]
    if not len(curves):
        curves = np.zeros((1, len(score), 9), dtype=np.float32)
    if variant == "score":
        return score[None, :, :]
    if variant == "curves":
        return curves
    if variant == "combined":
        score_stack = np.broadcast_to(score[None, :, :], (len(curves), *score.shape))
        return np.concatenate([score_stack, curves], axis=2)
    raise ValueError(f"Unknown TCN input variant: {variant}")


def predict_tcn(model: nn.Module, piece_data: dict[str, dict[str, np.ndarray]], normalizer: Normalizer, device: torch.device, input_variant: str = "combined") -> dict[str, np.ndarray]:
    model.eval()
    output: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for piece_id, data in sorted(piece_data.items()):
            features = compose_piece_features(data, input_variant)
            features = normalizer.apply(features).astype(np.float32)
            batch = torch.from_numpy(features).to(device)
            probabilities = torch.sigmoid(model(batch)).cpu().numpy()
            output[piece_id] = np.median(probabilities, axis=0)
    return output


def train_tcn(
    train_data: dict[str, dict[str, np.ndarray]],
    validation_data: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
    checkpoint_dir: Path,
    smoke_test: bool = False,
    resume: bool = False,
    initial_model_state: dict[str, torch.Tensor] | None = None,
) -> tuple[BeatBoundaryTCN, Normalizer, dict[str, Any]]:
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_variant = str(config["model"].get("input_variant", "combined"))
    feature_arrays = []
    training_sequences = []
    positive = 0.0
    negative = 0.0
    for piece_id, data in train_data.items():
        combined = compose_piece_features(data, input_variant)
        perf_count = len(combined)
        feature_arrays.append(combined)
        valid = data["label_mask"] > 0
        positive += float(data["labels"][valid].sum())
        negative += float(valid.sum() - data["labels"][valid].sum())
        for perf_features in combined:
            training_sequences.append((perf_features, data["labels"], data["label_mask"], 1.0 / perf_count))
    normalizer = fit_normalizer(feature_arrays)
    training_sequences = [(normalizer.apply(x), y, m, w) for x, y, m, w in training_sequences]
    training = config["training"]
    dataset = WindowDataset(training_sequences, int(config["features"]["window_beats"]), int(config["features"]["stride_beats"]))
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True, num_workers=0)
    model_cfg = config["model"]
    input_dim = training_sequences[0][0].shape[-1]
    model = BeatBoundaryTCN(input_dim, int(model_cfg["hidden_channels"]), list(model_cfg["dilations"]), int(model_cfg["kernel_size"]), float(model_cfg["dropout"])).to(device)
    if initial_model_state is not None and not (resume and (checkpoint_dir / "latest.pt").exists()):
        incompatible = model.load_state_dict(initial_model_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = set(incompatible.missing_keys)
        if unexpected or not missing.issubset({"output.weight", "output.bias"}):
            raise ValueError(f"Invalid pretrained TCN encoder state: missing={sorted(missing)}, unexpected={unexpected}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    pos_weight = min(negative / max(positive, 1.0), 10.0)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight, device=device))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    start_epoch = 0
    best_score = -1.0
    patience_count = 0
    history: list[dict[str, float | int]] = []
    if resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
        patience_count = int(state.get("patience_count", 0))
        history = state.get("history", [])
    maximum_epochs = 2 if smoke_test else int(training["maximum_epochs"])
    if resume and patience_count >= int(training["early_stopping_patience"]):
        maximum_epochs = start_epoch
    targets = {piece_id: (data["labels"], data["label_mask"]) for piece_id, data in validation_data.items()}
    for epoch in range(start_epoch, maximum_epochs):
        model.train()
        losses = []
        for features, labels, mask, sequence_weight in loader:
            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            sequence_weight = sequence_weight.to(device).view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss_values = criterion(logits, labels) * mask * sequence_weight
            loss = loss_values.sum() / torch.clamp((mask * sequence_weight).sum(), min=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_predictions = predict_tcn(model, validation_data, normalizer, device, input_variant)
        threshold, _ = choose_threshold(validation_predictions, targets, config["evaluation"]["threshold_grid"])
        _, validation_summary = evaluate_predictions(validation_predictions, targets, threshold)
        score = float(validation_summary["macro_f1_tol1"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_f1_tol1": score, "threshold": threshold})
        if score > best_score + 1e-9:
            best_score = score
            patience_count = 0
            torch.save({"model": model.state_dict(), "input_dim": input_dim, "normalizer_mean": normalizer.mean, "normalizer_std": normalizer.std, "epoch": epoch, "best_score": best_score, "threshold": threshold, "history": history}, best_path)
        else:
            patience_count += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "input_dim": input_dim, "normalizer_mean": normalizer.mean, "normalizer_std": normalizer.std, "epoch": epoch, "best_score": best_score, "patience_count": patience_count, "history": history}, latest_path)
        if not smoke_test and patience_count >= int(training["early_stopping_patience"]):
            break
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    normalizer = Normalizer(np.asarray(best["normalizer_mean"]), np.asarray(best["normalizer_std"]))
    info = {"device": str(device), "input_variant": input_variant, "input_dim": input_dim, "epochs_completed": len(history), "best_epoch": int(best["epoch"]), "validation_macro_f1_tol1": float(best["best_score"]), "threshold": float(best["threshold"]), "positive_weight": float(pos_weight), "history": history}
    return model, normalizer, info
