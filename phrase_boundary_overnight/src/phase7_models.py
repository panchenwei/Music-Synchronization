from __future__ import annotations

import copy
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .models import Normalizer, seed_everything
from .phase2_models import (
    CompactTransformerBlock,
    SinusoidalPositionEncoding,
    choose_single_threshold,
    evaluate_single_performance,
    fit_curve_normalizer,
)
from .phase6_models import positive_weight, window_starts


class ResidualTemporalConv(nn.Module):
    """Length-preserving 5-beat depthwise/pointwise residual frontend."""

    def __init__(self, channels: int = 32, kernel_size: int = 5, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, stride=1, padding=kernel_size // 2, groups=channels
        )
        self.activation = nn.GELU()
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if padding_mask is not None:
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        update = self.norm(hidden).transpose(1, 2)
        update = self.depthwise(update)
        update = self.activation(update)
        update = self.pointwise(update).transpose(1, 2)
        hidden = hidden + self.dropout(update)
        if padding_mask is not None:
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return hidden


class ResidualPointMLP(nn.Module):
    """Same-beat capacity control: never sees neighbouring beats."""

    def __init__(self, channels: int = 32, hidden: int = 16, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.net = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if padding_mask is not None:
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        hidden = hidden + self.dropout(self.net(self.norm(hidden)))
        if padding_mask is not None:
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return hidden


class Phase7BoundaryModel(nn.Module):
    """A=Transformer, B=temporal-conv+Transformer, C=temporal-conv head, D=point-MLP+Transformer."""

    def __init__(
        self,
        kind: str,
        input_dim: int = 9,
        d_model: int = 32,
        layers: int = 2,
        heads: int = 4,
        ffn_dim: int = 64,
        dropout: float = 0.2,
        kernel_size: int = 5,
        mlp_hidden: int = 16,
    ):
        super().__init__()
        if kind not in {"A", "B", "C", "D"}:
            raise ValueError(f"Unknown Phase7 model kind: {kind}")
        self.kind = kind
        # Construct shared modules in a fixed order so A/B/D share identical initial values per seed.
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = SinusoidalPositionEncoding(d_model)
        self.blocks = nn.ModuleList(
            [CompactTransformerBlock(d_model, heads, ffn_dim, dropout) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, 1)
        if kind == "C":
            self.blocks = nn.ModuleList()
        self.frontend: nn.Module | None = None
        if kind in {"B", "C"}:
            self.frontend = ResidualTemporalConv(d_model, kernel_size, dropout)
        elif kind == "D":
            self.frontend = ResidualPointMLP(d_model, mlp_hidden, dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        capture_attention: bool = False,
        return_hidden: bool = False,
    ):
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [B,T,C]")
        if padding_mask is None:
            padding_mask = torch.zeros(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        padding_mask = padding_mask.bool()
        if padding_mask.shape != inputs.shape[:2]:
            raise ValueError("padding_mask must have shape [B,T]")
        if padding_mask.all(dim=1).any():
            raise ValueError("Fully masked sequence is prohibited")
        hidden = self.input_projection(inputs)
        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        if self.frontend is not None:
            hidden = self.frontend(hidden, padding_mask)
        if self.kind != "C":
            hidden = self.position(hidden)
            for block in self.blocks:
                hidden = block(hidden, padding_mask, capture_attention=capture_attention)
                hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        hidden = self.final_norm(hidden)
        logits = self.output(hidden).squeeze(-1)
        logits = logits.masked_fill(padding_mask, 0.0)
        return (logits, hidden) if return_hidden else logits


class CurvePieceBalancedSampler:
    def __init__(self, data, normalizer: Normalizer, window: int, batch_size: int, seed: int):
        self.data = data
        self.normalizer = normalizer
        self.window = int(window)
        self.batch_size = int(batch_size)
        self.pieces = sorted(data)
        self.rng = np.random.default_rng(seed)

    def state(self) -> dict[str, Any]:
        return {"rng": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]

    def batch(self):
        samples = []
        for _ in range(self.batch_size):
            piece_id = str(self.rng.choice(self.pieces))
            item = self.data[piece_id]
            curves = item["curves"]
            if not len(curves):
                curves = np.zeros((1, len(item["labels"]), 9), np.float32)
            curve = curves[int(self.rng.integers(len(curves)))]
            starts = window_starts(len(item["labels"]), self.window, max(self.window // 2, 1))
            start = int(self.rng.choice(starts))
            valid = min(self.window, len(item["labels"]) - start)
            x = self.normalizer.apply(curve[start : start + self.window]).astype(np.float32)
            y = item["labels"][start : start + self.window].astype(np.float32)
            loss_mask = item["label_mask"][start : start + self.window].astype(np.float32)
            attention_valid = np.ones(valid, np.float32)
            if valid < self.window:
                pad = self.window - valid
                x = np.pad(x, ((0, pad), (0, 0)))
                y = np.pad(y, (0, pad))
                loss_mask = np.pad(loss_mask, (0, pad))
                attention_valid = np.pad(attention_valid, (0, pad))
            samples.append((x, y, loss_mask, attention_valid))
        return tuple(torch.as_tensor(np.stack([s[i] for s in samples])) for i in range(4))


def full_curve_predictions(model, data, normalizer: Normalizer, device):
    model.eval()
    output = {}
    with torch.no_grad():
        for piece_id, item in sorted(data.items()):
            curves = item["curves"]
            ids = [str(x) for x in item["performance_ids"]]
            if not len(curves):
                curves = np.zeros((1, len(item["labels"]), 9), np.float32)
                ids = ["missing_curve"]
            batch = torch.from_numpy(normalizer.apply(curves).astype(np.float32)).to(device)
            padding_mask = torch.zeros(batch.shape[:2], dtype=torch.bool, device=device)
            probabilities = torch.sigmoid(model(batch, padding_mask=padding_mask)).cpu().numpy()
            output[piece_id] = {pid: prob for pid, prob in zip(ids, probabilities)}
    return output


def model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    m = config["model"]
    return {
        "input_dim": int(config["data"]["input_dimensions"]),
        "d_model": int(m["d_model"]),
        "layers": int(m["layers"]),
        "heads": int(m["heads"]),
        "ffn_dim": int(m["ffn_dim"]),
        "dropout": float(m["dropout"]),
        "kernel_size": int(m["convolution_kernel"]),
        "mlp_hidden": int(config["conditional_control"]["mlp_hidden"]),
    }


def train_phase7_model(
    kind: str,
    train_data,
    validation_data,
    config: dict[str, Any],
    checkpoint_dir: Path,
    seed: int,
    deadline: datetime,
    resume: bool = True,
    contract_hash: str | None = None,
):
    started = time.perf_counter()
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    normalizer = fit_curve_normalizer(train_data)
    model = Phase7BoundaryModel(kind, **model_kwargs(config)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(
            positive_weight(train_data, float(config["training"]["positive_weight_cap"])), device=device
        ),
    )
    sampler = CurvePieceBalancedSampler(
        train_data,
        normalizer,
        int(config["data"]["window_beats"]),
        int(config["training"]["batch_size"]),
        seed,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path, latest_path = checkpoint_dir / "best.pt", checkpoint_dir / "latest.pt"
    step, best_score, history = 0, -1.0, []
    if resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        if contract_hash is not None and state.get("contract_hash") != contract_hash:
            raise ValueError("Refusing resume: Phase7 contract hash changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, best_score, history = int(state["step"]), float(state["best_score"]), list(state["history"])
        sampler.load_state(state["sampler"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])

    def save_latest():
        torch.save(
            {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "best_score": best_score, "history": history, "sampler": sampler.state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "normalizer_mean": normalizer.mean, "normalizer_std": normalizer.std,
                "kind": kind, "seed": seed, "contract_hash": contract_hash,
            }, latest_path,
        )

    maximum_steps = int(config["training"]["maximum_steps"])
    frequency = int(config["training"]["validation_every_steps"])
    losses = []
    while step < maximum_steps:
        remaining = (deadline - datetime.now().astimezone()).total_seconds()
        if remaining <= float(config["training"]["deadline_guard_minutes"]) * 60:
            save_latest()
            raise TimeoutError(f"Phase7 deadline guard reached ({remaining:.1f}s left)")
        features, labels, loss_mask, valid = sampler.batch()
        features, labels = features.to(device), labels.to(device)
        loss_mask, valid = loss_mask.to(device), valid.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, padding_mask=~valid.bool())
        loss_values = criterion(logits, labels) * loss_mask
        loss = loss_values.sum() / torch.clamp(loss_mask.sum(), min=1.0)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Phase7 loss")
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"])))
        if not np.isfinite(grad_norm):
            raise FloatingPointError("Non-finite Phase7 gradient")
        optimizer.step()
        step += 1
        losses.append(float(loss.detach().cpu()))
        if step % int(config["training"]["latest_every_steps"]) == 0:
            save_latest()
        if step % frequency == 0 or step == maximum_steps:
            raw = full_curve_predictions(model, validation_data, normalizer, device)
            threshold, _ = choose_single_threshold(raw, validation_data, config["evaluation"]["threshold_grid"])
            _, _, selected = evaluate_single_performance(raw, validation_data, threshold)
            _, _, fixed = evaluate_single_performance(raw, validation_data, float(config["evaluation"]["fixed_threshold"]))
            record = {
                "step": step, "train_loss": float(np.mean(losses)), "gradient_norm": grad_norm,
                "threshold": threshold, "validation_f1_tol0": selected["macro_f1_tol0"],
                "validation_precision_tol1": selected["macro_precision_tol1"],
                "validation_recall_tol1": selected["macro_recall_tol1"],
                "validation_f1_tol1": selected["macro_f1_tol1"],
                "validation_f1_tol2": selected["macro_f1_tol2"],
                "validation_pr_auc": selected["macro_pr_auc"], "fixed_f1_tol1": fixed["macro_f1_tol1"],
            }
            history.append(record)
            losses = []
            if record["validation_f1_tol1"] > best_score + 1e-9:
                best_score = float(record["validation_f1_tol1"])
                torch.save(
                    {
                        "model": model.state_dict(), "step": step, "best_score": best_score,
                        "threshold": threshold, "history": history, "normalizer_mean": normalizer.mean,
                        "normalizer_std": normalizer.std, "kind": kind, "seed": seed,
                        "contract_hash": contract_hash, "model_kwargs": model_kwargs(config),
                    }, best_path,
                )
            save_latest()
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    raw_validation = full_curve_predictions(model, validation_data, normalizer, device)
    threshold = float(best["threshold"])
    val_perf, val_piece, selected = evaluate_single_performance(raw_validation, validation_data, threshold)
    _, _, fixed = evaluate_single_performance(raw_validation, validation_data, float(config["evaluation"]["fixed_threshold"]))
    raw_train = full_curve_predictions(model, train_data, normalizer, device)
    _, _, train_summary = evaluate_single_performance(raw_train, train_data, threshold)
    elapsed = time.perf_counter() - started
    if torch.cuda.is_available():
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        peak_memory = 0
    info = {
        "kind": kind, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
        "best_step": int(best["step"]), "maximum_steps": maximum_steps, "validation_frequency": frequency,
        "threshold": threshold, "selected": selected, "fixed": fixed, "training": train_summary,
        "train_minus_validation_f1": float(train_summary["macro_f1_tol1"] - selected["macro_f1_tol1"]),
        "history": history, "training_and_validation_seconds": elapsed, "gpu_peak_bytes": peak_memory,
        "input_shape": [int(config["training"]["batch_size"]), int(config["data"]["window_beats"]), 9],
        "output_shape": [int(config["training"]["batch_size"]), int(config["data"]["window_beats"])],
    }
    return model, normalizer, raw_validation, val_perf, val_piece, info


def load_phase7_checkpoint(checkpoint: Path, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = Phase7BoundaryModel(state["kind"], **state["model_kwargs"]).to(device)
    model.load_state_dict(state["model"])
    normalizer = Normalizer(np.asarray(state["normalizer_mean"]), np.asarray(state["normalizer_std"]))
    return model, normalizer, state
