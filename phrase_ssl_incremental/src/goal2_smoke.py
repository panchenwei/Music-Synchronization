"""Unlabeled BiGRU reconstruction smoke test; no phrase annotations or full training."""
from __future__ import annotations

import json
import random
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.goal1_unlabeled import SourceAccessGuard, align_dynamics, discover_sources, read_frame, sha256

GROUPS = {"curves": (0, 9), "score": (9, 25), "multiscale": (25, 34), "motif": (34, 58)}
METADATA = [7, 8, 9, 10, 11, 30, 31, 32, 33, 37, 41, 45, 49, 53, 57]
ROOT = Path(__file__).resolve().parents[1]


def validity(values: np.ndarray, raw_times: np.ndarray, raw_dynamics: np.ndarray) -> np.ndarray:
    """Observed dependency masks, rather than the finite imputed cache values."""
    n = len(values)
    mask = np.isfinite(values)
    time_ok = np.isfinite(raw_times)
    interval = np.zeros(n, bool)
    interval[:-1] = time_ok[:-1] & time_ok[1:] & (np.diff(raw_times) > 0)
    mask[:, 0] &= interval
    mask[:, 1] &= interval & np.r_[False, interval[:-1]]
    mask[:, 2] &= mask[:, 1] & np.r_[False, mask[:-1, 1]]
    dyn_ok = np.isfinite(raw_dynamics)
    mask[:, 3] &= dyn_ok
    mask[:, 4] &= dyn_ok & np.r_[False, dyn_ok[:-1]]
    for i in range(n):
        mask[i, 5] &= interval[max(0, i - 4):min(n, i + 5)].all()
        mask[i, 6] &= dyn_ok[max(0, i - 4):min(n, i + 5)].all()
    # The cached multi-scale normalization uses a whole-piece reference median.
    # Conservatively exclude its targets when any required raw interval is missing.
    mask[:, 25:30] &= interval[:-1].all()
    for start in (34, 38, 42, 46, 50, 54):
        mask[:, start:start + 3] &= values[:, start + 3:start + 4] > 0.5
    mask[:, METADATA] = False
    return mask


def sample_spans(lengths: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, list[list[tuple[int, int]]]]:
    mask = torch.zeros(len(lengths), 64, dtype=torch.bool)
    records = []
    for row, length in enumerate(lengths.tolist()):
        if length < 4:
            raise ValueError("A smoke window must contain at least four valid beats")
        width = int(torch.randint(4, min(8, length) + 1, (), generator=generator))
        start = int(torch.randint(0, length - width + 1, (), generator=generator))
        mask[row, start:start + width] = True
        records.append([(start, start + width)])
    return mask, records


def corrupt(values: torch.Tensor, valid: torch.Tensor, spans: torch.Tensor, lengths: torch.Tensor):
    padding = torch.arange(values.shape[1])[None, :] >= lengths[:, None]
    hidden = spans[:, :, None].expand_as(values).clone()
    # Four-beat halo removes cached derivatives and local rolling statistics.
    for offset in range(1, 5):
        hidden[:, offset:] |= spans[:, :-offset, None]
        hidden[:, :-offset] |= spans[:, offset:, None]
    # Global recurrence and block summaries were computed before masking. Keep
    # their targets but withhold these input branches to prevent cached shortcuts.
    hidden[:, :, 25:58] = True
    hidden[:, :, METADATA] = True
    missing = ~valid & ~padding[:, :, None]
    observed = valid & ~hidden & ~padding[:, :, None]
    clean = torch.where(observed, values, torch.zeros_like(values))
    return clean, hidden, missing, padding


class ReconstructionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(58))
        self.missing_token = nn.Parameter(torch.zeros(58))
        self.projection = nn.Linear(58, 32)
        self.encoder = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        self.heads = nn.ModuleDict({name: nn.Linear(64, stop - start) for name, (start, stop) in GROUPS.items()})

    def forward(self, clean, hidden, missing, lengths):
        x = torch.where(hidden, self.mask_token, clean)
        x = torch.where(missing & ~hidden, self.missing_token, x)
        x = self.projection(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.encoder(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=64)
        return torch.cat([self.heads[name](output) for name in GROUPS], dim=-1)


def reconstruction_loss(prediction, target, valid, spans, lengths):
    selected = valid & spans[:, :, None] & (torch.arange(64)[None, :, None] < lengths[:, None, None])
    terms = {}
    for name, (start, stop) in GROUPS.items():
        use = selected[:, :, start:stop]
        if use.any():
            # Index before subtracting: NaN or absurd values outside the mask
            # cannot affect either loss or gradients.
            terms[name] = (prediction[:, :, start:stop][use] - target[:, :, start:stop][use]).square().mean()
    if not terms:
        raise ValueError("No masked and genuinely observed reconstruction targets")
    return torch.stack(list(terms.values())).mean(), terms, selected


def load_smoke_data(guard):
    paths = tomllib.loads((ROOT / "configs/data_paths.toml").read_text(encoding="utf-8"))
    sources, _ = discover_sources(Path(paths["raw"]["dcml_chopin"]), Path(paths["raw"]["mazurkabl"]))
    by_piece = {source.piece_id: source for source in sources}
    manifest = pd.read_csv(ROOT / "manifests/goal1/window_manifest.csv.gz")
    # Smoke-only partition by work, not window. No claim of downstream accuracy.
    split = {"train": ["chopin_op06_no1", "chopin_op06_no2"], "validation": ["chopin_op06_no4"]}
    batches = {}
    used_files = []
    for subset, pieces in split.items():
        samples = []
        for piece in pieces:
            source = by_piece[piece]
            raw = read_frame(source.beat_time, ",", guard)
            dyn = align_dynamics(raw, read_frame(source.beat_dyn, ",", guard))
            candidates = manifest[manifest.piece_id == piece]
            first_id = candidates.performance_id.iloc[0]
            first_perf = candidates[candidates.performance_id == first_id]
            # Include a padded tail in the real-data smoke test.
            rows = [first_perf.iloc[0], first_perf.iloc[-1]]
            cache_path = ROOT / "cache/unlabeled58" / f"{piece}.npz"
            used_files.append({"path": str(cache_path), "sha256": sha256(cache_path)})
            with np.load(cache_path, allow_pickle=False) as cache:
                if any("label" in key.lower() for key in cache.files):
                    raise ValueError("Unexpected label field in unlabeled cache")
                index = cache["performance_ids"].tolist().index(first_id)
                values = cache["inputs"][index].copy()
            times = pd.to_numeric(raw[first_id], errors="coerce").to_numpy(float)
            dynamics = pd.to_numeric(dyn[first_id], errors="coerce").to_numpy(float) if first_id in dyn else np.full(len(times), np.nan)
            observed = validity(values, times, dynamics)
            for row in rows:
                start, length = int(row.start_beat), int(row.valid_length)
                x, valid = np.zeros((64, 58), np.float32), np.zeros((64, 58), bool)
                x[:length] = values[start:start + length]
                valid[:length] = observed[start:start + length]
                samples.append((x, valid, length))
        batches[subset] = (
            torch.tensor(np.stack([s[0] for s in samples])),
            torch.tensor(np.stack([s[1] for s in samples])),
            torch.tensor([s[2] for s in samples]),
        )
    return batches, split, used_files


def fit_normalization(values, valid):
    count = valid.sum((0, 1)).clamp_min(1)
    mean = torch.where(valid, values, 0).sum((0, 1)) / count
    variance = torch.where(valid, (values - mean).square(), 0).sum((0, 1)) / count
    scale = variance.sqrt().clamp_min(1e-3)
    return mean, scale, variance > 1e-8


def save_checkpoint(path, model, optimizer, step, generator, mean, scale):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "config": {"input_dim": 58, "projection_dim": 32, "gru_hidden_per_direction": 32,
                           "gru_layers": 1, "bidirectional": True, "window": 64, "span_min": 4, "span_max": 8},
                "torch_rng": torch.get_rng_state(), "generator_rng": generator.get_state(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "normalization_mean": mean, "normalization_scale": scale}, path)


def smoke():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(20261013)
    np.random.seed(20261013)
    random.seed(20261013)
    generator = torch.Generator().manual_seed(42)
    report_dir = ROOT / "reports/goal2"
    checkpoint_dir = ROOT / "checkpoints/goal2_smoke"
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths = tomllib.loads((ROOT / "configs/data_paths.toml").read_text(encoding="utf-8"))
    guard = SourceAccessGuard(Path(paths["raw"]["dcml_chopin"]), Path(paths["raw"]["mazurkabl"]), Path(paths["legacy_read_only"]["experiment_root"]))
    guard.install()
    batches, split, files = load_smoke_data(guard)
    values, valid, lengths = batches["train"]
    mean, scale, varying = fit_normalization(values, valid)
    values = (values - mean) / scale
    valid &= varying
    spans, span_records = sample_spans(lengths, generator)
    clean, hidden, missing, padding = corrupt(values, valid, spans, lengths)
    model = ReconstructionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.eval()
    initial = float(reconstruction_loss(model(clean, hidden, missing, lengths), values, valid, spans, lengths)[0].detach())
    history = []
    gradient_max = 0.0
    for step in range(1, 251):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(clean, hidden, missing, lengths)
        loss, terms, selected = reconstruction_loss(prediction, values, valid, spans, lengths)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("Non-finite gradient")
                gradient_max = max(gradient_max, float(parameter.grad.abs().max()))
        optimizer.step()
        if step == 1 or step % 25 == 0:
            history.append({"step": step, "loss": float(loss.detach())})
    final_prediction = model(clean, hidden, missing, lengths)
    final, groups, selected = reconstruction_loss(final_prediction, values, valid, spans, lengths)
    final_value = float(final.detach())
    if final_value >= initial * 0.2:
        raise RuntimeError(f"Tiny-overfit gate failed: {initial} -> {final_value}")

    checkpoint = checkpoint_dir / "tiny_overfit.pt"
    save_checkpoint(checkpoint, model, optimizer, step, generator, mean, scale)
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    restored = ReconstructionModel()
    restored.load_state_dict(state["model"])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=3e-3)
    restored_optimizer.load_state_dict(state["optimizer"])
    torch.testing.assert_close(restored(clean, hidden, missing, lengths), final_prediction, rtol=0, atol=0)
    # Compare one optimizer update after recovery, including optimizer moments.
    next_models = [(model, optimizer), (restored, restored_optimizer)]
    for net, opt in next_models:
        opt.zero_grad(set_to_none=True)
        reconstruction_loss(net(clean, hidden, missing, lengths), values, valid, spans, lengths)[0].backward()
        opt.step()
    for left, right in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    saved_generator = torch.Generator()
    saved_generator.set_state(state["generator_rng"])
    torch.testing.assert_close(sample_spans(lengths, saved_generator)[0], sample_spans(lengths, generator)[0])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])

    # Target-copy probe: modifying every withheld value leaves network input unchanged.
    changed = values.clone()
    changed[hidden | ~valid] = 123456.0
    clean_changed = corrupt(changed, valid, spans, lengths)[0]
    torch.testing.assert_close(clean, clean_changed, rtol=0, atol=0)
    altered_target = values.clone()
    altered_target[~selected] = float("nan")
    original_loss = reconstruction_loss(final_prediction, values, valid, spans, lengths)[0]
    ignored_loss = reconstruction_loss(final_prediction, altered_target, valid, spans, lengths)[0]
    torch.testing.assert_close(original_loss, ignored_loss, rtol=0, atol=0)

    validation_x, validation_valid, validation_lengths = batches["validation"]
    validation_x = (validation_x - mean) / scale
    validation_valid &= varying
    validation_spans, _ = sample_spans(validation_lengths, generator)
    validation_input = corrupt(validation_x, validation_valid, validation_spans, validation_lengths)
    with torch.no_grad():
        validation_prediction = model(*validation_input[:3], validation_lengths)
        validation_loss = float(reconstruction_loss(validation_prediction, validation_x, validation_valid, validation_spans, validation_lengths)[0])
    guard.access_frame().to_csv(report_dir / "raw_source_access.csv", index=False)
    result = {"status": "passed", "scientific_status": "implemented_and_locally_reproduced_smoke_only",
              "model": "58->32 projection; 1-layer bidirectional GRU hidden32/direction; four linear reconstruction heads",
              "parameter_count": sum(p.numel() for p in model.parameters()), "device": "cpu", "split": split,
              "train_windows": len(values), "validation_windows": len(validation_x), "spans": span_records,
              "train_lengths": lengths.tolist(), "masked_observed_targets": int(selected.sum()),
              "initial_loss": initial, "final_loss": final_value, "loss_ratio": final_value / initial,
              "group_loss": {k: float(v.detach()) for k, v in groups.items()}, "validation_smoke_loss": validation_loss,
              "finite_gradient_max": gradient_max, "optimizer_steps": 250, "history": history,
              "checkpoint_prediction_exact": True, "checkpoint_next_update_exact": True,
              "sampler_restore_exact": True, "unselected_target_nan_invariance": True,
              "global_rng_restored": True,
              "hidden_target_copy_invariance": True, "metadata_columns_excluded": METADATA,
              "cached_derived_inputs_withheld": "all multiscale and motif columns; four-beat halo around masked span",
              "unlabeled_cache_sources": files, "seconds": time.monotonic() - started,
              "limitation": "Tiny-overfit proves trainability only; not generalization, phrase accuracy, or musical understanding"}
    (report_dir / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    smoke()
