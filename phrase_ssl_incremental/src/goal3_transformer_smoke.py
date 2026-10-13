"""Full-piece variable-length Transformer reconstruction: bounded smoke only."""
from __future__ import annotations

import json
import math
import random
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.goal1_unlabeled import SourceAccessGuard, align_dynamics, discover_sources, read_frame, sha256
from src.goal2_smoke import GROUPS, METADATA, ROOT, corrupt, fit_normalization, validity

CONFIG = {"input_dim": 58, "d_model": 64, "heads": 4, "blocks": 2, "ffn": 128,
          "dropout": 0.0, "position": "dynamic sinusoidal", "lr": 0.003,
          "smoke_steps": 300, "mask_fraction": 0.12, "span_min": 4, "span_max": 8}


def bucket_batches(samples, batch_size=2, bucket_width=128):
    buckets = {}
    for sample in samples:
        buckets.setdefault(len(sample[0]) // bucket_width, []).append(sample)
    batches = []
    for key in sorted(buckets):
        rows = sorted(buckets[key], key=lambda sample: len(sample[0]))
        for start in range(0, len(rows), batch_size):
            group = rows[start:start + batch_size]
            lengths = torch.tensor([len(sample[0]) for sample in group])
            maximum = int(lengths.max())
            values = torch.zeros(len(group), maximum, 58)
            observed = torch.zeros_like(values, dtype=torch.bool)
            for i, (x, valid, _) in enumerate(group):
                values[i, :len(x)] = torch.as_tensor(x)
                observed[i, :len(x)] = torch.as_tensor(valid)
            batches.append((values, observed, lengths, [sample[2] for sample in group]))
    return batches


def mask_spans(lengths, generator, fraction=0.12):
    maximum = int(lengths.max())
    spans = torch.zeros(len(lengths), maximum, dtype=torch.bool)
    records = []
    for row, length in enumerate(lengths.tolist()):
        if length < 4:
            raise ValueError("Sequence needs at least four valid beats")
        intervals = []
        for attempt in range(1000):
            if int(spans[row].sum()) >= max(4, math.ceil(length * fraction)):
                break
            width = int(torch.randint(4, min(8, length) + 1, (), generator=generator))
            start = int(torch.randint(0, length - width + 1, (), generator=generator))
            # Keep a gap so adjacent samples do not silently become >8-beat spans.
            if spans[row, max(0, start - 1):min(length, start + width + 1)].any():
                continue
            spans[row, start:start + width] = True
            intervals.append((start, start + width))
        if not intervals:
            raise ValueError("Failed to sample a valid span")
        records.append(intervals)
    return spans, records


class FullPieceTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(58))
        self.missing_token = nn.Parameter(torch.zeros(58))
        self.projection = nn.Linear(58, CONFIG["d_model"])
        layer = nn.TransformerEncoderLayer(CONFIG["d_model"], CONFIG["heads"], CONFIG["ffn"],
                                           dropout=CONFIG["dropout"], activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, CONFIG["blocks"], norm=nn.LayerNorm(64), enable_nested_tensor=False)
        self.heads = nn.ModuleDict({name: nn.Linear(64, stop - start) for name, (start, stop) in GROUPS.items()})

    def forward(self, clean, hidden, missing, lengths):
        maximum = clean.shape[1]
        padding = torch.arange(maximum, device=clean.device)[None] >= lengths.to(clean.device)[:, None]
        x = torch.where(hidden, self.mask_token, clean)
        x = torch.where(missing & ~hidden, self.missing_token, x)
        x = self.projection(x)
        positions = torch.arange(maximum, device=x.device, dtype=x.dtype)[:, None]
        frequencies = torch.exp(torch.arange(0, 64, 2, device=x.device, dtype=x.dtype) * (-math.log(10000) / 64))
        encoding = torch.zeros(maximum, 64, device=x.device, dtype=x.dtype)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        output = self.encoder((x + encoding).masked_fill(padding[:, :, None], 0), src_key_padding_mask=padding)
        predictions = torch.cat([self.heads[name](output) for name in GROUPS], dim=-1)
        return predictions.masked_fill(padding[:, :, None], 0)


def masked_loss(prediction, target, valid, spans, lengths):
    nonpadding = torch.arange(target.shape[1], device=target.device)[None] < lengths.to(target.device)[:, None]
    selected = valid & spans[:, :, None] & nonpadding[:, :, None]
    terms = {}
    for name, (start, stop) in GROUPS.items():
        use = selected[:, :, start:stop]
        if use.any():
            terms[name] = (prediction[:, :, start:stop][use] - target[:, :, start:stop][use]).square().mean()
    if not terms:
        raise ValueError("No observed masked target")
    return torch.stack(list(terms.values())).mean(), terms, selected


def load_full_pieces(guard):
    config = tomllib.loads((ROOT / "configs/data_paths.toml").read_text(encoding="utf-8"))
    sources, _ = discover_sources(Path(config["raw"]["dcml_chopin"]), Path(config["raw"]["mazurkabl"]))
    by_piece = {s.piece_id: s for s in sources}
    manifest_path = ROOT / "manifests/goal1/full_sequence_manifest.csv.gz"
    manifest = pd.read_csv(manifest_path)
    split = {"train": ["chopin_op06_no1", "chopin_op06_no2"], "validation": ["chopin_op06_no4"]}
    data, files = {}, [{"path": str(manifest_path), "sha256": sha256(manifest_path)}]
    for subset, pieces in split.items():
        samples = []
        for piece in pieces:
            row = manifest[manifest.piece_id == piece].iloc[0]
            path = ROOT / "cache/unlabeled58" / f"{piece}.npz"
            with np.load(path, allow_pickle=False) as cache:
                if any("label" in key.lower() for key in cache.files):
                    raise ValueError("Unexpected label field")
                index = cache["performance_ids"].tolist().index(row.performance_id)
                x = cache["inputs"][index].copy()
            if int(row.start_beat) != 0 or len(x) != int(row.valid_length) or len(x) != int(row.stop_beat_exclusive):
                raise ValueError("Full-piece manifest/cache mismatch")
            raw = read_frame(by_piece[piece].beat_time, ",", guard)
            dyn = align_dynamics(raw, read_frame(by_piece[piece].beat_dyn, ",", guard))
            times = pd.to_numeric(raw[row.performance_id], errors="coerce").to_numpy(float)
            dynamics = pd.to_numeric(dyn[row.performance_id], errors="coerce").to_numpy(float) if row.performance_id in dyn else np.full(len(x), np.nan)
            samples.append((x, validity(x, times, dynamics), {"piece_id": piece, "performance_id": row.performance_id, "full_length": len(x)}))
            files.append({"path": str(path), "sha256": sha256(path)})
        data[subset] = bucket_batches(samples)
    return data, split, files


def smoke():
    began = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(20261013)
    np.random.seed(20261013)
    random.seed(20261013)
    generator = torch.Generator().manual_seed(43)
    directory = ROOT / "reports/goal3"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ROOT / "checkpoints/goal3_transformer_smoke"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = tomllib.loads((ROOT / "configs/data_paths.toml").read_text(encoding="utf-8"))
    guard = SourceAccessGuard(Path(config["raw"]["dcml_chopin"]), Path(config["raw"]["mazurkabl"]), Path(config["legacy_read_only"]["experiment_root"]))
    guard.install()
    data, split, files = load_full_pieces(guard)
    if len(data["train"]) != 1:
        raise ValueError("Tiny-overfit contract expects one length-bucket batch")
    values, valid, lengths, ids = data["train"][0]
    mean, scale, varying = fit_normalization(values, valid)
    values = (values - mean) / scale
    valid &= varying
    spans, intervals = mask_spans(lengths, generator)
    clean, hidden, missing, padding = corrupt(values, valid, spans, lengths)
    model = FullPieceTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    initial = float(masked_loss(model(clean, hidden, missing, lengths), values, valid, spans, lengths)[0].detach())
    history, gradient_max = [], 0.0
    for step in range(1, CONFIG["smoke_steps"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = masked_loss(model(clean, hidden, missing, lengths), values, valid, spans, lengths)[0]
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
            print(f"step={step} loss={float(loss.detach()):.6f}", flush=True)
    model.eval()
    prediction = model(clean, hidden, missing, lengths)
    final, terms, selected = masked_loss(prediction, values, valid, spans, lengths)
    final_value = float(final.detach())
    if not final_value < initial * 0.2 or gradient_max <= 0:
        raise RuntimeError(f"Tiny-overfit failed: {initial} -> {final_value}")
    changed = values.clone()
    changed[hidden | ~valid] = 123456
    torch.testing.assert_close(corrupt(changed, valid, spans, lengths)[0], clean, rtol=0, atol=0)
    target = values.clone()
    target[~selected] = float("nan")
    torch.testing.assert_close(masked_loss(prediction, target, valid, spans, lengths)[0], final, rtol=0, atol=0)
    checkpoint = checkpoint_dir / "tiny_overfit.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": CONFIG,
                "step": step, "mean": mean, "scale": scale, "varying": varying, "split": split,
                "generator_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, checkpoint)
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    restored = FullPieceTransformer().eval()
    restored.load_state_dict(state["model"])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=CONFIG["lr"])
    restored_optimizer.load_state_dict(state["optimizer"])
    torch.testing.assert_close(restored(clean, hidden, missing, lengths), prediction, rtol=0, atol=0)
    for net, opt in ((model, optimizer), (restored, restored_optimizer)):
        opt.zero_grad(set_to_none=True)
        masked_loss(net(clean, hidden, missing, lengths), values, valid, spans, lengths)[0].backward()
        opt.step()
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    recovered_generator = torch.Generator()
    recovered_generator.set_state(state["generator_rng"])
    torch.testing.assert_close(mask_spans(lengths, generator)[0], mask_spans(lengths, recovered_generator)[0])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    validation = []
    for x, observed, lens, records in data["validation"]:
        x, observed = (x - mean) / scale, observed & varying
        masks, _ = mask_spans(lens, generator)
        inputs = corrupt(x, observed, masks, lens)
        with torch.no_grad():
            output = model(*inputs[:3], lens)
            val = float(masked_loss(output, x, observed, masks, lens)[0])
        if not math.isfinite(val):
            raise RuntimeError("Non-finite validation loss")
        validation.append({"samples": records, "loss": val})
    guard.access_frame().to_csv(directory / "raw_source_access.csv", index=False)
    result = {"status": "passed", "scientific_status": "implemented_and_locally_reproduced_smoke_only",
              "config": CONFIG, "parameters": sum(p.numel() for p in model.parameters()), "device": "cpu",
              "split": split, "train_samples": ids, "padded_shape": list(values.shape),
              "train_lengths": lengths.tolist(), "spans": intervals, "initial_loss": initial,
              "final_loss": final_value, "loss_ratio": final_value / initial,
              "group_loss": {k: float(v.detach()) for k, v in terms.items()}, "validation": validation,
              "gradient_max": gradient_max, "history": history, "optimizer_steps": step,
              "checkpoint_prediction_exact": True, "checkpoint_next_update_exact": True,
              "sampler_restore_exact": True, "hidden_target_copy_invariance": True,
              "unselected_target_nan_invariance": True, "metadata_excluded": METADATA,
              "derived_input_policy": "cached multiscale/motif withheld globally; local span halo4",
              "sources": files, "seconds": time.monotonic() - began,
              "limitation": "Fixed tiny sample memorization is possible; no phrase labels, no F1, no full pretraining"}
    (directory / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "parameters", "padded_shape", "train_lengths", "initial_loss", "final_loss", "validation", "seconds")}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    smoke()
