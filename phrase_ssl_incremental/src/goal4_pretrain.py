"""Bounded, work-disjoint unlabeled validation of full-piece Transformer SSL."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.goal1_unlabeled import SourceAccessGuard, align_dynamics, discover_sources, read_frame, sha256
from src.goal2_smoke import GROUPS, METADATA, ROOT, corrupt, validity
from src.goal3_transformer_smoke import FullPieceTransformer, bucket_batches, mask_spans, masked_loss

OUT = ROOT / "reports/goal4"
CHECKPOINTS = ROOT / "checkpoints/goal4_pretrain"
PLAN = {"split_seed": 20261014, "seeds": [17, 43, 89], "train_works": 30, "validation_works": 8,
        "reserved_works": 4, "performances_per_work": 2, "updates": 400, "eval_every": 100,
        "validation_mask_seed": 90210, "validation_views": 3, "mask_fraction": 0.12,
        "lr": 0.001, "dropout": 0.1, "batch_size": 2, "bucket_width": 128,
        "max_seconds": 1800, "gradient_clip": 1.0,
        "gate": "music beats own initialization, trained no-content, mean, interpolation by >=2% macro MSE, in all 3 seeds"}


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    pieces = pd.read_csv(ROOT / "manifests/goal1/piece_audit.csv")
    ids = sorted(pieces.loc[pieces.status == "usable", "piece_id"].tolist())
    if len(ids) != 42:
        raise ValueError("Registered cohort expects 42 usable works")
    shuffled = np.random.default_rng(PLAN["split_seed"]).permutation(ids).tolist()
    split = {"train": sorted(shuffled[:30]), "validation": sorted(shuffled[30:38]), "reserved_not_loaded": sorted(shuffled[38:])}
    for a, b in (("train", "validation"), ("train", "reserved_not_loaded"), ("validation", "reserved_not_loaded")):
        assert not set(split[a]) & set(split[b])
    contract = {"plan": PLAN, "split": split, "model": "2 blocks, d64, 4 heads, FFN128, dynamic sinusoidal positions, 74734 parameters",
                "source_sha256": sha256(Path(__file__).resolve()), "created_before_training": True,
                "input_policy": "all cached multiscale/motif columns hidden; all metadata hidden; core spans plus halo4",
                "controls": ["trained no-content, same init/order/masks", "train mean", "linear interpolation from only visible observed values", "trained music inference with zeroed content"],
                "stop": "400 updates per condition/seed; stop on nonfinite or 1800sec; no hyperparameter sweep"}
    (OUT / "registered_protocol.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return split


def load_data(split, guard):
    cfg = tomllib.loads((ROOT / "configs/data_paths.toml").read_text(encoding="utf-8"))
    sources, _ = discover_sources(Path(cfg["raw"]["dcml_chopin"]), Path(cfg["raw"]["mazurkabl"]))
    by_piece = {s.piece_id: s for s in sources}
    manifest_path = ROOT / "manifests/goal1/full_sequence_manifest.csv.gz"
    manifest = pd.read_csv(manifest_path)
    data, cache_access = {}, []
    for subset in ("train", "validation"):
        samples = []
        for piece in split[subset]:
            source = by_piece[piece]
            raw = read_frame(source.beat_time, ",", guard)
            dynamics = align_dynamics(raw, read_frame(source.beat_dyn, ",", guard))
            path = ROOT / "cache/unlabeled58" / f"{piece}.npz"
            cache_access.append({"piece_id": piece, "subset": subset, "path": str(path), "sha256": sha256(path)})
            with np.load(path, allow_pickle=False) as cache:
                if any("label" in key.lower() for key in cache.files):
                    raise ValueError("Unexpected phrase labels")
                rows = manifest[manifest.piece_id == piece].head(PLAN["performances_per_work"])
                for row in rows.itertuples(index=False):
                    index = cache["performance_ids"].tolist().index(row.performance_id)
                    x = cache["inputs"][index].copy()
                    assert row.start_beat == 0 and row.valid_length == len(x) == row.stop_beat_exclusive
                    t = pd.to_numeric(raw[row.performance_id], errors="coerce").to_numpy(float)
                    d = pd.to_numeric(dynamics[row.performance_id], errors="coerce").to_numpy(float) if row.performance_id in dynamics else np.full(len(t), np.nan)
                    samples.append((x, validity(x, t, d), {"piece_id": piece, "performance_id": row.performance_id, "full_length": len(x)}))
        data[subset] = samples
    pd.DataFrame(cache_access).to_csv(OUT / "cache_access.csv", index=False)
    return data


def normalization(samples):
    count, total, square = np.zeros(58), np.zeros(58), np.zeros(58)
    for x, valid, _ in samples:
        v = x.astype(np.float64)
        count += valid.sum(0)
        total += np.where(valid, v, 0).sum(0)
        square += np.where(valid, v ** 2, 0).sum(0)
    mean = total / np.maximum(count, 1)
    variance = np.maximum(square / np.maximum(count, 1) - mean ** 2, 0)
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(np.sqrt(variance).clip(1e-3), dtype=torch.float32), torch.tensor(variance > 1e-8)


def prepare(samples, mean, scale, varying):
    result = []
    for x, valid, ids in samples:
        result.append((((torch.tensor(x) - mean) / scale).numpy(), valid & varying.numpy(), ids))
    return bucket_batches(result, PLAN["batch_size"], PLAN["bucket_width"])


def interpolation(clean, valid, hidden, lengths):
    prediction = torch.zeros_like(clean)
    for row, length in enumerate(lengths.tolist()):
        for feature in range(58):
            visible = valid[row, :length, feature] & ~hidden[row, :length, feature]
            indices = torch.where(visible)[0].numpy()
            if len(indices):
                prediction[row, :length, feature] = torch.tensor(np.interp(np.arange(length), indices, clean[row, indices, feature].numpy()), dtype=clean.dtype)
    return prediction


def metric_rows(prediction, target, valid, spans, lengths, ids, view, method):
    selected = valid & spans[:, :, None] & (torch.arange(target.shape[1])[None, :, None] < lengths[:, None, None])
    rows = []
    for row, identifier in enumerate(ids):
        for name, (start, stop) in GROUPS.items():
            use = selected[row, :, start:stop]
            if not use.any():
                continue
            delta = prediction[row, :, start:stop][use] - target[row, :, start:stop][use]
            rows.append({**identifier, "view": view, "method": method, "group": name,
                         "mse": float(delta.square().mean()), "count": int(use.sum())})
    return rows


def macro(rows):
    frame = pd.DataFrame(rows)
    # Equal weight to groups, performances, views, and independent works.
    per_work = frame.groupby(["piece_id", "group"]).mse.mean().groupby("piece_id").mean()
    return float(per_work.mean())


def validation_views(batches):
    generator = torch.Generator().manual_seed(PLAN["validation_mask_seed"])
    views = []
    for view in range(PLAN["validation_views"]):
        for x, valid, lengths, ids in batches:
            spans, records = mask_spans(lengths, generator, PLAN["mask_fraction"])
            clean, hidden, missing, padding = corrupt(x, valid, spans, lengths)
            views.append((x, valid, lengths, ids, spans, clean, hidden, missing, view, records))
    return views


def make_model(seed, device):
    torch.manual_seed(seed)
    model = FullPieceTransformer()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = PLAN["dropout"]
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = PLAN["dropout"]
    return model.to(device)


@torch.no_grad()
def evaluate(model, views, device, condition, ablate=False, save_directory=None):
    model.eval()
    rows = []
    for index, (x, valid, lengths, ids, spans, clean, hidden, missing, view, records) in enumerate(views):
        content = torch.zeros_like(clean) if condition == "no_content" or ablate else clean
        prediction = model(content.to(device), hidden.to(device), missing.to(device), lengths.to(device)).cpu()
        if not torch.isfinite(prediction).all():
            raise RuntimeError("Non-finite evaluation prediction")
        rows.extend(metric_rows(prediction, x, valid, spans, lengths, ids, view, "ablation" if ablate else condition))
        if save_directory is not None:
            save_directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_directory / f"view_batch_{index:02d}.npz", prediction=prediction.numpy(), target=x.numpy(), valid=valid.numpy(),
                                spans=spans.numpy(), lengths=lengths.numpy(), piece_ids=np.array([i["piece_id"] for i in ids]),
                                performance_ids=np.array([i["performance_id"] for i in ids]), view=np.array(view))
    return macro(rows), rows


def run():
    began = time.monotonic()
    torch.set_num_threads(2)
    split = register()
    cfg = tomllib.loads((ROOT / "configs/data_paths.toml").read_text(encoding="utf-8"))
    guard = SourceAccessGuard(Path(cfg["raw"]["dcml_chopin"]), Path(cfg["raw"]["mazurkabl"]), Path(cfg["legacy_read_only"]["experiment_root"]))
    guard.install()
    samples = load_data(split, guard)
    mean, scale, varying = normalization(samples["train"])
    np.savez_compressed(OUT / "train_normalization.npz", mean=mean.numpy(), scale=scale.numpy(), varying=varying.numpy())
    train = prepare(samples["train"], mean, scale, varying)
    validation = prepare(samples["validation"], mean, scale, varying)
    views = validation_views(validation)
    mask_records = [{"view": row[8], "samples": row[3], "spans": row[9]} for row in views]
    (OUT / "validation_masks.json").write_text(json.dumps(mask_records, indent=2), encoding="utf-8")
    baseline_rows = []
    for x, valid, lengths, ids, spans, clean, hidden, missing, view, records in views:
        baseline_rows.extend(metric_rows(torch.zeros_like(x), x, valid, spans, lengths, ids, view, "mean"))
        baseline_rows.extend(metric_rows(interpolation(clean, valid, hidden, lengths), x, valid, spans, lengths, ids, view, "interpolation"))
    pd.DataFrame(baseline_rows).to_csv(OUT / "baselines.csv", index=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    all_history, summaries, all_rows = [], [], []
    for seed in PLAN["seeds"]:
        for condition in ("music", "no_content"):
            model = make_model(seed, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=PLAN["lr"])
            generator = torch.Generator().manual_seed(70000 + seed)
            order = np.random.default_rng(seed)
            batches = []
            exposure, train_fingerprints = 0, hashlib.sha256()
            initial, initial_rows = evaluate(model, views, device, condition)
            history = [{"seed": seed, "condition": condition, "step": 0, "train_loss": None, "validation_macro": initial}]
            best, best_step = initial, 0
            best_path = CHECKPOINTS / f"seed{seed}_{condition}.pt"
            for step in range(1, PLAN["updates"] + 1):
                if time.monotonic() - began > PLAN["max_seconds"]:
                    raise RuntimeError("Registered resource deadline reached; no new experiments")
                if not batches:
                    batches = order.permutation(len(train)).tolist()
                index = batches.pop()
                x, valid, lengths, ids = train[index]
                spans, _ = mask_spans(lengths, generator, PLAN["mask_fraction"])
                train_fingerprints.update(str(index).encode())
                train_fingerprints.update(spans.numpy().tobytes())
                clean, hidden, missing, padding = corrupt(x, valid, spans, lengths)
                if condition == "no_content":
                    clean = torch.zeros_like(clean)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(clean.to(device), hidden.to(device), missing.to(device), lengths.to(device))
                loss, terms, selected = masked_loss(prediction, x.to(device), valid.to(device), spans.to(device), lengths.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite train loss")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), PLAN["gradient_clip"])
                if not torch.isfinite(gradient):
                    raise RuntimeError("Non-finite gradient")
                optimizer.step()
                exposure += int(lengths.sum())
                if step % PLAN["eval_every"] == 0:
                    value, rows = evaluate(model, views, device, condition)
                    record = {"seed": seed, "condition": condition, "step": step, "train_loss": float(loss.detach()), "validation_macro": value,
                              **{f"validation_{group}": float(pd.DataFrame(rows).query("group == @group").groupby("piece_id").mse.mean().mean()) for group in GROUPS}}
                    history.append(record)
                    print(f"seed={seed} {condition} step={step} val={value:.6f}", flush=True)
                    if step == PLAN["eval_every"] or value < best:
                        best, best_step = value, step
                        state = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
                                 "step": step, "seed": seed, "condition": condition, "plan": PLAN, "split": split,
                                 "mean": mean, "scale": scale, "varying": varying, "mask_rng": generator.get_state(),
                                 "order_rng": order.bit_generator.state, "remaining_batches": batches.copy(), "exposure": exposure,
                                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
                        torch.save(state, best_path)
            state = torch.load(best_path, weights_only=False, map_location=device)
            model.load_state_dict(state["model"])
            saved_dir = OUT / "predictions" / f"seed{seed}_{condition}"
            best_recomputed, rows = evaluate(model, views, device, condition, save_directory=saved_dir)
            assert math.isclose(best_recomputed, best, rel_tol=1e-6, abs_tol=1e-7)
            ablated, ablation_rows = evaluate(model, views, device, condition, ablate=True)
            summaries.append({"seed": seed, "condition": condition, "initial": initial, "best_validation": best_recomputed,
                              "best_step": best_step, "last_validation": history[-1]["validation_macro"], "content_ablated": ablated,
                              "effective_beat_exposure": exposure, "train_order_mask_sha256": train_fingerprints.hexdigest()})
            for row in rows + ablation_rows:
                row["seed"] = seed
                row["trained_condition"] = condition
            all_rows.extend(rows + ablation_rows)
            all_history.extend(history)
            pd.DataFrame(all_history).to_csv(OUT / "learning_curves.csv", index=False)
            pd.DataFrame(summaries).to_csv(OUT / "run_summary.csv", index=False)
            pd.DataFrame(all_rows).to_csv(OUT / "validation_metrics.csv", index=False)
    guard.access_frame().to_csv(OUT / "raw_source_access.csv", index=False)
    frame = pd.DataFrame(summaries)
    for seed in PLAN["seeds"]:
        a, b = frame[frame.seed == seed].iloc[0], frame[frame.seed == seed].iloc[1]
        assert a.train_order_mask_sha256 == b.train_order_mask_sha256 and a.effective_beat_exposure == b.effective_beat_exposure
    baselines = {name: macro([row for row in baseline_rows if row["method"] == name]) for name in ("mean", "interpolation")}
    gates = []
    for seed in PLAN["seeds"]:
        pair = frame[frame.seed == seed].set_index("condition")
        music = float(pair.loc["music", "best_validation"])
        control = float(pair.loc["no_content", "best_validation"])
        gates.append({"seed": seed, "passed": music < 0.98 * min(control, baselines["mean"], baselines["interpolation"], float(pair.loc["music", "initial"])),
                      "music": music, "no_content": control, "relative_content_gain": (control - music) / control})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for seed in PLAN["seeds"]:
        for condition in ("music", "no_content"):
            curve = pd.DataFrame(all_history).query("seed == @seed and condition == @condition")
            axes[0].plot(curve.step, curve.validation_macro, label=f"{condition} seed{seed}", linestyle="-" if condition == "music" else "--")
    axes[0].axhline(baselines["mean"], color="gray", label="mean")
    axes[0].axhline(baselines["interpolation"], color="black", linestyle=":", label="interpolation")
    axes[0].set(xlabel="optimizer step", ylabel="validation work-macro MSE")
    axes[0].legend(fontsize=6)
    rows = pd.DataFrame(all_rows)
    groups = list(GROUPS)
    for offset, condition in enumerate(("music", "no_content")):
        subset = rows[(rows.trained_condition == condition) & (rows.method == condition)]
        scores = subset.groupby(["seed", "piece_id", "group"]).mse.mean().groupby("group").mean()
        axes[1].bar(np.arange(4) + (offset - .5) * .35, [scores[g] for g in groups], .35, label=condition)
    axes[1].set_xticks(np.arange(4), groups)
    axes[1].set_ylabel("validation group MSE")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", dpi=180, facecolor="white")
    plt.close(fig)
    result = {"status": "completed", "scientific_status": "preliminary_unlabeled_validation", "plan": PLAN,
              "split": split, "device": str(device), "train_performances": len(samples["train"]),
              "validation_performances": len(samples["validation"]), "baselines": baselines,
              "gates": gates, "promote": all(g["passed"] for g in gates), "runs": summaries,
              "seconds": time.monotonic() - began,
              "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
              "checkpoint_sha256": {path.name: sha256(path) for path in CHECKPOINTS.glob("*.pt")},
              "no_labels": True, "reserved_not_loaded": True}
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "baselines", "gates", "promote", "seconds", "peak_gpu_mib")}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
