"""Frozen work-disjoint Scratch versus SSL Transformer phrase-start comparison."""
from __future__ import annotations

import copy
import json
import math
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from src.goal1_unlabeled import SourceAccessGuard, _score_key, align_dynamics, discover_sources, read_frame, sha256, to_float
from src.goal2_smoke import ROOT, validity
from src.goal3_transformer_smoke import FullPieceTransformer
from src.goal4_pretrain import normalization, prepare

OUT = ROOT / "reports/goal5"
CHECKPOINTS = ROOT / "checkpoints/goal5_downstream"
PLAN = {"seeds": [17, 43, 89], "updates": 600, "eval_every": 100, "lr": 0.0003,
        "dropout": 0.1, "batch_size": 2, "thresholds": [round(i * .05, 2) for i in range(1, 20)],
        "decoder": "deterministic local maxima; equal-score plateau leftmost; minimum gap2",
        "labels": "DCML { phrase starts; floor quarterbeat; first beat excluded; annotated first-start to last-end coverage",
        "checkpoint_selection": "validation work-macro AP", "threshold_selection": "validation work-macro +/-1 beat F1",
        "test_policy": "only evaluate after all 6 validation-selected models and thresholds frozen",
        "max_seconds": 1800}


class DownstreamGuard(SourceAccessGuard):
    def validate_path(self, path):
        path = path.resolve()
        if self._under(path, self.dcml_root):
            relative = path.relative_to(self.dcml_root)
            if len(relative.parts) == 2 and relative.parts[0] == "harmonies" and path.name.endswith(".harmonies.tsv"):
                return "dcml_phrase_annotations_downstream_only"
        return super().validate_path(path)


def map_starts(harmonies, measures, beats, piece):
    required = {"mc", "mc_onset", "phraseend"}
    if not required <= set(harmonies.columns):
        raise ValueError(f"{piece}: missing label coordinate columns")
    events = []
    for index, row in harmonies.iterrows():
        marker = str(row.phraseend) if not pd.isna(row.phraseend) else ""
        if "{" not in marker and "}" not in marker:
            continue
        local = 4 * to_float(row.mc_onset)
        if not math.isfinite(local):
            raise ValueError(f"{piece}: label row {index} lacks mc_onset")
        occurrences = measures[measures.mc == int(row.mc)]
        if occurrences.empty:
            raise ValueError(f"{piece}: label mc {row.mc} absent from performed score")
        for occurrence in occurrences.itertuples(index=False):
            if local < 0 or local > occurrence.duration_qb + 1e-7:
                raise ValueError(f"{piece}: label onset outside measure")
            q = float(occurrence.start_qb) + local
            if not 0 <= q < beats:
                raise ValueError(f"{piece}: phrase marker outside beat grid at {q}")
            events.append({"piece_id": piece, "source_row": int(index), "mc": int(row.mc), "visit": int(occurrence.visit),
                           "mc_onset": str(row.mc_onset), "occurrence_start_qb": float(occurrence.start_qb),
                           "quarterbeat": q, "beat_index": int(math.floor(q)), "quantization_error": q - math.floor(q),
                           "marker": marker, "is_start": "{" in marker, "is_end": "}" in marker})
    starts = sorted({event["beat_index"] for event in events if event["is_start"]})
    ends = sorted({event["beat_index"] for event in events if event["is_end"]})
    if not starts or not ends or ends[-1] <= starts[0]:
        raise ValueError(f"{piece}: no usable explicitly bracketed annotation coverage")
    labels, coverage = np.zeros(beats, np.float32), np.zeros(beats, bool)
    coverage[starts[0]:ends[-1] + 1] = True
    coverage[0] = False
    for beat in starts:
        if coverage[beat]:
            labels[beat] = 1
    if labels.sum() == 0:
        raise ValueError(f"{piece}: no nontrivial phrase starts")
    return labels, coverage, events


def decode(probability, coverage, threshold):
    candidates = []
    for index, value in enumerate(probability):
        if not coverage[index] or value < threshold:
            continue
        left = probability[index - 1] if index and coverage[index - 1] else -np.inf
        right = probability[index + 1] if index + 1 < len(probability) and coverage[index + 1] else -np.inf
        if value > left and value >= right:
            candidates.append(index)
    chosen = []
    for index in sorted(candidates, key=lambda i: (-probability[i], i)):
        if all(abs(index - other) >= 2 for other in chosen):
            chosen.append(index)
    return np.asarray(sorted(chosen), int)


def match_f1(predicted, actual, tolerance):
    # Sorted interval matching gives maximum cardinality for fixed distance tolerance.
    a, b = sorted(predicted), sorted(actual)
    i = j = count = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j] - tolerance:
            i += 1
        elif a[i] > b[j] + tolerance:
            j += 1
        else:
            count += 1
            i += 1
            j += 1
    precision = count / len(a) if a else 0.0
    recall = count / len(b) if b else 0.0
    f1 = 2 * count / (len(a) + len(b)) if a or b else 1.0
    return precision, recall, f1


class BoundaryTransformer(FullPieceTransformer):
    def __init__(self):
        super().__init__()
        del self.heads
        self.classifier = nn.Linear(64, 1)

    def forward(self, clean, hidden, missing, lengths):
        n = clean.shape[1]
        padding = torch.arange(n, device=clean.device)[None] >= lengths.to(clean.device)[:, None]
        x = torch.where(hidden, self.mask_token, clean)
        x = torch.where(missing & ~hidden, self.missing_token, x)
        x = self.projection(x)
        position = torch.arange(n, device=x.device, dtype=x.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, 64, 2, device=x.device, dtype=x.dtype) * (-math.log(10000) / 64))
        encoding = torch.zeros(n, 64, device=x.device, dtype=x.dtype)
        encoding[:, ::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        encoded = self.encoder((x + encoding).masked_fill(padding[:, :, None], 0), src_key_padding_mask=padding)
        return self.classifier(encoded).squeeze(-1).masked_fill(padding, 0)


def load_subset(subset, split, guard, cfg, labels):
    sources, _ = discover_sources(Path(cfg["raw"]["dcml_chopin"]), Path(cfg["raw"]["mazurkabl"]))
    by_piece = {s.piece_id: s for s in sources}
    harmonies = {_score_key(path): path for path in (Path(cfg["raw"]["dcml_chopin"]) / "harmonies").glob("*.harmonies.tsv")}
    manifest = pd.read_csv(ROOT / "manifests/goal1/full_sequence_manifest.csv.gz")
    measures = pd.read_csv(ROOT / "manifests/goal1/measure_mapping.csv")
    samples, mappings = [], []
    for piece in split[subset]:
        source = by_piece[piece]
        rows = manifest[manifest.piece_id == piece].head(2)
        path = ROOT / "cache/unlabeled58" / f"{piece}.npz"
        annotations_path = harmonies[_score_key(source.notes)]
        guard.validate_path(annotations_path)
        annotation = pd.read_csv(annotations_path, sep="\t", usecols=["mc", "mc_onset", "phraseend"])
        raw = read_frame(source.beat_time, ",", guard)
        dynamics = align_dynamics(raw, read_frame(source.beat_dyn, ",", guard))
        with np.load(path, allow_pickle=False) as cache:
            beats = cache["inputs"].shape[1]
            y, coverage, mapping = map_starts(annotation, measures[measures.piece_id == piece], beats, piece)
            labels[piece] = (y, coverage)
            mappings.extend(mapping)
            for row in rows.itertuples(index=False):
                index = cache["performance_ids"].tolist().index(row.performance_id)
                x = cache["inputs"][index].copy()
                assert row.start_beat == 0 and row.valid_length == len(x) == row.stop_beat_exclusive
                t = pd.to_numeric(raw[row.performance_id], errors="coerce").to_numpy(float)
                d = pd.to_numeric(dynamics[row.performance_id], errors="coerce").to_numpy(float) if row.performance_id in dynamics else np.full(len(t), np.nan)
                samples.append((x, validity(x, t, d), {"piece_id": piece, "performance_id": row.performance_id, "full_length": len(x)}))
    pd.DataFrame(mappings).to_csv(OUT / f"{subset}_label_mapping.csv", index=False)
    return samples


def supervision(x, valid, lengths, ids, labels):
    padding = torch.arange(x.shape[1])[None] >= lengths[:, None]
    missing = ~valid & ~padding[:, :, None]
    clean = torch.where(valid & ~padding[:, :, None], x, 0)
    hidden = torch.zeros_like(valid)
    y, mask = torch.zeros(x.shape[:2]), torch.zeros(x.shape[:2], dtype=torch.bool)
    for i, info in enumerate(ids):
        target, coverage = labels[info["piece_id"]]
        y[i, :len(target)] = torch.tensor(target)
        mask[i, :len(target)] = torch.tensor(coverage)
    return clean, hidden, missing, y, mask


@torch.no_grad()
def predict(model, batches, labels, device):
    model.eval()
    records = []
    for x, valid, lengths, ids in batches:
        clean, hidden, missing, y, coverage = supervision(x, valid, lengths, ids, labels)
        probability = torch.sigmoid(model(clean.to(device), hidden.to(device), missing.to(device), lengths.to(device))).cpu().numpy()
        assert np.isfinite(probability).all()
        for i, info in enumerate(ids):
            n = int(lengths[i])
            records.append({**info, "probability": probability[i, :n].copy(), "labels": y[i, :n].numpy(), "coverage": coverage[i, :n].numpy()})
    return records


def score_records(records, threshold):
    rows = []
    for record in records:
        probability, y, coverage = (record[k] for k in ("probability", "labels", "coverage"))
        actual = np.flatnonzero((y > .5) & coverage)
        predicted = decode(probability, coverage, threshold)
        p, r, f1 = match_f1(predicted, actual, 1)
        exact = match_f1(predicted, actual, 0)[2]
        ap = average_precision_score(y[coverage], probability[coverage]) if actual.size else float("nan")
        rows.append({"piece_id": record["piece_id"], "performance_id": record["performance_id"], "P": p, "R": r, "F1": f1,
                     "Exact_F1": exact, "AP": ap, "predictions": len(predicted), "truth": len(actual)})
    frame = pd.DataFrame(rows)
    piece = frame.groupby("piece_id")[["P", "R", "F1", "Exact_F1", "AP"]].mean()
    return piece.mean().to_dict(), frame, piece


def make_model(seed, device, condition):
    torch.manual_seed(seed)
    model = BoundaryTransformer()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = PLAN["dropout"]
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = PLAN["dropout"]
    if condition == "ssl":
        source = ROOT / "checkpoints/goal4_pretrain" / f"seed{seed}_music.pt"
        state = torch.load(source, weights_only=False, map_location="cpu")
        transferred = {k: v for k, v in state["model"].items() if not k.startswith("heads.")}
        incompatible = model.load_state_dict(transferred, strict=False)
        assert set(incompatible.missing_keys) == {"classifier.weight", "classifier.bias"}
        assert not incompatible.unexpected_keys
    return model.to(device)


def run():
    began = time.monotonic()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    previous = json.loads((ROOT / "reports/goal4/registered_protocol.json").read_text())
    split = {"train": previous["split"]["train"], "validation": previous["split"]["validation"], "test": previous["split"]["reserved_not_loaded"]}
    contract = {"plan": PLAN, "split": split, "pretrain_sources": {str(seed): sha256(ROOT / "checkpoints/goal4_pretrain" / f"seed{seed}_music.pt") for seed in PLAN["seeds"]},
                "pretrain_prior_gate": "failed; this downstream comparison is diagnostic, not a promised improvement",
                "input": "same full-piece 58-D normalized valid semantic features for both arms; missing metadata token; no masked spans downstream",
                "source_sha256": sha256(Path(__file__).resolve())}
    (OUT / "registered_protocol.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    cfg = tomllib.loads((ROOT / "configs/data_paths.toml").read_text(encoding="utf-8"))
    guard = DownstreamGuard(Path(cfg["raw"]["dcml_chopin"]), Path(cfg["raw"]["mazurkabl"]), Path(cfg["legacy_read_only"]["experiment_root"]))
    guard.install()
    labels = {}
    training_samples = load_subset("train", split, guard, cfg, labels)
    validation_samples = load_subset("validation", split, guard, cfg, labels)
    mean, scale, varying = normalization(training_samples)
    np.savez_compressed(OUT / "train_normalization.npz", mean=mean.numpy(), scale=scale.numpy(), varying=varying.numpy())
    train = prepare(training_samples, mean, scale, varying)
    validation = prepare(validation_samples, mean, scale, varying)
    positive = sum(float(labels[piece][0].sum()) for piece in split["train"])
    negative = sum(float(labels[piece][1].sum() - labels[piece][0].sum()) for piece in split["train"])
    pos_weight = min(negative / max(positive, 1), 50)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight), reduction="none")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = criterion.to(device)
    frozen, history = [], []
    # All validation choices complete before loading any test annotation/cache.
    for seed in PLAN["seeds"]:
        for condition in ("scratch", "ssl"):
            model = make_model(seed, device, condition)
            optimizer = torch.optim.Adam(model.parameters(), lr=PLAN["lr"])
            order = np.random.default_rng(seed)
            available, best, exposure = [], -1.0, 0
            for step in range(1, PLAN["updates"] + 1):
                if time.monotonic() - began > PLAN["max_seconds"]:
                    raise RuntimeError("Resource deadline reached")
                if not available:
                    available = order.permutation(len(train)).tolist()
                x, valid, lengths, ids = train[available.pop()]
                clean, hidden, missing, y, use = supervision(x, valid, lengths, ids, labels)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(clean.to(device), hidden.to(device), missing.to(device), lengths.to(device))
                loss = criterion(logits, y.to(device))[use.to(device)].mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite supervised loss")
                loss.backward()
                gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gradient):
                    raise RuntimeError("Non-finite supervised gradient")
                optimizer.step()
                exposure += int(lengths.sum())
                if step % PLAN["eval_every"] == 0:
                    records = predict(model, validation, labels, device)
                    metrics, _, _ = score_records(records, .5)
                    history.append({"seed": seed, "condition": condition, "step": step, "train_batch_loss": float(loss.detach()), **metrics})
                    print(f"{seed} {condition} step{step} validationAP={metrics['AP']:.4f}", flush=True)
                    if metrics["AP"] > best:
                        best = metrics["AP"]
                        path = CHECKPOINTS / f"seed{seed}_{condition}.pt"
                        torch.save({"model": copy.deepcopy(model.state_dict()), "optimizer": optimizer.state_dict(), "step": step,
                                    "plan": PLAN, "split": split, "mean": mean, "scale": scale, "varying": varying, "pos_weight": pos_weight}, path)
            state = torch.load(path, weights_only=False, map_location=device)
            model.load_state_dict(state["model"])
            records = predict(model, validation, labels, device)
            scans = [(threshold, score_records(records, threshold)[0]["F1"]) for threshold in PLAN["thresholds"]]
            threshold, score = max(scans, key=lambda pair: (pair[1], pair[0]))
            frozen.append({"seed": seed, "condition": condition, "checkpoint": str(path), "checkpoint_sha256": sha256(path),
                           "best_step": state["step"], "validation_AP": best, "threshold": threshold, "validation_F1": score,
                           "threshold_scan": scans, "effective_beat_exposure": exposure})
    (OUT / "frozen_models_and_thresholds.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(OUT / "learning_curves.csv", index=False)
    guard.access_frame().to_csv(OUT / "sources_before_test.csv", index=False)
    test_samples = load_subset("test", split, guard, cfg, labels)
    test = prepare(test_samples, mean, scale, varying)
    summary, piece_rows = [], []
    for choice in frozen:
        model = make_model(choice["seed"], device, "scratch")
        state = torch.load(choice["checkpoint"], weights_only=False, map_location=device)
        model.load_state_dict(state["model"])
        records = predict(model, test, labels, device)
        metrics, perf, pieces = score_records(records, choice["threshold"])
        summary.append({"seed": choice["seed"], "condition": choice["condition"], "threshold": choice["threshold"], **metrics})
        for piece, row in pieces.iterrows():
            piece_rows.append({"seed": choice["seed"], "condition": choice["condition"], "piece_id": piece, **row.to_dict()})
        name = f"seed{choice['seed']}_{choice['condition']}"
        perf.to_csv(OUT / f"{name}_test_performance_metrics.csv", index=False)
        for record in records:
            np.savez_compressed(OUT / f"{name}_{record['piece_id']}_{record['performance_id']}.npz",
                                probability=record["probability"], labels=record["labels"], coverage=record["coverage"])
    pd.DataFrame(summary).to_csv(OUT / "test_summary.csv", index=False)
    pd.DataFrame(piece_rows).to_csv(OUT / "test_piece_metrics.csv", index=False)
    guard.access_frame().to_csv(OUT / "all_source_access.csv", index=False)
    differences = []
    for seed in PLAN["seeds"]:
        rows = pd.DataFrame(summary).query("seed == @seed").set_index("condition")
        differences.append({"seed": seed, **{metric: float(rows.loc["ssl", metric] - rows.loc["scratch", metric]) for metric in ("F1", "Exact_F1", "AP")}})
    result = {"status": "completed", "scientific_status": "preliminary_four_work_test_comparison", "split": split,
              "model_parameters": sum(p.numel() for p in model.parameters()), "pos_weight": pos_weight,
              "runs": summary, "ssl_minus_scratch": differences, "seconds": time.monotonic() - began,
              "limitation": "Only four held-out works, two performances/work; one frozen development split; no architecture tuning on test"}
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
