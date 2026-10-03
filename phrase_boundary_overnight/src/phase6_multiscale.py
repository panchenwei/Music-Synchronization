from __future__ import annotations

import copy

import numpy as np


SCALES = (4, 8, 16)
TEMPO_INDEX = 0
DYNAMICS_INDEX = 3
TIME_MASK_INDEX = 7
DYNAMICS_MASK_INDEX = 8


def segment_novelty(values: np.ndarray, valid: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """Offline absolute mean contrast between preceding/following beat segments."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    out = np.zeros(len(values), dtype=np.float32)
    quality = np.zeros(len(values), dtype=np.float32)
    minimum = max(1, scale // 2)
    for index in range(len(values)):
        left = np.arange(max(0, index - scale), index)
        right = np.arange(index, min(len(values), index + scale))
        left = left[valid[left]]
        right = right[valid[right]]
        if len(left) >= minimum and len(right) >= minimum:
            out[index] = abs(float(values[right].mean() - values[left].mean()))
            quality[index] = 1.0
    return out, quality


def augment_piece(item: dict[str, np.ndarray], groups: str) -> dict[str, np.ndarray]:
    if groups not in {"tempo", "dynamics", "both"}:
        raise ValueError(groups)
    augmented = {key: value for key, value in item.items()}
    curves = np.asarray(item["curves"], dtype=np.float32)
    names = [str(x) for x in item["curve_feature_names"]]
    added = []
    added_names = []
    added_quality = []
    specifications = []
    if groups in {"tempo", "both"}:
        specifications.append(("tempo", TEMPO_INDEX, TIME_MASK_INDEX))
    if groups in {"dynamics", "both"}:
        specifications.append(("dynamics_proxy", DYNAMICS_INDEX, DYNAMICS_MASK_INDEX))
    for prefix, value_index, mask_index in specifications:
        per_scale = []
        quality_scale = []
        for scale in SCALES:
            values = []
            qualities = []
            for performance in curves:
                novelty, quality = segment_novelty(performance[:, value_index], performance[:, mask_index] > 0.5, scale)
                values.append(novelty)
                qualities.append(quality)
            per_scale.append(np.stack(values, axis=0))
            quality_scale.append(np.stack(qualities, axis=0))
            added_names.append(f"{prefix}_novelty_s{scale}")
        added.extend(per_scale)
        added_quality.extend(quality_scale)
    novelty = np.stack(added, axis=-1).astype(np.float32)
    quality = np.stack(added_quality, axis=-1).astype(np.float32)
    augmented["curves"] = np.concatenate([curves, novelty], axis=-1)
    augmented["curve_feature_names"] = np.asarray(names + added_names)
    augmented["phase6_multiscale_quality"] = quality
    return augmented


def augment_dataset(data: dict[str, dict[str, np.ndarray]], groups: str) -> dict[str, dict[str, np.ndarray]]:
    return {piece_id: augment_piece(item, groups) for piece_id, item in data.items()}
