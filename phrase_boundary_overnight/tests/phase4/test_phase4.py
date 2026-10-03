from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.phase4_models import ExperimentDeadline, _deadline_guard, soft_target, softened_training_data
from src.run_phase4 import Phase4Pipeline
from src.evaluation import one_to_one_counts
from scripts.plot_best_test_piece import matching_pairs


def synthetic_piece(offset: float = 0.0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    beats = 20
    curves = rng.normal(offset, 1.0, size=(2, beats, 9)).astype(np.float32)
    score = rng.normal(offset, 1.0, size=(beats, 16)).astype(np.float32)
    labels = np.zeros(beats, np.float32)
    labels[[4, 10, 16]] = 1
    return {
        "curves": curves,
        "score_phase3": score,
        "labels": labels,
        "label_mask": np.ones(beats, np.float32),
        "performance_ids": np.array(["p0", "p1"]),
    }


def test_soft_target_respects_edges_and_mask_without_mutation():
    labels = np.array([1, 0, 0, 1, 0], np.float32)
    mask = np.array([1, 1, 0, 1, 1], np.float32)
    original = labels.copy()
    result = soft_target(labels, mask, radius=1, neighbor_weight=0.5)
    assert np.allclose(result, [1, 0.5, 0, 1, 0.5])
    assert np.array_equal(labels, original)


def test_softened_training_data_does_not_mutate_source():
    source = {"work": synthetic_piece()}
    before = source["work"]["labels"].copy()
    result = softened_training_data(source, radius=1, neighbor_weight=0.5)
    assert np.array_equal(source["work"]["labels"], before)
    assert result["work"]["labels"].sum() > before.sum()


def test_deadline_guard_fails_before_training_cutoff():
    with pytest.raises(ExperimentDeadline):
        _deadline_guard(datetime.now().astimezone() + timedelta(seconds=1), 1)


def test_real_split_has_zero_piece_and_opus_overlap():
    root = Path(__file__).resolve().parents[2]
    frame = pd.read_csv(root / "artifacts/phase2/splits/opus_split_manifest.csv")
    for fold in range(5):
        parts = {name: frame[(frame.fold == fold) & (frame.split == name)] for name in ["train", "validation", "test"]}
        for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            assert set(parts[left].piece_id).isdisjoint(parts[right].piece_id)
            assert set(parts[left].opus.astype(str)).isdisjoint(parts[right].opus.astype(str))


def test_svm_sampler_is_work_balanced_and_finite():
    pipeline = object.__new__(Phase4Pipeline)
    data = {"work_a": synthetic_piece(0.0), "work_b": synthetic_piece(20.0)}
    x, y, weights, counts = pipeline._sample_svm(data, cap=18, seed=42)
    assert x.shape[1] == 41
    assert set(counts) == set(data)
    assert len(set(counts.values())) == 1
    assert np.isfinite(x).all() and np.isfinite(weights).all()
    assert np.isclose(sum(weights[: counts["work_a"]]), 1.0)
    assert np.isclose(sum(weights[counts["work_a"] :]), 1.0)
    assert set(np.unique(y)) == {0, 1}


def test_registry_update_preserves_preregistered_fields(tmp_path: Path):
    pipeline = object.__new__(Phase4Pipeline)
    pipeline.reports = tmp_path
    pipeline.register({"id": "candidate", "hypothesis": "falsifiable", "cost": "bounded"})
    pipeline.register({"id": "candidate", "status": "negative", "fold0_validation_f1": 0.2})
    row = pipeline.registry()["experiments"][0]
    assert row["hypothesis"] == "falsifiable"
    assert row["cost"] == "bounded"
    assert row["status"] == "negative"


def test_visual_matching_reuses_evaluator_assignment_rule():
    predicted = np.array([9, 10, 20, 30])
    truth = np.array([10, 21, 40])
    pairs = matching_pairs(predicted, truth, tolerance=1)
    counts = one_to_one_counts(predicted, truth, tolerance=1)
    assert len(pairs) == counts.tp
    assert len({p for p, _, _ in pairs}) == len(pairs)
    assert len({t for _, t, _ in pairs}) == len(pairs)
    assert pairs[0] == (10, 10, 0)
