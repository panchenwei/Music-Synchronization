import numpy as np
import torch

from src.models import Normalizer
from src.phase2_models import CompactBoundaryTransformer, PerformanceWindowDataset, evaluate_single_performance


def test_transformer_preserves_token_shape_and_padding_invariance():
    torch.manual_seed(42)
    model = CompactBoundaryTransformer(9, 16, 2, 4, 32, 0.0).eval()
    inputs = torch.randn(2, 12, 9)
    padding = torch.zeros(2, 12, dtype=torch.bool)
    padding[1, 8:] = True
    changed = inputs.clone()
    changed[1, 8:] = 100 * torch.randn_like(changed[1, 8:])
    with torch.no_grad():
        first = model(inputs, padding)
        second = model(changed, padding)
    assert first.shape == (2, 12)
    assert torch.allclose(first[1, :8], second[1, :8], atol=1e-6)


def test_transformer_rejects_fully_masked_attention_row():
    model = CompactBoundaryTransformer(9, 16, 1, 4, 32, 0.0)
    with torch.no_grad():
        try:
            model(torch.zeros(1, 4, 9), torch.ones(1, 4, dtype=torch.bool))
        except ValueError as exc:
            assert "Fully masked" in str(exc)
        else:
            raise AssertionError("Fully masked row was accepted")


def test_attention_mask_and_loss_mask_have_distinct_semantics():
    data = {
        "piece": {
            "curves": np.zeros((1, 4, 9), dtype=np.float32),
            "labels": np.array([0, 1, 0, 0], dtype=np.float32),
            "label_mask": np.array([1, 1, 0, 1], dtype=np.float32),
        }
    }
    dataset = PerformanceWindowDataset(data, Normalizer(np.zeros(9), np.ones(9)), window=6, stride=3)
    _, _, loss_mask, attention_valid, _ = dataset[0]
    assert loss_mask.tolist() == [1, 1, 0, 1, 0, 0]
    assert attention_valid.tolist() == [1, 1, 1, 1, 0, 0]


def test_single_performance_aggregation_is_work_balanced():
    probabilities = {
        "a": {"a1": np.array([0, 1, 0]), "a2": np.array([0, 0, 0])},
        "b": {"b1": np.array([0, 1, 0])},
    }
    data = {
        key: {"labels": np.array([0, 1, 0], dtype=np.float32), "label_mask": np.ones(3)}
        for key in probabilities
    }
    perf, pieces, summary = evaluate_single_performance(probabilities, data, 0.5)
    assert len(perf) == 3 and len(pieces) == 2
    assert summary["macro_f1_tol0"] == 0.75
