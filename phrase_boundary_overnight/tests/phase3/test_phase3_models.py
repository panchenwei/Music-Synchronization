from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.models import BeatBoundaryTCN
from src.phase3_frontend_adapter import beat_interval_energy
from src.phase3_models import ResidualGatedFusionTCN, SmallBiGRU
from src.run_phase3 import (
    Phase3Pipeline,
    boundary_rate_summary,
    context_features,
    empirical_noise_scales,
    minimum_gap_dp_probabilities,
    paired_empirical_noise,
    with_boundary_rates,
)


def test_residual_fusion_initially_equals_curve_model():
    curve = BeatBoundaryTCN(9, 32, [1, 2, 4, 8], 3, 0.0)
    model = ResidualGatedFusionTCN(curve)
    model.eval(); x = torch.randn(2, 20, 9); s = torch.randn(2, 20, 16)
    with torch.no_grad():
        combined, baseline, residual, gate = model(x, s, return_parts=True)
    assert torch.allclose(residual, torch.zeros_like(residual))
    assert torch.allclose(combined, baseline)
    assert 0 < float(gate) < 1


def test_bigru_preserves_token_resolution_and_is_compact():
    model = SmallBiGRU(25, 32)
    output = model(torch.randn(3, 64, 25))
    assert output.shape == (3, 64)
    assert sum(p.numel() for p in model.parameters()) < 60000


def test_context_features_are_fixed_width_and_finite():
    output = context_features(np.random.default_rng(1).normal(size=(11, 9)), np.random.default_rng(2).normal(size=(11, 16)))
    assert output.shape == (11, 41)
    assert np.isfinite(output).all()


def test_bayesian_posterior_is_finite_probability():
    posterior = Phase3Pipeline.bayesian_posterior(np.sin(np.linspace(0, 8, 48)))
    assert posterior.shape == (48,)
    assert np.isfinite(posterior).all()
    assert ((posterior >= 0) & (posterior <= 1)).all()


def test_boundary_rate_summary_derives_rates_from_counts():
    frame = pd.DataFrame({
        "beats": [100, 200],
        "true_boundaries": [10, 10],
        "predicted_boundaries": [20, 10],
    })
    result = boundary_rate_summary({"candidate": frame}).iloc[0]
    assert np.isclose(result.true_boundary_rate, (0.10 + 0.05) / 2)
    assert np.isclose(result.predicted_boundary_rate, (0.20 + 0.05) / 2)
    detailed = with_boundary_rates(frame, "candidate")
    assert np.allclose(detailed.true_boundary_rate, [0.10, 0.05])
    assert np.allclose(detailed.predicted_boundary_rate, [0.20, 0.05])


def test_empirical_noise_conditions_are_paired_and_monotonic():
    errors = np.array([-0.8, -0.2, 0.1, 0.4, 2.0])
    scales = empirical_noise_scales(errors)
    assert list(scales) == ["clean", "typical", "long_tail", "catastrophic"]
    assert np.all(np.diff(list(scales.values())) >= 0)
    draws = [paired_empirical_noise(errors, 32, "same-curve", scale) for scale in scales.values()]
    assert np.allclose(draws[0], 0)
    assert all(np.all(np.abs(draws[i]) <= np.abs(draws[i + 1]) + 1e-12) for i in range(3))


def test_minimum_gap_dp_keeps_strongest_compatible_peaks():
    probabilities = np.array([0.0, 0.7, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0])
    output = minimum_gap_dp_probabilities(probabilities, threshold=0.5, min_gap=4)
    selected = np.flatnonzero(output)
    assert selected.tolist() == [3, 8]
    assert np.all(np.diff(selected) >= 4)
