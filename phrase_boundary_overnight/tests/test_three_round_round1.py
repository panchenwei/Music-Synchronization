import numpy as np
import torch

from src.slice_energy_study import model_for
from src.three_round_round1 import SCALES, extra_block


def fixture_arrays(length=26):
    tempo = np.r_[np.ones(24) * 100, np.ones(length - 24) * 50]
    curves = np.zeros((1, length, 9), np.float32)
    curves[0, :, 0] = np.log(tempo)
    curves[0, :, 7] = 1
    rmsq = np.zeros((1, length, 6), np.float32); rmsq[..., 5] = 1
    return curves, rmsq


def test_round1_five_blocks_have_fixed_shape_and_semantics():
    curves, rmsq = fixture_arrays()
    blocks = {kind: extra_block(curves, rmsq, kind) for kind in ["Z", "Q", "NQ", "MeanQ", "RMSQ"]}
    assert all(value.shape == (1, 26, 6) for value in blocks.values())
    assert np.count_nonzero(blocks["Z"]) == 0
    assert np.flatnonzero(np.any(blocks["Q"] != 0, axis=(0, 1))).tolist() == [5]
    assert np.flatnonzero(np.any(blocks["NQ"] != 0, axis=(0, 1))).tolist() == [0, 5]
    assert np.flatnonzero(np.any(blocks["MeanQ"] != 0, axis=(0, 1))).tolist() == [0, 1, 2, 3, 4, 5]
    np.testing.assert_array_equal(blocks["RMSQ"], rmsq)


def test_meanq_tail_uses_real_length_without_zero_padding():
    curves, rmsq = fixture_arrays()
    value = extra_block(curves, rmsq, "MeanQ")
    np.testing.assert_allclose(value[0, -2:, SCALES.index(24)], np.log(0.5), atol=1e-6)


def test_all_31d_conditions_share_step0_logits_and_new_columns_are_zero():
    x25 = torch.randn(2, 19, 25)
    extras = [torch.randn(2, 19, 6) for _ in range(5)]
    models = [model_for(31, 42).eval() for _ in range(5)]
    for model in models:
        assert torch.count_nonzero(model.input_projection.weight[:, 25:]) == 0
    with torch.no_grad():
        outputs = [model(torch.cat([x25, extra], -1)) for model, extra in zip(models, extras)]
    for output in outputs[1:]:
        torch.testing.assert_close(outputs[0], output)


def test_same_seed_common_weights_equal_across_fresh_models():
    a, b = model_for(31, 43).state_dict(), model_for(31, 43).state_dict()
    assert all(torch.equal(value, b[key]) for key, value in a.items())
