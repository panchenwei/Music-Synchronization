import numpy as np
import pandas as pd
import torch

from src.data import _nearest_beat, align_dynamics_to_beats, canonical_piece_id, choose_measure_path, make_piece_splits, to_float, unfold_measure_path
from src.evaluation import one_to_one_counts
from src.models import BeatBoundaryTCN


def test_fraction_parsing_and_piece_id():
    assert to_float("5/8") == 0.625
    assert canonical_piece_id("Op. 6", 1) == "chopin_op06_no1"


def test_repeat_graph_unfolding():
    measures = pd.DataFrame(
        [
            {"mc": 1, "duration_qb": 1, "next": "2"},
            {"mc": 2, "duration_qb": 3, "next": "1, 3"},
            {"mc": 3, "duration_qb": 2, "next": "-1"},
        ]
    )
    path = unfold_measure_path(measures)
    assert [x["mc"] for x in path] == [1, 2, 1, 2, 3]
    assert sum(x["duration_qb"] for x in path) == 10


def test_repeat_graph_can_exit_with_second_negative_candidate():
    measures = pd.DataFrame(
        [
            {"mc": 1, "duration_qb": 1, "next": "2"},
            {"mc": 2, "duration_qb": 1, "next": "1, -1"},
        ]
    )
    assert [x["mc"] for x in unfold_measure_path(measures)] == [1, 2, 1, 2]


def test_jump_backward_stops_at_play_until_marker():
    measures = pd.DataFrame(
        [
            {"mc": 1, "duration_qb": 1, "next": "2", "markers": "segno"},
            {"mc": 2, "duration_qb": 1, "next": "3", "markers": "fine"},
            {"mc": 3, "duration_qb": 1, "next": "1", "jump_bwd": "segno", "play_until": "fine"},
        ]
    )
    assert [x["mc"] for x in unfold_measure_path(measures)] == [1, 2, 3, 1, 2]


def test_length_selection_can_choose_folded_and_reject_mismatch():
    measures = pd.DataFrame(
        [
            {"mc": 1, "duration_qb": 3, "next": "2"},
            {"mc": 2, "duration_qb": 3, "next": "1, 3"},
            {"mc": 3, "duration_qb": 3, "next": "-1"},
        ]
    )
    _, mode, error = choose_measure_path(measures, n_beats=9)
    assert mode == "folded" and error == 0
    path, mode, _ = choose_measure_path(measures, n_beats=30)
    assert path == [] and mode == "score_length_mismatch"


def test_dynamics_aligns_by_measure_and_beat_not_row_count():
    beats = pd.DataFrame({"measure_number": [1, 2, 2], "beat_number": [2, 0, 1], "p": [0.1, 0.8, 1.5]})
    dynamics = pd.DataFrame({"measure_number": [2, 2], "beat_number": [0, 1], "p": [0.4, 0.6]})
    aligned = align_dynamics_to_beats(beats, dynamics)
    assert len(aligned) == 3
    assert np.isnan(aligned.loc[0, "p"])
    assert aligned.loc[1:, "p"].tolist() == [0.4, 0.6]


def test_mapping_masks_exact_half_beat_tie():
    status, index, distance = _nearest_beat(4.5, 20, 0.5)
    assert (status, index, distance) == ("ambiguous_tie", None, 0.5)
    assert _nearest_beat(4.49, 20, 0.5)[1] == 4
    assert _nearest_beat(float("nan"), 20, 0.5)[0] == "invalid_coordinate"


def test_piece_splits_have_zero_overlap():
    pieces = [f"piece_{i:02d}" for i in range(45)]
    splits = make_piece_splits(pieces, seed=42, folds=5, validation_count=6)
    assert len(splits) == 5
    for groups in splits.values():
        train, validation, test = map(set, [groups["train"], groups["validation"], groups["test"]])
        assert len(train) == 30 and len(validation) == 6 and len(test) == 9
        assert not train & validation and not train & test and not validation & test


def test_one_to_one_boundary_matching():
    counts = one_to_one_counts([9, 10, 20], [10, 21], tolerance=1)
    assert (counts.tp, counts.fp, counts.fn) == (2, 1, 0)


def test_tcn_preserves_token_resolution_and_loss_mask_semantics():
    model = BeatBoundaryTCN(input_dim=9, hidden_channels=8, dilations=[1, 2], kernel_size=3, dropout=0.0)
    inputs = torch.randn(2, 64, 9)
    logits = model(inputs)
    assert logits.shape == (2, 64)

    labels_a = torch.zeros_like(logits)
    labels_b = labels_a.clone()
    labels_b[:, 48:] = 1.0
    loss_mask = torch.ones_like(logits)
    loss_mask[:, 48:] = 0.0
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss_a = (criterion(logits, labels_a) * loss_mask).sum() / loss_mask.sum()
    loss_b = (criterion(logits, labels_b) * loss_mask).sum() / loss_mask.sum()
    assert torch.allclose(loss_a, loss_b)
