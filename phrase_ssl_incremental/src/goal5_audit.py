"""Independent probability/event audit after all downstream choices are frozen."""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from sklearn.metrics import average_precision_score

from src.goal1_unlabeled import sha256
from src.goal2_smoke import ROOT

OUT = ROOT / "reports/goal5"


def independent_candidates(probability, mask, threshold):
    eligible = []
    for i in np.flatnonzero(mask & (probability >= threshold)):
        prior = probability[i - 1] if i and mask[i - 1] else -np.inf
        after = probability[i + 1] if i + 1 < len(probability) and mask[i + 1] else -np.inf
        if probability[i] > prior and probability[i] >= after:
            eligible.append(int(i))
    accepted = []
    for i in sorted(eligible, key=lambda k: (-probability[k], k)):
        if not any(abs(i - k) < 2 for k in accepted):
            accepted.append(i)
    return np.asarray(accepted, dtype=int)


def independent_event_scores(predicted, actual, tolerance):
    if len(predicted) and len(actual):
        edges = np.abs(predicted[:, None] - actual[None, :]) <= tolerance
        matching = maximum_bipartite_matching(csr_matrix(edges.astype(int)), perm_type="column")
        tp = int((matching >= 0).sum())
    else:
        tp = 0
    p = tp / len(predicted) if len(predicted) else 0.
    r = tp / len(actual) if len(actual) else 0.
    f1 = 2 * tp / (len(predicted) + len(actual)) if len(predicted) + len(actual) else 1.
    return p, r, f1


def run():
    protocol = json.loads((OUT / "registered_protocol.json").read_text())
    frozen = json.loads((OUT / "frozen_models_and_thresholds.json").read_text())
    split = protocol["split"]
    assert not set(split["train"]) & set(split["validation"])
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["validation"]) & set(split["test"])
    before = pd.read_csv(OUT / "sources_before_test.csv")
    all_access = pd.read_csv(OUT / "all_source_access.csv")
    assert len(before) == 114 and len(all_access) == 126
    assert set(all_access.role) == {"mazurkabl_beat_time", "mazurkabl_beat_dyn", "dcml_phrase_annotations_downstream_only"}
    test_map = pd.read_csv(OUT / "test_label_mapping.csv")
    for piece in split["test"]:
        # Source paths include op/number identifiers from raw filenames.
        opus = piece.split("_op")[1].split("_")[0]
        number = piece.rsplit("no", 1)[1]
        harmonic = all_access[all_access.role == "dcml_phrase_annotations_downstream_only"]
        matching = harmonic[harmonic.path.str.contains(f"op{int(opus)}-{number}.", regex=False)]
        assert len(matching) == 1
        assert matching.path.iloc[0] not in set(before.path)
    for subset in ("train", "validation", "test"):
        mapping = pd.read_csv(OUT / f"{subset}_label_mapping.csv")
        local = mapping.mc_onset.map(lambda value: float(Fraction(str(value)))) * 4
        np.testing.assert_allclose(mapping.quarterbeat, mapping.occurrence_start_qb + local, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(mapping.beat_index, np.floor(mapping.quarterbeat))
        assert set(mapping.piece_id) == set(split[subset])
    summary, all_pieces = [], []
    max_difference = 0.
    original = pd.read_csv(OUT / "test_summary.csv")
    for choice in frozen:
        assert sha256(Path(choice["checkpoint"])) == choice["checkpoint_sha256"]
        assert choice["threshold"] in protocol["plan"]["thresholds"]
        scans = choice["threshold_scan"]
        expected = max(scans, key=lambda pair: (pair[1], pair[0]))[0]
        assert choice["threshold"] == expected
        rows = []
        for piece in split["test"]:
            pattern = f"seed{choice['seed']}_{choice['condition']}_{piece}_*.npz"
            paths = sorted(OUT.glob(pattern))
            assert len(paths) == 2
            for path in paths:
                with np.load(path) as z:
                    probability, labels, coverage = (z[k] for k in ("probability", "labels", "coverage"))
                    assert len(probability) == len(labels) == len(coverage)
                    assert np.isfinite(probability).all() and not coverage[0]
                    ground_truth = np.flatnonzero((labels == 1) & coverage)
                    mapped = test_map[(test_map.piece_id == piece) & test_map.is_start].beat_index.unique()
                    expected_truth = sorted(int(index) for index in mapped if coverage[int(index)])
                    assert ground_truth.tolist() == expected_truth
                    predicted = independent_candidates(probability, coverage, choice["threshold"])
                    p, r, f1 = independent_event_scores(predicted, ground_truth, 1)
                    exact = independent_event_scores(predicted, ground_truth, 0)[2]
                    ap = float(average_precision_score(labels[coverage], probability[coverage]))
                    rows.append({"piece_id": piece, "P": p, "R": r, "F1": f1, "Exact_F1": exact, "AP": ap})
        piece_frame = pd.DataFrame(rows).groupby("piece_id").mean()
        metrics = piece_frame.mean().to_dict()
        record = {"seed": choice["seed"], "condition": choice["condition"], **metrics}
        summary.append(record)
        expected = original[(original.seed == choice["seed"]) & (original.condition == choice["condition"])].iloc[0]
        for metric in metrics:
            delta = abs(metrics[metric] - expected[metric])
            max_difference = max(max_difference, delta)
            assert delta < 1e-10
        for piece, row in piece_frame.iterrows():
            all_pieces.append({"seed": choice["seed"], "condition": choice["condition"], "piece_id": piece, **row.to_dict()})
    frame = pd.DataFrame(summary)
    pieces = pd.DataFrame(all_pieces)
    frame.to_csv(OUT / "independent_test_summary.csv", index=False)
    grouped = pieces.groupby(["condition", "piece_id"])[["F1", "Exact_F1", "AP"]].mean()
    grouped.to_csv(OUT / "seed_mean_piece_metrics.csv")
    # Paired bootstrap on independent works, not performances or random seeds.
    by_piece = pieces.groupby(["condition", "piece_id"]).F1.mean().unstack(0)
    difference = (by_piece.ssl - by_piece.scratch).to_numpy()
    rng = np.random.default_rng(20261015)
    draws = difference[rng.integers(0, len(difference), (10000, len(difference)))].mean(1)
    interval = np.quantile(draws, [.025, .975]).tolist()
    audit = {"status": "passed", "independent_bipartite_match_recomputation": True, "max_metric_difference": max_difference,
             "mapping_formula_checked_all_subsets": True, "test_annotation_not_read_before_freeze": True,
             "six_checkpoint_hashes_match_frozen_choices": True, "thresholds_validation_only": True,
             "work_disjoint": True, "mean_scores": frame.groupby("condition")[["P", "R", "F1", "Exact_F1", "AP"]].mean().to_dict("index"),
             "paired_work_F1_delta": float(difference.mean()), "paired_four_work_bootstrap_95_CI": interval,
             "limitation": "Four independent works only; interval unstable and not evidence of broad generalization"}
    (OUT / "independent_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    print(grouped.to_string())


if __name__ == "__main__":
    run()
