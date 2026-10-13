from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.goal1_unlabeled import (
    FEATURE_NAMES,
    SourceAccessGuard,
    build_curve_features,
    build_score_cues,
    motif_recurrence,
    positioned_events,
    tempo_multiscale,
    unfold_measure_path,
    window_starts,
)


class Goal1Tests(unittest.TestCase):
    def test_repeat_and_local_coordinate_formula(self):
        measures = pd.DataFrame({"mc": [1, 2], "duration_qb": [3, 3], "next": ["2", "1, -1"]})
        path, decisions = unfold_measure_path(measures)
        self.assertEqual([row["mc"] for row in path], [1, 2, 1, 2])
        self.assertEqual([row["start_qb"] for row in path], [0, 3, 6, 9])
        self.assertEqual(len(decisions), 2)
        notes = pd.DataFrame(
            {"mc": [1], "mc_onset": ["1/2"], "duration_qb": [1], "midi": [60], "staff": [1], "voice": [1], "tied": [np.nan]}
        )
        events, _, mapping = positioned_events(notes, path, 12)
        self.assertEqual([event[0] for event in events], [2, 8])
        self.assertEqual(mapping.global_qb.tolist(), [2, 8])
        self.assertEqual(mapping.beat_index.tolist(), [2, 8])

    def test_feature_shapes(self):
        times = np.arange(8, dtype=float) * 0.6
        curve, _ = build_curve_features(times, np.linspace(0.1, 0.9, 8))
        curves = curve[None]
        self.assertEqual(curves.shape, (1, 8, 9))
        self.assertEqual(tempo_multiscale(curves).shape, (1, 8, 9))
        events = [(0.0, 1.0, 60, 1, 1), (1.25, 0.5, 64, 1, 1), (3.0, 2.0, 40, 2, 1)]
        score = build_score_cues(events, np.arange(8) % 3, 8)
        self.assertEqual(score.shape, (8, 16))
        motif, subticks = motif_recurrence([(*event, 0.0) for event in events], 8)
        self.assertEqual(motif.shape, (8, 24))
        self.assertEqual(subticks.shape, (2, 8, 4, 24))
        self.assertEqual(len(FEATURE_NAMES), 58)

    def test_window_coverage(self):
        starts = window_starts(337)
        self.assertEqual(starts, [0, 32, 64, 96, 128, 160, 192, 224, 256, 288])
        self.assertTrue(all(right - left == 32 for left, right in zip(starts, starts[1:])))
        self.assertEqual(window_starts(40), [0])

    def test_access_guard_rejects_labels_and_legacy(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            guard = SourceAccessGuard(root / "dcml", root / "mazurka", root / "legacy")
            allowed = root / "dcml" / "notes" / "x.notes.tsv"
            self.assertEqual(guard.validate_path(allowed), "dcml_notes")
            with self.assertRaises(PermissionError):
                guard.validate_path(root / "dcml" / "harmonies" / "x.harmonies.tsv")
            with self.assertRaises(PermissionError):
                guard.validate_path(root / "legacy" / "cache" / "x.npz")


if __name__ == "__main__":
    unittest.main()
