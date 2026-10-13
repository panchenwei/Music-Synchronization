from __future__ import annotations

import math
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "goal1"
CACHE = ROOT / "cache" / "unlabeled58"
REPORT = ROOT / "reports" / "goal1"


def parse_number(value: object) -> float:
    text = str(value).strip()
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return float(text)


class Goal1OutputAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pieces = pd.read_csv(MANIFEST / "piece_audit.csv")
        cls.usable = cls.pieces[cls.pieces.status == "usable"].copy()
        cls.windows = pd.read_csv(MANIFEST / "window_manifest.csv.gz")
        cls.full = pd.read_csv(MANIFEST / "full_sequence_manifest.csv.gz")

    def test_piece_counts_exclusions_and_length_rule(self):
        self.assertEqual(len(self.pieces), 45)
        self.assertEqual(len(self.usable), 42)
        excluded = self.pieces[self.pieces.status == "excluded"].set_index("piece_id")
        self.assertEqual(set(excluded.index), {"chopin_op17_no3", "chopin_op33_no3", "chopin_op50_no2"})
        self.assertIn("Cannot uniquely select next edge", excluded.loc["chopin_op17_no3", "reason"])
        self.assertIn("Score/beat length mismatch", excluded.loc["chopin_op33_no3", "reason"])
        self.assertIn("Score/beat length mismatch", excluded.loc["chopin_op50_no2", "reason"])
        self.assertTrue((self.usable.length_error_beats <= 2.0).all())

    def test_cache_is_unlabeled_finite_and_exactly_58d(self):
        cache_paths = sorted(CACHE.glob("*.npz"))
        self.assertEqual(len(cache_paths), 42)
        for path in cache_paths:
            with self.subTest(piece=path.stem), np.load(path, allow_pickle=False) as data:
                self.assertFalse(any("label" in key.lower() for key in data.files))
                performances, beats, dimensions = data["inputs"].shape
                self.assertEqual(dimensions, 58)
                self.assertEqual(data["curves"].shape, (performances, beats, 9))
                self.assertEqual(data["score_cues"].shape, (beats, 16))
                self.assertEqual(data["tempo_multiscale"].shape, (performances, beats, 9))
                self.assertEqual(data["motif_recurrence"].shape, (beats, 24))
                self.assertEqual(data["motif_subticks"].shape, (2, beats, 4, 24))
                self.assertEqual(data["feature_names"].shape, (58,))
                np.testing.assert_array_equal(data["quarterbeat"], np.arange(beats, dtype=np.float32))
                expected = np.concatenate(
                    [
                        data["curves"],
                        np.broadcast_to(data["score_cues"], (performances, beats, 16)),
                        data["tempo_multiscale"],
                        np.broadcast_to(data["motif_recurrence"], (performances, beats, 24)),
                    ],
                    axis=-1,
                )
                np.testing.assert_allclose(data["inputs"], expected, rtol=0, atol=0)
                self.assertTrue(np.isfinite(data["inputs"]).all())

    def test_measure_and_event_coordinate_formulas(self):
        measures = pd.read_csv(MANIFEST / "measure_mapping.csv")
        for piece_id, rows in measures.groupby("piece_id", sort=False):
            rows = rows.sort_values("sequence_index")
            self.assertEqual(rows.sequence_index.tolist(), list(range(len(rows))), piece_id)
            expected_starts = rows.start_qb.to_numpy()[:-1] + rows.duration_qb.to_numpy()[:-1]
            np.testing.assert_allclose(rows.start_qb.to_numpy()[1:], expected_starts, rtol=0, atol=1e-7)

        events = pd.read_csv(MANIFEST / "event_mapping.csv.gz")
        local = events.mc_onset.map(parse_number).to_numpy() * 4.0
        np.testing.assert_allclose(events.local_qb.to_numpy(), local, rtol=0, atol=1e-7)
        np.testing.assert_allclose(
            events.global_qb.to_numpy(),
            events.occurrence_start_qb.to_numpy() + events.local_qb.to_numpy(),
            rtol=0,
            atol=1e-7,
        )
        mapped = events[events.status == "mapped"]
        expected_beat = np.floor(mapped.global_qb.to_numpy()).astype(int)
        expected_subtick = np.floor(mapped.global_qb.to_numpy() * 4.0 + 1e-8).astype(int) % 4
        np.testing.assert_array_equal(mapped.beat_index.to_numpy(dtype=int), expected_beat)
        np.testing.assert_array_equal(mapped.subtick.to_numpy(dtype=int), expected_subtick)

    def test_fixed_windows_and_variable_full_sequences(self):
        self.assertEqual(len(self.windows), 19653)
        self.assertEqual(len(self.full), 1954)
        self.assertTrue((self.windows.window_length == 64).all())
        self.assertTrue((self.windows.stride == 32).all())
        self.assertTrue((self.windows.valid_length == self.windows.stop_beat_exclusive - self.windows.start_beat).all())
        self.assertTrue((self.windows.padded == (self.windows.valid_length < 64)).all())
        for _, rows in self.windows.groupby(["piece_id", "performance_id"], sort=False):
            starts = rows.start_beat.to_numpy()
            if len(starts) > 1:
                np.testing.assert_array_equal(np.diff(starts), np.full(len(starts) - 1, 32))

        self.assertTrue((self.full.start_beat == 0).all())
        self.assertTrue((self.full.stop_beat_exclusive == self.full.valid_length).all())
        self.assertTrue((self.full.sequence_mode == "full_piece_variable_length").all())
        self.assertTrue(self.full.requires_padding_mask.all())
        self.assertGreater(self.full.valid_length.nunique(), 1)
        expected = int(self.usable.performances.sum())
        self.assertEqual(len(self.full), expected)
        self.assertEqual(len(self.full[["piece_id", "performance_id"]].drop_duplicates()), expected)

    def test_access_log_visualizations_and_no_goal1_training_artifacts(self):
        access = pd.read_csv(MANIFEST / "source_access.csv")
        allowed = {"dcml_notes", "dcml_measures", "mazurkabl_beat_time", "mazurkabl_beat_dyn"}
        self.assertEqual(set(access.role), allowed)
        self.assertEqual(access.groupby("role").size().to_dict(), {role: 45 for role in allowed})
        self.assertTrue(access.sha256.str.fullmatch(r"[0-9a-f]{64}").all())
        lower_paths = access.path.str.lower()
        for forbidden in ("harmon", "phrase", "label", "prediction", "checkpoint"):
            self.assertFalse(lower_paths.str.contains(forbidden, regex=False).any(), forbidden)

        visual_paths = sorted((REPORT / "visualizations").glob("*_alignment.png"))
        self.assertGreaterEqual(len(visual_paths), 3)
        self.assertTrue(all(path.stat().st_size > 10_000 for path in visual_paths))
        trained = [
            path
            for suffix in ("*.pt", "*.pth", "*.ckpt")
            for path in (ROOT / "checkpoints" / "goal1").rglob(suffix)
        ]
        self.assertEqual(trained, [])


if __name__ == "__main__":
    unittest.main()
