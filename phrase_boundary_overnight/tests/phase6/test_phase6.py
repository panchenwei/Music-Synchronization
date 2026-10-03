from pathlib import Path

import numpy as np
import pandas as pd

from src.phase3_models import fit_train_normalizer
from src.phase6_models import StepSampler, triangular_weights, window_starts
from src.phase6_multiscale import augment_piece, segment_novelty


def test_window_coverage_and_nonzero_triangular_weights():
    for length in [1, 63, 64, 65, 121, 673]:
        cover=np.zeros(length,float)
        for start in window_starts(length,64,32):
            valid=min(64,length-start);cover[start:start+valid]+=triangular_weights(64)[:valid]
        assert np.all(cover>0)


def test_window_starts_include_final_endpoint():
    assert window_starts(121,64,32)[-1]==57
    assert window_starts(64,64,32)==[0]


def test_real_opus_split_remains_disjoint():
    root=Path(__file__).resolve().parents[2];frame=pd.read_csv(root/"artifacts/phase2/splits/opus_split_manifest.csv")
    for fold in range(5):
        parts={s:frame[(frame.fold==fold)&(frame.split==s)] for s in ["train","validation","test"]}
        for a,b in [("train","validation"),("train","test"),("validation","test")]:
            assert set(parts[a].piece_id).isdisjoint(parts[b].piece_id)
            assert set(parts[a].opus.astype(str)).isdisjoint(parts[b].opus.astype(str))


def test_multiscale_novelty_is_aligned_finite_and_constant_zero():
    values=np.ones(64,np.float32);novelty,quality=segment_novelty(values,np.ones(64,bool),8)
    assert novelty.shape==(64,) and quality.shape==(64,)
    assert np.isfinite(novelty).all() and np.allclose(novelty,0)
    item={"curves":np.ones((2,64,9),np.float32),"curve_feature_names":np.asarray([f"f{i}" for i in range(9)])}
    item["curves"][:,:,7:9]=1
    augmented=augment_piece(item,"both")
    assert augmented["curves"].shape==(2,64,15)
    assert augmented["phase6_multiscale_quality"].shape==(2,64,6)
    assert len(augmented["curve_feature_names"])==15


def test_piece_sampler_resume_is_exact_and_split_local():
    data={}
    for index,piece_id in enumerate(["train_a","train_b"]):
        data[piece_id]={
            "curves":np.full((2,8,9),index+1,dtype=np.float32),
            "performance_ids":np.asarray(["p0","p1"]),
            "score_phase3":np.full((8,16),index+1,dtype=np.float32),
            "labels":np.asarray([0,0,1,0,0,1,0,0],dtype=np.float32),
            "label_mask":np.ones(8,dtype=np.float32),
        }
    curve_norm=fit_train_normalizer([item["curves"] for item in data.values()])
    score_norm=fit_train_normalizer([item["score_phase3"] for item in data.values()])
    sampler=StepSampler(data,curve_norm,score_norm,4,2,3,"piece_balanced",42)
    sampler.batch();state=sampler.state();expected=sampler.batch()
    resumed=StepSampler(data,curve_norm,score_norm,4,2,3,"piece_balanced",999,state=state)
    actual=resumed.batch()
    assert resumed.pieces==["train_a","train_b"]
    for left,right in zip(expected,actual):
        assert np.array_equal(left.numpy(),right.numpy())
