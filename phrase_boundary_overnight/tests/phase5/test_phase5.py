from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.phase4_models import ExperimentDeadline
from src.phase5_models import apply_modality, deadline_guard, loss_balance, target_data


def item():
    labels=np.zeros(40,np.float32);labels[[5,25]]=1;mask=np.ones(40,np.float32);mask[26]=0
    return {"labels":labels,"label_mask":mask}


def test_modality_controls_preserve_shape_and_zero_only_requested_group():
    values=torch.randn(2,7,25)
    score=apply_modality(values,"score_only");curves=apply_modality(values,"curves_only");combined=apply_modality(values,"combined")
    assert score.shape==curves.shape==combined.shape==values.shape
    assert torch.count_nonzero(score[...,:9])==0 and torch.allclose(score[...,9:],values[...,9:])
    assert torch.count_nonzero(curves[...,9:])==0 and torch.allclose(curves[...,:9],values[...,:9])
    assert torch.allclose(combined,values)


def test_target_weight_2x2_is_numerically_distinct_and_masked():
    source={"x":item()};hard=target_data(source,"hard");soft=target_data(source,"soft")
    assert soft["x"]["labels"][26]==0
    for data in [hard,soft]:
        capped=loss_balance(data,"original_capped",10);balanced=loss_balance(data,"mass_balanced",10)
        assert balanced["pos_weight"]>capped["pos_weight"]
        assert np.isclose(balanced["effective_positive_coefficient"],balanced["effective_negative_coefficient"])


def test_constant_logit_matches_documented_bce_positive_weight_formula():
    labels=torch.tensor([0.0,0.5,1.0]);logits=torch.zeros(3);weight=3.0
    actual=nn.BCEWithLogitsLoss(reduction="none",pos_weight=torch.tensor(weight))(logits,labels)
    expected=np.log(2)*((1-labels.numpy())+weight*labels.numpy())
    assert np.allclose(actual.numpy(),expected)


def test_deadline_guard():
    with pytest.raises(ExperimentDeadline):deadline_guard(datetime.now().astimezone()+timedelta(seconds=1),1)


def test_real_opus_split_is_disjoint():
    root=Path(__file__).resolve().parents[2];frame=pd.read_csv(root/"artifacts/phase2/splits/opus_split_manifest.csv")
    for fold in range(5):
        parts={s:frame[(frame.fold==fold)&(frame.split==s)] for s in ["train","validation","test"]}
        for a,b in [("train","validation"),("train","test"),("validation","test")]:
            assert set(parts[a].piece_id).isdisjoint(parts[b].piece_id)
            assert set(parts[a].opus.astype(str)).isdisjoint(parts[b].opus.astype(str))
