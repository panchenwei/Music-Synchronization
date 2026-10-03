import numpy as np
import torch

from src.three_round_round2 import assemble_extra, interval_observed, make_model34, normalizer, timing_features


def arrays(tempo):
    tempo=np.asarray(tempo,float);curves=np.zeros((1,len(tempo),9),np.float32);curves[0,:,0]=np.log(tempo);curves[0,:,7]=1
    rmsq=np.arange(len(tempo)*6,dtype=np.float32).reshape(1,len(tempo),6)
    return curves,rmsq


def test_constant_tempo_direction_zero_and_no_wrap():
    curves,rmsq=arrays([100]*8);original,lag,lag_ok,direction,direction_ok=timing_features(curves,rmsq)
    np.testing.assert_allclose(direction,0,atol=1e-7);np.testing.assert_array_equal(lag[:,0],0);np.testing.assert_array_equal(lag[:,1:],original[:,:-1]);assert lag_ok[0,0]==direction_ok[0,0]==0


def test_slowdown_speedup_direction_sign():
    up,_=arrays([60,60,60,120,120,120,120]);down,_=arrays([120,120,120,60,60,60,60]);r=np.zeros((1,7,6),np.float32)
    assert timing_features(up,r)[3][0,3]>0
    assert timing_features(down,r)[3][0,3]<0


def test_flags_identical_all_conditions_and_values_only_change_as_registered():
    curves,rmsq=arrays([80,90,100,120,110,100,90]);blocks={k:assemble_extra(curves,rmsq,k) for k in ['B','L','D','LD']}
    for k in blocks:np.testing.assert_array_equal(blocks[k][...,[6,8]],blocks['B'][...,[6,8]])
    np.testing.assert_array_equal(blocks['B'][...,:6],blocks['D'][...,:6]);np.testing.assert_array_equal(blocks['L'][...,:6],blocks['LD'][...,:6])
    assert np.count_nonzero(blocks['B'][...,7])==np.count_nonzero(blocks['L'][...,7])==0


def test_observation_flag_excludes_final_estimate():
    assert interval_observed(np.ones(5)).tolist()==[1,1,1,1,0]


def test_step0_equal_and_nine_new_columns_zero():
    x25=torch.randn(2,13,25);models=[make_model34(42).eval() for _ in range(4)]
    for model in models:assert torch.count_nonzero(model.input_projection.weight[:,25:])==0
    with torch.no_grad():outputs=[model(torch.cat([x25,torch.randn(2,13,9)],-1)) for model in models]
    for output in outputs[1:]:torch.testing.assert_close(output,outputs[0])


def test_train_only_normalizer_does_not_use_validation_extreme():
    def item(value):
        curves=np.zeros((1,4,34),np.float32);curves[...,25:]=value
        return {'curves':curves,'selected_score':np.zeros((4,16),np.float32),'round2_extra':curves[...,25:]}
    train={'a':item(1)};before=normalizer(train);validation={'v':item(999)};after=normalizer(train)
    np.testing.assert_array_equal(before.mean,after.mean);np.testing.assert_array_equal(before.std,after.std);assert validation['v']['round2_extra'].mean()==999
