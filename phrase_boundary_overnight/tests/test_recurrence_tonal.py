import numpy as np,torch
from src.recurrence_tonal_features import features,key_estimate
from src.run_recurrence_tonal_study import make_model
from src.recurrence_depth_models import make_model as baseline


def test_tonal_transposition_and_matched_side_channels():
    events=np.array([[0,4,60,1],[4,1,62,1],[5,2,64,1],[7,1,65,1],[8,3,67,1],[11,1,69,1],[12,1,71,1],[13,4,60,1]],float)
    a,r,k=features(events,20);assert k['margin']>1e-4
    for shift in range(12):
        moved=events.copy();moved[:,2]+=shift
        aa,rr,kk=features(moved,20)
        assert kk['tonic']==(k['tonic']+shift)%12 and kk['minor']==k['minor']
        np.testing.assert_allclose(rr,r,atol=1e-6);np.testing.assert_array_equal(aa[:,24:],rr[:,24:])
    assert np.isfinite(features(np.empty((0,4)),20)[1]).all()


def test_initial_function_rng_and_added_gradient():
    for seed in (42,43):
        old=baseline('C3',seed).eval();rng=torch.get_rng_state().clone();new=make_model('TA',seed).eval()
        assert torch.equal(rng,torch.get_rng_state()) and sum(p.numel() for p in new.parameters())==6753
        x=torch.randn(2,64,84);mask=torch.zeros(2,64,dtype=torch.bool);mask[0,-9:]=True
        torch.testing.assert_close(old(x[:,:,:58],mask),new(x,mask),atol=1e-6,rtol=1e-6)
        new(x,mask).sum().backward();assert new.input_projection.weight.grad[:,58:].abs().sum()>0
