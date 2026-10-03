"""Audit initialization stochastic-stream equivalence, without changing training."""
import torch
from .recurrence_depth_models import make_model as base
from .harmony_auxiliary import make_model
from .score_context_study import ROOT,write


def main():
    rows=[]
    for seed in (42,43):
        old=base('C3',seed);cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all()
        group=[]
        for kind in ('G','D','Z'):
            m=make_model(kind,seed);c=torch.get_rng_state();g=torch.cuda.get_rng_state_all()
            assert all(torch.equal(v,m.core.state_dict()[k]) for k,v in old.state_dict().items())
            group.append((c,g))
            rows.append(dict(seed=seed,kind=kind,core_weights_equal=True,cpu_equal_old=bool(torch.equal(cpu,c)),
                cuda_equal_old=all(torch.equal(a,b) for a,b in zip(gpu,g))))
        assert all(torch.equal(group[0][0],c) and all(torch.equal(a,b) for a,b in zip(group[0][1],g)) for c,g in group[1:])
    write(ROOT/'reports/harmony_auxiliary/rng_diagnostic.json',dict(rows=rows,groups_G_D_Z_share_same_streams=True,training_modified=False,
        caution='fork_rng(devices=[]) restores CPU but manual_seed also resets CUDA. Z is the matched control; historical C3 is not a step-for-step RNG replay.'))
    print(rows,flush=True)


if __name__=='__main__':main()
