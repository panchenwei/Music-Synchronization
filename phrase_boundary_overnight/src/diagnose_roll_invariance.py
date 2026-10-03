"""Synthetic structural probe, not a musical-validity or held-out-score experiment."""
import numpy as np
import pandas as pd
import torch
from .score_context_study import ROOT,read,write,sha
from .score_roll_branch import RollBoundary


def example(shifts,short=False):
    x=torch.zeros(1,len(shifts),2,128,4)
    for b,shift in enumerate(shifts):
        for p in (48,52,55):
            x[0,b,0,p+shift,:1 if short else 4]=1
            x[0,b,1,p+shift,0]=1
    return x


def main():
    torch.set_num_threads(2);out=ROOT/'reports/roll_information_probe';out.mkdir(parents=True,exist_ok=True)
    asc=[0,2,4,5,7,9,11,12];desc=[12,11,9,7,5,4,2,0]
    a=example(asc);b=example(desc);short=example(asc,True);rows=[];hashes={}
    for kind,seed in (('random',42),('external',42),('external',43)):
        model=RollBoundary('L',seed).eval()
        if kind=='external':
            p=ROOT/'artifacts/external_stem_transfer/external'/f'seed{seed}'/'best.pt';hashes[str(p)]=sha(p);model.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model'])
        with torch.no_grad():
            ea=model.stem(a);eb=model.stem(b);ec=model.stem(short)
            x=torch.zeros(1,8,58);pa=torch.sigmoid(model(x,a));pb=torch.sigmoid(model(x,b))
        equal=float((ea-eb).abs().max());contrast=float((ea-ec).abs().max())
        assert equal<1e-5 and contrast>1e-5
        rows.append(dict(kind=kind,seed=seed,ascending_vs_descending_roll_cells_changed=int((a!=b).sum()),stem_max_delta_pitch_direction=equal,stem_max_delta_duration_change=contrast,output_max_delta_with_original34_held_equal=float((pa-pb).abs().max())))
    assert all(sha(p)==h for p,h in hashes.items());pd.DataFrame(rows).to_csv(out/'synthetic_results.csv',index=False)
    write(out/'source_hashes.json',hashes);write(out/'STATE.json',dict(status='complete',synthetic_examples_only=True,training_runs=0,target_data_accessed=False,external_checkpoints_unchanged=True,scope='Per-beat pitch-translation information is removed by this stem; full model may still receive pitch information through original34 inputs.'))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
