"""Frozen R branch-zero/averaging sensitivity; OOD diagnostic, not ablation retraining."""
import time
import numpy as np
import pandas as pd
import torch
from . import ordered_attack_study as s
from .score_context_study import ROOT,read,write,sha
from .three_round_round2 import split_ids
from .local_context_study import predictions,metrics
from .audit_external_stem_transfer import checked_raw

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    out=ROOT/'reports/ordered_attack_usage';out.mkdir(exist_ok=True);began=time.monotonic()
    assert read(s.OUT/'completion_audit.json')['status']=='complete'
    rows=[];hashes={str(__file__):sha(__file__)}
    for f in (0,1):
        ids=split_ids(f);norm=s.normalizer(s.dataset(ids['train'],'R'));val=s.dataset(ids['validation'],'R')
        for seed in (42,43):
            run=f'R_seed{seed}_fold{f}';cp=s.ART/'checkpoints'/run/'best.pt';mp=s.ART/'metrics'/f'{run}.json';pp=s.ART/'metrics'/f'{run}_predictions.csv.gz'
            hashes.update({str(p):sha(p) for p in (cp,mp,pp)})
            state=torch.load(cp,map_location='cpu',weights_only=False);meta=read(mp);old=checked_raw(pd.read_csv(pp),val)
            np.testing.assert_array_equal(norm.mean,state['mean']);np.testing.assert_array_equal(norm.std,state['std'])
            model=s.ArcBoundary('R',seed).cuda().eval();model.load_state_dict(state['model'])
            for mode in ('U','Z'):
                changed=s.dataset(ids['validation'],mode)
                for p,v in val.items():np.testing.assert_array_equal(v['curves'][...,:58],changed[p]['curves'][...,:58])
                raw=predictions(model,changed,norm,torch.device('cuda'));score=metrics(raw,changed,meta['threshold'])[2]
                delta=np.concatenate([np.abs(raw[p][q]-old[p][q]) for p in raw for q in raw[p]])
                rows.append(dict(fold=f,seed=seed,mode=mode,raw_f1=score['macro_f1_tol1'],raw_f1_delta=score['macro_f1_tol1']-meta['macro_f1_tol1'],ap_delta=score['raw_ap']-meta['raw_ap'],mean_abs_probability_change=float(delta.mean()),max_abs_probability_change=float(delta.max()),new_weight_norm=float(model.core.input_projection.weight[:,58:].norm().detach())))
    assert all(sha(p)==h for p,h in hashes.items())
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False);write(out/'source_hashes.json',hashes);write(out/'completion_audit.json',dict(status='complete',new_training=0,perturbed_gpu_inferences=8,weights_unchanged=True,test_used=False,seconds=time.monotonic()-began))
    print(pd.DataFrame(rows).groupby('mode').mean(numeric_only=True).to_string())

if __name__=='__main__':main()
