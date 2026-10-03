"""Independently reconstruct selected averaged weights and endpoint comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .score_context_study import ROOT,read,write,sha

def main():
    out=ROOT/'reports/trajectory_average_study';art=ROOT/'artifacts/trajectory_average_study';rows=[];checks=[]
    for f in (0,1):
        for s in (42,43):
            name=f'A_seed{s}_fold{f}';meta=read(art/'metrics'/f'{name}.json')
            cp=torch.load(art/'checkpoints'/name/'best.pt',map_location='cpu',weights_only=False)
            hist=cp['history'];assert [h['step'] for h in hist]==[50,100,150,200,250,300]
            chosen=max(hist,key=lambda h:h['macro_f1_tol1']);assert cp['step']==chosen['step']==meta['best_step']
            assert chosen['macro_f1_tol1']==meta['macro_f1_tol1']
            times=[50] if cp['step']==50 else [cp['step']-50,cp['step']-25,cp['step']]
            files=[art/'snapshots'/f'C3_seed{s}_fold{f}'/f'step{t}.pt' for t in times]
            values=[torch.load(p,map_location='cpu',weights_only=True) for p in files]
            for k,v in cp['model'].items():torch.testing.assert_close(torch.stack([x[k] for x in values]).mean(0),v,atol=0,rtol=0)
            original=torch.load(ROOT/'artifacts/recurrence_depth_study/checkpoints'/f'C3_seed{s}_fold{f}'/'latest.pt',map_location='cpu',weights_only=False)
            np.testing.assert_array_equal(original['mean'],cp['mean']);np.testing.assert_array_equal(original['std'],cp['std'])
            for kind,hs in [('original',original['history']),('average',hist)]:
                for select,h in [('terminal',hs[-1]),('best',max(hs,key=lambda h:h['macro_f1_tol1']))]:
                    rows.append(dict(fold=f,seed=s,kind=kind,selection=select,step=h['step'],f1=h['macro_f1_tol1'],exact=h['macro_f1_tol0'],ap=h['raw_ap']))
            checks.append(dict(fold=f,seed=s,selected_step=cp['step'],source_steps=times,parameters_exactly_reconstructed=True,hashes={str(p):sha(p) for p in files}))
    frame=pd.DataFrame(rows);frame.to_csv(out/'endpoint_comparisons.csv',index=False)
    means=frame.groupby(['kind','selection'])[['f1','exact','ap']].mean();means.to_csv(out/'endpoint_means.csv')
    write(out/'parameter_reconstruction_audit.json',dict(status='complete',checks=checks,selection_points_equal=6,train_normalization_matches=True))
    print(means.to_string())

if __name__=='__main__':main()
