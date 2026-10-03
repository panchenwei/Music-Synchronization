"""Frozen MERT-content dependence and generalization diagnostics."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from . import external_mert_trial as study
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/external_mert_trial_probe'


def main():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    assert read(study.OUT/'completion_audit.json')['status']=='complete'
    base=study.base;base.mask_modalities=study.mask_modalities;data=base.load_data();splits=read(study.OLD_OUT/'splits.json');rows=[];removals=[]
    hashes=dict(read(study.OUT/'checkpoint_hashes.json'));hashes.update(read(study.OUT/'contract.json')['hashes'])
    for p in (Path(__file__),OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    for f in range(3):
        for kind in ('S','E','F'):
            d=study.load_fold(data,f,kind);train=base.subset(d,splits[str(f)]['train']);val=base.subset(d,splits[str(f)]['validation']);norm=base.normalizer(train)
            for seed in (42,43):
                name=f'{kind}_seed{seed}_fold{f}';meta=read(study.ART/'metrics'/f'{name}.json');cp=torch.load(study.ART/'checkpoints'/name/'best.pt',map_location='cpu',weights_only=False)
                np.testing.assert_array_equal(norm.mean,cp['mean']);np.testing.assert_array_equal(norm.std,cp['std'])
                model=base.make_model(seed).cuda();model.load_state_dict(cp['model'])
                for split,part in [('train',train),('validation',val)]:
                    raw=base.predict(model,part,norm,kind);score=base.metrics(raw,part,meta['threshold'])[2];rows.append(dict(fold=f,seed=seed,kind=kind,split=split,**score))
                    if split=='validation':assert max(abs(score[c]-meta[c]) for c in base.COLS)<1e-10
                    removed=base.predict(model,part,norm,'S');rs=base.metrics(removed,part,meta['threshold'])[2];bywork={}
                    for key,v in part.items():
                        mask=v['label_mask']>.5;bywork.setdefault(v['piece_id'],[]).append(float(abs(removed[key][mask]-raw[key][mask]).mean()))
                    change=float(np.mean([np.mean(v) for v in bywork.values()]));assert kind!='S' or change==0
                    removals.append(dict(fold=f,seed=seed,kind=kind,split=split,probability_change=change,f1_delta=rs['macro_f1_tol1']-score['macro_f1_tol1'],ap_delta=rs['raw_ap']-score['raw_ap']))
                print('PROBED',name,flush=True);assert time.monotonic()-began<600
    pd.DataFrame(rows).to_csv(OUT/'train_validation_metrics.csv',index=False);pd.DataFrame(removals).to_csv(OUT/'feature_removal.csv',index=False)
    means=pd.DataFrame(rows).groupby(['kind','split'])[base.COLS].mean();means.to_csv(OUT/'means.csv')
    pd.DataFrame(removals).groupby(['kind','split'])[['probability_change','f1_delta','ap_delta']].mean().to_csv(OUT/'removal_means.csv')
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',models=18,training_runs=0,held_test_inferences=0,hashes_unchanged=True,seconds=time.monotonic()-began))
    print(means.to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
