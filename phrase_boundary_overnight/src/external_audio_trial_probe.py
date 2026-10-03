"""Post-result frozen train/validation diagnostics, no new test model selection."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from . import external_audio_trial as trial
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/external_audio_trial_probe'


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);data=trial.load_data();splits=read(trial.OUT/'splits.json');rows=[];probes=[]
    assert read(trial.OUT/'completion_audit.json')['status']=='complete';hashes=read(trial.OUT/'checkpoint_hashes.json')
    hashes.update(read(trial.OUT/'contract.json')['hashes'])
    for p in (Path(__file__),OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    for fold in range(3):
        train=trial.subset(data,splits[str(fold)]['train']);val=trial.subset(data,splits[str(fold)]['validation']);norm=trial.normalizer(train)
        for seed in (42,43):
            for kind in ('S','E','F'):
                name=f'{kind}_seed{seed}_fold{fold}';cp=torch.load(trial.ART/'checkpoints'/name/'best.pt',map_location='cpu',weights_only=False)
                meta=read(trial.ART/'metrics'/f'{name}.json');m=trial.make_model(seed).cuda();m.load_state_dict(cp['model'])
                for split,d in [('train',train),('validation',val)]:
                    raw=trial.predict(m,d,norm,kind);score=trial.metrics(raw,d,meta['threshold'])[2];rows.append(dict(fold=fold,seed=seed,kind=kind,split=split,**score))
                    if split=='validation':assert max(abs(score[c]-meta[c]) for c in trial.COLS)<1e-10
                    if kind!='S':
                        controls=['S'] if kind=='E' else ['E','S']
                        for control in controls:
                            changed=trial.predict(m,d,norm,control);other=trial.metrics(changed,d,meta['threshold'])[2]
                            bywork={}
                            for key,v in d.items():
                                mask=v['label_mask']>.5;bywork.setdefault(v['piece_id'],[]).append(float(abs(changed[key][mask]-raw[key][mask]).mean()))
                            probes.append(dict(fold=fold,seed=seed,kind=kind,split=split,removed_to=control,
                                probability_delta=float(np.mean([np.mean(v) for v in bywork.values()])),f1_delta=other['macro_f1_tol1']-score['macro_f1_tol1'],ap_delta=other['raw_ap']-score['raw_ap']))
                assert time.monotonic()-started<600
                print('PROBED',name,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'train_validation_metrics.csv',index=False);pd.DataFrame(probes).to_csv(OUT/'feature_removal.csv',index=False)
    pd.DataFrame(rows).groupby(['kind','split'])[trial.COLS].mean().to_csv(OUT/'means.csv')
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',models=18,training_runs=0,test_inferences=0,source_hashes_unchanged=True,seconds=time.monotonic()-started,
        caveat='Post-result diagnostic, not preregistered primary evidence; train/validation means average folds, primary test report averages held-out movements.'))
    print(pd.DataFrame(rows).groupby(['kind','split'])[trial.COLS].mean().to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
