"""Describe real frozen training coverage, not nominal epochs."""
import json
import time
import pandas as pd
import torch
from .run_motif_recurrence_study import dataset, ART
from .phrase_end_auxiliary import ROOT, split_ids, read, write, sha
from .window_exposure_audit import exposure


def main():
    began=time.monotonic();out=ROOT/'reports/training_exposure_audit';out.mkdir(parents=True,exist_ok=True)
    source=read(ROOT/'reports/motif_recurrence_study/contract.json')['hashes']
    assert all(sha(p)==h for p,h in source.items())
    rows=[];cps={};checks=[]
    for f in (0,1):
        data=dataset(split_ids(f)['train'],'O')
        for seed in (42,43):
            cp=ART/'checkpoints'/f'O_seed{seed}_fold{f}'/'latest.pt';cps[str(cp)]=sha(cp)
            saved=torch.load(cp,map_location='cpu',weights_only=False);assert saved['step']==300
            for steps in (50,100,300):
                r=exposure(data,seed,steps)
                if steps==300:assert r['sampler']==saved['sampler']
                r.pop('sampler');draws=r.pop('piece_draws')
                rows.append(dict(fold=f,seed=seed,**r,min_piece_draws=min(draws.values()),max_piece_draws=max(draws.values())))
            checks.append(dict(fold=f,seed=seed,step300_rng_matches_checkpoint=True))
    df=pd.DataFrame(rows);df.to_csv(out/'exposure.csv',index=False)
    assert all(sha(p)==h for p,h in cps.items()) and all(sha(p)==h for p,h in source.items())
    write(out/'audit.json',dict(status='complete',rng_checks=checks,checkpoints=cps,seconds=time.monotonic()-began,
          observations='Sampling exposure only; no gradient updates, no test predictions, no added independent works.'))
    print(df[['fold','seed','steps','score_window_coverage','performance_window_coverage','score_beat_pass_equivalent','performance_beat_pass_equivalent']].to_string(index=False))


if __name__=='__main__':main()
