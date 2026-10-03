"""Freeze new curve-scale models; distinguish coefficient and quality effects."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import time,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
from . import curve_scale_study as study
from .curve_scale_models import CurveScaleBoundary
from .local_context_study import predictions,metrics
from .score_context_study import ROOT,read,write,sha
from .three_round_round2 import split_ids
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/curve_scale_information';ART=ROOT/'artifacts/curve_scale_information'
VARIANTS=('intact','zero_signed','zero_all_new')


def intervene(data,variant):
    assert variant in VARIANTS
    out={}
    for pid,item in data.items():
        curves=item['curves'].copy()
        if variant=='zero_signed':curves[...,58:68]=0
        elif variant=='zero_all_new':curves[...,58:]=0
        out[pid]={**item,'curves':curves}
    return out


def main():
    began=time.monotonic();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    assert read(study.OUT/'completion_audit.json')['status']=='complete'
    for p in (OUT,ART):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(study.OUT/'contract.json')['hashes']);hashes.update(read(study.OUT/'checkpoint_hashes.json'))
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_curve_scale_information.py'):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    write(OUT/'STATE.json',dict(status='running',pid=os.getpid()));rows=[]
    for fold in (0,1):
        ids=split_ids(fold);val=study.dataset(ids['validation'],'F');norm=study.normalizer(study.dataset(ids['train'],'F'))
        for seed in (42,43):
            for kind in study.KINDS:
                run=f'{kind}_seed{seed}_fold{fold}';meta=read(study.ART/'metrics'/f'{run}.json')
                cp=torch.load(study.ART/'checkpoints'/run/'best.pt',map_location='cpu',weights_only=False)
                m=CurveScaleBoundary(kind,seed).cuda();m.load_state_dict(cp['model'])
                saved=checked_raw(pd.read_csv(study.ART/'metrics'/f'{run}_predictions.csv.gz'),val)
                for variant in VARIANTS:
                    data=intervene(val,variant);raw=predictions(m,data,norm,torch.device('cuda'))
                    if variant=='intact':
                        error=max(float(abs(raw[p][q]-saved[p][q]).max()) for p in raw for q in raw[p]);assert error<2e-4
                    _,pieces,score=metrics(raw,val,meta['threshold'])
                    if variant=='intact':assert max(abs(score[c]-meta[c]) for c in study.COLS)<1e-10
                    change=float(np.mean([np.mean([np.mean(abs(raw[p][q]-saved[p][q])) for q in raw[p]]) for p in raw]))
                    pieces.to_csv(ART/f'{run}_{variant}_pieces.csv',index=False)
                    rows.append(dict(run_id=run,kind=kind,fold=fold,seed=seed,variant=variant,mean_abs_probability_change=change,**score))
                pd.DataFrame(rows).to_csv(ART/'summary.csv',index=False);print('SCALE INFORMATION',run,flush=True)
    df=pd.DataFrame(rows);assert len(df)==36
    df.groupby(['kind','variant'])[study.COLS+['mean_abs_probability_change']].mean().to_csv(OUT/'means.csv')
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',models=12,conditions=36,hashes_unchanged=True,thresholds_refitted=False,test_used=False,seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));print(df.groupby(['kind','variant'])[study.COLS+['mean_abs_probability_change']].mean().to_string(),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        write(OUT/'STATE.json',dict(status='failed',pid=None,traceback=traceback.format_exc()));raise
