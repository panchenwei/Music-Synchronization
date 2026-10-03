"""Post-training feature-use diagnostics; never retrains or selects thresholds."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import time,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
from . import axis_auxiliary_study as study
from .axis_split_music import features_in_blocks
from .axis_auxiliary_model import AuxiliaryBoundary as AxisBoundary
from .score_context_study import ROOT,read,write,sha,normalizer
from .three_round_round2 import split_ids
from .audit_external_stem_transfer import checked_raw
from .local_context_study import metrics

OUT=ROOT/'reports/axis_auxiliary_information';ART=ROOT/'artifacts/axis_auxiliary_information'


def diagnose(model,val,norm):
    variants=('intact','zero_embedding','silent_roll');outputs={k:{} for k in variants};feature_rows=[]
    model.eval()
    with torch.no_grad():
        for pid,item in sorted(val.items()):
            roll=torch.from_numpy(item['piano_roll'][None]).cuda()
            actual=features_in_blocks(model,roll)
            features={'intact':actual,'zero_embedding':torch.zeros_like(actual),'silent_roll':features_in_blocks(model,torch.zeros_like(roll))}
            projection=model.core.input_projection.weight[:,58:]
            delta=actual@projection.T
            feature_rows.append(dict(piece_id=pid,embedding_time_std=float(actual.std(1,unbiased=False).mean()),projection_time_std=float(delta.std(1,unbiased=False).mean()),projection_absolute_mean=float(delta.mean(1).abs().mean())))
            for variant in variants:
                rows=[]
                for start in range(0,len(item['curves']),8):
                    x=torch.from_numpy(norm.apply(item['curves'][start:start+8]).astype(np.float32)).cuda()
                    z=features[variant].expand(len(x),-1,-1)
                    rows.extend(torch.sigmoid(model.forward_embedded(x,z)).cpu().numpy())
                outputs[variant][pid]={str(k):v for k,v in zip(item['performance_ids'],rows)}
    return outputs,feature_rows


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);began=time.monotonic()
    assert read(study.OUT/'completion_audit.json')['status']=='complete'
    for p in (OUT,ART):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(study.OUT/'checkpoint_hashes.json'))
    hashes.update(read(study.OUT/'contract.json')['hashes'])
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_axis_auxiliary_information.py'):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'source_hashes.json',hashes);results=[];feature_rows=[]
    write(OUT/'STATE.json',dict(status='running',pid=os.getpid()))
    for fold in (0,1):
        ids=split_ids(fold);val=study.dataset(ids['validation'],'A');norm=normalizer(study.dataset(ids['train'],'A'))
        for seed in (42,43):
            for kind in study.KINDS:
                run=f'{kind}_seed{seed}_fold{fold}';meta=read(study.ART/'metrics'/f'{run}.json')
                checkpoint=torch.load(study.ART/'checkpoints'/run/'best.pt',map_location='cpu',weights_only=False)
                model=AxisBoundary(kind,seed).cuda();model.load_state_dict(checkpoint['model'])
                raw_saved=checked_raw(pd.read_csv(study.ART/'metrics'/f'{run}_predictions.csv.gz'),val)
                outputs,fr=diagnose(model,val,norm)
                error=max(float(abs(outputs['intact'][p][q]-raw_saved[p][q]).max()) for p in raw_saved for q in raw_saved[p]);assert error<2e-4
                feature_rows.extend(dict(run_id=run,kind=kind,fold=fold,seed=seed,**r) for r in fr)
                for variant,raw in outputs.items():
                    _,pieces,score=metrics(raw,val,meta['threshold']);pieces.to_csv(ART/f'{run}_{variant}_pieces.csv',index=False)
                    if variant=='intact':assert max(abs(score[c]-meta[c]) for c in study.COLS)<1e-10
                    piece_change=[np.mean([np.mean(abs(raw[p][q]-outputs['intact'][p][q])) for q in raw[p]]) for p in raw]
                    results.append(dict(run_id=run,kind=kind,fold=fold,seed=seed,variant=variant,mean_abs_probability_change=float(np.mean(piece_change)),replay_error=error,**score))
                pd.DataFrame(results).to_csv(ART/'summary.csv',index=False);pd.DataFrame(feature_rows).to_csv(ART/'feature_variation.csv',index=False)
                print('INFORMATION AUDIT',run,flush=True)
    df=pd.DataFrame(results);assert len(df)==12
    z=df[df.variant=="zero_embedding"].set_index("run_id")
    s=df[df.variant=="silent_roll"].set_index("run_id")
    assert abs(z[study.COLS+["mean_abs_probability_change"]]-s[study.COLS+["mean_abs_probability_change"]]).to_numpy().max()<1e-12
    df.groupby(['kind','variant'])[study.COLS+['mean_abs_probability_change']].mean().to_csv(OUT/'means.csv')
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',checkpoints_unchanged=True,models=4,conditions=12,centered_silent_equals_zero_embedding=True,thresholds_refitted=False,test_used=False,seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));print(df.groupby(['kind','variant'])[study.COLS+['mean_abs_probability_change']].mean().to_string(),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        write(OUT/'STATE.json',dict(status='failed',pid=None,traceback=traceback.format_exc()));raise
