"""Bounded development-only end threshold sweep; original starts fixed."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .confidence_pair_decoder import decode
from .phrase_end_auxiliary import ROOT,read,write,sha,dataset,end_data,split_ids
from .audit_external_stem_transfer import checked_raw
from .phase2_models import evaluate_single_performance

OUT=ROOT/'reports/pair_threshold_sweep';ART=ROOT/'artifacts/pair_threshold_sweep'
COLS=['macro_precision_tol1','macro_recall_tol1','macro_f1_tol1','macro_f1_tol0']
OFFSETS=(0.,.1,.2,.3,.4,.5)


def main():
    began=time.monotonic();ART.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/confidence_pairing/completion_audit.json')['status']=='complete'
    hashes=dict(read(ROOT/'reports/confidence_pairing/contract.json')['hashes'])
    assert all(sha(p)==h for p,h in hashes.items())
    for p in (Path(__file__),OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes))
    write(OUT/'STATE.json',dict(status='running',contract=contract,completed=0))
    reference=pd.read_csv(ROOT/'reports/confidence_pairing/comparison.csv').set_index(['fold','seed'])
    rows=[];audits=[]
    for fold in (0,1):
        val=dataset(split_ids(fold)['validation']);ev=end_data(val)
        for seed in (42,43):
            cp=ROOT/f'artifacts/phrase_end_auxiliary/metrics/C_seed{seed}_fold{fold}'
            dp=ROOT/f'artifacts/independent_phrase_end/metrics/D_seed{seed}_fold{fold}'
            c,d=read(str(cp)+'.json'),read(str(dp)+'.json')
            cr=checked_raw(pd.read_csv(str(cp)+'_predictions.csv.gz'),val)
            dr=checked_raw(pd.read_csv(str(dp)+'_predictions.csv.gz'),ev)
            for offset in OFFSETS:
                assert time.monotonic()-began<600
                threshold=max(.05,round(d['threshold']-offset,8));paired={};arrays={};keymap=[];spans_total=0
                for pid,perfs in cr.items():
                    paired[pid]={}
                    for perfid,start in perfs.items():
                        prob,spans,_=decode(start,dr[pid][perfid],c['threshold'],threshold)
                        paired[pid][perfid]=prob;spans_total+=len(spans)
                        key=f'p{len(keymap)}';arrays[key]=prob;keymap.append([key,pid,perfid])
                run=f'offset{offset:.1f}_seed{seed}_fold{fold}'
                dest=ART/f'{run}.npz';np.savez_compressed(dest,**arrays);write(ART/f'{run}_keys.json',keymap)
                pp,pieces,summary=evaluate_single_performance(paired,val,c['threshold'])
                pp.to_csv(ART/f'{run}_performances.csv',index=False);pieces.to_csv(ART/f'{run}_pieces.csv',index=False)
                replay={p:{} for p in cr}
                with np.load(dest,allow_pickle=False) as z:
                    for key,pid,perfid in read(ART/f'{run}_keys.json'):replay[pid][perfid]=z[key].copy()
                rs=evaluate_single_performance(replay,val,c['threshold'])[2]
                err=max(abs(rs[k]-summary[k]) for k in COLS);assert err<1e-10
                if offset==0:assert max(abs(summary[k]-reference.loc[(fold,seed),'paired_'+k]) for k in COLS)<1e-10
                rows.append(dict(run_id=run,fold=fold,seed=seed,offset=offset,start_threshold=c['threshold'],end_threshold=threshold,**{k:summary[k] for k in COLS},**{'delta_'+k:summary[k]-c[k] for k in COLS},raw_start_ap_unchanged=c['raw_ap'],spans=spans_total))
                audits.append(dict(run_id=run,metric_replay_error=err,arrays=len(keymap),saved_sha256=sha(dest)))
                pd.DataFrame(rows).to_csv(OUT/'all_results.csv',index=False)
                write(OUT/'STATE.json',dict(status='running',contract=contract,completed=len(rows)))
                print(run,'F1',round(summary['macro_f1_tol1'],5),'P/R',round(summary['macro_precision_tol1'],4),round(summary['macro_recall_tol1'],4),flush=True)
    df=pd.DataFrame(rows);assert len(df)==24
    means=df.groupby('offset')[COLS+['delta_'+k for k in COLS]].mean()
    means['f1_positive']=df.groupby('offset').delta_macro_f1_tol1.apply(lambda x:int((x>0).sum()))
    means['promotion']=(means.delta_macro_f1_tol1>=.015)&(means.delta_macro_f1_tol0>0)&(means.f1_positive>=3)
    means.to_csv(OUT/'means.csv');best=means.reset_index().sort_values(['macro_f1_tol1','offset'],ascending=[False,True]).iloc[0]
    write(OUT/'selected_development_setting.json',json.loads(best.to_json()))
    pd.DataFrame(audits).to_csv(OUT/'run_audit.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',configurations=24,metric_replays=24,original_offset_reproduced=4,source_input_hashes_unchanged=True,training_steps=0,test_predictions_accessed=False,development_selected_not_unbiased=True,seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',contract=contract,completed=24));print(means.to_string(),flush=True)


if __name__=='__main__':main()
