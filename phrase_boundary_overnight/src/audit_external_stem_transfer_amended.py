"""Supplemental audit: retain failed CPU check; require same-device replay and IEEE cross-check."""
import time
import numpy as np
import pandas as pd
import torch
from .external_stem_transfer import OUT,ART,COLS,ROOT,read,write,sha,dataset,external_dataset,external_split,build_model,split_ids,normalizer,Normalizer,RollBoundary,roll_predictions,metrics
from .fixed_ensemble_study import aligned_average,raw_from_frame


def checked_raw(frame,data):
    assert set(frame.piece_id)==set(data)
    raw={}
    for (pid,perf),g in frame.groupby(['piece_id','performance_id']):
        g=g.sort_values('beat');item=data[pid]
        np.testing.assert_array_equal(g.beat,np.arange(len(item['labels'])))
        np.testing.assert_array_equal(g.label,item['labels']);np.testing.assert_array_equal(g.valid,item['label_mask'])
        assert np.isfinite(g.probability).all() and g.probability.between(0,1).all()
        raw.setdefault(pid,{})[str(perf)]=g.probability.to_numpy()
    for pid,item in data.items():assert set(raw[pid])==set(item['performance_ids'].astype(str))
    return raw


def main():
    start=time.monotonic();torch.set_num_threads(2);state=read(OUT/'STATE.json');assert state['status'] in ('training_complete','complete','failed','complete_with_numerical_caveat')
    contract=read(OUT/'contract.json');assert state['contract']==contract['contract']
    assert all(sha(p)==h for p,h in contract['hashes'].items())
    checkpoints=list((ART/'checkpoints').glob('*/*.pt'))+list((ART/'external').glob('*/*.pt'));assert len(checkpoints)==20
    before={str(p):sha(p) for p in checkpoints};cpu=torch.device('cpu');gpu=torch.device('cuda');rows=[];histories=[];external_rows=[]
    torch.backends.cudnn.allow_tf32=True
    split=external_split();saved_split=pd.read_csv(OUT/'external_split.csv');pd.testing.assert_frame_equal(split,saved_split)
    val=external_dataset('validation');train=external_dataset('train');zero_norm=Normalizer(np.zeros(58,np.float32),np.ones(58,np.float32))
    for seed in (42,43):
        dest=ART/'external'/f'seed{seed}';r=read(dest/'result.json');s=torch.load(dest/'best.pt',map_location=cpu,weights_only=False);latest=torch.load(dest/'latest.pt',map_location=cpu,weights_only=False)
        assert latest['step']==600 and s['step']==r['best_step'] and s['contract']==state['contract']==r['contract']==latest['contract']
        assert s['seed']==r['seed']==seed and r['selection_metric']=='raw_ap'
        expected=max(latest['history'],key=lambda h:h['raw_ap']);assert expected['step']==s['step']
        frame=pd.read_csv(dest/'validation_predictions.csv.gz');raw=checked_raw(frame,val)
        _,_,score=metrics(raw,val,r['threshold']);error=max(abs(score[c]-r[c]) for c in COLS);assert error<1e-10
        model=RollBoundary('L',seed);initial={k:v.clone() for k,v in model.stem.state_dict().items()};model.load_state_dict(s['model']);model.eval()
        replay=roll_predictions(model,val,zero_norm,cpu)
        cpu_delta=max(float(np.max(abs(replay[p]['score']-raw[p]['score']))) for p in val)
        cpu_score=metrics(replay,val,r['threshold'])[2]
        model.to(gpu);gpu_replay=roll_predictions(model,val,zero_norm,gpu)
        delta=max(float(np.max(abs(gpu_replay[p]['score']-raw[p]['score']))) for p in val);assert delta<2e-4
        gpu_score=metrics(gpu_replay,val,r['threshold'])[2];assert max(abs(gpu_score[c]-r[c]) for c in COLS)<1e-10
        torch.backends.cudnn.allow_tf32=False
        ieee=roll_predictions(model,val,zero_norm,gpu)
        ieee_delta=max(float(np.max(abs(ieee[p]['score']-replay[p]['score']))) for p in val);assert ieee_delta<2e-4
        torch.backends.cudnn.allow_tf32=True
        _,_,ts=metrics(roll_predictions(model,train,zero_norm,gpu),train,r['threshold'])
        change=max(float((v.detach().cpu()-initial[k]).abs().max()) for k,v in model.stem.state_dict().items());assert change>0
        external_rows.append(dict(seed=seed,best_step=s['step'],validation_f1=r['macro_f1_tol1'],validation_ap=r['raw_ap'],train_f1=ts['macro_f1_tol1'],train_ap=ts['raw_ap'],stem_max_change=change,metric_error=error,gpu_replay_error=delta,cpu_replay_error=cpu_delta,cpu_original_tolerance_pass=cpu_delta<2e-4,cpu_vs_ieee_gpu_error=ieee_delta,cpu_f1_delta=cpu_score['macro_f1_tol1']-r['macro_f1_tol1']))
        for h in latest['history']:histories.append(dict(run_id=f'external_seed{seed}',**h))
        print('AUDITED external',seed,flush=True)
    pd.DataFrame(external_rows).to_csv(OUT/'external_audit.csv',index=False)
    results=pd.read_csv(ART/'summary.csv');assert len(results)==8 and set(zip(results.kind,results.fold,results.seed))=={(k,f,s) for k in ('R','T') for f in (0,1) for s in (42,43)}
    for r in results.to_dict('records'):
        run=r['run_id'];ids=split_ids(r['fold']);assert not set(ids['train'])&set(ids['validation']);data=dataset(ids['validation'],r['kind'])
        frame=pd.read_csv(ART/'metrics'/f'{run}_predictions.csv.gz');raw=checked_raw(frame,data);_,_,score=metrics(raw,data,r['threshold']);error=max(abs(score[c]-r[c]) for c in COLS);assert error<1e-10
        s=torch.load(ART/'checkpoints'/run/'best.pt',map_location=cpu,weights_only=False);latest=torch.load(ART/'checkpoints'/run/'latest.pt',map_location=cpu,weights_only=False)
        assert (s['contract'],s['kind'],s['seed'],s['fold'],s['step'])==(state['contract'],r['kind'],r['seed'],r['fold'],r['best_step'])
        assert latest['step']==300 and latest['contract']==state['contract']
        model=build_model(r['kind'],r['seed']);initial={k:v.clone() for k,v in model.stem.state_dict().items()};model.load_state_dict(s['model']);model.eval()
        assert sum(p.numel() for p in model.parameters())==6121 and sum(p.numel() for p in model.parameters() if p.requires_grad)==3297
        change=max(float((v-initial[k]).abs().max()) for k,v in model.stem.state_dict().items());assert change==0
        norm=Normalizer(s['mean'],s['std']);train=dataset(ids['train'],r['kind']);rebuilt=normalizer(train);np.testing.assert_array_equal(norm.mean,rebuilt.mean);np.testing.assert_array_equal(norm.std,rebuilt.std)
        pid=sorted(data)[0];perf=sorted(raw[pid])[len(raw[pid])//2];idx=list(data[pid]['performance_ids'].astype(str)).index(perf)
        subset={pid:{**data[pid],'curves':data[pid]['curves'][idx:idx+1],'performance_ids':np.array([perf])}}
        replay=roll_predictions(model,subset,norm,cpu)[pid][perf];cpu_delta=float(np.max(abs(replay-raw[pid][perf])))
        model.to(gpu);full_gpu_replay=roll_predictions(model,data,norm,gpu);gpu_replay=full_gpu_replay[pid][perf]
        delta=max(float(np.max(abs(full_gpu_replay[p][k]-raw[p][k]))) for p in raw for k in raw[p]);assert delta<2e-4
        full_gpu_score=metrics(full_gpu_replay,data,r['threshold'])[2]
        assert max(abs(full_gpu_score[c]-r[c]) for c in COLS)<1e-10
        torch.backends.cudnn.allow_tf32=False
        ieee=roll_predictions(model,subset,norm,gpu)[pid][perf]
        ieee_delta=float(np.max(abs(ieee-replay)));assert ieee_delta<2e-4
        torch.backends.cudnn.allow_tf32=True
        _,_,rs=metrics({pid:{perf:replay}},subset,r['threshold']);_,_,cs=metrics({pid:{perf:raw[pid][perf]}},subset,r['threshold']);assert abs(metrics({pid:{perf:gpu_replay}},subset,r['threshold'])[2]['macro_f1_tol1']-cs['macro_f1_tol1'])<1e-12
        _,_,ts=metrics(roll_predictions(model,train,norm,gpu),train,r['threshold']);rows.append(dict(run_id=run,kind=r['kind'],fold=r['fold'],seed=r['seed'],metric_error=error,gpu_replay_error=delta,cpu_replay_error=cpu_delta,cpu_original_tolerance_pass=cpu_delta<2e-4,cpu_vs_ieee_gpu_error=ieee_delta,cpu_f1_delta=rs['macro_f1_tol1']-cs['macro_f1_tol1'],stem_max_change=change,train_f1=ts['macro_f1_tol1'],validation_f1=r['macro_f1_tol1'],gap=ts['macro_f1_tol1']-r['macro_f1_tol1'],train_ap=ts['raw_ap'],validation_ap=r['raw_ap']))
        for h in latest['history']:histories.append(dict(run_id=run,**h))
        pd.DataFrame(rows).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',run,'gap',rows[-1]['gap'],flush=True)
    ens=pd.read_csv(ART/'ensemble_summary.csv');assert len(ens)==8;ens_rows=[]
    for r in ens.to_dict('records'):
        kind=r['kind'][-1];paths=[ROOT/'artifacts/label_repaired_rebaseline/metrics'/f"N_seed{r['seed']}_fold{r['fold']}_predictions.csv.gz",ART/'metrics'/f"{kind}_seed{r['seed']}_fold{r['fold']}_predictions.csv.gz"]
        rebuilt=aligned_average([pd.read_csv(p) for p in paths]).sort_values(['piece_id','performance_id','beat']).reset_index(drop=True)
        frame=pd.read_csv(ART/'ensemble'/f"{r['run_id']}_predictions.csv.gz").sort_values(['piece_id','performance_id','beat']).reset_index(drop=True)
        error=float(np.max(abs(rebuilt.probability-frame.probability)));assert error<1e-12
        data=dataset(split_ids(r['fold'])['validation'],'R');raw=checked_raw(frame,data);_,_,score=metrics(raw,data,r['threshold']);delta=max(abs(score[c]-r[c]) for c in COLS);assert delta<1e-10
        ens_rows.append(dict(run_id=r['run_id'],probability_error=error,metric_error=delta))
    pd.DataFrame(ens_rows).to_csv(OUT/'ensemble_audit.csv',index=False);pd.DataFrame(histories).to_csv(OUT/'training_history.csv',index=False)
    assert before=={str(p):sha(p) for p in checkpoints};assert all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',before);write(OUT/'completion_audit.json',dict(status='complete_with_numerical_caveat',original_cpu_tolerance_not_passed=True,same_device_replays_pass=True,cpu_vs_ieee_gpu_replays_pass=True,amended_audit_sha256=sha(__file__),external_pretrains=2,target_runs=8,checkpoints=20,target_checkpoint_replays=8,external_full_validation_replays=2,ensemble_replays=8,all_metric_recomputations_pass=True,frozen_stems_unchanged=True,source_and_checkpoint_hashes_unchanged=True,test_predictions_accessed=False,audit_training_runs=0,audit_seconds=time.monotonic()-start))
    state.update(status='complete_with_numerical_caveat',pid=None);write(OUT/'STATE.json',state)


if __name__=='__main__':main()
