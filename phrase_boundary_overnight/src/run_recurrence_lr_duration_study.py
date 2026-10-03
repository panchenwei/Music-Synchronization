"""Isolated, state-preserving 300-to-1000 matched continuation; no test prediction."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import run_motif_recurrence_study as prior
from . import run_recurrence_depth_study as depth
from .score_context_study import ROOT, read, write, sha, normalizer, GRID
from .three_round_round2 import split_ids, checkpoint_threshold
from .recurrence_lr_training import make_model
from . import run_recurrence_lr_study as lr
from .recurrence_duration_core import restore, update, improves, selected_step
from .window_exposure_audit import exposure
from .phase7_models import CurvePieceBalancedSampler
from .phase6_models import positive_weight
from .phase2_models import choose_single_threshold
from .local_context_study import predictions, metrics
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/recurrence_lr_duration_study';ART=ROOT/'artifacts/recurrence_lr_duration_study'
COLS=list(prior.COLS);DEVICE=torch.device('cuda')
CELLS=('best_300','terminal_300','best_1000','terminal_1000')


def source(kind):
    return lr.ART,lr.OUT


def load(path):return torch.load(path,map_location=DEVICE,weights_only=False)


def prepare():
    began=time.monotonic()
    for p in (OUT,ART/'runs',ART/'metrics',ART/'checkpoints'):p.mkdir(parents=True,exist_ok=True)
    assert read(lr.OUT/'completion_audit.json')['status']=='complete'
    hashes=dict(read(lr.OUT/'contract.json')['hashes'])
    assert all(sha(p)==h for p,h in hashes.items())
    for kind in ('L3',):
        art,out=source(kind);expected=read(out/'contract.json')['contract']
        for fold in (0,1):
            ids=split_ids(fold);assert not set(ids['train'])&set(ids['validation'])
            manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv');part=manifest[manifest.fold==fold]
            assert not set(part[part.split=='train'].opus)&set(part[part.split=='validation'].opus)
            norm=normalizer(prior.dataset(ids['train'],'O'))
            for seed in (42,43):
                run=f'{kind}_seed{seed}_fold{fold}';cp=art/'checkpoints'/run
                last=load(cp/'latest.pt');best=load(cp/'best.pt')
                assert last['step']==300 and selected_step(last['history'])==best['step']
                for s in (last,best):
                    assert (s['contract'],s['kind'],s['fold'],s['seed'])==(expected,kind,fold,seed)
                    np.testing.assert_array_equal(s['mean'],norm.mean);np.testing.assert_array_equal(s['std'],norm.std)
                for p in (cp/'latest.pt',cp/'best.pt'):hashes[str(p)]=sha(p)
    for p in (Path(__file__),ROOT/'src/recurrence_duration_core.py',ROOT/'tests/test_recurrence_duration_core.py',
              ROOT/'src/window_exposure_audit.py',OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    assert time.monotonic()-began<600
    write(OUT/'STATE.json',dict(status='ready',pid=os.getpid(),contract=digest,prepare_seconds=time.monotonic()-began))
    return digest


def train_one(kind,fold,seed,contract):
    run=f'{kind}_seed{seed}_fold{fold}';done=ART/'runs'/f'{run}.json'
    if done.exists():assert read(done)['contract']==contract;print('CACHED',run,flush=True);return
    dest=ART/'checkpoints'/run;dest.mkdir(exist_ok=True)
    ids=split_ids(fold);train=prior.dataset(ids['train'],'O');val=prior.dataset(ids['validation'],'O');norm=normalizer(train)
    m=make_model(kind,seed).to(DEVICE);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.0001)
    sampler=CurvePieceBalancedSampler(train,norm,64,32,seed)
    crit=torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(train,10),device=DEVICE))
    src=source(kind)[0]/'checkpoints'/run
    original=load(src/'latest.pt');exp300=exposure(train,seed,300)
    assert original['sampler']==exp300['sampler']
    # Two independent restores of a single original GPU step must agree exactly.
    probe=OUT/f'{run}_restore_probe.json'
    if not probe.exists():
        restore(original,m,opt,sampler);a=update(m,opt,sampler,crit,DEVICE)
        weights=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
        cuda=[v.clone() for v in torch.cuda.get_rng_state_all()];sample_state=sampler.state()
        restore(original,m,opt,sampler);b=update(m,opt,sampler,crit,DEVICE)
        assert a==b and sample_state==sampler.state() and torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
        assert all(torch.equal(w,m.state_dict()[k]) for k,w in weights.items())
        write(probe,dict(contract=contract,source_step=300,loss=a[0],gradient=a[1],identical_gpu_update=True))
        del weights
    else:assert read(probe)['contract']==contract
    if not (dest/'latest.pt').exists():
        for cell,name in (('best_300','best.pt'),('terminal_300','latest.pt')):
            s=load(src/name);s['origin_contract']=s['contract'];s['contract']=contract
            s['origin_seconds']=s['seconds'];s['seconds']=0.
            torch.save(s,dest/f'{cell}.pt')
        torch.save(load(dest/'best_300.pt'),dest/'best.pt')
        torch.save(load(dest/'terminal_300.pt'),dest/'latest.pt')
    saved=load(dest/'latest.pt');assert saved['contract']==contract and 300<=saved['step']<=1000
    restore(saved,m,opt,sampler)
    assert all(g['lr']==.0003 for g in opt.param_groups)
    step=saved['step'];history=saved['history'];best_score=saved['best_score'];elapsed_before=saved['seconds']
    completed_seconds=sum(read(p)['seconds'] for p in (ART/'runs').glob('*.json'))
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    write(OUT/'STATE.json',dict(status='running',pid=os.getpid(),current_run=run,step=step,contract=contract))
    def snapshot():
        return dict(model=m.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),step=step,history=history,
                    best_score=best_score,seconds=elapsed_before+time.monotonic()-start,contract=contract,kind=kind,
                    fold=fold,seed=seed,mean=norm.mean,std=norm.std,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    origin_contract=original['contract'],origin_seconds=original['seconds'])
    while step<1000:
        if completed_seconds+elapsed_before+time.monotonic()-start>=2400:
            torch.save(snapshot(),dest/'latest.pt');raise TimeoutError('2400s cumulative continuation cap')
        loss,grad=update(m,opt,sampler,crit,DEVICE);step+=1
        if step%50==0:
            raw=predictions(m,val,norm,DEVICE);threshold,_=choose_single_threshold(raw,val,GRID);score=metrics(raw,val,threshold)[2]
            history.append(dict(step=step,loss=loss,grad=grad,**score))
            if improves(score['macro_f1_tol1'],best_score):best_score=score['macro_f1_tol1'];torch.save(snapshot(),dest/'best.pt')
            print(run,step,'F1',round(score['macro_f1_tol1'],4),'exact',round(score['macro_f1_tol0'],4),'AP',round(score['raw_ap'],4),flush=True)
        if step%25==0:torch.save(snapshot(),dest/'latest.pt')
    seconds=elapsed_before+time.monotonic()-start
    last=load(dest/'latest.pt');best=load(dest/'best.pt');assert selected_step(last['history'])==best['step']
    torch.save(last,dest/'terminal_1000.pt');torch.save(best,dest/'best_1000.pt')
    exp1000=exposure(train,seed,1000);assert last['sampler']==exp1000['sampler']
    write(OUT/f'{run}_exposure.json',dict(contract=contract,at300=exp300,at1000=exp1000,sampler_rng_matched=True))
    write(done,dict(run_id=run,kind=kind,fold=fold,seed=seed,contract=contract,seconds=seconds,
                    best_step=best['step'],gpu_peak_bytes=torch.cuda.max_memory_allocated()))


def audit():
    start=time.monotonic();contract=read(OUT/'contract.json');hashes=contract['hashes']
    assert all(sha(p)==h for p,h in hashes.items());rows=[];checks=[];cp_hashes={}
    runs=[read(p) for p in sorted((ART/'runs').glob('*.json'))];assert len(runs)==4
    write(OUT/'STATE.json',dict(status='auditing',pid=os.getpid(),contract=contract['contract']))
    for r in runs:
        assert r['contract']==contract['contract'];run=r['run_id'];kind=r['kind'];ids=split_ids(r['fold'])
        train=prior.dataset(ids['train'],'O');val=prior.dataset(ids['validation'],'O');norm=normalizer(train)
        dest=ART/'checkpoints'/run
        for cell in CELLS:
            assert time.monotonic()-start<1800
            path=dest/f'{cell}.pt';cp_hashes[str(path)]=sha(path);s=load(path)
            assert (s['contract'],s['kind'],s['fold'],s['seed'])==(contract['contract'],kind,r['fold'],r['seed'])
            np.testing.assert_array_equal(s['mean'],norm.mean);np.testing.assert_array_equal(s['std'],norm.std)
            assert all(g['lr']==.0003 for g in s['optimizer']['param_groups'])
            budget=int(cell.split('_')[1]);policy=cell.split('_')[0];assert s['step']<=budget
            if policy=='terminal':assert s['step']==budget
            m=make_model(kind,r['seed']).to(DEVICE);m.load_state_dict(s['model']);threshold=checkpoint_threshold(s)
            raw=predictions(m,val,norm,DEVICE);perfs,pieces,score=metrics(raw,val,threshold)
            stem=f'{run}_{cell}';predpath=ART/'metrics'/f'{stem}_predictions.csv.gz'
            if not predpath.exists():
                values=[(pid,perf,b,float(p),int(val[pid]['labels'][b]),int(val[pid]['label_mask'][b]))
                        for pid,pp in raw.items() for perf,arr in pp.items() for b,p in enumerate(arr)]
                pd.DataFrame(values,columns=['piece_id','performance_id','beat','probability','label','valid']).to_csv(predpath,index=False)
            cached=checked_raw(pd.read_csv(predpath),val);score_cached=metrics(cached,val,threshold)[2]
            replay=predictions(m,val,norm,DEVICE)
            error=max(float(abs(replay[p][q]-cached[p][q]).max()) for p in raw for q in raw[p]);assert error<2e-4
            assert max(abs(metrics(replay,val,threshold)[2][c]-score_cached[c]) for c in COLS)<1e-10
            expected=next(h for h in s['history'] if h['step']==s['step'])
            assert max(abs(score[c]-expected[c]) for c in COLS)<1e-10
            if cell=='best_300':
                old=read(source(kind)[0]/'metrics'/f'{run}.json')
                assert max(abs(score[c]-old[c]) for c in COLS)<1e-10
            train_score=metrics(predictions(m,train,norm,DEVICE),train,threshold)[2]
            result=dict(**r,cell=cell,budget=budget,policy=policy,selected_step=s['step'],params=sum(p.numel() for p in m.parameters()),
                        train_f1=train_score['macro_f1_tol1'],gap=train_score['macro_f1_tol1']-score['macro_f1_tol1'],**score)
            perfs.to_csv(ART/'metrics'/f'{stem}_performances.csv',index=False);pieces.to_csv(ART/'metrics'/f'{stem}_pieces.csv',index=False)
            write(ART/'metrics'/f'{stem}.json',result);rows.append(result);checks.append(dict(run_id=run,cell=cell,replay_error=error))
            pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',stem,flush=True)
    assert len(cp_hashes)==16 and all(sha(p)==h for p,h in cp_hashes.items())
    assert all(sha(p)==h for p,h in hashes.items())
    df=pd.DataFrame(rows);df.to_csv(ART/'summary.csv',index=False)
    means=df.groupby(['kind','policy','budget'])[COLS+['train_f1','gap']].mean();means.to_csv(OUT/'means.csv');comparisons=[]
    for kind in ('L3',):
        for policy in ('best','terminal'):
            v=df[(df.kind==kind)&(df.policy==policy)];a=v[v.budget==1000].set_index(['fold','seed']);b=v[v.budget==300].set_index(['fold','seed']);d=a[COLS]-b[COLS]
            d.to_csv(OUT/f'delta_{kind}_{policy}_1000_300.csv')
            if policy=='best':assert (d.macro_f1_tol1>=-1e-9).all()
            passed=policy=='best' and d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3 and (a.selected_step>300).sum()>=3
            comparisons.append(dict(kind=kind,policy=policy,**d.mean().to_dict(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),new_selected=int((a.selected_step>300).sum()),passed=bool(passed)))
    v=df[(df.kind=='L3')&(df.policy=='best')&(df.budget==1000)].set_index(['fold','seed'])
    control=pd.DataFrame([read(depth.ART/'metrics'/f'C3_seed{seed}_fold{fold}.json') for fold in (0,1) for seed in (42,43)]).set_index(['fold','seed'])
    d=v[COLS]-control[COLS]
    d.to_csv(OUT/'delta_L3_1000_C3_300.csv')
    write(OUT/'best_reference_comparison.json',dict(**d.mean().to_dict(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),equal_compute=False))
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False);write(OUT/'checkpoint_hashes.json',cp_hashes)
    write(OUT/'completion_audit.json',dict(status='complete',continuation_trajectories=4,evaluated_cells=16,
          full_gpu_replays=16,source_hashes_unchanged=True,norms_rebuilt=True,test_predictions_accessed=False,
          source_and_final_sampler_rng_verified=True,restore_gpu_probes=4,new_training_validation_seconds=sum(r['seconds'] for r in runs),
          export_and_audit_seconds=time.monotonic()-start))
    write(OUT/'STATE.json',dict(status='complete',pid=None,contract=contract['contract']));print(means.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','all','audit'),default='all');args=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    try:
        if args.stage=='audit':audit();return
        digest=prepare()
        if args.stage=='prepare':print(digest,flush=True);return
        for fold in (0,1):
            for seed in (42,43):
                for kind in ('L3',):train_one(kind,fold,seed,digest)
        audit()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise


if __name__=='__main__':main()
