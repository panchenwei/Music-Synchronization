"""Replay only differing checkpoint selections under a uniform AP policy."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from .score_context_study import *
from . import curve_arc_study as arc
from . import run_halo_decoder_composition as dec
from .fixed_ensemble_study import raw_from_frame
from .interstart_decoder import fit_prior
OUT=ROOT/'reports/curve_arc_ap_selection';ART=ROOT/'artifacts/curve_arc_ap_selection'
dataset=arc.dataset;normalizer=arc.normalizer;make_model=arc.ArcBoundary;CAP=1200.
COLS=arc.COLS

def train_one(kind,fold,seed,contract):
    run=f'{kind}_seed{seed}_fold{fold}';respath=ART/'metrics'/f'{run}.json'
    if respath.exists():assert read(respath)['contract']==contract;print('CACHED',run,flush=True);return
    state=read(OUT/'STATE.json');dest=ART/'checkpoints'/run;dest.mkdir(exist_ok=True);best=dest/'best.pt';latest=dest/'latest.pt'
    ids=split_ids(fold);train=dataset(ids['train'],kind);val=dataset(ids['validation'],kind)
    norm=normalizer(train);model=make_model(kind,seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    sampler=CurvePieceBalancedSampler(train,norm,64,32,seed);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    crit=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(train,10),device=device))
    step=0;history=[];best_score=-1.;prior=0.
    if latest.exists():
        s=torch.load(latest,map_location=device,weights_only=False);assert s['contract']==contract
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);sampler.load_state(s['sampler'])
        step=s['step'];history=s['history'];best_score=s['best_score'];prior=s['seconds'];torch.set_rng_state(s['rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']])
    start=time.monotonic();state.update(status='running',current_run=run,pid=os.getpid());write(OUT/'STATE.json',state)
    def snapshot():return dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),step=step,history=history,best_score=best_score,seconds=prior+time.monotonic()-start,contract=contract,kind=kind,fold=fold,seed=seed,mean=norm.mean,std=norm.std,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    while step<300:
        if state['seconds']+prior+time.monotonic()-start>=CAP:
            torch.save(snapshot(),latest);raise TimeoutError('Frozen per-batch 2400s cap reached')
        model.train();x,y,mask,valid=(v.to(device) for v in sampler.batch());opt.zero_grad(set_to_none=True);logit=model(x,padding_mask=~valid.bool());loss=(crit(logit,y)*mask).sum()/mask.sum().clamp_min(1)
        assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);opt.step();step+=1
        if step%50==0:
            raw=predictions(model,val,norm,device);threshold,_=choose_single_threshold(raw,val,GRID);_,_,score=metrics(raw,val,threshold)
            history.append(dict(step=step,loss=float(loss.detach()),grad=float(gn),**score))
            if score['raw_ap']>best_score+1e-9:best_score=score['raw_ap'];torch.save(snapshot(),best)
            print(run,step,'F1',round(score['macro_f1_tol1'],4),'exact',round(score['macro_f1_tol0'],4),'AP',round(score['raw_ap'],4),flush=True)
        if step%25==0:torch.save(snapshot(),latest)
    s=torch.load(best,map_location=device,weights_only=False);model.load_state_dict(s['model']);threshold=checkpoint_threshold(s);raw=predictions(model,val,norm,device)
    perfs,pieces,score=metrics(raw,val,threshold);elapsed=prior+time.monotonic()-start
    result=dict(run_id=run,kind=kind,fold=fold,seed=seed,contract=contract,best_step=int(s['step']),params=sum(p.numel() for p in model.parameters()),seconds=elapsed,gpu_peak_bytes=int(torch.cuda.max_memory_allocated()) if device.type=='cuda' else 0,**score)
    perfs.to_csv(ART/'metrics'/f'{run}_performances.csv',index=False);pieces.to_csv(ART/'metrics'/f'{run}_pieces.csv',index=False)
    prediction_rows=[(pid,perf,b,float(p),int(val[pid]['labels'][b]),int(val[pid]['label_mask'][b])) for pid,pp in raw.items() for perf,arr in pp.items() for b,p in enumerate(arr)]
    pd.DataFrame(prediction_rows,columns=['piece_id','performance_id','beat','probability','label','valid']).to_csv(ART/'metrics'/f'{run}_predictions.csv.gz',index=False)
    write(respath,result);state['completed'].append(run);state['seconds']+=elapsed;state.update(status='between_runs',pid=os.getpid());write(OUT/'STATE.json',state)

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);started=time.monotonic()
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/curve_arc_fusion/completion_audit.json')['status']=='complete'
    hashes=dict(read(arc.OUT/'contract.json')['hashes']);hashes.update(read(arc.OUT/'checkpoint_hashes.json'))
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'src/score_context_study.py'):hashes[str(p)]=sha(p)
    for p in (arc.ART/'metrics').glob('R*'):hashes[str(p)]=sha(p)
    assert all(sha(p)==v for p,v in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',completed=[],seconds=0.,pid=None))
    rows=[];audit=[];dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder';newruns=0
    for fold in (0,1):
        ids=split_ids(fold);train=dataset(ids['train'],'R');val=dataset(ids['validation'],'R');norm=normalizer(train);prior=fit_prior(train)
        for seed in (42,43):
            run=f'R_seed{seed}_fold{fold}';old=torch.load(arc.ART/'checkpoints'/run/'latest.pt',map_location='cpu',weights_only=False)
            ap_step=max(old['history'],key=lambda h:h['raw_ap'])['step'];f1_step=max(old['history'],key=lambda h:h['macro_f1_tol1'])['step']
            source=arc.ART
            if ap_step!=f1_step:
                guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
                train_one('R',fold,seed,digest);newruns+=1;source=ART
                last=torch.load(ART/'checkpoints'/run/'latest.pt',map_location='cpu',weights_only=False)
                assert last['step']==300
                for name,value in old['model'].items():torch.testing.assert_close(last['model'][name],value,atol=0,rtol=0)
                for a,b in zip(old['history'],last['history']):
                    for key in ('step','raw_ap','macro_f1_tol1','macro_f1_tol0'):assert a[key]==b[key]
            best=torch.load(source/'checkpoints'/run/'best.pt',map_location='cpu',weights_only=False);meta=read(source/'metrics'/f'{run}.json')
            assert best['step']==meta['best_step']==ap_step
            np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
            raw=raw_from_frame(pd.read_csv(source/'metrics'/f'{run}_predictions.csv.gz'),val)
            score=metrics(raw,val,meta['threshold'])[2];assert max(abs(score[c]-meta[c]) for c in COLS)<1e-10
            if source==ART:
                m=make_model('R',seed).cuda().eval();m.load_state_dict(best['model']);replay=predictions(m,val,norm,torch.device('cuda'))
                err=max(float(abs(replay[p][q]-raw[p][q]).max()) for p in raw for q in raw[p]);assert err<2e-4
                assert max(abs(metrics(replay,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
            audit.append(dict(run_id=run,old_f1_step=f1_step,ap_step=ap_step,replayed_training=source==ART))
            for strength in (0.,.5):
                row=dec.evaluate(raw,val,meta['threshold'],prior,strength,f'AP_{run}_lambda{strength:g}',digest)
                if strength==0:assert max(abs(row[c]-score[c]) for c in COLS)<1e-10
                rows.append(dict(kind='AP',fold=fold,seed=seed,**row))
    d=pd.DataFrame(rows);assert len(d)==8 and newruns==1
    d.to_csv(ART/'summary.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'checkpoint_policy.csv',index=False)
    old=pd.read_csv(arc.ART/'decoder/summary.csv');old=old[old.kind=='R']
    ref=pd.read_csv(ROOT/'artifacts/interstart_decoder_study/summary.csv');ref=ref[ref.strength.isin((0.,.5))].copy();ref['kind']='C3'
    allrows=pd.concat([d,old,ref],ignore_index=True);means=allrows.groupby(['kind','strength'])[COLS].mean();means.to_csv(OUT/'means.csv')
    comparisons=[]
    for control in ('R','C3'):
        a=d[d.strength==.5].set_index(['fold','seed'])[COLS];b=allrows[(allrows.kind==control)&(allrows.strength==.5)].set_index(['fold','seed'])[COLS];delta=a-b
        comparisons.append(dict(control=control,**delta.mean().to_dict(),positive=int((delta.macro_f1_tol1>0).sum())))
    write(OUT/'comparisons.json',comparisons)
    assert all(sha(p)==v for p,v in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',replayed_training_runs=1,reused_runs=3,identical_terminal_weights_and_metrics=True,full_gpu_replays=1,metric_recomputations=4,decoder_cells=8,hashes_unchanged=True,test_used=False,seconds=time.monotonic()-started))
    write(OUT/'STATE.json',dict(status='complete',pid=None));write(dec.OUT/'STATE.json',dict(status='complete',pid=None));print(means.to_string(),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);write(OUT/'STATE.json',dict(status='failed',traceback=traceback.format_exc()));raise
