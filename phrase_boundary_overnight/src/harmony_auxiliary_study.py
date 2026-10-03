"""Frozen C3 plus training-only harmonic-degree objective and matched controls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from . import run_recurrence_depth_study as base
from . import start_guided_attention_study as audit_engine
from .harmony_auxiliary import targets,make_model,HarmonySampler
from .score_context_study import ROOT,read,write,sha,normalizer,GRID,positive_weight,choose_single_threshold,checkpoint_threshold
from .three_round_round2 import split_ids
from .slice_energy_study import DCML
from .data import discover_dcml_pieces
from .local_context_study import predictions,metrics

OUT=ROOT/'reports/harmony_auxiliary';ART=ROOT/'artifacts/harmony_auxiliary';CAP=4800.


def dataset(ids,kind):
    data=base.dataset(ids,'C3')
    for pid,v in data.items():
        with np.load(ART/'cache'/f'{pid}.npz',allow_pickle=False) as z:
            v['harmony_labels']=z['labels'].copy();v['harmony_mask']=z['mask'].copy()
    return data


def prepare():
    began=time.monotonic()
    for p in (OUT,ART/'cache',ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(base.OUT/'contract.json')['hashes']);assert all(sha(p)==h for p,h in hashes.items())
    ids=sorted({p for f in (0,1) for split in ('train','validation') for p in split_ids(f)[split]})
    sources=discover_dcml_pieces(DCML);counts=[];examples=[];files=[]
    for pid in ids:
        piece=sources[pid];original=ROOT/'artifacts/coordinate_repair_preview'/f'{pid}.npz'
        with np.load(original,allow_pickle=False) as z:n=len(z['labels'])
        y,m,rows,mode,error=targets(pd.read_csv(piece.harmony_path,sep='\t'),pd.read_csv(piece.measures_path,sep='\t'),n)
        assert m.sum()>0
        dest=ART/'cache'/f'{pid}.npz'
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:np.testing.assert_array_equal(z['labels'],y);np.testing.assert_array_equal(z['mask'],m)
        else:np.savez_compressed(dest,labels=y,mask=m)
        counts.append(dict(piece_id=pid,beats=n,known=int(m.sum()),coverage=float(m.mean()),classes=json.dumps(np.bincount(y[m>.5],minlength=7).tolist()),mode=mode,length_error=error))
        examples.extend([dict(piece_id=pid,**r) for r in rows])
        files.extend([original,piece.harmony_path,piece.measures_path,dest])
    pd.DataFrame(counts).to_csv(OUT/'label_coverage.csv',index=False);pd.DataFrame(examples).to_csv(OUT/'source_intervals.csv',index=False)
    splitrows=[]
    for f in (0,1):
        idsf=split_ids(f);a=base.dataset(idsf['train'],'C3');b=dataset(idsf['train'],'G');an=normalizer(a);bn=normalizer(b)
        np.testing.assert_array_equal(an.mean,bn.mean);np.testing.assert_array_equal(an.std,bn.std)
        for p in a:
            for field in ('curves','pitch_profiles','labels','label_mask'):np.testing.assert_array_equal(a[p][field],b[p][field])
        manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv');part=manifest[manifest.fold==f]
        assert not set(part[part.split=='train'].opus)&set(part[part.split=='validation'].opus)
        splitrows.append(dict(fold=f,train=len(a),validation=len(idsf['validation']),inputs_unchanged=True,opus_disjoint=True))
    pd.DataFrame(splitrows).to_csv(OUT/'split_audit.csv',index=False)
    files+=[Path(__file__),ROOT/'src/harmony_auxiliary.py',ROOT/'src/start_guided_attention_study.py',ROOT/'tests/test_harmony_auxiliary.py',OUT/'PROTOCOL.md',
        ROOT/'src/interstart_decoder.py',ROOT/'src/run_halo_decoder_composition.py',ROOT/'src/recurrence_message_probe.py',ROOT/'src/context_inference_audit_v2.py']
    files+=list((ROOT/'artifacts/recurrence_mean_control/graphs').glob('*.npz'))
    for f in (0,1):
        for s in (42,43):
            for suffix in ('.json','_predictions.csv.gz'):files.append(base.ART/'metrics'/f'C3_seed{s}_fold{f}{suffix}')
    for p in files:hashes[str(p)]=sha(p)
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',completed=[],seconds=0.,pid=None))
    write(OUT/'label_audit.json',dict(status='complete',pieces=len(counts),known=sum(r['known'] for r in counts),beats=sum(r['beats'] for r in counts),
        input_labels_unchanged=True,training_only_auxiliary=True,seconds=time.monotonic()-began))
    return digest


def preflight():
    if (OUT/'preflight.json').exists():assert read(OUT/'preflight.json')['passed'];return
    # Tiny artificial sanity check only; not a held-out music score.
    torch.manual_seed(20260914);x=torch.randn(2,16,58);y=(x[...,0]>0).float();h=x[...,:7].argmax(-1)
    model=make_model('G',20260914);opt=torch.optim.AdamW(model.parameters(),lr=.01)
    def objective():
        a,b=model(x,both=True)
        return nn.functional.binary_cross_entropy_with_logits(a,y)+.25*nn.functional.cross_entropy(b.reshape(-1,7),h.reshape(-1))/np.log(7)
    model.eval();before=float(objective().detach())
    for _ in range(160):
        model.train();opt.zero_grad();loss=objective();assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    model.eval();after=float(objective().detach());assert after<before*.25
    write(OUT/'preflight.json',dict(passed=True,tiny_artificial_steps=160,loss_before=before,loss_after=after,params=6152))


def train_one(kind,fold,seed,contract):
    run=f'{kind}_seed{seed}_fold{fold}';respath=ART/'metrics'/f'{run}.json'
    if respath.exists():assert read(respath)['contract']==contract;print('CACHED',run,flush=True);return
    state=read(OUT/'STATE.json');dest=ART/'checkpoints'/run;dest.mkdir(exist_ok=True);best=dest/'best.pt';latest=dest/'latest.pt'
    ids=split_ids(fold);train=dataset(ids['train'],kind);val=dataset(ids['validation'],kind)
    norm=normalizer(train);model=make_model(kind,seed).cuda();device=torch.device('cuda');sampler=HarmonySampler(train,norm,seed,kind)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    crit=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(train,10),device=device))
    step=0;history=[];best_score=-1.;prior=0.;weight=0. if kind=='Z' else .25
    if latest.exists():
        s=torch.load(latest,map_location=device,weights_only=False);assert s['contract']==contract
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);sampler.load_state(s['sampler'])
        step=s['step'];history=s['history'];best_score=s['best_score'];prior=s['seconds'];torch.set_rng_state(s['rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']])
    start=time.monotonic();state.update(status='running',current_run=run,pid=os.getpid());write(OUT/'STATE.json',state)
    def snapshot():return dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),step=step,history=history,best_score=best_score,seconds=prior+time.monotonic()-start,contract=contract,kind=kind,fold=fold,seed=seed,mean=norm.mean,std=norm.std,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    torch.cuda.reset_peak_memory_stats()
    while step<300:
        if state['seconds']+prior+time.monotonic()-start>=CAP:torch.save(snapshot(),latest);raise TimeoutError('Frozen 4800s training cap')
        model.train();x,y,mask,valid,hy,hm=(v.to(device) for v in sampler.batch());opt.zero_grad(set_to_none=True)
        logit,hlogit=model(x,padding_mask=~valid.bool(),both=True)
        main=(crit(logit,y)*mask).sum()/mask.sum().clamp_min(1)
        aux=(nn.functional.cross_entropy(hlogit.transpose(1,2),hy,reduction='none')*hm).sum()/hm.sum().clamp_min(1)/np.log(7)
        loss=main+weight*aux;assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);opt.step();step+=1
        if step%50==0:
            raw=predictions(model,val,norm,device);threshold,_=choose_single_threshold(raw,val,GRID);_,_,score=metrics(raw,val,threshold)
            history.append(dict(step=step,loss=float(loss.detach()),main_bce=float(main.detach()),auxiliary_ce=float(aux.detach()),grad=float(gn),**score))
            if score['macro_f1_tol1']>best_score+1e-9:best_score=score['macro_f1_tol1'];torch.save(snapshot(),best)
            print(run,step,'F1',round(score['macro_f1_tol1'],4),'exact',round(score['macro_f1_tol0'],4),'AP',round(score['raw_ap'],4),'aux',round(float(aux.detach()),4),flush=True)
        if step%25==0:torch.save(snapshot(),latest)
    s=torch.load(best,map_location=device,weights_only=False);model.load_state_dict(s['model']);threshold=checkpoint_threshold(s);raw=predictions(model,val,norm,device)
    perfs,pieces,score=metrics(raw,val,threshold);elapsed=prior+time.monotonic()-start
    result=dict(run_id=run,kind=kind,fold=fold,seed=seed,contract=contract,best_step=int(s['step']),params=6152,seconds=elapsed,gpu_peak_bytes=int(torch.cuda.max_memory_allocated()),**score)
    perfs.to_csv(ART/'metrics'/f'{run}_performances.csv',index=False);pieces.to_csv(ART/'metrics'/f'{run}_pieces.csv',index=False)
    pd.DataFrame([(pid,perf,b,float(p),int(val[pid]['labels'][b]),int(val[pid]['label_mask'][b])) for pid,pp in raw.items() for perf,arr in pp.items() for b,p in enumerate(arr)],
        columns=['piece_id','performance_id','beat','probability','label','valid']).to_csv(ART/'metrics'/f'{run}_predictions.csv.gz',index=False)
    write(respath,result);state['completed'].append(run);state['seconds']+=elapsed;state.update(status='between_runs',pid=os.getpid());write(OUT/'STATE.json',state)


def audit():
    # Reuse the unchanged, already audited G/D/Z x raw/M10 replay matrix.
    audit_engine.OUT=OUT;audit_engine.ART=ART;audit_engine.dataset=dataset;audit_engine.make_model=make_model
    audit_engine.audit()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.stage=='audit':audit();return
    digest=prepare();preflight()
    if args.stage=='prepare':print(digest,read(OUT/'label_audit.json'),flush=True);return
    for f in (0,1):
        for s in (42,43):
            for k in ('G','D','Z'):
                guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
                train_one(k,f,s,digest)
    audit()


if __name__=='__main__':
    try:
        with threadpool_limits(2):main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',{**(read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else {}),'status':'failed','pid':None});raise
