"""Bounded full-work note-graph versus node-self control, with independent audit."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .note_relation_graph import build_graph,tensors,NoteRelationBoundary
from .label_repaired_rebaseline import dataset as base_data
from .phrase_end_auxiliary import ROOT,read,write,sha,normalizer,split_ids
from .score_local_coordinates import local_events
from .data import discover_dcml_pieces
from .slice_energy_study import DCML
from .phase6_models import positive_weight
from .phase2_models import choose_single_threshold
from .local_context_study import metrics
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/note_relation_study';ART=ROOT/'artifacts/note_relation_study'
GRID=np.arange(.1,.91,.05).round(2).tolist();COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']


def dataset(ids):
    data=base_data(ids,'B')
    for pid,v in data.items():
        with np.load(ART/'cache'/f'{pid}.npz',allow_pickle=False) as z:v['graph']={k:z[k].copy() for k in z.files}
        assert int(v['graph']['n_beats'])==len(v['labels'])
    return data


def prepare():
    for p in (OUT,ART/'cache',ART/'metrics',ART/'checkpoints'):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/note_input_correction/completion_audit.json')['status']=='complete'
    hashes=dict(read(ROOT/'reports/label_repaired_rebaseline/contract.json')['hashes']);assert all(sha(p)==h for p,h in hashes.items())
    pieces=discover_dcml_pieces(DCML);rows=[];began=time.monotonic()
    for source in sorted((ROOT/'artifacts/slice_energy_study/cache').glob('*.npz')):
        assert time.monotonic()-began<300
        pid=source.stem;piece=pieces[pid]
        with np.load(source,allow_pickle=False) as z:bn=z['beat_number'].copy()
        events,ties=local_events(pd.read_csv(piece.notes_path,sep='\t'),pd.read_csv(piece.measures_path,sep='\t'),len(bn))
        g=build_graph(events,ties,bn);dest=ART/'cache'/f'{pid}.npz'
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:
                assert set(z.files)==set(g)
                for k,v in g.items():np.testing.assert_array_equal(z[k],v)
        else:np.savez_compressed(dest,**g)
        rows.append(dict(piece_id=pid,notes=len(g['node_features']),edges=g['edge_index'].shape[1],beats=len(bn),merged_ties=int(g['merged_ties'])))
        for p in (source,piece.notes_path,piece.measures_path,dest):hashes[str(p)]=sha(p)
    counts=pd.DataFrame(rows);assert len(counts)==43;counts.to_csv(OUT/'graph_counts.csv',index=False)
    for p in (Path(__file__),ROOT/'src/note_relation_graph.py',ROOT/'src/resolve_tie_audit.py',ROOT/'tests/test_note_relation_graph.py',OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes))
    done=[read(p) for p in (ART/'metrics').glob('*_fold*.json')];assert all(r['contract']==contract for r in done)
    write(OUT/'STATE.json',dict(status='ready',contract=contract,completed=[r['run_id'] for r in done],seconds=sum(r['seconds'] for r in done),pid=None))
    return contract


def predictions(model,data,norm):
    was=model.training;model.eval();out={}
    try:
        with torch.no_grad():
            for pid,v in sorted(data.items()):
                g=tensors(v['graph'],'cuda');features=model.encode_beats(g)[None];rows=[]
                for first in range(0,len(v['curves']),8):
                    x=torch.from_numpy(norm.apply(v['curves'][first:first+8]).astype(np.float32)).cuda()
                    rows.extend(torch.sigmoid(model.forward_embedded(x,features)).cpu().numpy())
                out[pid]={str(k):v for k,v in zip(v['performance_ids'],rows)}
    finally:model.train(was)
    return out


def train_one(kind,fold,seed,contract):
    run=f'{kind}_seed{seed}_fold{fold}';rp=ART/'metrics'/f'{run}.json'
    if rp.exists():assert read(rp)['contract']==contract;print('CACHED',run,flush=True);return
    cp=ART/'checkpoints'/run;cp.mkdir(exist_ok=True);ids=split_ids(fold);assert not set(ids['train'])&set(ids['validation'])
    train=dataset(ids['train']);val=dataset(ids['validation']);norm=normalizer(train)
    model=NoteRelationBoundary(kind=='G',seed).cuda();rng=np.random.default_rng(seed);names=sorted(train)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    criterion=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(train,10),device='cuda'))
    step=0;history=[];best_score=-1.;prior=0.
    if (cp/'latest.pt').exists():
        c=torch.load(cp/'latest.pt',map_location='cuda',weights_only=False);assert c['contract']==contract
        model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);rng.bit_generator.state=c['numpy_rng'];step=c['step'];history=c['history'];best_score=c['best_score'];prior=c['seconds'];torch.set_rng_state(c['rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in c['cuda_rng']])
    state=read(OUT/'STATE.json');state.update(status='running',current_run=run,pid=os.getpid());write(OUT/'STATE.json',state);began=time.monotonic()
    def snapshot():return dict(model=model.state_dict(),optimizer=opt.state_dict(),numpy_rng=rng.bit_generator.state,step=step,history=history,best_score=best_score,seconds=prior+time.monotonic()-began,contract=contract,kind=kind,fold=fold,seed=seed,mean=norm.mean,std=norm.std,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    torch.cuda.reset_peak_memory_stats()
    while step<300:
        elapsed=prior+time.monotonic()-began
        if elapsed>=300 or state['seconds']+elapsed>=2400:torch.save(snapshot(),cp/'latest.pt');raise TimeoutError('Frozen work-graph budget exceeded')
        model.train();v=train[str(rng.choice(names))];inds=rng.integers(len(v['curves']),size=4)
        x=torch.from_numpy(norm.apply(v['curves'][inds]).astype(np.float32)).cuda();g=tensors(v['graph'],'cuda')
        y=torch.from_numpy(v['labels']).cuda()[None].expand(4,-1);mask=torch.from_numpy(v['label_mask']).cuda()[None].expand(4,-1)
        opt.zero_grad(set_to_none=True);logits=model(x,g);loss=(criterion(logits,y)*mask).sum()/mask.sum().clamp_min(1)
        assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(gn);opt.step();step+=1
        if step%50==0:
            raw=predictions(model,val,norm);threshold,_=choose_single_threshold(raw,val,GRID);_,_,score=metrics(raw,val,threshold);history.append(dict(step=step,loss=float(loss.detach()),grad=float(gn),**score))
            if score['macro_f1_tol1']>best_score+1e-9:best_score=score['macro_f1_tol1'];torch.save(snapshot(),cp/'best.pt')
            print(run,step,'F1',round(score['macro_f1_tol1'],4),'exact',round(score['macro_f1_tol0'],4),'AP',round(score['raw_ap'],4),flush=True)
        if step%25==0:torch.save(snapshot(),cp/'latest.pt')
    best=torch.load(cp/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(best['model']);threshold=max(best['history'],key=lambda h:h['macro_f1_tol1'])['threshold']
    raw=predictions(model,val,norm);perfs,pieces,score=metrics(raw,val,threshold)
    frame=pd.DataFrame([(p,k,b,float(prob),int(val[p]['labels'][b]),int(val[p]['label_mask'][b])) for p,pp in raw.items() for k,arr in pp.items() for b,prob in enumerate(arr)],columns=['piece_id','performance_id','beat','probability','label','valid'])
    frame.to_csv(ART/'metrics'/f'{run}_predictions.csv.gz',index=False);perfs.to_csv(ART/'metrics'/f'{run}_performances.csv',index=False);pieces.to_csv(ART/'metrics'/f'{run}_pieces.csv',index=False)
    elapsed=prior+time.monotonic()-began;write(rp,dict(run_id=run,kind=kind,fold=fold,seed=seed,contract=contract,best_step=best['step'],params=sum(p.numel() for p in model.parameters()),seconds=elapsed,gpu_peak_bytes=int(torch.cuda.max_memory_allocated()),history=history,**score))
    state['completed'].append(run);state['seconds']+=elapsed;state.update(status='between_runs');write(OUT/'STATE.json',state)


def probe():
    began=time.monotonic();pid=split_ids(0)['train'][0];data=dataset([pid]);v=data[pid];norm=normalizer(data);model=NoteRelationBoundary(True,42).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001)
    x=torch.from_numpy(norm.apply(v['curves'][:1]).astype(np.float32)).cuda();g=tensors(v['graph'],'cuda');y=torch.from_numpy(v['labels'][None]).cuda();mask=torch.from_numpy(v['label_mask'][None]).cuda();losses=[]
    for _ in range(20):
        model.train();opt.zero_grad(set_to_none=True);loss=(nn.functional.binary_cross_entropy_with_logits(model(x,g),y,reduction='none')*mask).sum()/mask.sum().clamp_min(1);assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);opt.step();losses.append(float(loss.detach()))
    write(OUT/'numerical_probe.json',dict(status='complete',training_piece=pid,steps=20,losses=losses,seconds=time.monotonic()-began,params=sum(p.numel() for p in model.parameters()),formal_checkpoint_saved=False,validation_accessed=False,scope='Finite optimization smoke, NOT tiny-overfit proof'))


def audit_report():
    began=time.monotonic();contract=read(OUT/'contract.json');assert all(sha(p)==h for p,h in contract['hashes'].items());cps=list((ART/'checkpoints').glob('*/*.pt'));assert len(cps)==16;before={str(p):sha(p) for p in cps};rows=[];audits=[]
    for p in sorted((ART/'metrics').glob('*_fold*.json')):
        r=read(p);ids=split_ids(r['fold']);train=dataset(ids['train']);val=dataset(ids['validation']);norm=normalizer(train);cp=ART/'checkpoints'/r['run_id']
        best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False);last=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False);assert last['step']==300 and best['contract']==last['contract']==contract['contract'];assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
        np.testing.assert_array_equal(best['mean'],norm.mean);np.testing.assert_array_equal(best['std'],norm.std)
        raw=checked_raw(pd.read_csv(ART/'metrics'/f"{r['run_id']}_predictions.csv.gz"),val);score=metrics(raw,val,r['threshold'])[2];assert max(abs(score[k]-r[k]) for k in COLS)<1e-10
        model=NoteRelationBoundary(r['kind']=='G',r['seed']).cuda();model.load_state_dict(best['model']);replay=predictions(model,val,norm);delta=max(float(np.max(abs(replay[p][k]-raw[p][k]))) for p in raw for k in raw[p]);assert delta<2e-4
        assert max(abs(metrics(replay,val,r['threshold'])[2][k]-score[k]) for k in COLS)<1e-10
        ts=metrics(predictions(model,train,norm),train,r['threshold'])[2];rows.append({k:v for k,v in r.items() if k!='history'});audits.append(dict(run_id=r['run_id'],gpu_replay_error=delta,train_f1=ts['macro_f1_tol1'],gap=ts['macro_f1_tol1']-r['macro_f1_tol1']));pd.DataFrame(audits).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',r['run_id'],flush=True)
    df=pd.DataFrame(rows);assert len(df)==8 and set(zip(df.kind,df.fold,df.seed))=={(k,f,s) for k in ('S','G') for f in (0,1) for s in (42,43)};df.to_csv(ART/'summary.csv',index=False);means=df.groupby('kind')[COLS+['macro_precision_tol1','macro_recall_tol1']].mean();means.to_csv(OUT/'means.csv')
    a=df[df.kind=='G'].set_index(['fold','seed']);b=df[df.kind=='S'].set_index(['fold','seed']);d=a[COLS]-b[COLS];d.to_csv(OUT/'paired_deltas.csv')
    write(OUT/'comparison.json',dict(mean_delta=d.mean().to_dict(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),promotion=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)))
    assert before=={str(p):sha(p) for p in cps} and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'completion_audit.json',dict(status='complete',runs=8,checkpoints=16,full_gpu_replays=8,norms_rebuilt=True,hashes_unchanged=True,test_predictions_accessed=False,training_validation_seconds=float(df.seconds.sum()),audit_seconds=time.monotonic()-began));write(OUT/'STATE.json',dict(status='complete',contract=contract['contract'],pid=None));print(means.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=p.parse_args();torch.set_num_threads(2)
    try:
        if args.stage=='audit':audit_report();return
        contract=prepare()
        if args.stage=='prepare':probe();print(contract);return
        if not (OUT/'numerical_probe.json').exists():probe()
        for fold in (0,1):
            for seed in (42,43):
                for kind in ('S','G'):train_one(kind,fold,seed,contract)
        write(OUT/'STATE.json',{**read(OUT/'STATE.json'),'status':'auditing','pid':os.getpid()});audit_report()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise


if __name__=='__main__':main()
