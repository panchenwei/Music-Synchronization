"""Isolated, resumable fixed-matrix slice and tempo-hierarchy experiment."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .data import discover_dcml_pieces, load_piece_cache
from .models import Normalizer
from .phase2_models import fit_curve_normalizer, choose_single_threshold
from .phase3_models import fit_train_normalizer
from .phase6_models import positive_weight
from .phase7_models import CurvePieceBalancedSampler
from .local_context_study import make_model, predictions, metrics, sha, write
from .slice_energy_features import map_starts, voiced_events, corrected_score, tempo_hierarchy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/slice_energy_study'
ART = ROOT/'artifacts/slice_energy_study'
DCML = Path(r'C:\Users\pa1018\Desktop\learn\柴柴\TEST2\DCMLab-chopin_mazurkas-5127700')
KINDS = ['E25','S25','SR25','SF25','SE31','SFZ31','SFE31']
GRID = np.arange(.1,.91,.05).round(2).tolist()

def prepare():
    (ART/'cache').mkdir(parents=True,exist_ok=True)
    pieces = discover_dcml_pieces(DCML)
    audits=[]; mappings=[]; segments=[]; provenance=[]
    for path in sorted((ROOT/'cache/piece_features').glob('*.npz')):
        pid=path.stem; item=load_piece_cache(ROOT/'cache',pid); piece=pieces[pid]
        with np.load(ROOT/f'cache/phase3/piece_features/{pid}.npz',allow_pickle=False) as source:
            old=source['score_phase3'].copy()
        labels,mask,rows,segs,prov,mode,error=map_starts(piece,len(old))
        rest,fixed,voices=corrected_score(old,voiced_events(piece,len(old)))
        energy=[]; prior=[]; distribution=[]
        for curve in item['curves']:
            a,b,c=tempo_hierarchy(np.exp(curve[:,0].astype(float)),curve[:,7])
            energy.append(a);prior.append(b);distribution.append(c)
        payload={**item,'start_labels':labels,'start_mask':mask,'old_score':old,'rest_score':rest,
                 'fixed_score':fixed,'energy':np.stack(energy),'paper_prior':np.stack(prior),
                 'paper_distribution':np.stack(distribution)}
        np.savez_compressed(ART/'cache'/f'{pid}.npz',**payload)
        for row in rows: mappings.append({'piece_id':pid,**row})
        segments.extend(segs);provenance.extend(prov)
        audits.append({'piece_id':pid,'beats':len(old),'performances':len(item['curves']),
                       'old_ends':int(item['labels'].sum()),'start_markers':len(rows),
                       'interior_start_positives':int((labels*mask).sum()),'valid_start_beats':int(mask.sum()),
                       'ambiguous_starts':sum(r['status']=='ambiguous_tie' for r in rows),
                       'old_false_rest_beats':int(((old[:,4]>0)&(rest[:,4]==0)).sum()),
                       'rest_feature_changed':int((abs(old[:,4]-rest[:,4])>1e-6).sum()),
                       'voice_pitch_changed':int((abs(old[:,7]-fixed[:,7])>1e-6).sum()),
                       'mode':mode,'length_error':error,**voices})
    pd.DataFrame(audits).to_csv(OUT/'data_audit.csv',index=False)
    pd.DataFrame(mappings).to_csv(OUT/'start_mapping.csv',index=False)
    pd.DataFrame(segments).to_csv(OUT/'phrase_slices.csv',index=False)
    pd.DataFrame(provenance).to_csv(OUT/'start_provenance.csv',index=False)
    print('PREPARED',len(audits),'pieces',len(segments),'slices',flush=True)

def dataset(ids,kind):
    output={}
    for pid in ids:
        with np.load(ART/'cache'/f'{pid}.npz',allow_pickle=False) as f:
            item={k:f[k] for k in f.files}
        if kind!='E25': item['labels']=item['start_labels'];item['label_mask']=item['start_mask']
        score=item['fixed_score'] if kind.startswith('SF') else item['rest_score'] if kind=='SR25' else item['old_score']
        x=item['curves']; x=np.concatenate([x,np.broadcast_to(score,(*x.shape[:2],16))],axis=-1)
        if kind.endswith('31'):
            energy=np.zeros_like(item['energy']) if kind=='SFZ31' else item['energy']
            x=np.concatenate([x,energy],axis=-1)
        item['curves']=x;item['selected_score']=score
        assert (item['labels']*item['label_mask']).sum()>0
        assert np.isfinite(x).all()
        output[pid]=item
    return output

def normalization(data):
    curves=fit_curve_normalizer({p:{**i,'curves':i['curves'][...,:9]} for p,i in data.items()})
    score=fit_train_normalizer([i['selected_score'] for i in data.values()])
    mean=np.r_[curves.mean,score.mean];std=np.r_[curves.std,score.std]
    if next(iter(data.values()))['curves'].shape[-1]>25:
        energy=fit_train_normalizer([i['curves'][...,25:].reshape(-1,6) for i in data.values()])
        mean=np.r_[mean,energy.mean];std=np.r_[std,energy.std]
    return Normalizer(mean,std)

def model_for(dim,seed):
    model=make_model('C25',seed)
    if dim==31:
        old=model.input_projection
        with torch.random.fork_rng(devices=[]): expanded=nn.Linear(31,32)
        with torch.no_grad():
            expanded.weight.zero_();expanded.weight[:,:25].copy_(old.weight);expanded.bias.copy_(old.bias)
        model.input_projection=expanded
    return model

def run_one(kind,fold,seed,train,val,contract,state):
    run=f'{kind}_seed{seed}_fold{fold}';dest=ART/'checkpoints'/run;dest.mkdir(parents=True,exist_ok=True)
    metric_path=ART/'metrics'/f'{run}.json'
    if metric_path.exists():
        result=json.loads(metric_path.read_text(encoding='utf-8'))
        assert result['contract']==contract
        print('CACHED',run,flush=True);return result
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim=next(iter(train.values()))['curves'].shape[-1]
    model=model_for(dim,seed).to(device);norm=normalization(train)
    sampler=CurvePieceBalancedSampler(train,norm,64,32,seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    criterion=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(train,10),device=device))
    history=[];step=0;best=-1.;past=0.;start=time.monotonic()
    latest=dest/'latest.pt'
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=False);assert saved['contract']==contract
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);sampler.load_state(saved['sampler'])
        step=saved['step'];history=saved['history'];best=saved['best'];past=saved['seconds']
        torch.set_rng_state(saved['rng'].cpu())
        if torch.cuda.is_available():torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
    def snapshot():
        return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'sampler':sampler.state(),
                'step':step,'history':history,'best':best,'contract':contract,'seconds':past+time.monotonic()-start,
                'mean':norm.mean,'std':norm.std,'rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    while step<300:
        if time.time()>state['deadline_unix']-60 or state['seconds']+past+time.monotonic()-start>2700:
            torch.save(snapshot(),latest);raise TimeoutError('Fixed resource budget reached')
        model.train()
        x,y,mask,valid=(a.to(device) for a in sampler.batch())
        optimizer.zero_grad(set_to_none=True)
        loss=(criterion(model(x,padding_mask=~valid.bool()),y)*mask).sum()/mask.sum().clamp_min(1)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
        optimizer.step();step+=1
        if step%50==0:
            raw=predictions(model,val,norm,device);threshold,_=choose_single_threshold(raw,val,GRID)
            _,_,score=metrics(raw,val,threshold)
            history.append({'step':step,'loss':float(loss),'threshold':threshold,**score})
            if score['macro_f1_tol1']>best+1e-9:
                best=score['macro_f1_tol1'];torch.save(snapshot(),dest/'best.pt')
            assert model.training
            print(run,step,round(score['macro_f1_tol1'],4),flush=True)
        if step%25==0:torch.save(snapshot(),latest)
    saved=torch.load(dest/'best.pt',map_location=device,weights_only=False);model.load_state_dict(saved['model'])
    threshold=saved['history'][-1]['threshold'];raw=predictions(model,val,norm,device)
    perfs,pieces,score=metrics(raw,val,threshold)
    _,_,train_score=metrics(predictions(model,train,norm,device),train,threshold)
    seconds=past+time.monotonic()-start
    result={'kind':kind,'fold':fold,'seed':seed,'contract':contract,'threshold':threshold,'best_step':saved['step'],
            'params':sum(p.numel() for p in model.parameters()),'seconds':seconds,
            'gpu_peak_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            'train_f1':train_score['macro_f1_tol1'],'train_gap':train_score['macro_f1_tol1']-score['macro_f1_tol1'],
            'history':history,**score}
    perfs.to_csv(ART/'metrics'/f'{run}_performances.csv',index=False)
    pieces.to_csv(ART/'metrics'/f'{run}_pieces.csv',index=False)
    predrows=[]
    for pid,performances in raw.items():
        for perf,prob in performances.items():
            for b in range(len(prob)):
                predrows.append((pid,perf,b,float(prob[b]),int(val[pid]['labels'][b]),int(val[pid]['label_mask'][b])))
    pd.DataFrame(predrows,columns=['piece_id','performance_id','beat','probability','label','valid']).to_csv(ART/'metrics'/f'{run}_predictions.csv.gz',index=False)
    write(metric_path,result);state['seconds']+=seconds;state['completed'].append(run);write(OUT/'STATE.json',state)
    return result

def paper_baseline(val,fold,target):
    raw={pid:{str(perf):score for perf,score in zip(item['performance_ids'],item['paper_prior'])} for pid,item in val.items()}
    threshold,_=choose_single_threshold(raw,val,GRID)
    perfs,pieces,score=metrics(raw,val,threshold)
    query=[]
    for pid,item in val.items():
        valid=item['label_mask'].astype(bool);positive=(item['labels']>.5)&valid
        for p in item['paper_distribution']:
            # Re-normalize over the same known interior region for this audit.
            p=p/max(p[valid].sum(),1e-12)
            query.append({'piece_id':pid,'log_query_above_uniform':float(np.log(np.maximum(p[positive],1e-30)).mean()+np.log(valid.sum()))})
    score.update({'fold':fold,'target':target,'threshold':threshold,
                  'mean_log_query_above_uniform':float(pd.DataFrame(query).groupby('piece_id').mean().mean().iloc[0])})
    pieces.to_csv(ART/'metrics'/f'paper_{target}_fold{fold}_pieces.csv',index=False)
    return score

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(ART/'metrics').mkdir(parents=True,exist_ok=True)
    if args.prepare_only:prepare();return
    if not (OUT/'data_audit.csv').exists():prepare()
    splits_path=ROOT/'artifacts/phase2/splits/opus_split_manifest.csv';splits=pd.read_csv(splits_path)
    tracked=[Path(__file__),ROOT/'src/slice_energy_features.py',OUT/'PROTOCOL.md',splits_path]
    tracked+=sorted((ART/'cache').glob('*.npz'))
    tracked += [ROOT/'src'/s for s in ['local_context_study.py','data.py','models.py','phase2_models.py','phase3_models.py','phase6_models.py','phase7_models.py','evaluation.py']]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in tracked}
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    statepath=OUT/'STATE.json'
    if statepath.exists():
        if not args.resume:raise ValueError('Existing study requires --resume')
        state=json.loads(statepath.read_text(encoding='utf-8'));assert state['contract']==contract
    else:
        state={'started_unix':time.time(),'deadline_unix':time.time()+5400,'seconds':0.,'completed':[],
               'status':'running','contract':contract,'outer_test_evaluated':False,'usage_start_percent':47}
        write(statepath,state);write(OUT/'input_contract.json',hashes)
        # Original raw inputs are fingerprinted separately, not copied or altered.
        sourcepaths=[DCML/'metadata.tsv']+list((DCML/'notes').glob('*.tsv'))+list((DCML/'harmonies').glob('*.tsv'))+list((DCML/'measures').glob('*.tsv'))
        write(OUT/'source_hashes.json',{str(p):sha(p) for p in sourcepaths})
    results=[];split_audit=[];paper=[]
    for fold in [0,1]:
        part=splits[splits.fold==fold]
        sets={s:set(part[part.split==s].piece_id) for s in ['train','validation','test']}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            overlap=len(sets[a]&sets[b]);opus_overlap=len(set(part[part.split==a].opus)&set(part[part.split==b].opus))
            assert overlap==opus_overlap==0
            split_audit.append({'fold':fold,'a':a,'b':b,'piece_overlap':overlap,'opus_overlap':opus_overlap})
        for kind in KINDS:
            train=dataset(sorted(sets['train']),kind);val=dataset(sorted(sets['validation']),kind)
            for seed in [42,43]:
                result=run_one(kind,fold,seed,train,val,contract,state);results.append(result)
                pd.DataFrame([{k:v for k,v in r.items() if k!='history'} for r in results]).to_csv(ART/'summary.csv',index=False)
            if kind in ['E25','S25']:paper.append(paper_baseline(val,fold,kind))
            del train,val
    pd.DataFrame(split_audit).to_csv(OUT/'split_audit.csv',index=False)
    pd.DataFrame(paper).to_csv(OUT/'paper_baseline.csv',index=False)
    assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in tracked}
    state.update(status='complete',completed_unix=time.time(),input_hashes_unchanged=True)
    write(statepath,state);print('COMPLETE',round(state['seconds'],2),flush=True)

if __name__=='__main__':main()
