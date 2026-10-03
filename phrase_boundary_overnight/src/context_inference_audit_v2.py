"""Frozen-weight length/position audit, with no modifications to historical studies."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import pure_transformer_reference as ref
from . import run_recurrence_depth_study as base
from . import run_halo_decoder_composition as dec
from .score_context_study import ROOT, read, write, sha, normalizer
from .three_round_round2 import split_ids
from .local_context_study import metrics, predictions
from .audit_external_stem_transfer import checked_raw
from .interstart_decoder import fit_prior

OUT=ROOT/'reports/context_inference_audit_v2'
ART=ROOT/'artifacts/context_inference_audit_v2'
COLS=['macro_f1_tol1','macro_f1_tol0','macro_precision_tol1','macro_recall_tol1','raw_ap']

def segments(length, window=64, core=32):
    if length<1:raise ValueError('Empty timeline')
    for begin in range(0,length,core):
        end=min(begin+core,length)
        start=max(0,min(begin-(window-core)//2,length-window))
        yield start,min(start+window,length),begin,end

def offset_logits(model,x,offset):
    """Same pure-Transformer forward, but absolute position starts at offset."""
    mask=torch.zeros(x.shape[:2],dtype=torch.bool,device=x.device)
    h=model.input_projection(x)
    assert model.frontend is None and model.kind=='A'
    h=h+model.position.encoding[:,offset:offset+x.shape[1]].to(h.dtype)
    for block in model.blocks:h=block(h,mask)
    return model.output(model.final_norm(h)).squeeze(-1)

def window_logits(model,x,global_positions=False):
    result=torch.empty(x.shape[:2],device=x.device,dtype=x.dtype)
    for lo,hi,a,b in segments(x.shape[1]):
        y=offset_logits(model,x[:,lo:hi],lo) if global_positions else model(x[:,lo:hi])
        result[:,a:b]=y[:,a-lo:b-lo]
    return result

def window_predictions(model,data,norm,device,global_positions=False):
    model.eval();result={}
    with torch.no_grad():
        for pid,item in sorted(data.items()):
            values=[]
            for a in range(0,len(item['curves']),8):
                x=torch.from_numpy(norm.apply(item['curves'][a:a+8]).astype(np.float32)).to(device)
                values.extend(torch.sigmoid(window_logits(model,x,global_positions)).cpu().numpy())
            result[pid]={str(k):v for k,v in zip(item['performance_ids'],values)}
    return result

def error(a,b):
    assert set(a)==set(b)
    return max(float(abs(a[p][q]-b[p][q]).max()) for p in a for q in a[p])

def save_raw(raw,path,data):
    rows=[(p,q,i,float(v),float(data[p]['labels'][i]),float(data[p]['label_mask'][i])) for p,qs in raw.items() for q,vs in qs.items() for i,v in enumerate(vs)]
    # checked_raw expects the historical columns below.
    pd.DataFrame(rows,columns=['piece_id','performance_id','beat','probability','label','valid']).to_csv(path,index=False)

def main():
    start=time.monotonic();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    for p in (OUT,ART,ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(ref.OUT/'contract.json')['hashes'])
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_context_inference_audit.py',ROOT/'src/run_halo_decoder_composition.py'):
        hashes[str(p)]=sha(p)
    for source in (ref.ART,base.ART):
        kind='T' if source==ref.ART else 'C3'
        for f in (0,1):
            for s in (42,43):
                run=f'{kind}_seed{s}_fold{f}'
                for p in (source/'checkpoints'/run/'best.pt',source/'metrics'/f'{run}.json',source/'metrics'/f'{run}_predictions.csv.gz'):
                    hashes[str(p)]=sha(p)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    if (OUT/'completion_audit.json').exists():
        assert read(OUT/'completion_audit.json')['status']=='complete'
        print('ALREADY COMPLETE; hashes checked');return
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    result=[];audit=[]
    for f in (0,1):
        ids=split_ids(f);train=base.dataset(ids['train'],'C3');val=base.dataset(ids['validation'],'C3')
        norm=normalizer(train);prior=fit_prior(train)
        for s in (42,43):
            for kind,source,factory in [('T',ref.ART,ref.make_model),('C3',base.ART,base.make_model)]:
                assert time.monotonic()-start<1800
                guard=read(ROOT/'reports/research_resource_guard.json')
                assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
                run=f'{kind}_seed{s}_fold{f}';meta=read(source/'metrics'/f'{run}.json')
                cp=torch.load(source/'checkpoints'/run/'best.pt',map_location='cpu',weights_only=False)
                np.testing.assert_array_equal(norm.mean,cp['mean']);np.testing.assert_array_equal(norm.std,cp['std'])
                m=factory(kind,s).cuda().eval();m.load_state_dict(cp['model'])
                frozen=checked_raw(pd.read_csv(source/'metrics'/f'{run}_predictions.csv.gz'),val)
                full=predictions(m,val,norm,torch.device('cuda'))
                err=error(full,frozen);assert err<2e-4
                assert max(abs(metrics(full,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
                modes=['FULL','W64_LOCAL']+(['W64_GLOBAL'] if kind=='T' else [])
                for mode in modes:
                    raw=full if mode=='FULL' else window_predictions(m,val,norm,torch.device('cuda'),mode=='W64_GLOBAL')
                    dist=error(full,raw)
                    if kind=='C3' and mode!='FULL':
                        old_tf32=torch.backends.cudnn.allow_tf32
                        try:
                            torch.backends.cudnn.allow_tf32=False
                            strict_full=predictions(m,val,norm,torch.device('cuda'))
                            strict_window=window_predictions(m,val,norm,torch.device('cuda'))
                            strict_error=error(strict_full,strict_window)
                            assert strict_error<2e-5,strict_error
                        finally:torch.backends.cudnn.allow_tf32=old_tf32
                        write(OUT/f'{run}_precision.json',dict(default_error=dist,fp32_error=strict_error))
                        # The strict FP32 check, not a widened default tolerance, validates stitching.
                    name=f'{run}_{mode}'
                    ppath=ART/f'{name}_predictions.csv.gz';save_raw(raw,ppath,val)
                    reread=checked_raw(pd.read_csv(ppath),val);assert error(raw,reread)<1e-7
                    for strength in (0.,.5):
                        r=dec.evaluate(reread,val,meta['threshold'],prior,strength,f'{name}_lambda{strength:g}',digest)
                        result.append(dict(kind=kind,mode=mode,fold=f,seed=s,**r))
                    audit.append(dict(run=run,mode=mode,original_replay_error=err,change_from_full=dist,probabilities_sha256=sha(ppath)))
                    pd.DataFrame(result).to_csv(ART/'summary.csv',index=False)
                    pd.DataFrame(audit).to_csv(OUT/'run_audit.csv',index=False)
                    print('DONE',name,'max_change',dist,flush=True)
    frame=pd.DataFrame(result);assert len(frame)==40
    means=frame.groupby(['kind','mode','strength'])[COLS].mean();means.to_csv(OUT/'means.csv')
    comparison=[]
    for kind in ('T','C3'):
        for a,b in ([('W64_LOCAL','FULL'),('W64_GLOBAL','FULL'),('W64_LOCAL','W64_GLOBAL')] if kind=='T' else [('W64_LOCAL','FULL')]):
            for strength in (0.,.5):
                left=frame[(frame.kind==kind)&(frame['mode']==a)&(frame.strength==strength)].set_index(['fold','seed'])
                right=frame[(frame.kind==kind)&(frame['mode']==b)&(frame.strength==strength)].set_index(['fold','seed'])
                delta=left[COLS]-right[COLS]
                delta.to_csv(OUT/f'{kind}_{a}_minus_{b}_lambda{strength:g}.csv')
                comparison.append(dict(kind=kind,candidate=a,control=b,strength=strength,**delta.mean().to_dict(),positive=int((delta.macro_f1_tol1>0).sum())))
    write(OUT/'comparisons.json',comparison)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=0,old_full_replays=8,inference_conditions=20,decoder_cells=40,hashes_unchanged=True,norms_rebuilt=True,saved_probabilities_reloaded=True,c3_window_invariant=True,test_used=False,seconds=time.monotonic()-start))
    write(OUT/'STATE.json',dict(status='complete',pid=None));write(dec.OUT/'STATE.json',dict(status='complete',pid=None))
    print(means.to_string(),flush=True)

if __name__=='__main__':main()


