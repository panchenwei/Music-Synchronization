"""Independent, bounded local-context experiment. Never writes Phase1--7 artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score
from torch import nn
from .models import Normalizer, seed_everything
from .phase2_models import fit_curve_normalizer, choose_single_threshold, evaluate_single_performance, nms_probabilities
from .phase3_models import fit_train_normalizer
from .phase6_models import positive_weight
from .phase7_models import Phase7BoundaryModel, CurvePieceBalancedSampler, load_phase7_checkpoint
from .data import load_piece_cache

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/local_context_study'
ART = ROOT / 'artifacts/local_context_study'
KINDS = ['A9', 'B9', 'C9', 'A25', 'B25', 'C25', 'M25', 'L25']

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class MaskedConv(nn.Module):
    def __init__(self, kernels=(5,), channels=32, dropout=.2):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.convs = nn.ModuleList([nn.Conv1d(channels, channels, k, padding=k//2, groups=channels) for k in kernels])
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, h, padding_mask=None):
        mask = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device) if padding_mask is None else padding_mask
        h = h.masked_fill(mask[...,None], 0)
        z = self.norm(h).masked_fill(mask[...,None], 0).transpose(1,2)
        z = sum(conv(z) for conv in self.convs) / len(self.convs)
        z = self.pointwise(torch.nn.functional.gelu(z)).transpose(1,2)
        return (h + self.dropout(z)).masked_fill(mask[...,None], 0)

class RecurrentFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(32)
        self.rnn = nn.LSTM(32, 16, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(.2)
    def forward(self, h, padding_mask=None):
        mask = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device) if padding_mask is None else padding_mask
        lengths = (~mask).sum(1)
        expected = torch.arange(h.shape[1], device=h.device)[None,:] >= lengths[:,None]
        if (lengths == 0).any() or not torch.equal(mask, expected):
            raise ValueError('LSTM requires nonempty sequences with right padding only')
        h = h.masked_fill(mask[...,None], 0)
        z = self.norm(h).masked_fill(mask[...,None], 0)
        packed = nn.utils.rnn.pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.rnn(packed)
        z, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=h.shape[1])
        return (h + self.dropout(z)).masked_fill(mask[...,None], 0)

def make_model(kind, seed=42):
    seed_everything(seed)
    dim = 9 if kind.endswith('9') else 25
    base = kind[0] if kind[0] in 'ABC' else 'B'
    model = Phase7BoundaryModel(base, input_dim=dim)
    if base in 'BC':
        old = model.frontend
        if kind[0] == 'M':
            new = MaskedConv((3,9,21))
        else:
            # Copying the existing weights must not advance the dropout RNG.
            with torch.random.fork_rng(devices=[]):
                new = MaskedConv((5,))
        if kind[0] != 'M':
            new.norm.load_state_dict(old.norm.state_dict())
            new.convs[0].load_state_dict(old.depthwise.state_dict())
            new.pointwise.load_state_dict(old.pointwise.state_dict())
        model.frontend = new
    if kind[0] == 'L':
        model.frontend = RecurrentFrontend()
    return model

def data_for(ids, dim):
    result = {}
    for pid in ids:
        item = load_piece_cache(ROOT / 'cache', pid)
        if not len(item['curves']):
            raise ValueError(f'No curves: {pid}')
        if dim == 25:
            with np.load(ROOT / f'cache/phase3/piece_features/{pid}.npz', allow_pickle=False) as source:
                for key in ['labels', 'label_mask', 'measure_number', 'beat_number']:
                    np.testing.assert_array_equal(item[key], source[key])
                score = source['score_phase3'].copy()
            assert score.shape == (len(item['labels']), 16)
            assert np.isfinite(score).all()
            item['score_local_study'] = score
            item['curves'] = np.concatenate([item['curves'], np.broadcast_to(score, (*item['curves'].shape[:2],16))], axis=-1)
        assert item['curves'].shape[-1] == dim
        assert np.isfinite(item['curves']).all()
        result[pid] = item
    return result

def normalizer_for(data, dim):
    curves = {p:{**item,'curves':item['curves'][...,:9]} for p,item in data.items()}
    c = fit_curve_normalizer(curves)
    if dim == 9:
        return c
    s = fit_train_normalizer([x['score_local_study'] for x in data.values()])
    return Normalizer(np.concatenate([c.mean,s.mean]), np.concatenate([c.std,s.std]))

def predictions(model, data, norm, device):
    was_training = model.training
    model.eval()
    result = {}
    try:
        with torch.no_grad():
            for pid,item in sorted(data.items()):
                probs = []
                for first in range(0,len(item['curves']),8):
                    x = torch.from_numpy(norm.apply(item['curves'][first:first+8]).astype(np.float32)).to(device)
                    probs.extend(torch.sigmoid(model(x)).cpu().numpy())
                result[pid] = {str(k):v for k,v in zip(item['performance_ids'], probs)}
    finally:
        model.train(was_training)
    return result

def max_match_count(pred, truth, tolerance):
    p,t = sorted(set(pred)),sorted(set(truth))
    i=j=tp=0
    while i<len(p) and j<len(t):
        if p[i] < t[j]-tolerance: i+=1
        elif t[j] < p[i]-tolerance: j+=1
        else: tp+=1; i+=1; j+=1
    return tp

def metrics(raw, data, threshold):
    perf, pieces, summary = evaluate_single_performance(raw, data, threshold)
    rows = []
    for pid, perfs in raw.items():
        item=data[pid]; valid=item['label_mask'].astype(bool); y=item['labels'][valid]
        truth=np.flatnonzero((item['labels']>.5)&valid)
        for perfid, probs in perfs.items():
            pred=np.flatnonzero((nms_probabilities(probs)>=threshold)&valid)
            tp=max_match_count(pred,truth,1)
            rows.append({'piece_id':pid,'performance_id':perfid,
                         'raw_ap':float(average_precision_score(y,probs[valid])) if y.sum()>0 else float('nan'),
                         'maxmatch_f1_tol1':2*tp/max(len(pred)+len(truth),1)})
    extra=pd.DataFrame(rows)
    perf=perf.merge(extra,on=['piece_id','performance_id'])
    mean=extra.groupby('piece_id')[['raw_ap','maxmatch_f1_tol1']].mean().mean()
    summary.update({'raw_ap':float(mean.raw_ap),'maxmatch_f1_tol1':float(mean.maxmatch_f1_tol1),
                    'legacy_nms_ap':summary['macro_pr_auc'],
                    'maxmatch_delta':float(mean.maxmatch_f1_tol1-summary['macro_f1_tol1'])})
    return perf,pieces,summary

def probe_groups(model,data,norm,threshold,device,baseline,kind):
    groups={'tempo':[0,1,2,5],'dynamics':[3,4,6]}
    if kind.endswith('25'):
        groups.update(score=list(range(9,25)), rest_duration=[12,13,14,15], pitch_harmony=list(range(16,25)))
    rows=[]
    for name,indices in groups.items():
        changed={}
        for pid,item in data.items():
            x=item['curves'].copy(); x[...,indices]=norm.mean[indices]
            changed[pid]={**item,'curves':x}
        _,_,score=metrics(predictions(model,changed,norm,device),changed,threshold)
        rows.append({'group':name,'f1_delta':score['macro_f1_tol1']-baseline['macro_f1_tol1'],
                     'raw_ap_delta':score['raw_ap']-baseline['raw_ap']})
    return rows

def run_one(kind, fold, train, val, cfg, contract, state):
    run=f'{kind}_seed42_fold{fold}'; directory=ART/'checkpoints'/run
    directory.mkdir(parents=True,exist_ok=True)
    result_path=ART/'metrics'/f'{run}.json'
    if result_path.exists():
        result=json.loads(result_path.read_text(encoding='utf-8'))
        if result['contract']!=contract: raise ValueError('Completed run contract changed')
        print('[cached]',run,flush=True)
        return result
    start=time.monotonic(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=make_model(kind).to(device); norm=normalizer_for(train,train[next(iter(train))]['curves'].shape[-1])
    sampler=CurvePieceBalancedSampler(train,norm,64,32,42)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    loss_fn=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(train,10),device=device))
    step=0; best=-1.; history=[]; prior_seconds=0.
    latest=directory/'latest.pt'
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=False)
        if saved['contract']!=contract: raise ValueError('Resume contract changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        sampler.load_state(saved['sampler']); step=saved['step']; best=saved['best']; history=saved['history']
        torch.set_rng_state(saved['rng'].cpu())
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
        prior_seconds=saved['elapsed_seconds']
    def snapshot():
        return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'sampler':sampler.state(),
                'step':step,'best':best,'history':history,'rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                'mean':norm.mean,'std':norm.std,'contract':contract,
                'elapsed_seconds':prior_seconds+time.monotonic()-start}
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    while step<300:
        elapsed=prior_seconds+time.monotonic()-start
        if time.time()>state['deadline_unix']-120 or state['completed_run_seconds']+elapsed>1800:
            torch.save(snapshot(),latest); raise TimeoutError('Study time budget reached')
        model.train()  # Explicit restoration after every validation is essential.
        x,y,mask,valid=(a.to(device) for a in sampler.batch())
        optimizer.zero_grad(set_to_none=True)
        logits=model(x,padding_mask=~valid.bool())
        loss=(loss_fn(logits,y)*mask).sum()/mask.sum().clamp_min(1)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
        loss.backward(); grad=nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad): raise FloatingPointError('nonfinite gradient')
        optimizer.step(); step+=1
        if step%50==0:
            raw=predictions(model,val,norm,device)
            assert model.training, 'Validation failed to restore mode'
            threshold,_=choose_single_threshold(raw,val,cfg['evaluation']['threshold_grid'])
            _,_,scores=metrics(raw,val,threshold)
            history.append({'step':step,'train_loss':float(loss),'threshold':threshold,**scores})
            if scores['macro_f1_tol1']>best+1e-9:
                best=scores['macro_f1_tol1']; torch.save(snapshot(),directory/'best.pt')
            print(run,step,round(scores['macro_f1_tol1'],4),'train_mode',model.training,flush=True)
        if step%25==0: torch.save(snapshot(),latest)
    saved=torch.load(directory/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(saved['model']); threshold=saved['history'][-1]['threshold']
    raw=predictions(model,val,norm,device); perf,pieces,score=metrics(raw,val,threshold)
    _,_,train_score=metrics(predictions(model,train,norm,device),train,threshold)
    probe=probe_groups(model,val,norm,threshold,device,score,kind)
    elapsed=prior_seconds+time.monotonic()-start
    result={'kind':kind,'fold':fold,'seed':42,'contract':contract,'threshold':threshold,'best_step':saved['step'],
            'params':sum(p.numel() for p in model.parameters()),'seconds':elapsed,
            'gpu_peak_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            'train_f1':train_score['macro_f1_tol1'],'train_gap':train_score['macro_f1_tol1']-score['macro_f1_tol1'],
            'history':history,'probe':probe,**score}
    perf.to_csv(ART/'metrics'/f'{run}_performances.csv',index=False)
    pieces.to_csv(ART/'metrics'/f'{run}_pieces.csv',index=False)
    write(result_path,result)
    state['completed_run_seconds']+=elapsed; state['completed_runs'].append(run)
    write(OUT/'STATE.json',state)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(ART/'metrics').mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/'configs/phase7/protocol.yaml').read_text(encoding='utf-8'))
    split_path=ROOT/cfg['data']['split_source']; splits=pd.read_csv(split_path)
    tracked=[Path(__file__),OUT/'PROTOCOL.md',split_path]
    tracked += [ROOT/'src'/s for s in ['phase7_models.py','phase2_models.py','phase3_models.py','phase6_models.py','evaluation.py','models.py','data.py']]
    # Fingerprint all inputs; reading these hashes does not use test outcomes in selection.
    for folder in ['cache/piece_features','cache/phase3/piece_features']:
        tracked+=sorted((ROOT/folder).glob('*.npz'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in tracked}
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    statepath=OUT/'STATE.json'
    if statepath.exists():
        state=json.loads(statepath.read_text(encoding='utf-8'))
        if not args.resume: raise ValueError('Use --resume; study already exists')
        if state['contract']!=contract: raise ValueError('Study contract changed')
    else:
        state={'started_unix':time.time(),'deadline_unix':time.time()+3600,'status':'running','contract':contract,
               'completed_run_seconds':0.,'completed_runs':[],'usage_start_percent':44,
               'outer_test_evaluated':False,'training_mode_restored_each_step':True}
        write(statepath,state);write(OUT/'input_contract.json',hashes)
    frozen_paths=[ROOT/'reports/phase7/final_report.md',ROOT/'reports/phase7/completion_audit.json']
    frozen_paths+=sorted((ROOT/'checkpoints/phase7').rglob('*.pt'))
    frozen={str(p.relative_to(ROOT)):sha(p) for p in frozen_paths}
    if not (OUT/'frozen_phase7.json').exists():write(OUT/'frozen_phase7.json',frozen)
    else:
        assert frozen==json.loads((OUT/'frozen_phase7.json').read_text(encoding='utf-8'))
    all_results=[]; audits=[]
    for fold in [0,1]:
        part=splits[splits.fold==fold]
        train_ids=part[part.split=='train'].piece_id.tolist(); val_ids=part[part.split=='validation'].piece_id.tolist()
        test_ids=part[part.split=='test'].piece_id.tolist()
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            x,y=part[part.split==a],part[part.split==b]
            row={'fold':fold,'a':a,'b':b,'piece_overlap':len(set(x.piece_id)&set(y.piece_id)),
                 'opus_overlap':len(set(x.opus)&set(y.opus))}; audits.append(row)
            assert row['piece_overlap']==row['opus_overlap']==0
        for kind in KINDS:
            dim=9 if kind.endswith('9') else 25
            train=data_for(train_ids,dim);val=data_for(val_ids,dim)
            assert not(set(train)&set(test_ids)) and not(set(val)&set(test_ids))
            result=run_one(kind,fold,train,val,cfg,contract,state);all_results.append(result)
            pd.DataFrame([{k:v for k,v in r.items() if k not in ['history','probe']} for r in all_results]).to_csv(ART/'summary.csv',index=False)
            del train,val
    pd.DataFrame(audits).to_csv(OUT/'split_audit.csv',index=False)
    assert frozen=={str(p.relative_to(ROOT)):sha(p) for p in frozen_paths}
    state['status']='complete';state['completed_unix']=time.time();state['frozen_phase7_unchanged']=True
    state['hashes_unchanged']=hashes=={str(p.relative_to(ROOT)):sha(p) for p in tracked}
    assert state['hashes_unchanged']
    write(statepath,state)
    print('COMPLETE',state['completed_run_seconds'],flush=True)

if __name__=='__main__':main()
