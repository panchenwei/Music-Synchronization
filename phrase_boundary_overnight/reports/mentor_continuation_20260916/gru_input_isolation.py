"""Two input-source ablations on frozen internal G64 protocol; no old-source edits."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,copy,json,hashlib,time,types,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from src.mentor_sequence_v2 import engine
from src.mentor_sequence_models import make_model as original_model,ContextSampler
from src.current_gru_models import CurrentBiGRU
from src.mentor_sequence_study import splits,digest_array
from src.run_recurrence_depth_study import dataset
from src.score_context_study import read,write,sha,normalizer
from src.local_context_study import predictions,metrics
BASE=Path(__file__).parent;OUT=BASE/'gru_input_isolation';ART=ROOT/'artifacts/mentor_gru_input_isolation_20260916'
OLD=ROOT/'artifacts/mentor_sequence_20260916_v2';COLS=engine.COLS
KEEP={'score':list(range(9,25))+list(range(34,58)),'performance':list(range(9))+list(range(25,34)),'all':list(range(58))}


def masked_forward(self,x,padding_mask=None):
    return CurrentBiGRU.forward(self,x*self.input_keep,padding_mask)


def model(arm,seed):
    m=original_model('G64',seed)
    keep=torch.zeros(58);keep[KEEP[arm]]=1
    m.register_buffer('input_keep',keep,persistent=False)
    m.forward=types.MethodType(masked_forward,m)
    assert sum(p.numel() for p in m.parameters())==6005
    return m


def guard(new=False):
    b=read(BASE/'BUDGET.json')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(b['observed_at_utc'].replace('Z','+00:00'))).total_seconds()
    if age>1800 or b['observed_used_percent']>=b['stop_new_runs_used_percent' if new else 'absolute_stop_used_percent']:
        raise TimeoutError('Budget limit or stale usage')
    seconds=sum(read(p)['seconds'] for p in ART.glob('*/metrics/*_fold*.json'))
    assert seconds<12000
    return seconds


def preflight(train):
    tests=[]
    assert set(KEEP['score']).isdisjoint(KEEP['performance']) and sorted(KEEP['score']+KEEP['performance'])==KEEP['all']
    ref=original_model('G64',42).eval();control=model('all',42).eval()
    x=torch.randn(2,16,58);pad=torch.zeros(2,16,dtype=torch.bool);pad[1,12:]=True
    torch.testing.assert_close(ref(x,pad),control(x,pad),rtol=0,atol=0)
    for arm in ('score','performance'):
        m=model(arm,42).eval();old=original_model('G64',42)
        assert set(m.state_dict())==set(old.state_dict()) and all(torch.equal(v,old.state_dict()[k]) for k,v in m.state_dict().items())
        drop=sorted(set(range(58))-set(KEEP[arm]));changed=x.clone();changed[...,drop]=12345;changed[pad]=-888
        torch.testing.assert_close(m(x,pad),m(changed,pad),rtol=0,atol=0)
        z=x.clone().requires_grad_(True);m(z,pad).sum().backward();assert not z.grad[...,drop].any() and not z.grad[pad].any()
        opt=torch.optim.AdamW(m.parameters(),lr=.003);y=torch.zeros(2,16);y[:,[4,9]]=1
        for _ in range(180):
            m.train();opt.zero_grad(set_to_none=True);p=m(x,pad);loss=torch.nn.functional.binary_cross_entropy_with_logits(p[~pad],y[~pad]);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        m.eval();fit=float(((torch.sigmoid(m(x,pad))[~pad]>.5)==y[~pad].bool()).float().mean());assert fit>.95
        norm=normalizer(train);sampler=ContextSampler(train,norm,42,False);a,b,mask,valid=[v.cuda() for v in sampler.batch()]
        m=model(arm,42).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
        def update():
            m.train();opt.zero_grad(set_to_none=True);p=m(a,~valid.bool());loss=(torch.nn.functional.binary_cross_entropy_with_logits(p,b,pos_weight=torch.tensor(10.,device='cuda'),reduction='none')*mask).sum()/mask.sum();assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1);assert torch.isfinite(gn);opt.step();return float(loss.detach())
        update();saved=copy.deepcopy(m.state_dict());osaved=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state();crng=torch.cuda.get_rng_state_all();expected=update();after=copy.deepcopy(m.state_dict());m.load_state_dict(saved);opt.load_state_dict(osaved);torch.set_rng_state(rng);torch.cuda.set_rng_state_all(crng)
        assert update()==expected and all(torch.equal(v,m.state_dict()[k]) for k,v in after.items())
        tests.append(dict(arm=arm,retained=len(KEEP[arm]),initial_values_equal=True,dropped_input_and_padding_invariance=True,dropped_input_gradients_zero=True,tiny_overfit=fit,real_optimizer_rng_resume_exact=True))
    write(OUT/'preflight.json',dict(status='passed',all_channel_wrapper_exact=True,tests=tests))


def prepare():
    guard(True);OUT.mkdir(exist_ok=True)
    parent=read(BASE/'sequence_v2/sequence_contract.json');assert all(sha(p)==h for p,h in parent['sources'].items())
    frame,_=splits();data=dataset(sorted(frame.piece_id.unique()),'C3')
    inputs={p:{k:digest_array(v[k]) for k in ('curves','labels','label_mask','performance_ids','pitch_profiles')} for p,v in data.items()}
    assert inputs==parent['inputs']
    sources={str(p):sha(p) for p in [Path(__file__),OUT/'PROTOCOL.md']}
    for seed in (42,43):
        for fold in (0,1):
            run=f'G64_seed{seed}_fold{fold}'
            for p in [OLD/'metrics'/f'{run}.json',OLD/'checkpoints'/run/'best.pt',OLD/'checkpoints'/run/'latest.pt']:sources[str(p)]=sha(p)
    payload=dict(parent_contract=parent['contract'],inputs=inputs,keep=KEEP,sources=sources)
    contract=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    else:write(OUT/'contract.json',dict(contract=contract,**payload))
    frame.to_csv(OUT/'splits.csv',index=False)
    return frame,data,contract


def audit(frame,data):
    rows=[];diagnostics=[];hashes={}
    for arm in ('score','performance'):
        for seed in (42,43):
            for fold in (0,1):
                run=f'G64_seed{seed}_fold{fold}';a=read(ART/arm/'metrics'/f'{run}.json');b=read(OLD/'metrics'/f'{run}.json')
                cp=torch.load(ART/arm/'checkpoints'/run/'latest.pt',map_location='cpu',weights_only=False);ref=torch.load(OLD/'checkpoints'/run/'latest.pt',map_location='cpu',weights_only=False)
                assert cp['step']==600 and a['params']==6005 and cp['sampler']==ref['sampler'] and a['supervised_positions']==b['supervised_positions']
                np.testing.assert_array_equal(cp['mean'],ref['mean']);np.testing.assert_array_equal(cp['std'],ref['std'])
                for p in [ART/arm/'checkpoints'/run/'best.pt',ART/arm/'checkpoints'/run/'latest.pt',ART/arm/'metrics'/f'{run}.json']:hashes[str(p)]=sha(p)
                part=frame[frame.fold==fold];tr={p:data[p] for p in part[part.split=='train'].piece_id};va={p:data[p] for p in part[part.split=='validation'].piece_id};norm=normalizer(tr)
                m=model(arm,seed).cuda();m.load_state_dict(cp['model']);et=metrics(predictions(m,tr,norm,torch.device('cuda')),tr,a['threshold'])[2];ev=metrics(predictions(m,va,norm,torch.device('cuda')),va,a['threshold'])[2]
                rows.append(dict(arm=arm,seed=seed,fold=fold,**{k+'_candidate':a[k] for k in COLS},**{k+'_reference':b[k] for k in COLS},**{k+'_delta':a[k]-b[k] for k in COLS}))
                diagnostics.append(dict(arm=arm,seed=seed,fold=fold,best_step=a['best_step'],threshold=a['threshold'],best_train_f1=a['train_at_dev_threshold']['macro_f1_tol1'],best_dev_f1=a['macro_f1_tol1'],end_train_f1=et['macro_f1_tol1'],end_dev_f1=ev['macro_f1_tol1']))
                print('INPUT_AUDITED',arm,run,flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'comparisons.csv',index=False);pd.DataFrame(diagnostics).to_csv(OUT/'diagnostics.csv',index=False)
    results=[]
    for arm,part in f.groupby('arm'):
        means={c:float(part[c].mean()) for c in part if c.startswith(('macro_','raw_'))};positive=int((part.macro_f1_tol1_delta>0).sum())
        results.append(dict(arm=arm,cells=4,positive_cells=positive,passed=bool(positive>=3 and means['macro_f1_tol1_delta']>=.015 and means['macro_f1_tol0_delta']>=0 and means['raw_ap_delta']>=0),**means))
    write(OUT/'summary.json',results)
    sources={**read(BASE/'sequence_v2/sequence_contract.json')['sources'],**read(OUT/'contract.json')['sources'],**hashes}
    assert all(sha(p)==h for p,h in sources.items());write(OUT/'checkpoint_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',runs=8,source_hashes_unchanged=True,normalizers_and_sampler_exposure_equal=True,original_best_references_not_rerun=True,independent_test=False))
    write(OUT/'STATE.json',dict(status='complete',pid=None))


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    frame,data,contract=prepare();part=frame[frame.fold==0]
    if not (OUT/'preflight.json').exists():preflight({p:data[p] for p in part[part.split=='train'].piece_id})
    engine.guard=guard
    for arm in ('score','performance'):
        engine.OUT=OUT/arm;engine.ART=ART/arm;engine.make_model=lambda kind,seed:model(arm,seed)
        for p in [engine.OUT,engine.ART/'metrics',engine.ART/'checkpoints']:p.mkdir(parents=True,exist_ok=True)
        for seed in (42,43):
            for fold in (0,1):
                write(OUT/'STATE.json',dict(status='running',pid=os.getpid(),arm=arm,seed=seed,fold=fold))
                part=frame[frame.fold==fold];tr={p:data[p] for p in part[part.split=='train'].piece_id};va={p:data[p] for p in part[part.split=='validation'].piece_id}
                engine.train_one('G64',fold,seed,tr,va,contract)
    audit(frame,data)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()));raise
