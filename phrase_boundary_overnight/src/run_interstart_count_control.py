"""Fixed-C3 boundary-count penalty control: no distance-specific rewards."""
import hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from . import run_recurrence_depth_study as base
from .score_context_study import ROOT,read,write,sha
from .three_round_round2 import split_ids
from .audit_external_stem_transfer import checked_raw
from .evaluation import evaluate_piece
from .interstart_decoder import fit_prior,decode,log_ratio

OUT=ROOT/'reports/interstart_count_control';ART=ROOT/'artifacts/interstart_count_control'
STRENGTHS=(0.,.5);COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']


def main():
    began=time.monotonic()
    for path in (OUT,ART):path.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/interstart_decoder_study/completion_audit.json')['status']=='complete'
    hashes=dict(read(ROOT/'reports/interstart_decoder_study/contract.json')['hashes'])
    reference_path=ROOT/'artifacts/interstart_decoder_study/summary.csv';hashes[str(reference_path)]=sha(reference_path)
    for fold in (0,1):
        for seed in (42,43):
            for suffix in ('.json','_predictions.csv.gz'):
                path=base.ART/'metrics'/f'C3_seed{seed}_fold{fold}{suffix}';hashes[str(path)]=sha(path)
    for path in (Path(__file__),ROOT/'src/interstart_decoder.py',ROOT/'tests/test_interstart_decoder.py',OUT/'PROTOCOL.md'):
        hashes[str(path)]=sha(path)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));rows=[];decode_hashes={}
    for fold in (0,1):
        ids=split_ids(fold);assert not set(ids['train'])&set(ids['validation'])
        manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv');part=manifest[manifest.fold==fold]
        assert not set(part[part.split=='train'].opus)&set(part[part.split=='validation'].opus)
        prior=fit_prior(base.dataset(ids['train'],'C3'))
        prior.update(histogram=[0.]*256,mode='constant_log_half_count_penalty_not_a_normalized_prior')
        np.testing.assert_allclose(log_ratio(prior,np.arange(1,600)),np.log(.5),atol=1e-12,rtol=0)
        write(OUT/f'prior_fold{fold}.json',dict(prior=prior,train_ids=ids['train'],contract=digest))
        val=base.dataset(ids['validation'],'C3')
        for seed in (42,43):
            run=f'C3_seed{seed}_fold{fold}';meta=read(base.ART/'metrics'/f'{run}.json')
            raw=checked_raw(pd.read_csv(base.ART/'metrics'/f'{run}_predictions.csv.gz'),val)
            for strength in STRENGTHS:
                name=f'{run}_lambda{strength:g}';dest=ART/f'{name}.json'
                if dest.exists():
                    row=read(dest);assert row['contract']==digest
                    assert all(sha(p)==h for p,h in row['artifact_hashes'].items())
                    rows.append(row);decode_hashes.update(row['artifact_hashes']);continue
                write(OUT/'STATE.json',dict(status='running',pid=os.getpid(),run=name,contract=digest))
                evaluated=[];outputs=[];started=time.monotonic()
                for pid,perfs in raw.items():
                    for perf,probs in perfs.items():
                        if time.monotonic()-began>1800:raise TimeoutError('1800s decoder-study cap')
                        selected=decode(probs,meta['threshold'],prior,strength)
                        # Decoder receives neither validation labels nor their validity mask.
                        binary=np.zeros(len(probs));binary[selected]=1.
                        score=evaluate_piece(pid,binary,val[pid]['labels'],val[pid]['label_mask'],.5)
                        evaluated.append(dict(**score,performance_id=perf))
                        outputs.extend((pid,perf,int(b)) for b in selected)
                pf=pd.DataFrame(evaluated);pieces=pf.groupby('piece_id').mean(numeric_only=True)
                score={f'macro_{key}':float(pieces[key].mean()) for key in ('f1_tol1','f1_tol0','precision_tol1','recall_tol1')}
                score['raw_ap']=meta['raw_ap'] # Original neural score ranking is unchanged, NOT AP of the decoded binary vector.
                if strength==0:assert max(abs(score[c]-meta[c]) for c in COLS)<1e-10
                ppath=ART/f'{name}_positions.csv.gz';pd.DataFrame(outputs,columns=['piece_id','performance_id','beat']).to_csv(ppath,index=False)
                pf.to_csv(ART/f'{name}_performances.csv',index=False);pieces.to_csv(ART/f'{name}_pieces.csv')
                # Read back positions, regenerate event counts, independently compare each saved row.
                saved=pd.read_csv(ppath);groups={(p,q):v.beat.to_numpy(int) for (p,q),v in saved.groupby(['piece_id','performance_id'])}
                for row in evaluated:
                    pid=row['piece_id'];perf=row['performance_id'];binary=np.zeros(len(val[pid]['labels']))
                    binary[groups.get((pid,perf),np.array([],int))]=1
                    check=evaluate_piece(pid,binary,val[pid]['labels'],val[pid]['label_mask'],.5)
                    assert all(check[k]==row[k] for k in ('tp_tol0','fp_tol0','fn_tol0','tp_tol1','fp_tol1','fn_tol1'))
                ah={str(ppath):sha(ppath)};decode_hashes.update(ah)
                row=dict(run_id=run,fold=fold,seed=seed,strength=strength,contract=digest,threshold=meta['threshold'],
                         seconds=time.monotonic()-started,artifact_hashes=ah,**score)
                write(dest,row);rows.append(row);print(name,score,flush=True)
    assert len(rows)==8 and all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in decode_hashes.items())
    frame=pd.DataFrame(rows);frame.to_csv(ART/'summary.csv',index=False)
    means=frame.groupby('strength')[COLS].mean();means.to_csv(OUT/'means.csv');comparisons=[]
    for strength in STRENGTHS[1:]:
        a=frame[frame.strength==strength].set_index(['fold','seed']);b=frame[frame.strength==0].set_index(['fold','seed']);d=a[COLS]-b[COLS]
        d.to_csv(OUT/f'deltas_lambda{strength:g}.csv')
        passed=d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>=0 and (d.macro_f1_tol1>0).sum()>=3
        comparisons.append(dict(strength=strength,**d.mean().to_dict(),f1_positive=int((d.macro_f1_tol1>0).sum()),passed=bool(passed)))
    reference=pd.read_csv(reference_path);a=reference[reference.strength==.5].set_index(['fold','seed'])
    b=frame[frame.strength==.5].set_index(['fold','seed']);d=a[COLS]-b[COLS]
    d.to_csv(OUT/'distance_specific_minus_count_only.csv')
    write(OUT/'specificity_comparison.json',dict(**d.mean().to_dict(),f1_positive=int((d.macro_f1_tol1>0).sum()),exact_positive=int((d.macro_f1_tol0>0).sum())))
    write(OUT/'comparisons.json',comparisons)
    write(OUT/'completion_audit.json',dict(status='complete',decode_cells=8,new_neural_training_runs=0,
          original_neural_ap_unchanged=True,zero_strength_original_f1_exact=True,source_hashes_unchanged=True,
          saved_positions_event_counts_recomputed=True,priors_train_only=True,validation_masks_only_for_evaluation=True,
          test_used=False,elapsed_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None,contract=digest));print(means.to_string())


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise
