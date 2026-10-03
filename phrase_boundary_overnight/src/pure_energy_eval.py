"""Read-only execution of user tempo algorithms and documented Li2016 adapter."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .local_context_study import max_match_count
from .phase2_models import nms_probabilities

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/pure_energy_20260910'
SOURCE = Path(r'C:\Users\pa1018\Desktop\learn\柴柴\复现')
SCALES = [(1,), (1,3), (1,3,6), (1,3,6,12), (1,3,6,12,24),
          (1,3,6,12,24,48), (1,6,12,24), (1,4,12,24)]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def paper(tempo, scales):
    """Keep literal window-normalized log product separate from normalized beat pdf."""
    t = np.asarray(tempo, dtype=np.float64)
    assert np.isfinite(t).all() and (t > 0).all()
    log_product = np.zeros(len(t))
    for size in scales:
        starts = range(0, len(t), size)
        energies = [np.sqrt(np.mean(t[a:a+size]**2))-np.std(t[a:a+size], ddof=0) for a in starts]
        assert np.min(energies) > 0
        weights = np.asarray(energies)**-2
        weights /= weights.sum()
        for a, weight in zip(starts, weights):
            log_product[a:a+size] += np.log(weight)
    relative = np.exp(log_product-log_product.max())
    distribution = relative/relative.sum()
    return relative, distribution, log_product

def tests(mapped, original):
    t = np.array([60.,120.,80.,90.,65.,110.,75.])
    r,p,_ = paper(t,SCALES[4])
    assert np.allclose(r,paper(t*3,SCALES[4])[0])
    assert np.allclose(paper(np.ones(7)*90,SCALES[4])[1],np.ones(7)/7)
    assert np.isclose(p.sum(),1) and len(r)==7
    assert max_match_count(np.array([2,3]),np.array([2]),1)==1
    assert max_match_count(np.array([1,4]),np.array([2,3]),1)==2
    assert mapped.calc_energy(np.array([3.,4.])) == np.sqrt(12.5)
    assert len(mapped.compute_energy_and_minima(np.ones(7))[2][0])==0
    assert original.compute_energy_and_minima(np.ones(7))[1][0].tolist()==[3]
    return {'scale_invariance':True,'constant_uniform':True,'partial_block':True,
            'matching_no_double_count':True,'external_constant_difference_verified':True}

def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [SOURCE/'CPOPY.py',SOURCE/'复现/data_analysis.py', SOURCE/'复现/main.py', SOURCE/'content.pdf',
                    Path(__file__),OUT/'PROTOCOL.md',ROOT/'src/local_context_study.py',ROOT/'src/phase2_models.py',
                    ROOT/'artifacts/phase2/splits/opus_split_manifest.csv']
    hashes = {str(p):sha(p) for p in source_paths}
    mapped = load_module('user_mapped_energy',source_paths[1])
    original = load_module('user_original_energy',source_paths[0])
    checks = tests(mapped,original)
    split = pd.read_csv(source_paths[-1])
    chosen = split[(split.fold.isin([0,1])) & (split.split=='validation')]
    if chosen.empty:
        chosen = split[(split.fold.isin([0,1])) & (split.split=='val')]
    assert len(chosen)==19 and chosen.piece_id.nunique()==19
    rows=[]; outputs=[]; cache_max_diff=0.0; perf_total=0
    for entry in chosen.itertuples(index=False):
        cache_path=ROOT/f'artifacts/slice_energy_study/cache/{entry.piece_id}.npz'
        hashes[str(cache_path)]=sha(cache_path)
        with np.load(cache_path,allow_pickle=False) as z:
            data={k:z[k] for k in ['curves','performance_ids','start_labels','start_mask','labels','label_mask','paper_prior']}
        for j,perfid in enumerate(data['performance_ids']):
            if time.perf_counter()-started>300: raise TimeoutError('CPU evaluation cap300s')
            tempo=np.exp(data['curves'][j,:,0].astype(float))
            norm=mapped.normalize_curve(tempo)
            assert len(norm)==len(tempo)
            energy,mins,mappings,index_map=mapped.compute_energy_and_minima(norm)
            user_points=np.array(sorted(set(int(v) for layer in mappings for v in layer)),dtype=int)
            old_energy,old_minima,old_levels=original.compute_energy_and_minima(tempo)
            old_points=np.asarray(old_minima[0],dtype=int)
            assert np.all((user_points>=0)&(user_points<len(tempo)))
            outputs.append({'piece_id':entry.piece_id,'performance_id':str(perfid),
                            'mapped_levels':mappings,'mapped_union':user_points.tolist(),
                            'original_level_indices':[np.asarray(v).tolist() for v in old_minima],
                            'original_energy_levels':[np.asarray(v).tolist() for v in old_energy],
                            'mapped_energy_levels':energy})
            methods=[('user_mapped_union',user_points,None,None,None),('user_CPOPY_level0',old_points,None,None,None)]
            for level,scales in enumerate(SCALES):
                relative,distribution,log_product=paper(tempo,scales)
                nms=nms_probabilities(relative)
                methods.append((f'paper_L{level}_fixed010',np.flatnonzero(nms>=.10),relative,distribution,log_product))
                if level==4:
                    cache_max_diff=max(cache_max_diff,float(np.max(np.abs(relative-data['paper_prior'][j]))))
                    methods.append(('paper_L4_all_positive_peaks',np.flatnonzero(nms>0),relative,distribution,log_product))
            for target,label_key,mask_key in [('phrase_start','start_labels','start_mask'),('structural_end','labels','label_mask')]:
                valid=data[mask_key].astype(bool)
                y=data[label_key]>.5
                truth=np.flatnonzero(y&valid)
                for method,points,relative,dist,logp in methods:
                    prediction=points[valid[points]]
                    row={'fold':int(entry.fold),'piece_id':entry.piece_id,'performance_id':str(perfid),
                         'target':target,'method':method,'true_count':len(truth),'pred_count':len(prediction),
                         'valid_beats':int(valid.sum())}
                    for tol in [0,1,2]:
                        tp=max_match_count(prediction,truth,tol)
                        row.update({f'tp{tol}':tp,f'fp{tol}':len(prediction)-tp,f'fn{tol}':len(truth)-tp,
                                    f'p{tol}':tp/max(len(prediction),1),f'r{tol}':tp/max(len(truth),1),
                                    f'f1_{tol}':2*tp/max(len(prediction)+len(truth),1)})
                    row['raw_ap']=float(average_precision_score(y[valid],relative[valid])) if relative is not None else np.nan
                    row['paper_log_query_sum']=float(logp[truth].sum()) if logp is not None else np.nan
                    row['paper_log_query_mean']=float(logp[truth].mean()) if logp is not None else np.nan
                    row['normalized_query_above_uniform']=float(np.log(dist[truth]/dist[valid].sum()).mean()+np.log(valid.sum())) if dist is not None else np.nan
                    rows.append(row)
            perf_total+=1
        print(entry.piece_id,perf_total,'performances processed',flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'per_performance.csv',index=False)
    groups=['target','method']
    works=frame.groupby(groups+['fold','piece_id']).mean(numeric_only=True).reset_index()
    folds=works.groupby(groups+['fold']).mean(numeric_only=True).reset_index()
    means=folds.groupby(groups).mean(numeric_only=True).reset_index().drop(columns='fold')
    works.to_csv(OUT/'per_work.csv',index=False);folds.to_csv(OUT/'per_fold.csv',index=False)
    means.to_csv(OUT/'summary_equal_fold.csv',index=False)
    works.groupby(groups).mean(numeric_only=True).to_csv(OUT/'summary_equal_work.csv')
    with (OUT/'original_outputs.jsonl').open('w',encoding='utf-8') as f:
        for record in outputs:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    checks['existing_L4_cache_agreement']=cache_max_diff<1e-6
    old=pd.read_csv(ROOT/'reports/slice_energy_study/paper_baseline.csv')
    for fold in [0,1]:
        for target,key in [('phrase_start','S25'),('structural_end','E25')]:
            a=folds[(folds.fold==fold)&(folds.target==target)&(folds.method=='paper_L4_fixed010')].iloc[0].f1_1
            b=old[(old.fold==fold)&(old.target==key)].iloc[0].macro_f1_tol1
            checks[f'prior_adapter_F1_{fold}_{key}']=bool(abs(a-b)<1e-10)
    checks['sources_and_inputs_unchanged']=all(sha(p)==h for p,h in hashes.items())
    audit={'status':'complete' if all(checks.values()) else 'failed_checks','checks':checks,
           'pieces':len(chosen),'performances':perf_total,'runtime_seconds':time.perf_counter()-started,
           'L4_cache_max_abs_difference':cache_max_diff,'trained_models':0,'outer_test_evaluated':False,
           'source_input_hashes':hashes,'protocol':'PROTOCOL.md'}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(means[['target','method','p1','r1','f1_0','f1_1','f1_2','raw_ap']].to_string(index=False))
    assert all(checks.values()),checks
    print('COMPLETE',round(audit['runtime_seconds'],2),'seconds')

if __name__=='__main__':main()
