"""Frozen target predictions: distinguish full-batch replay from subset arithmetic."""
import time
import numpy as np
import pandas as pd
import torch
from .external_stem_transfer import OUT,ART,read,write,sha,dataset,split_ids,build_model,Normalizer,roll_predictions,metrics,COLS
from .audit_external_stem_transfer import checked_raw


def main():
    began=time.monotonic();torch.set_num_threads(2);rows=[];hashes={}
    original=torch.backends.cudnn.allow_tf32
    for r in pd.read_csv(ART/'summary.csv').to_dict('records'):
        run=r['run_id'];cp=ART/'checkpoints'/run/'best.pt';fp=ART/'metrics'/f'{run}_predictions.csv.gz'
        hashes[str(cp)]=sha(cp);hashes[str(fp)]=sha(fp)
        s=torch.load(cp,map_location='cpu',weights_only=False);data=dataset(split_ids(r['fold'])['validation'],r['kind'])
        raw=checked_raw(pd.read_csv(fp),data);norm=Normalizer(s['mean'],s['std'])
        pid=sorted(data)[0];perf=sorted(raw[pid])[len(raw[pid])//2];idx=list(data[pid]['performance_ids'].astype(str)).index(perf)
        subset={pid:{**data[pid],'curves':data[pid]['curves'][idx:idx+1],'performance_ids':np.array([perf])}}
        model=build_model(r['kind'],r['seed']);model.load_state_dict(s['model']);model.eval()
        replays={}
        for mode,dev,tf32,items in [('cpu_subset','cpu',False,subset),('gpu_subset','cuda',True,subset),('gpu_ieee_subset','cuda',False,subset),('gpu_original_full','cuda',True,data)]:
            torch.backends.cudnn.allow_tf32=tf32;model.to(dev)
            pred=roll_predictions(model,items,norm,torch.device(dev));replays[mode]=pred[pid][perf]
            ref={p:{k:raw[p][k] for k in item['performance_ids'].astype(str)} for p,item in items.items()}
            score=metrics(pred,items,r['threshold'])[2];base=metrics(ref,items,r['threshold'])[2]
            delta=max(float(np.max(abs(pred[p][k]-ref[p][k]))) for p in ref for k in ref[p])
            rows.append(dict(run_id=run,mode=mode,max_probability_error=delta,**{c+'_delta':score[c]-base[c] for c in COLS}))
        cross=float(np.max(abs(replays['cpu_subset']-replays['gpu_ieee_subset'])))
        for row in rows[-4:]:row['cpu_vs_ieee_subset_error']=cross
        pd.DataFrame(rows).to_csv(OUT/'target_replay_diagnostic.csv',index=False)
        print(run,rows[-1], 'CPU_IEEE',cross,flush=True)
    torch.backends.cudnn.allow_tf32=original
    assert all(sha(p)==v for p,v in hashes.items())
    write(OUT/'target_replay_diagnostic_manifest.json',dict(training_runs=0,test_predictions_accessed=False,source_hashes=hashes,script_sha256=sha(__file__),seconds=time.monotonic()-began,torch_version=torch.__version__,original_cudnn_tf32=original))


if __name__=='__main__':main()
