"""Label-free prototype adapter for the frozen U ensemble; no training or scoring."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,argparse,json
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;sys.path[:0]=[str(ROOT),str(BASE)]
import numpy as np
import pandas as pd
import torch
from src.data import to_float
from src.slice_energy_study import DCML
from src.score_context_study import read,write,sha
from src.mentor_sequence_models import make_model
from src.models import Normalizer
from src.local_context_study import predictions
from src.interstart_decoder import decode
from score_repeat_bridge import curve_export
from score_only_features_export import extract

def build_features(piece_id):
    # Approved source geometry only. No score labels, label masks or split lookup.
    approved=pd.read_csv(BASE/'score_only_export/feature_parity.csv');assert piece_id in set(approved.piece_id),'Unsupported geometry: run a source-only bridge audit first'
    manifest=pd.read_csv(ROOT/'manifests/piece_manifest.csv').set_index('piece_id');r=manifest.loc[piece_id];key=r.dcml_key;piece=SimpleNamespace(piece_id=piece_id,notes_path=DCML/'notes'/f'{key}.notes.tsv',measures_path=DCML/'measures'/f'{key}.measures.tsv')
    sources={str(p):sha(p) for p in (Path(r.beat_time_path),Path(r.beat_dyn_path),piece.notes_path,piece.measures_path)}
    beat,c,e,ids=curve_export(r);measures=pd.read_csv(piece.measures_path,sep='\t');origin=float(beat.beat_number.iloc[0])-4*to_float(measures.iloc[0].mc_offset);assert origin>=0 and np.isfinite(origin);score,meta=extract(piece,len(beat),origin);np.testing.assert_allclose(score['beat_number'],beat.beat_number.to_numpy(),atol=1e-8,rtol=0)
    x=np.concatenate([c,np.broadcast_to(score['score16'],(*c.shape[:2],16)),e,np.broadcast_to(score['motif24'],(*c.shape[:2],24))],axis=-1).astype(np.float32);assert x.shape==(*c.shape[:2],58) and np.isfinite(x).all();assert all(sha(p)==h for p,h in sources.items())
    return dict(curves=x,performance_ids=ids),beat,dict(input_version='U',piece_id=piece_id,origin_qb=origin,geometry=meta,sources=sources,labels_used=False)

def infer(piece_id,fold,device='cuda'):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);export=read(BASE/'corrected_seed_ensemble/export.json');spec=next(x for x in export['folds'] if x['fold']==fold);item,beat,provenance=build_features(piece_id);raws=[]
    for c in spec['components']:
        assert sha(c['checkpoint'])==c['sha256'];cp=torch.load(c['checkpoint'],map_location='cpu',weights_only=False);m=make_model('G64',c['seed']).to(device);m.load_state_dict(cp['model']);raws.append(predictions(m,{piece_id:item},Normalizer(np.asarray(cp['mean']),np.asarray(cp['std'])),torch.device(device))[piece_id])
    probabilities={k:np.stack([r[k] for r in raws]).mean(0) for k in raws[0]};rows=[]
    for pid,p in probabilities.items():
        for b in decode(p,spec['threshold'],spec['prior'],1.):
            second=float(beat[pid].iloc[int(b)]);rows.append(dict(piece_id=piece_id,performance_id=pid,beat_index=int(b),measure_number=float(beat.measure_number.iloc[int(b)]),beat_in_measure=float(beat.beat_number.iloc[int(b)]),recording_seconds=second if np.isfinite(second) else None,mean_model_probability=float(p[b])))
    provenance.update(fold=fold,ensemble_components=spec['components'],threshold=spec['threshold'],decoder='B10 strength1; fixed train prior',labels_used=False,waveform_alignment_independently_validated=False,fold_is_explicit_not_selected_by_this_piece=True)
    return probabilities,pd.DataFrame(rows),provenance

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--piece-id',required=True);p.add_argument('--fold',type=int,choices=(0,1),required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--out',type=Path,required=True);a=p.parse_args();dest=a.out.resolve();assert not dest.exists(),'Use a new output directory to avoid overwrites';dest.mkdir(parents=True);probs,rows,meta=infer(a.piece_id,a.fold,a.device);rows.to_csv(dest/'phrase_starts.csv',index=False);np.savez_compressed(dest/'probabilities.npz',performance_ids=np.asarray(list(probs)),probabilities=np.stack(list(probs.values())));write(dest/'provenance.json',meta);print(dest)

if __name__=='__main__':main()
