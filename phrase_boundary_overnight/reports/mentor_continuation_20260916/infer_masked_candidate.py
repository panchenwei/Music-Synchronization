"""Label-free inference for the internally promoted masked-training ensemble."""
import os,sys,argparse
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;sys.path[:0]=[str(ROOT),str(BASE)]
import numpy as np
import pandas as pd
import torch
from src.score_context_study import read,write,sha
from src.mentor_sequence_models import make_model
from src.models import Normalizer
from src.local_context_study import predictions
from src.interstart_decoder import decode
from infer_corrected_candidate import build_features


def infer(piece_id,fold):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);export_path=BASE/'temporal_mask_replication/export.json';export=read(export_path);assert not export['candidate_only_not_promoted'];spec=next(x for x in export['folds'] if x['fold']==fold);item,beat,provenance=build_features(piece_id);raws=[]
    for c in spec['components']:
        assert sha(c['checkpoint'])==c['sha256'];cp=torch.load(c['checkpoint'],map_location='cpu',weights_only=False);m=make_model('G64',c['seed']).cuda();m.load_state_dict(cp['model']);raws.append(predictions(m,{piece_id:item},Normalizer(np.asarray(cp['mean']),np.asarray(cp['std'])),torch.device('cuda'))[piece_id])
    probabilities={k:np.stack([r[k] for r in raws]).mean(0) for k in raws[0]};rows=[]
    for perf,p in probabilities.items():
        for b in decode(p,spec['threshold'],spec['prior'],1.):
            second=float(beat[perf].iloc[int(b)]);rows.append(dict(piece_id=piece_id,performance_id=perf,beat_index=int(b),measure_number=float(beat.measure_number.iloc[int(b)]),beat_in_measure=float(beat.beat_number.iloc[int(b)]),recording_seconds=second if np.isfinite(second) else None,mean_model_probability=float(p[b])))
    provenance.update(candidate_version='U58_B4mask_4seed_B10_20260916',export_path=str(export_path),export_sha256=sha(export_path),fold=fold,ensemble_components=spec['components'],threshold=spec['threshold'],decoder='B10 strength1; fixed train prior',training_only_augmentation='contiguous 4 normalized-zero time positions',inference_augmentation=False,labels_used=False,independent_test=False,waveform_alignment_independently_validated=False,fold_is_explicit_not_selected_by_this_piece=True,cpu_inference_validated=False)
    return probabilities,pd.DataFrame(rows),provenance


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--piece-id',required=True);p.add_argument('--fold',type=int,choices=(0,1),required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();dest=a.out.resolve();assert not dest.exists(),'Use a new output directory';dest.mkdir(parents=True);probs,rows,meta=infer(a.piece_id,a.fold);rows.to_csv(dest/'phrase_starts.csv',index=False);np.savez_compressed(dest/'probabilities.npz',performance_ids=np.asarray(list(probs)),probabilities=np.stack(list(probs.values())));write(dest/'provenance.json',meta);print(dest)


if __name__=='__main__':main()
