"""Label-free array API for the versioned small CNN; offline beat input only."""
from __future__ import annotations
import numpy as np
import torch
from .models import Normalizer
from .phase2_models import nms_probabilities
from .slice_energy_features import corrected_score, tempo_hierarchy
from .slice_energy_study import model_for

def assemble_inputs(curves_9, original_score_16, note_events, kind='SFE31'):
    curves=np.asarray(curves_9,dtype=np.float32)
    score=np.asarray(original_score_16,dtype=np.float32)
    if curves.ndim!=2 or curves.shape[1]!=9 or score.shape!=(len(curves),16):
        raise ValueError('Expected one performance [beats,9] and score [beats,16]')
    if not np.isfinite(curves).all() or not np.isfinite(score).all():raise ValueError('Nonfinite input')
    if not note_events or any(len(e)!=5 for e in note_events):
        raise ValueError('Score events need onset,duration,MIDI pitch,staff,voice; merged MIDI is insufficient')
    note_events=[tuple(e) for e in note_events]
    rest,fixed,_=corrected_score(score,note_events)
    selected=fixed if kind.startswith('SF') else rest if kind=='SR25' else score
    x=np.concatenate([curves,selected],axis=1)
    if kind.endswith('31'):
        extra=tempo_hierarchy(np.exp(curves[:,0].astype(float)),curves[:,7])[0]
        if kind=='SFZ31':extra=np.zeros_like(extra)
        x=np.concatenate([x,extra],axis=1)
    return x

def predict_single(checkpoint_path, curves_9, original_score_16, note_events, kind='SFE31'):
    """No ground-truth boundary, phrase start, harmony label or split is accepted.

    Input beat alignment must be supplied by the existing system. This API does
    not imply the alignment/audio-domain performance has been validated.
    """
    x=assemble_inputs(curves_9,original_score_16,note_events,kind)
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model=model_for(x.shape[1],42).eval();model.load_state_dict(saved['model'])
    norm=Normalizer(saved['mean'],saved['std'])
    threshold=float(saved['history'][-1]['threshold'])
    with torch.no_grad():prob=torch.sigmoid(model(torch.from_numpy(norm.apply(x).astype(np.float32))[None]))[0].numpy()
    boundaries=np.flatnonzero(nms_probabilities(prob)>=threshold)
    # All positions are inferred; unlike evaluation, no ground-truth interior mask.
    slices=[{'start_beat':int(a),'stop_beat':int(b),'interval':'[start,stop)'} for a,b in zip(boundaries[:-1],boundaries[1:])]
    return {'probabilities':prob,'threshold':threshold,'predicted_starts':boundaries,'slices':slices,
            'prefix_suffix':'not automatically declared complete phrases','kind':kind}
