"""Transfer a recurrence-only CNN while preserving the target normalizer."""
import hashlib
import numpy as np
import torch


def shuffled_labels(labels, mask, piece_id):
    """Fixed within-valid-position permutation, preserving per-work class count."""
    labels=np.asarray(labels).copy();idx=np.flatnonzero(np.asarray(mask)>.5)
    seed=int.from_bytes(hashlib.sha256(('external-label-control:'+piece_id).encode()).digest()[:8],'little')
    labels[idx]=np.random.default_rng(seed).permutation(labels[idx])
    return labels


def transfer(model, external_state, external_norm, target_norm):
    original=model.input_projection.weight.detach().clone()
    model.load_state_dict(external_state)
    with torch.no_grad():
        weights=model.input_projection.weight[:,34:].clone()
        ratio=torch.as_tensor(target_norm.std[34:]/external_norm.std[34:],dtype=weights.dtype)
        offset=torch.as_tensor((target_norm.mean[34:]-external_norm.mean[34:])/external_norm.std[34:],dtype=weights.dtype)
        model.input_projection.weight[:,34:]=weights*ratio[None]
        model.input_projection.bias.add_(weights@offset)
        # These channels were unavailable in the external corpus. Keep original
        # target initialization rather than treating zeros as measured curves.
        model.input_projection.weight[:,:34]=original[:,:34]
    return model
