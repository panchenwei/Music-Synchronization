"""Explicit state restoration and unchanged masked training step for continuation."""
import copy
import torch
from torch import nn


def restore(saved, model, optimizer, sampler):
    model.load_state_dict(saved['model'])
    # load_state_dict may reuse tensors already on the target device. Never let
    # the next AdamW step mutate the source snapshot used for another restore.
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    sampler.load_state(saved['sampler'])
    torch.set_rng_state(saved['rng'].cpu())
    if next(model.parameters()).is_cuda:
        torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])


def update(model, optimizer, sampler, criterion, device):
    model.train()
    x,y,mask,valid=(v.to(device) for v in sampler.batch())
    optimizer.zero_grad(set_to_none=True)
    logits=model(x,padding_mask=~valid.bool())
    loss=(criterion(logits,y)*mask).sum()/mask.sum().clamp_min(1)
    assert torch.isfinite(loss)
    loss.backward()
    grad=nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert torch.isfinite(grad)
    optimizer.step()
    return float(loss.detach()),float(grad)


def improves(score, best):
    return score>best+1e-9


def selected_step(history):
    best=-1.;step=None
    for row in history:
        if improves(row['macro_f1_tol1'],best):
            best=row['macro_f1_tol1'];step=row['step']
    return step
