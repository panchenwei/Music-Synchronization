"""First-order L2 SAM with common dropout randomness; no persistent perturbation."""
import torch


def step(model, optimizer, batch, criterion, rho):
    assert rho >= 0
    model.train()
    x,y,mask,valid=batch
    params=list(model.parameters())
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    cpu=torch.get_rng_state()
    devices=[x.device.index if x.device.index is not None else torch.cuda.current_device()] if x.is_cuda else []
    cuda=[torch.cuda.get_rng_state(d) for d in devices]
    def loss():
        logits=model(x,padding_mask=~valid.bool())
        result=(criterion(logits,y)*mask).sum()/mask.sum().clamp_min(1)
        assert torch.isfinite(result)
        return result
    optimizer.zero_grad(set_to_none=True)
    original_loss=loss();original_loss.backward()
    second=original_loss.detach()
    radius=0.
    if rho:
        norm=torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params if p.grad is not None]))
        assert torch.isfinite(norm)
        originals=[p.detach().clone() for p in params]
        try:
            with torch.no_grad():
                for p in params:
                    if p.grad is not None:p.add_(p.grad*(rho/(norm+1e-12)))
                radius=float(torch.linalg.vector_norm(torch.stack([(p-o).norm() for p,o in zip(params,originals)])))
            optimizer.zero_grad(set_to_none=True)
            # Second pass uses the same dropout realization but consumes no extra
            # randomness in the next training step or validation operation.
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(cpu)
                for d,state in zip(devices,cuda):torch.cuda.set_rng_state(state,d)
                second=loss();second.backward()
        finally:
            with torch.no_grad():
                for p,o in zip(params,originals):p.copy_(o)
        assert all(torch.equal(p,o) for p,o in zip(params,originals))
    grad=torch.nn.utils.clip_grad_norm_(params,1.)
    assert torch.isfinite(grad)
    optimizer.step()
    return dict(loss=float(original_loss.detach()),perturbed_loss=float(second.detach()),grad=float(grad),radius=radius)
