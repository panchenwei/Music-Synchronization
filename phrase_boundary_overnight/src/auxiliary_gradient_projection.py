"""Main-task-priority projection inspired by, but not identical to, PCGrad."""
import torch


def merge(main,aux,project):
    shared=[(a,b) for a,b in zip(main,aux) if a is not None and b is not None]
    u=torch.cat([a.reshape(-1) for a,b in shared]);v=torch.cat([b.reshape(-1) for a,b in shared])
    dot=u@v;den=u@u
    coefficient=torch.minimum(dot,torch.zeros_like(dot))/den.clamp_min(1e-20) if project else torch.zeros_like(dot)
    result=[]
    for a,b in zip(main,aux):
        if a is None:result.append(b)
        elif b is None:result.append(a)
        else:result.append(a+b-coefficient*a)
    stats=dict(cosine=float(dot/(u.norm()*v.norm()).clamp_min(1e-20)),conflicting=float(dot<0),
        applied=float(project and dot<0),aux_main_ratio=float(v.norm()/u.norm().clamp_min(1e-20)),
        projected_dot=float(dot-coefficient*den))
    assert all(a is None or torch.isfinite(a).all() for a in result)
    return result,stats
