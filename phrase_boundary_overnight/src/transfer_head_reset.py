"""Reset only the binary decision head, retaining the frozen transfer mapping."""
import torch
from .recurrence_transfer_core import transfer


def transfer_without_head(model, external_state, external_norm, target_norm):
    initial = {k: v.detach().clone() for k, v in model.output.state_dict().items()}
    transfer(model, external_state, external_norm, target_norm)
    model.output.load_state_dict(initial)
    return model
