"""Same 5921-parameter C3; change temporal dilation only to (1,2,4)."""
from .recurrence_depth_models import make_model as original


def make_model(kind,seed):
    assert kind=='D'
    model=original('C3',seed)
    for layer,dilation in zip(model.frontend.layers,(1,2,4)):
        conv=layer.convs[0]
        assert conv.kernel_size==(5,) and conv.stride==(1,)
        conv.dilation=(dilation,);conv.padding=(2*dilation,)
    return model
