import inspect
import builtins,dis,types
import torch
from src import recurrence_lr_training as candidate
from src import score_context_study as original


def test_training_loop_diff_is_only_learning_rate_lookup():
    expected=inspect.getsource(original.train_one).replace('lr=.001,weight_decay=.0001','lr=LEARNING_RATES[kind],weight_decay=.0001')
    assert inspect.getsource(candidate.train_one).strip()==expected.strip()
    assert candidate.LEARNING_RATES=={'C3':.001,'L3':.0003}


def test_model_weights_and_rng_are_matched_across_learning_rates():
    for seed in (42,43):
        a=candidate.make_model('C3',seed);rng=torch.get_rng_state().clone();b=candidate.make_model('L3',seed)
        assert torch.equal(rng,torch.get_rng_state())
        assert sum(p.numel() for p in b.parameters())==5921
        for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0)


def test_copied_loop_and_nested_snapshot_have_all_global_dependencies():
    def codes(code):
        yield code
        for value in code.co_consts:
            if isinstance(value,types.CodeType):yield from codes(value)
    missing={i.argval for code in codes(candidate.train_one.__code__) for i in dis.get_instructions(code)
             if i.opname=='LOAD_GLOBAL' and i.argval not in candidate.train_one.__globals__ and not hasattr(builtins,i.argval)}
    assert not missing
