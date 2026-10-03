import inspect,math
import torch
from src import recurrence_schedule_training as candidate
from src import recurrence_lr_training as original


def test_only_per_step_schedule_changes_loop():
    expected=inspect.getsource(original.train_one).replace('        model.train();x,y,mask,valid=',"        for group in opt.param_groups:group['lr']=learning_rate(kind,step)\n        model.train();x,y,mask,valid=")
    assert inspect.getsource(candidate.train_one).strip()==expected.strip()
    for step in range(300):assert candidate.learning_rate('C3',step)==.001
    assert math.isclose(candidate.learning_rate('K3',0),.001)
    assert candidate.learning_rate('K3',299)==.0001
    assert all(candidate.learning_rate('K3',s)>candidate.learning_rate('K3',s+1) for s in range(299))


def test_initial_weights_rng_and_step_resume_schedule():
    a=candidate.make_model('C3',42);rng=torch.get_rng_state();b=candidate.make_model('K3',42)
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0)
    full=[candidate.learning_rate('K3',s) for s in range(300)]
    assert full[:175]+[candidate.learning_rate('K3',s) for s in range(175,300)]==full
