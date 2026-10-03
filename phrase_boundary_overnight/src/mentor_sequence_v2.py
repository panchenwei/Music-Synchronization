"""Versioned repair: remove a duplicate threshold metadata key, no method change.

V1 stopped at its first validation (step100), before saving any validation result.
Its source, checkpoint and failure record are retained. V2 uses new output paths
and starts from seed; no completed run is repeated or rewritten.
"""
from . import mentor_sequence_study as engine
from .score_context_study import ROOT,read
from .local_context_study import metrics as original_metrics

PARENT=ROOT/'reports/mentor_continuation_20260916'
engine.OUT=PARENT/'sequence_v2'
engine.ART=ROOT/'artifacts/mentor_sequence_20260916_v2'


def metrics_without_duplicate_threshold(*args,**kwargs):
    a,b,c=original_metrics(*args,**kwargs)
    c=dict(c);c.pop('threshold',None)
    return a,b,c


def guard(new=False):
    b=read(PARENT/'BUDGET.json')
    stop=b['stop_new_runs_used_percent'] if new else b['absolute_stop_used_percent']
    if b['observed_used_percent']>=stop:raise TimeoutError('Account budget stop')
    return sum(read(p)['seconds'] for p in (engine.ART/'metrics').glob('*_fold*.json'))


engine.metrics=metrics_without_duplicate_threshold
engine.guard=guard

if __name__=='__main__':engine.main()
