"""Reuse frozen dependence protocol with an explicitly rebound study target."""
from pathlib import Path
from . import external_mert_trial_probe as engine
from . import external_novelty_trial as target
from .score_context_study import ROOT,write,sha


def main():
    out=ROOT/'reports/external_novelty_trial_probe';out.mkdir(parents=True,exist_ok=True)
    sources={str(p):sha(p) for p in (Path(__file__),Path(engine.__file__),Path(target.__file__),out/'PROTOCOL.md')}
    engine.OUT=out;engine.study.OUT=target.OUT;engine.study.ART=target.ART;engine.study.load_fold=target.load_fold
    engine.main();assert all(sha(p)==h for p,h in sources.items());write(out/'wrapper_audit.json',dict(status='complete',target='external_novelty_trial',sources=sources))


if __name__=='__main__':main()
