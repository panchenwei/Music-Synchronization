"""Read-only model/result diagnosis: AP and decoded boundary F1 differ."""
import pandas as pd
from .score_context_study import ROOT,read,write,sha


def main():
    out=ROOT/'reports/recurrence_dropout_study'
    contract=read(out/'contract.json');assert all(sha(p)==h for p,h in contract['hashes'].items())
    df=pd.read_csv(ROOT/'artifacts/recurrence_dropout_study/summary.csv')
    cols=['macro_f1_tol1','macro_f1_tol0','raw_ap','macro_precision_tol1','macro_recall_tol1','threshold','best_step']
    a=df[df.kind=='D4'].set_index(['fold','seed']);b=df[df.kind=='C3'].set_index(['fold','seed'])
    assert len(a)==len(b)==4 and not a.index.duplicated().any() and a.index.equals(b.index)
    delta=a[cols]-b[cols];delta.to_csv(out/'metric_tradeoff_deltas.csv')
    audit=pd.read_csv(out/'run_audit.csv');audit['kind']=audit.run_id.str.split('_').str[0]
    gaps=audit.groupby('kind')[['train_f1','gap']].mean()
    write(out/'metric_tradeoff_diagnosis.json',dict(status='complete',means=df.groupby('kind')[cols].mean().to_dict('index'),
          deltas=delta.mean().to_dict(),precision_positive=int((delta.macro_precision_tol1>0).sum()),
          recall_positive=int((delta.macro_recall_tol1>0).sum()),ap_positive=int((delta.raw_ap>0).sum()),
          changed_thresholds=int((delta.threshold!=0).sum()),changed_best_steps=int((delta.best_step!=0).sum()),
          training_diagnosis=gaps.to_dict('index'),source_hashes_unchanged=True,threshold_retuned=False))
    print(delta.to_string());print(gaps.to_string())


if __name__=='__main__':main()
