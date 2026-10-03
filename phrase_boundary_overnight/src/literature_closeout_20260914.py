"""Evidence-index and figures only; never starts model training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/literature_continuation_20260914'
TRAIN_ROUNDS=['harmony_auxiliary','harmony_projection','predictive_information_boundary','current_tabular','tabular_seed_confirmation','external_audio_trial','external_mert_trial','external_novelty_trial','learned_boundary_contrast','external_linear_novelty']
OTHER_ROUNDS=['current_tabular_fusion','current_threshold_diagnostic','external_structure_probe','external_cbm_probe','external_interval_probe','external_adaptive_peak_probe','external_linear_interval_probe','learned_boundary_contrast_probe']


def metric(folder,kind,policy=None):
    df=pd.read_csv(ROOT/'reports'/folder/'means.csv');sel=df.kind==kind
    if policy is not None:sel&=df.policy==policy
    row=df[sel];assert len(row)==1
    fcol='macro_f1_tol1' if 'macro_f1_tol1' in df else 'f1_tol1'
    value=float(row.iloc[0][fcol]);assert 0<=value<=1;return value


def main():
    OUT.mkdir(parents=True,exist_ok=True);evidence=[];sources={}
    for name in TRAIN_ROUNDS+OTHER_ROUNDS:
        folder=ROOT/'reports'/name;p=folder/'completion_audit.json';audit=read(p);assert audit['status']=='complete',(name,audit)
        sources[str(p)]=sha(p)
        for q in (folder/'means.csv',folder/'comparisons.json',folder/'final_report.md'):
            if q.exists():sources[str(q)]=sha(q)
        # This count explicitly covers the ten named training families only.
        fits=int(audit.get('new_training_runs',audit.get('training_runs',0))) if name in TRAIN_ROUNDS else 0
        if 'training_seconds' in audit:seconds=float(audit['training_seconds'])+float(audit.get('audit_seconds',0))
        else:seconds=float(audit.get('seconds',audit.get('audit_seconds',0)))
        evidence.append(dict(round=name,status='complete',new_fits_in_named_family=fits,reported_stage_seconds=seconds,source=str(p)))
    table=pd.DataFrame(evidence);table.to_csv(OUT/'evidence_index.csv',index=False)
    main=[('CNN reference','learned_boundary_contrast','C3'),('Harmony auxiliary','harmony_auxiliary','G'),('Gradient projection','harmony_projection','G'),('Predictive information','predictive_information_boundary','G'),('Shallow tree','current_tabular','H0'),('CNN + tree (50/50)','current_tabular_fusion','T0'),('Learned left/right contrast','learned_boundary_contrast','G')]
    external=[('Score CNN','external_audio_trial','S'),('+ basic audio','external_audio_trial','E'),('+ CQT','external_audio_trial','F'),('+ MERT embedding','external_mert_trial','E'),('MERT novelty only','external_structure_probe','M'),('Novelty + CNN','external_novelty_trial','E'),('MERT global CBM','external_cbm_probe','M'),('Novelty + interval prior','external_interval_probe','MP'),('Adaptive defaults','external_adaptive_peak_probe','A'),('Novelty + linear head','external_linear_novelty','E'),('Linear + interval prior','external_linear_interval_probe','E')]
    rows=[]
    for task,items in [('Chopin development',main),('External audio development',external)]:
        for label,folder,kind in items:
            policy='M10' if task.startswith('Chopin') else ('interval' if folder=='external_linear_interval_probe' else None)
            val=metric(folder,kind,policy);rows.append(dict(task=task,label=label,f1=val,round=folder,kind=kind,policy=policy or 'raw'))
            sources[str(ROOT/'reports'/folder/'means.csv')]=sha(ROOT/'reports'/folder/'means.csv')
    points=pd.DataFrame(rows);points.to_csv(OUT/'figure_data.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(15,7.5),gridspec_kw={'wspace':.7})
    for ax,task in zip(axes,points.task.unique()):
        subset=points[points.task==task];values=subset.f1.to_numpy()*100;labels=subset.label.tolist()
        colors=['#2563eb' if label in ('CNN reference','MERT novelty only') else '#64748b' for label in labels]
        ax.barh(np.arange(len(values)),values,color=colors,height=.68);ax.invert_yaxis();ax.set_yticks(np.arange(len(values)),labels)
        ax.set_xlim(0,70 if task.startswith('Chopin') else 30);ax.set_xlabel('Event F1 (%) at +/-1 quarter-note beat');ax.xaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
        for i,v in enumerate(values):ax.text(v+.4,i,f'{v:.2f}',va='center',fontsize=9)
        ax.set_title(task+'\n'+('Same 2 folds x 2 seeds; fixed M10' if task.startswith('Chopin') else '21 movements; source-grouped 3 folds'),loc='left',fontsize=12,pad=15)
    fig.suptitle('Literature-guided experiments: improvements must survive controls',x=.05,ha='left',fontsize=16)
    fig.text(.05,.03,'Panels are DIFFERENT tasks; do not compare scores across panels. Reused development sets, not new blind tests.\nAuxiliary runs have a documented RNG caveat; matched controls and full results are in the linked reports.',fontsize=9,color='#475569')
    fig.subplots_adjust(left=.18,right=.96,top=.86,bottom=.15)
    fig.savefig(OUT/'experiment_comparison.png',dpi=180,facecolor='white');fig.savefig(OUT/'experiment_comparison.svg',facecolor='white');plt.close(fig)
    sources[str(Path(__file__))]=sha(Path(__file__));assert all(sha(p)==h for p,h in sources.items());write(OUT/'evidence_hashes.json',sources)
    write(OUT/'completion_audit.json',dict(status='complete',named_training_families=len(TRAIN_ROUNDS),fits_in_named_families=int(table.new_fits_in_named_family.sum()),other_audited_rounds=len(OTHER_ROUNDS),figure_points=len(points),new_training_this_builder=0,sources_unchanged=True,reported_stage_seconds=float(table.reported_stage_seconds.sum()),cost_caution='Sum of the listed stage counters, not elapsed wall time; excludes downloads, feature preparation, some diagnostic probes, historical earlier rounds and agent reasoning.'))
    print(table.to_string(index=False));print(read(OUT/'completion_audit.json'))


if __name__=='__main__':main()
