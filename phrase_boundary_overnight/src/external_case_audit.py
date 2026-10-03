"""Deterministic case selection and signal/score traces; no model fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import external_audio_trial as base
from .external_audio_candidates import ASAP
from .external_audio_tie_audit import midi_intervals
from .score_context_study import ROOT,read,write,sha
from .evaluation import one_to_one_counts

OUT=ROOT/'reports/external_case_audit'
SELECTED={0:('beethoven_piano_sonatas_03-1','3674aec3b9da281f'),1:('beethoven_piano_sonatas_05-1','13002c407ad4ec88'),2:('beethoven_piano_sonatas_08-1','bc97ccf01bded15c')}


def pairing(predicted,truth):
    candidates=sorted((abs(int(p)-int(t)),int(p),int(t)) for p in predicted for t in truth if abs(p-t)<=1)
    usedp=set();usedt=set();pairs=[]
    for _,p,t in candidates:
        if p not in usedp and t not in usedt:usedp.add(p);usedt.add(t);pairs.append((p,t))
    check=one_to_one_counts(predicted,truth,1);assert len(pairs)==check.tp
    return sorted(pairs),sorted(set(predicted)-usedp),sorted(set(truth)-usedt)


def main():
    OUT.mkdir(parents=True,exist_ok=True);hashes={};data=base.load_data();splits=read(base.OUT/'splits.json')
    sources=[Path(__file__),OUT/'PROTOCOL.md',ROOT/'reports/external_audio_projection_v2/verified_bar_intervals.csv',ROOT/'reports/external_linear_novelty/run_audit.csv']
    regions=pd.read_csv(sources[2]);choices=pd.read_csv(sources[3]);rows=[];traces=[];selection=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    for fold,(pid,key) in SELECTED.items():
        item=data[key];assert item['piece_id']==pid and item['group'] in splits[str(fold)]['test']
        allowed=sorted((v['piece_id'],k) for k,v in base.subset(data,splits[str(fold)]['test']).items() if v['labels'].sum()>=3)
        assert allowed[0]==(pid,key)
        selected_regions=regions[regions.performance==item['performance_id']].sort_values('score_quarter_start');folder=selected_regions.iloc[0].folder;mp=ASAP/folder/'midi_score.mid';events=midi_intervals(mp);sources.append(mp)
        cutsfile=ROOT/f'artifacts/external_linear_interval_probe/E_raw_fold{fold}_{key}.npz';probfile=ROOT/f'artifacts/external_linear_novelty/E_fold{fold}_test.npz';novfile=ROOT/f'artifacts/external_structure_probe/M_fold{fold}_predictions.npz';sources.extend([cutsfile,probfile,novfile])
        with np.load(cutsfile,allow_pickle=False) as z:cuts=z['cuts'].copy()
        with np.load(probfile,allow_pickle=False) as z:prob=z[key].copy()
        with np.load(novfile,allow_pickle=False) as z:nov=z[key].copy()
        mask=item['label_mask']>.5;truth=np.flatnonzero((item['labels']>.5)&mask);pred=cuts[mask[cuts]];pairs,fp,fn=pairing(pred,truth);matched={p for p,t in pairs}
        cases=[('TP',pairs[0][0] if pairs else None),('FP',fp[0] if fp else None),('FN',fn[0] if fn else None)]
        threshold=float(choices[(choices.fold==fold)&(choices.kind=='E')].iloc[0].threshold)
        fig,axes=plt.subplots(3,3,figsize=(13,7),gridspec_kw={'height_ratios':[2,1.6,1],'hspace':.48,'wspace':.3})
        for col,(category,center) in enumerate(cases):
            if center is None:
                for ax in axes[:,col]:ax.set_axis_off()
                axes[0,col].set_title(category+' unavailable');continue
            center=int(center);lo=max(0,center-6);hi=min(len(mask)-1,center+6);q=np.arange(lo,hi+1)
            scope=selected_regions[(selected_regions.score_quarter_start<=center+1e-5)&(selected_regions.score_quarter_end>center+1e-5)]
            at=events[(events[:,0]<center+1)&(events[:,1]>center)];attacks=at[at[:,0]>=center-1e-5]
            prev=events[(events[:,0]<center)&(events[:,1]>center-1)]
            left=item['features'][max(0,center-2):center,28];right=item['features'][center:min(len(mask),center+2),28]
            rms_change=float(right.mean()-left.mean()) if len(left) and len(right) else None
            row=dict(fold=fold,piece_id=pid,record=key,category=category,quarterbeat=center,
                source_mc=int(scope.iloc[0].source_mc) if len(scope) else None,xml_index=int(scope.iloc[0].xml_index) if len(scope) else None,
                probability=float(prob[center]),threshold=threshold,novelty=float(nov[center]),nearest_truth_distance=int(np.min(abs(truth-center))),
                mean_log_rms_after_minus_before=rms_change,active_notes_before=len(prev),active_notes_after=len(at),attacks_in_beat=len(attacks),
                highest_pitch_before=float(prev[:,2].max()) if len(prev) else None,highest_pitch_after=float(at[:,2].max()) if len(at) else None)
            rows.append(row)
            for i in q:traces.append(dict(fold=fold,category=category,beat=int(i),known=bool(mask[i]),label=int(item['labels'][i]),probability=float(prob[i]),novelty=float(nov[i]),log_rms=float(item['features'][i,28])))
            ax=axes[0,col];local=events[(events[:,0]<hi+1)&(events[:,1]>lo)]
            for a,b,pitch in local:ax.plot([max(a,lo),min(b,hi+1)],[pitch,pitch],color='black',lw=1.2,solid_capstyle='butt')
            ax.axvline(center,color='#f59e0b',lw=1.5);ax.set_ylabel('MIDI pitch');ax.set_title(f'{category} at q={center} | DCML mc={row["source_mc"]}');ax.set_xlim(lo,hi)
            ax=axes[1,col];ax.plot(q,prob[q],color='#2563eb',lw=1.5,label='Linear probability');ax.plot(q,nov[q],color='#a855f7',lw=1.3,label='MERT novelty');ax.axhline(threshold,color='#64748b',ls='--',lw=.8);ax.set_ylim(0,1);ax.set_xlim(lo,hi)
            if col==0:ax.legend(loc='upper left',fontsize=7,frameon=False)
            ax=axes[2,col];knownq=q[mask[q]];unknownq=q[~mask[q]];ax.scatter(knownq,np.zeros(len(knownq)),s=13,color='black');ax.scatter(unknownq,np.zeros(len(unknownq)),s=13,facecolors='none',edgecolors='#94a3b8')
            for boundary in pred[(pred>=lo)&(pred<=hi)]:ax.scatter(boundary,0,s=45,color='#55df77' if int(boundary) in matched else '#ef4444',zorder=4)
            for boundary in truth[(truth>=lo)&(truth<=hi)]:ax.plot([boundary,boundary],[-.38,-.15],color='black',lw=2)
            for bar in selected_regions.score_quarter_start.unique():
                if lo<=bar<=hi:ax.axvline(bar,color='black',lw=.6,alpha=.5)
            ax.set_ylim(-.5,.3);ax.set_yticks([]);ax.set_xlabel('Quarter-note beat');ax.set_xlim(lo,hi)
            for ax in axes[:,col]:
                for unknown in unknownq:ax.axvspan(unknown-.5,unknown+.5,color='#e2e8f0',alpha=.35,zorder=-1)
        fig.suptitle(f'Fixed case audit | {pid} | fold {fold}',x=.06,ha='left',fontsize=13)
        fig.text(.06,.02,'Piano roll, not a complete engraved score. Green = matched prediction (+/-1); red = unmatched prediction.\nBlack lower ticks = labels; hollow dots / shaded cells = unknown labels; thin vertical lines = verified bar starts.',fontsize=8,color='#475569')
        fig.subplots_adjust(left=.07,right=.98,top=.87,bottom=.16);fig.savefig(OUT/f'fold{fold}_cases.png',dpi=160,facecolor='white');fig.savefig(OUT/f'fold{fold}_cases.svg',facecolor='white');plt.close(fig)
        selection.append(dict(fold=fold,piece_id=pid,record=key,tp=len(pairs),fp=len(fp),fn=len(fn),selected_cases=[(k,None if q is None else int(q)) for k,q in cases]))
    pd.DataFrame(rows).to_csv(OUT/'case_observations.csv',index=False);pd.DataFrame(traces).to_csv(OUT/'case_traces.csv',index=False);write(OUT/'selection.json',selection)
    for p in sources:hashes[str(p)]=sha(p)
    # The trial contracts already audit transitive data; this artifact records the exact viewed files.
    write(OUT/'source_hashes.json',hashes);write(OUT/'completion_audit.json',dict(status='complete',works=3,cases=len(rows),training_runs=0,new_blind_test=False,selection_deterministic=True,one_to_one_rule_unchanged=True,notation_limitation='Piano roll omits written voices/slurs and tonal interpretation. These are diagnostics, not a new performance estimate.'))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
