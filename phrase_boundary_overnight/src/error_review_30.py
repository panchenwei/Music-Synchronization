"""Read-only diagnosis of frozen development outputs; writes only a new report folder."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evaluation import one_to_one_counts
from .phase2_models import nms_probabilities
from .local_context_study import max_match_count
from .data import discover_dcml_pieces, choose_measure_path, to_float
from .slice_energy_features import voiced_events
from .slice_energy_study import DCML

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/error_review_30'
ART=ROOT/'artifacts/three_round_study/round2'
CACHE=ROOT/'artifacts/slice_energy_study/cache'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rank(value):
    return hashlib.sha256(('error-review-30-v1|'+value).encode()).hexdigest()

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def matches(pred,truth,tol=1):
    usedp=set();usedt=set();pairs=[]
    for _,p,t in sorted((abs(int(p)-int(t)),int(p),int(t)) for p in pred for t in truth if abs(p-t)<=tol):
        if p not in usedp and t not in usedt:
            pairs.append((p,t));usedp.add(p);usedt.add(t)
    return pairs,sorted(set(map(int,pred))-usedp),sorted(set(map(int,truth))-usedt)

def decode(prob,label,valid,threshold):
    nms=nms_probabilities(prob)
    pred=np.flatnonzero((nms>=threshold)&valid)
    truth=np.flatnonzero((label>.5)&valid)
    pairs,fp,fn=matches(pred,truth)
    assert len(pairs)==one_to_one_counts(pred,truth,1).tp
    return dict(prob=prob,nms=nms,pred=pred,truth=truth,pairs=pairs,fp=fp,fn=fn,threshold=threshold,valid=valid)

def subtype(d,category,beat):
    if category=='FN':
        idx=np.arange(max(0,beat-1),min(len(d['prob']),beat+2));idx=idx[d['valid'][idx]]
        if d['prob'][idx].max()<d['threshold']:return 'below_threshold'
        if d['nms'][idx].max()<d['threshold']:return 'nms_removed'
        return 'matching_competition'
    if category=='FP':
        distance=int(np.min(abs(d['truth']-beat)))
        return 'matching_competition' if distance<=1 else 'near_2_3_beats' if distance<=3 else 'far_over_3_beats'
    return 'exact' if (beat,beat) in d['pairs'] else 'within_1_beat'

def per_event(d,pid,perf,fold,category,beat,truthbeat=None):
    nearest=int(d['truth'][np.argmin(abs(d['truth']-beat))])
    return dict(piece_id=pid,performance_id=perf,fold=fold,category=category,beat=int(beat),
                matched_truth=truthbeat,nearest_truth=nearest,distance_to_truth=abs(nearest-beat),
                probability=float(d['prob'][beat]),threshold=d['threshold'],
                subtype=subtype(d,category,beat))

def note_context(piece,n):
    events=voiced_events(piece,n)
    measures=pd.read_csv(piece.measures_path,sep='\t')
    traversal,_,_=choose_measure_path(measures,n)
    harm=pd.read_csv(piece.harmony_path,sep='\t')
    folded={int(r.mc):to_float(r.quarterbeats) for r in measures.itertuples()}
    expanded=[]
    for block in traversal:
        for r in harm[harm.mc==int(block['mc'])].to_dict('records'):
            r.update(unfolded_beat=float(block['start_qb']+to_float(r['quarterbeats'])-folded[int(block['mc'])]),visit=block['visit'])
            expanded.append(r)
    return events,traversal,pd.DataFrame(expanded)

def draw_case(row,item,d,events,traversal,path):
    b=int(row['beat']);n=len(d['prob']);lo=max(0,b-12);hi=min(n-1,b+12);x=np.arange(lo,hi+1)
    pi=list(item['performance_ids'].astype(str)).index(row['performance_id'])
    curves=item['curves'][pi];fig,axs=plt.subplots(5,1,figsize=(14,10),sharex=True,gridspec_kw={'height_ratios':[2,1,1,1,1.2]},layout='constrained')
    for onset,duration,pitch,staff,voice in events:
        if onset<=hi and onset+duration>=lo:
            axs[0].plot([onset,onset+duration],[pitch,pitch],color='#111827' if (staff,voice)==(1,1) else '#9ca3af',lw=2)
    axs[0].set_ylabel('Source notes\nMIDI pitch')
    axs[0].text(.01,.97,'Black: staff 1 / voice 1; grey: other voices. Symbolic notes, not original engraving.',transform=axs[0].transAxes,fontsize=8,va='top',bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    axs[1].plot(x,np.exp(curves[x,0]),color='#2563eb');axs[1].set_ylabel('Tempo\nBPM')
    axs[2].plot(x,curves[x,3],color='#7c3aed');axs[2].set_ylabel('Dynamics\nsource units')
    for j,scale in enumerate((1,3,6,12,24)):
        axs[3].step(x,np.exp(item['energy'][pi,x,j]),where='post',label=f'{scale} beats',lw=1)
    axs[3].legend(ncol=5,fontsize=8,loc='upper right');axs[3].set_ylabel('RMS-std\n/ median tempo')
    axs[4].plot(x,d['prob'][x],color='#111827');axs[4].axhline(d['threshold'],color='#f59e0b',ls='--',label='Frozen threshold')
    axs[4].scatter(x,np.full(len(x),-.08),s=9,color='black')
    paired={p for p,t in d['pairs']}
    for p in d['pred']:
        if lo<=p<=hi:axs[4].scatter(p,d['prob'][p],color='#4ade80' if p in paired else '#ef4444',edgecolor='#166534' if p in paired else '#991b1b',s=65,zorder=5)
    for t in d['truth']:
        if lo<=t<=hi:axs[4].plot([t,t],[-.01,.08],color='black',lw=2)
    for t in d['fn']:
        if lo<=t<=hi:axs[4].scatter(t,-.08,facecolors='white',edgecolors='black',s=55,zorder=6)
    for a in axs:
        a.axvspan(max(lo,b-2),min(hi,b+2),color='#dbeafe',alpha=.45,zorder=-2)
        a.axvline(b,color='#0f172a',lw=.8,ls=':')
        for m in traversal:
            if lo<=m['start_qb']<=hi:a.axvline(m['start_qb'],color='black',lw=.4,alpha=.3)
        a.set_xlim(lo-.3,hi+.3);a.spines[['top','right']].set_visible(False)
    for m in traversal:
        if lo<=m['start_qb']<=hi:axs[0].text(m['start_qb'],1.01,f"mc{m['mc']}v{m['visit']}",transform=axs[0].get_xaxis_transform(),fontsize=7)
    axs[4].set_ylim(-.15,1.04);axs[4].set_ylabel('Boundary\nscore');axs[4].set_xlabel('Unfolded beat index (0-based); thin vertical lines = source measure starts')
    fig.suptitle(f"{row['case_id']} | {row['category']} | {row['piece_id']} | {row['performance_id']} | beat {b}\n{row['subtype']} | green: one-to-one hit within 1 beat; red: FP; hollow: FN",fontsize=12)
    fig.savefig(path,dpi=125);plt.close(fig)

def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True);(OUT/'contexts').mkdir(exist_ok=True)
    sources={};track=lambda p:sources.setdefault(str(p),digest(p))
    track(Path(__file__));track(OUT/'PROTOCOL.md')
    splits=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv');track(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv')
    metadata=discover_dcml_pieces(DCML);data={};decoded={};events=[];summaries=[];maxmatch_differences=0
    for fold in (0,1):
        ids=sorted(splits[(splits.fold==fold)&(splits.split=='validation')].piece_id)
        assert not set(ids)&set(splits[(splits.fold==fold)&(splits.split!='validation')].piece_id)
        for pid in ids:
            p=CACHE/f'{pid}.npz';track(p)
            with np.load(p,allow_pickle=False) as source:data[pid]={k:source[k] for k in source.files}
        for seed in (42,43):
            run=f'B_seed{seed}_fold{fold}';mp=ART/'metrics'/f'{run}.json';pp=ART/'metrics'/f'{run}_predictions.csv.gz';track(mp);track(pp)
            track(ART/'checkpoints'/run/'best.pt')
            saved=json.loads(mp.read_text());threshold=float(saved['threshold']);frame=pd.read_csv(pp)
            assert set(frame.piece_id)==set(ids)
            for (pid,perf),g in frame.groupby(['piece_id','performance_id']):
                g=g.sort_values('beat');item=data[pid];n=len(item['start_labels'])
                np.testing.assert_array_equal(g.beat.to_numpy(),np.arange(n))
                np.testing.assert_array_equal(g.label.to_numpy(),item['start_labels'])
                np.testing.assert_array_equal(g.valid.to_numpy(),item['start_mask'])
                prob=g.probability.to_numpy();assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
                d=decode(prob,item['start_labels'],item['start_mask'].astype(bool),threshold);decoded[(seed,pid,perf)]=d
                maxmatch_differences+=int(len(d['pairs'])!=max_match_count(d['pred'],d['truth'],1))
                if seed==42:
                    summaries.append(dict(piece_id=pid,performance_id=perf,fold=fold,tp=len(d['pairs']),fp=len(d['fp']),fn=len(d['fn']),f1=2*len(d['pairs'])/max(1,len(d['pred'])+len(d['truth']))))
                    for p,t in d['pairs']:events.append(per_event(d,pid,perf,fold,'TP',p,t))
                    for p in d['fp']:events.append(per_event(d,pid,perf,fold,'FP',p))
                    for t in d['fn']:events.append(per_event(d,pid,perf,fold,'FN',t))
            if seed==42:
                check=pd.DataFrame([r for r in summaries if r['fold']==fold]).groupby('piece_id').f1.mean().mean()
                assert abs(check-saved['macro_f1_tol1'])<1e-10,(check,saved['macro_f1_tol1'])
    ev=pd.DataFrame(events);sf=pd.DataFrame(summaries)
    ev.to_csv(OUT/'all_development_events.csv',index=False);sf.to_csv(OUT/'performance_counts.csv',index=False)
    subtype_counts=ev.groupby(['category','subtype']).size().rename('events').reset_index();subtype_counts.to_csv(OUT/'event_subtypes.csv',index=False)
    perwork=ev.groupby(['piece_id','category','subtype']).size().rename('count').reset_index()
    perwork['within_category_fraction']=perwork['count']/perwork.groupby(['piece_id','category'])['count'].transform('sum')
    perwork.to_csv(OUT/'work_subtypes.csv',index=False)
    # Deterministic case selection from one non-performance-selected representative per work.
    representatives={pid:sorted(item['performance_ids'].astype(str))[len(item['performance_ids'])//2] for pid,item in data.items()}
    pool=ev[[r.performance_id==representatives[r.piece_id] for r in ev.itertuples()]].copy()
    selected=[];used=set();work_load={}
    for category in ('FP','FN','TP'):
        candidates=pool[pool.category==category]
        works=sorted(candidates.piece_id.unique(),key=lambda pid:(work_load.get(pid,0),rank(category+'|'+pid)))
        for pid in works:
            rows=candidates[candidates.piece_id==pid].to_dict('records')
            rows=sorted(rows,key=lambda r:rank(f"{category}|{pid}|{r['beat']}"))
            row=next((r for r in rows if (pid,r['beat']) not in used),None)
            if row is None:continue
            row['case_id']=f'{category}{1+sum(r["category"]==category for r in selected):02d}'
            selected.append(row);used.add((pid,row['beat']));work_load[pid]=work_load.get(pid,0)+1
            if sum(r['category']==category for r in selected)==10:break
    assert len(selected)==30 and len(used)==30
    selected_context={};all_site=[]
    for pid,item in data.items():
        perfids=item['performance_ids'].astype(str)
        for t in np.flatnonzero((item['start_labels']>.5)&(item['start_mask']>.5)):
            flags=[any(tt==t for _,tt in decoded[(42,pid,p)]['pairs']) for p in perfids]
            all_site.append(dict(piece_id=pid,truth_beat=int(t),performances=len(perfids),hit_fraction=float(np.mean(flags))))
    pd.DataFrame(all_site).to_csv(OUT/'boundary_consistency.csv',index=False)
    for row in selected:
        pid=row['piece_id'];perf=row['performance_id'];b=int(row['beat']);item=data[pid];n=len(item['start_labels']);d=decoded[(42,pid,perf)];other=decoded[(43,pid,perf)]
        pi=list(item['performance_ids'].astype(str)).index(perf);curves=item['curves'][pi]
        if pid not in selected_context:
            piece=metadata[pid]
            for p in (piece.notes_path,piece.measures_path,piece.harmony_path):track(p)
            selected_context[pid]=note_context(piece,n)
        note_events,traversal,harm=selected_context[pid]
        lo=max(0,b-12);hi=min(n-1,b+12);idx=np.arange(lo,hi+1)
        row['seed43_probability']=float(other['prob'][b]);row['seed43_threshold']=other['threshold']
        row['seed43_has_prediction_near_center']=any(abs(p-b)<=1 for p in other['pred'])
        target=int(row['matched_truth']) if row['category']=='TP' else b
        row['seed43_target_hit']=any(t==target for p,t in other['pairs']) if row['category']!='FP' else None
        row['same_site_prediction_fraction']=float(np.mean([any(abs(p-b)<=1 for p in decoded[(42,pid,pidperf)]['pred']) for pidperf in item['performance_ids'].astype(str)]))
        if row['category']!='FP':row['same_truth_hit_fraction']=float(np.mean([any(t==target for p,t in decoded[(42,pid,pidperf)]['pairs']) for pidperf in item['performance_ids'].astype(str)]))
        oldends=np.flatnonzero(item['labels']>.5);row['nearest_old_end_distance']=int(np.min(abs(oldends-b))) if len(oldends) else None
        row['tempo_bpm']=float(np.exp(curves[b,0]));row['tempo_clip_nearby']=bool((np.exp(curves[max(0,b-2):min(n,b+3),0])>=399.9).any())
        row['time_missing_nearby']=bool((curves[max(0,b-2):min(n,b+3),7]<.5).any());row['dynamics_missing_nearby']=bool((curves[max(0,b-2):min(n,b+3),8]<.5).any())
        row['global_rest_beats']=float(item['fixed_score'][b,4]*8);row['old_note_end_gap_beats']=float(item['old_score'][b,4]*8)
        row['sounding_notes_before']=sum(o<b and o+dur>=b-1e-9 and dur>0 for o,dur,*_ in note_events)
        row['source_measure_number']=float(item['measure_number'][b]);row['source_beat_number']=float(item['beat_number'][b])
        block=[m for m in traversal if m['start_qb']<=b][-1];row['source_mc']=int(block['mc']);row['source_visit']=int(block['visit'])
        row['distance_to_24_block_edge']=min(b%24,24-b%24)
        local_harm=harm[(harm.unfolded_beat>=lo)&(harm.unfolded_beat<=hi)].copy();local_harm.to_csv(OUT/'contexts'/f"{row['case_id']}_annotations.csv",index=False)
        local_notes=[e for e in note_events if e[0]<=hi and e[0]+e[1]>=lo]
        pd.DataFrame(local_notes,columns=['onset','duration','midi','staff','voice']).to_csv(OUT/'contexts'/f"{row['case_id']}_notes.csv",index=False)
        pd.DataFrame(dict(beat=idx,probability=d['prob'][idx],nms=d['nms'][idx],label=item['start_labels'][idx],mask=item['start_mask'][idx],tempo=np.exp(curves[idx,0]),dynamics=curves[idx,3])).to_csv(OUT/'contexts'/f"{row['case_id']}_signals.csv",index=False)
        draw_case(row,item,d,note_events,traversal,OUT/'figures'/f"{row['case_id']}.png")
    cases=pd.DataFrame(selected);cases.to_csv(OUT/'cases.csv',index=False)
    review=cases[['case_id','category','piece_id','performance_id','beat','source_mc','source_visit','subtype']].copy()
    for name in ['musical_boundary_is_clear','annotation_issue','missing_musical_cue','notes','reviewer']:review[name]=''
    review.to_csv(OUT/'human_review_template.csv',index=False)
    # Descriptive population and work-macro counts, not inference about independent performances.
    site=pd.DataFrame(all_site);counts={str(k):int(v) for k,v in ev.category.value_counts().items()}
    stats=dict(works=len(data),performances=len(sf),counts=counts,selected_works=int(cases.piece_id.nunique()),cases=30,
               macro_f1_seed42_equal_folds=float(sf.groupby(['fold','piece_id']).f1.mean().groupby('fold').mean().mean()),
               unique_truth_sites=len(site),truth_sites_hit_at_most_20pct=int((site.hit_fraction<=.2).sum()),truth_sites_hit_at_least_80pct=int((site.hit_fraction>=.8).sum()),
               max_matching_count_disagreements=maxmatch_differences,seconds=time.monotonic()-started)
    dump('summary.json',stats)
    lines=['# 30个位置图册：冻结句首模型错误审查','', '每类10例；不是按错误率抽样，不能用10/30当真实误报率。原始谱面事件时间轴不是完整五线谱；音乐层级和听觉判断均待复核。','',
           '绿色=精确或±1拍一对一命中；红色=误报；空心=漏报；浅蓝区=中心前后2拍。所有拍索引为展开后0起始，mc为源文件小节计数，visit为反复访问次数。','']
    for r in selected:
        lines += [f"## {r['case_id']} — {r['piece_id']} / {r['performance_id']}",'',f"- 类别：{r['category']}；中心拍：{r['beat']}；源mc={r['source_mc']}，visit={r['source_visit']}；原beat表小节={r['source_measure_number']}，拍={r['source_beat_number']}。",
        f"- 原分数={r['probability']:.3f}，冻结阈值={r['threshold']:.2f}；现象分类={r['subtype']}；最近标注句首距离={r['distance_to_truth']}拍，最近旧句末距离={r['nearest_old_end_distance']}拍。",
        f"- seed43同拍分数={r['seed43_probability']:.3f}（其阈值={r['seed43_threshold']:.2f}）；同曲所有演奏在此位置±1拍有预测的比例={r['same_site_prediction_fraction']:.1%}。",
        f"- 全局谱面休止={r['global_rest_beats']:.2f}拍，旧音符结束间隔={r['old_note_end_gap_beats']:.2f}拍；邻近速度限幅={r['tempo_clip_nearby']}，时间缺失={r['time_missing_nearby']}，力度缺失={r['dynamics_missing_nearby']}。",
        '- 音乐判断：待复核，不以自动规则替代人工判断。', '',f"![{r['case_id']}](<{(OUT/'figures'/(r['case_id']+'.png')).as_posix()}>)",'',
        f"[局部音符事件](contexts/{r['case_id']}_notes.csv) · [原标注上下文](contexts/{r['case_id']}_annotations.csv) · [曲线与分数](contexts/{r['case_id']}_signals.csv)",'']
    (OUT/'30个位置图册.md').write_text('\n'.join(lines),encoding='utf-8')
    dump('source_hashes.json',sources)
    assert all(digest(p)==h for p,h in sources.items())
    dump('audit.json',dict(status='complete',source_files=len(sources),source_hashes_unchanged=True,counts_reproduced=True,
                          validation_only=True,training_runs=0,threshold_changes=0,label_changes=0,
                          category_counts=cases.category.value_counts().to_dict(),unique_cases=len(used),figures=len(list((OUT/'figures').glob('*.png'))),
                          musical_manual_review='pending',**stats))
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
