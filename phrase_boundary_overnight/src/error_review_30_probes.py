"""CPU inference-only probes and source-label audit for the frozen 30 cases."""
from __future__ import annotations
import json
import copy
import time
import numpy as np
import pandas as pd
import torch
from .error_review_30 import ROOT,OUT,ART,CACHE,DCML,digest,dump,plt
from .data import discover_dcml_pieces
from .slice_energy_features import map_starts
from .three_round_round2 import make_model34,assemble_extra
from .models import Normalizer

def main():
    started=time.monotonic();torch.set_num_threads(2)
    cases=pd.read_csv(OUT/'cases.csv');sources=json.loads((OUT/'source_hashes.json').read_text(encoding='utf-8'))
    sources[str(__file__)]=digest(__file__)
    pieces=discover_dcml_pieces(DCML);models={};saved={};rows=[];labels_audit=[];max_error=0.;max_far_grad=0.;numeric_checks=[]
    groups={'meter':[9,10,11],'score_timing':[12,13,14,15],
            'score_pitch_density':list(range(16,25)),'tempo':[0,1,2,5],
            'dynamics':[3,4,6],'hierarchy':list(range(25,31)),'score_all':list(range(9,25))}
    for fold in (0,1):
        path=ART/'checkpoints'/f'B_seed42_fold{fold}'/'best.pt'
        saved[fold]=torch.load(path,map_location='cpu',weights_only=False)
        model=make_model34(42).eval();model.load_state_dict(saved[fold]['model']);models[fold]=model
    for pid,sub in cases.groupby('piece_id'):
        with np.load(CACHE/f'{pid}.npz',allow_pickle=False) as cache:item={k:cache[k] for k in cache.files}
        rebuilt,mask,mapping,*_=map_starts(pieces[pid],len(item['start_labels']))
        np.testing.assert_array_equal(rebuilt,item['start_labels']);np.testing.assert_array_equal(mask,item['start_mask'])
        labels_audit.append(dict(piece_id=pid,labels_equal=True,mask_equal=True,positives=int((rebuilt*mask).sum())))
        for r in sub.to_dict('records'):
            fold=int(r['fold']);model=models[fold];state=saved[fold];norm=Normalizer(state['mean'],state['std']);b=int(r['beat'])
            pi=list(item['performance_ids'].astype(str)).index(r['performance_id'])
            curve=item['curves'][pi:pi+1];score=item['fixed_score'][None,:,:]
            raw=np.concatenate([curve,score,assemble_extra(curve,item['energy'][pi:pi+1],'B')],axis=-1)
            x=torch.from_numpy(norm.apply(raw).astype('float32'))
            with torch.no_grad():base=torch.sigmoid(model(x))[0].numpy()
            max_error=max(max_error,abs(float(base[b])-r['probability']))
            error=abs(float(base[b])-r['probability'])
            if error>=2e-6:
                gpu_probability=None
                if torch.cuda.is_available():
                    gpu_model=copy.deepcopy(model).to('cuda')
                    with torch.no_grad():gpu_probability=float(torch.sigmoid(gpu_model(x.to('cuda')))[0,b])
                    del gpu_model
                numeric_checks.append(dict(case_id=r['case_id'],cpu=float(base[b]),saved=r['probability'],gpu=gpu_probability,cpu_error=error))
                assert error<1e-4 and (gpu_probability is None or abs(gpu_probability-r['probability'])<1e-4),numeric_checks[-1]
            assert (float(base[b])>=r['threshold'])==(r['probability']>=r['threshold'])
            record={'case_id':r['case_id'],'category':r['category'],'reproduced_probability':float(base[b])}
            for name,indices in groups.items():
                changed=x.clone();changed[:,:,indices]=0
                with torch.no_grad():prob=float(torch.sigmoid(model(changed))[0,b])
                record[name+'_delta']=prob-float(base[b])
            # Gradient is with respect to assembled features, not the original waveform.
            z=x.clone().requires_grad_(True);value=model(z)[0,b];value.backward()
            g=z.grad.abs().sum(-1)[0].numpy();outside=np.ones(len(g),bool);outside[max(0,b-2):b+3]=False
            max_far_grad=max(max_far_grad,float(g[outside].max(initial=0)))
            record['outside_5beat_gradient_sum']=float(g[outside].sum())
            local=pd.read_csv(OUT/'contexts'/f"{r['case_id']}_annotations.csv")
            target=int(r['matched_truth']) if r['category']=='TP' else b
            prior=local[(local.unfolded_beat>=target-3)&(local.unfolded_beat<=target)]
            record['source_cadences_prev3']=';'.join(prior.cadence.dropna().astype(str)) if 'cadence' in prior else ''
            exact=local[abs(local.unfolded_beat-target)<1e-6]
            record['source_marker_at_target']=';'.join(exact.phraseend.dropna().astype(str))
            record['source_labels_at_target']=';'.join(exact.label.dropna().astype(str))
            record['nearest_start_quantization']=min(abs(float(m['target_qb'])-target) for m in mapping if not m['first_start'])
            rows.append(record)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'feature_probes.csv',index=False)
    pd.DataFrame(labels_audit).to_csv(OUT/'source_label_rebuild.csv',index=False)
    ev=pd.read_csv(OUT/'all_development_events.csv');site=pd.read_csv(OUT/'boundary_consistency.csv')
    # Fill zero subtype counts in each work before macro averaging.
    stats=[]
    for cat,g in ev.groupby('category'):
        table=g.groupby(['piece_id','subtype']).size().unstack(fill_value=0)
        fractions=table.div(table.sum(axis=1),axis=0)
        for kind in table.columns:stats.append(dict(category=cat,subtype=kind,event_fraction=int(table[kind].sum())/len(g),work_macro_fraction=float(fractions[kind].mean())))
    pd.DataFrame(stats).to_csv(OUT/'subtype_fractions.csv',index=False)
    combined=cases.merge(frame,on=['case_id','category']);combined.to_csv(OUT/'case_evidence.csv',index=False)
    # Per-case summaries contain source evidence, not judgments of musical truth.
    lines=['# 逐例诊断补充','', '置训练均值探针只量化模型敏感性，有分布外风险，不能替代重训消融或证明错误原因。和声标签仅用于审查，不进入模型。','']
    for r in combined.to_dict('records'):
        influences=sorted(((name,r[name+'_delta']) for name in groups),key=lambda x:abs(x[1]),reverse=True)[:3]
        lines += [f"## {r['case_id']}",'',f"源标注中心：{r['source_labels_at_target'] or '无和声标注行'}；前3拍终止式标记：{r['source_cadences_prev3'] or '无'}。",
                  '最大分数敏感性（置均值后减原值）：'+ '；'.join(f'{n}: {v:+.3f}' for n,v in influences)+'。',
                  ('该真句首在同曲多个演奏中仍常漏掉；优先核查可表达的谱面结构线索，不能仅解释成此演奏偶然波动。' if r['category']=='FN' and r['same_truth_hit_fraction']<=.2 else
                   '同曲该句首在多数演奏中可以命中，本例需留意演奏特征与阈值附近的变化。' if r['category']=='FN' and r['same_truth_hit_fraction']>=.8 else
                   '这是相对当前标注的误报；不等于已经证明此位置在音乐上完全不构成任何层级的分界。' if r['category']=='FP' else
                   '命中是正对照，不能由单次命中推断理解了音乐语义。'), '']
    (OUT/'逐例诊断补充.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axs=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    for ax,cat,title in zip(axs[:2],['FN','FP'],['Why did a true start get missed?','How far are false predictions?']):
        table=pd.DataFrame(stats);table=table[table.category==cat]
        names={'below_threshold':'Raw scores below threshold','nms_removed':'Removed by NMS','far_over_3_beats':'>3 beats from nearest label','near_2_3_beats':'2-3 beats from nearest label','matching_competition':'One-to-one competition'}
        y=np.arange(len(table));vals=table.event_fraction.to_numpy()*100
        ax.barh(y,vals,color='#2563eb' if cat=='FN' else '#ef4444');ax.set_yticks(y,[names[k] for k in table.subtype],fontsize=8);ax.invert_yaxis();ax.set_xlim(0,115)
        for yy,v in zip(y,vals):ax.text(v+1,yy,f'{v:.2f}%',va='center',fontsize=9)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Event-weighted percentage')
    counts=[int((site.hit_fraction<=.2).sum()),int(((site.hit_fraction>.2)&(site.hit_fraction<.8)).sum()),int((site.hit_fraction>=.8).sum())]
    axs[2].bar(['<=20%','20-80%','>=80%'],counts,color=['#ef4444','#94a3b8','#4ade80'])
    for i,v in enumerate(counts):axs[2].text(i,v+1,str(v),ha='center')
    axs[2].set_ylim(0,max(counts)+15);axs[2].set_title('Same labeled site across performances',fontsize=10);axs[2].set_xlabel('Fraction of performances with a hit');axs[2].set_ylabel('Distinct labeled sites')
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Frozen CNN diagnosis | 19 development works, 906 performances, 225 distinct starts\nRepeated performances are not independent works; no retraining or threshold change.',fontsize=12)
    fig.savefig(OUT/'diagnosis_overview.png',dpi=140);plt.close(fig)
    assert all(digest(p)==h for p,h in sources.items())
    dump('numeric_crosscheck.json',numeric_checks)
    dump('probe_audit.json',dict(status='complete',cases=len(frame),source_label_works=len(labels_audit),max_cpu_probability_error=max_error,numeric_checks=numeric_checks,
                                max_assembled_gradient_outside_5beats=max_far_grad,source_hashes_unchanged=True,training_runs=0,seconds=time.monotonic()-started))
    dump('source_hashes.json',sources)
    print(pd.DataFrame(stats).to_string(index=False));print(frame.groupby('category')[[name+'_delta' for name in groups]].mean().to_string())
    print(json.dumps(json.loads((OUT/'probe_audit.json').read_text(encoding='utf-8')),indent=2))

if __name__=='__main__':main()
