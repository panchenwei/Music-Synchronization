"""Read-only audit of local trusted checkpoints; curves are not invented epochs."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .recurrence_schedule_training import learning_rate

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research_continuation_20260915'

def main():
    OUT.mkdir(exist_ok=True);rows=[];audit=[]
    for label,folder,kind in [('CNN','recurrence_depth_study','C3'),('BiGRU','current_gru_study','G')]:
        for f in (0,1):
            for s in (42,43):
                p=ROOT/'artifacts'/folder/'checkpoints'/f'{kind}_seed{s}_fold{f}'/'latest.pt'
                before=hashlib.sha256(p.read_bytes()).hexdigest();cp=torch.load(p,map_location='cpu',weights_only=False)
                assert cp['step']==300 and cp['optimizer']['param_groups'][0]['lr']==.001
                for h in cp['history']:rows.append(dict(model=label,fold=f,seed=s,**h))
                assert hashlib.sha256(p.read_bytes()).hexdigest()==before
                audit.append(dict(path=str(p),sha256=before,last_step=cp['step'],last_optimizer_lr=.001))
    df=pd.DataFrame(rows);df.to_csv(OUT/'actual_training_history.csv',index=False)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(15,4.8));colors={'CNN':'#2369ad','BiGRU':'#cf6b33'}
    steps=np.arange(1,301);axs[0].plot(steps,np.full(300,.001),color='#2369ad',lw=2.5,label='当前主模型：固定0.001')
    axs[0].plot(steps,[learning_rate('K3',int(s)-1) for s in steps],color='#777777',lw=1.5,ls='--',label='历史余弦实验（未采用）')
    axs[0].set_ylim(0,.00115);axs[0].ticklabel_format(axis='y',style='sci',scilimits=(-3,-3));axs[0].set_ylabel('优化器设定的学习率');axs[0].set_title('(a) 学习率不是loss');axs[0].legend(fontsize=8)
    for label in colors:
        part=df[df.model==label]
        for ax,col,scale in [(axs[1],'loss',1),(axs[2],'macro_f1_tol1',100)]:
            stat=part.groupby('step')[col].agg(['mean','min','max'])
            ax.plot(stat.index,stat['mean']*scale,'o-',color=colors[label],label=label,lw=2,ms=4)
            ax.fill_between(stat.index,stat['min']*scale,stat['max']*scale,color=colors[label],alpha=.10)
    axs[1].set_title('(b) 实际训练批次loss');axs[1].set_ylabel('加权二分类损失');axs[1].legend(fontsize=8)
    axs[2].set_title('(c) 实际开发F1（未加M10/融合）');axs[2].set_ylabel('F1 @ ±1拍（%）');axs[2].legend(fontsize=8)
    for ax in axs:ax.set_xlabel('参数更新次数 step（不是epoch）');ax.grid(axis='y',alpha=.2)
    fig.suptitle('当前模型怎样学习：步长恒定，但loss与开发表现会变化',fontsize=15,y=.99)
    fig.text(.5,.01,'来源：2个开发折 × seed42/43；实线为4组均值，阴影为范围（非置信区间）。loss每50步只记录当批值，不是整轮平均。\n学习率按源码重建并核对8个末次检查点；灰色为已完成的历史对照。不同分支独立训练，图中没有合成epoch。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.10,1,.94]);fig.savefig(OUT/'actual_learning_curves.png',dpi=180);fig.savefig(OUT/'actual_learning_curves.svg');plt.close(fig)
    (OUT/'learning_curve_audit.json').write_text(json.dumps(dict(status='reproduced',checkpoint_count=8,source_engine=str(ROOT/'src/score_context_study.py'),lr_reconstructed_from_code=True,history_actual=True,checkpoints=audit),ensure_ascii=False,indent=2),encoding='utf-8')
    print(df.groupby(['model','step'])[['loss','macro_f1_tol1']].mean().to_string())

if __name__=='__main__':main()
