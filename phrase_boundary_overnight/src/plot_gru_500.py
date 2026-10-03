"""Actual CNN/GRU histories through step500; no epoch interpolation."""
from pathlib import Path
import hashlib,json
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/gru_500_study'

def main():
    rows=[];hashes={}
    for model,folder,kind in [('CNN','recurrence_duration_study','C3'),('BiGRU','gru_500_study','G')]:
        for fold in (0,1):
            for seed in (42,43):
                p=ROOT/'artifacts'/folder/'checkpoints'/f'{kind}_seed{seed}_fold{fold}'/'latest.pt';hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
                cp=torch.load(p,map_location='cpu',weights_only=False);assert cp['step']>=500
                for h in cp['history']:
                    if h['step']<=500:rows.append(dict(model=model,fold=fold,seed=seed,**h))
    df=pd.DataFrame(rows);df.to_csv(OUT/'history_to500.csv',index=False)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for model,color in [('CNN','#2467a9'),('BiGRU','#cf6b33')]:
        for ax,col,scale in [(axs[0],'loss',1),(axs[1],'macro_f1_tol1',100)]:
            g=df[df.model==model].groupby('step')[col].agg(['mean','min','max'])
            ax.plot(g.index,g['mean']*scale,'o-',color=color,label=model,lw=2,ms=4);ax.fill_between(g.index,g['min']*scale,g['max']*scale,color=color,alpha=.1)
    axs[0].set_title('训练批次loss');axs[1].set_title('原始开发F1 @ ±1拍（未融合/未加M10）')
    axs[1].set_ylabel('F1（%）');axs[0].set_ylabel('加权BCE')
    for ax in axs:ax.axvline(300,ls='--',color='#777',lw=1);ax.set_xlabel('参数更新次数step，不是epoch');ax.legend();ax.grid(axis='y',alpha=.15)
    fig.suptitle('把训练增加到500步：CNN与BiGRU的真实轨迹',fontsize=14)
    fig.text(.5,.01,'每个分支为2开发折×2seed的均值，阴影是范围而非置信区间。虚线表示本次BiGRU续训起点。\nloss是每50步记录的当批值；CNN读取已完成旧续训，BiGRU读取本轮完整恢复续训；两者输入均58维、学习率均0.001。',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.09,1,.94]);fig.savefig(OUT/'learning_to500.png',dpi=180);fig.savefig(OUT/'learning_to500.svg');plt.close(fig)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (OUT/'curve_audit.json').write_text(json.dumps(dict(status='complete',actual_histories=True,checkpoints_unmodified=True,hashes=hashes),ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
