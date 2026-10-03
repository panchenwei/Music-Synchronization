"""Build result report only from completed GRU500 audit artifacts."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/gru_500_study'


def main():
    audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8'));assert audit['status']=='complete'
    df=pd.read_csv(OUT/'results.csv');checks=pd.read_csv(OUT/'run_audit.csv');assert len(df)==64 and len(checks)==16
    means=df.groupby(['kind','cell','policy']).mean(numeric_only=True)
    lines=['# BiGRU从300到500步：完整结果','',
           '本轮唯一变化是延长BiGRU训练；输入、架构、学习率、数据切分和解码保持固定。2开发折×2seed，不是独立测试。四条轨迹从原300点恢复模型、AdamW、采样器和CPU/CUDA随机数。', '',
           '|模型与评价|选取方式|300步预算F1|500步预算F1|差值（百分点）|','|---|---|---:|---:|---:|']
    for kind,title in [('G','BiGRU'),('E','原CNN/GRU等权融合')]:
        for policy in ('raw','M10'):
            for select,st in [('best','预算内最佳'),('terminal','训练终点')]:
                a=means.loc[(kind,select+'_300',policy),'macro_f1_tol1'];b=means.loc[(kind,select+'_500',policy),'macro_f1_tol1']
                lines.append(f'|{title} / {policy}|{st}|{100*a:.4f}%|{100*b:.4f}%|{100*(b-a):+.4f}|')
    comps=json.loads((OUT/'comparisons.json').read_text(encoding='utf-8'));assert not any(c['passed'] for c in comps)
    lines+=['','## 结论','',
            '本轮未达到事先规定的晋级条件，不替换原模型。只有seed43/fold0在350步刷新了单模型最佳，其他三组仍用原检查点；增加训练上限不能保证泛化改善。原四seed主模型开发F1仍为64.1397%，72%目标未达到。两seed探索均值不能直接与四seed主成绩相减。', '',
            '最佳点小涨还增加了选择机会（6个点变10个点）；所有成绩仍来自反复使用的开发集。exact与AP详见means.csv；raw_ap列在M10行表示调整后概率的AP，不能混同原始概率AP。', '',
            '## 训练与开发差距','', '|状态|训练F1均值|开发F1均值|','|---|---:|---:|']
    for cell in ('terminal_300','terminal_500'):
        g=checks[checks.cell==cell];lines.append(f'|{cell}|{100*g.train_f1.mean():.4f}%|{100*g.dev_f1.mean():.4f}%|')
    lines+=['','训练集F1使用相应开发集选定阈值，是差距诊断而非新评价协议。loss继续降低、开发终点表现下降与过拟合相符，但不能据此解释全部音乐语义识别问题。', '',
            '## 三个诊断问题','',
            '1. 是否真的续训了？是。四条恢复探针通过，训练更新至500；每条采样器曝光计数、历史步数与状态一致。',
            '2. 是否证明学会更多音乐内容？没有。输入和监督未变，单模型仅一组刷新最佳，融合收益不足且增加了选择机会。模型误差互补可影响融合，不能只看单分支分数。',
            '3. 何种结果会改变判断？预注册要求+1.5个百分点、至少3/4同向且3/4在300后有新最佳、exact/AP不下降。本轮不满足，不降低门槛。下一候选是独立固定rho的SAM优化对照，尚需查看其实际运行状态。', '',
            '## 核验与代价','',
            f"4条续训、16检查点状态、32格M10解码完成；保存概率回放最大误差{checks.replay_error.max():.3g}。原源码及输出检查点哈希保持不变。3项原恢复/选择测试通过；图表8检查点只读哈希核验通过。",
            '学习率始终0.001，没有scheduler。500是optimizer step，batch32时累计16000次窗口抽样（含重复），不是500epoch，也不是16000首作品。', '',
            '## 文献复盘','',
            '核读[Cawley与Talbot 2010官方摘要](https://www.jmlr.org/papers/v11/cawley10a.html)：有限验证集上反复选择也可能过拟合，因此必须区分预算内best与独立泛化。另已读[SAM论文第2与3.1节](https://arxiv.org/html/2010.01412v3)，其参数邻域优化机制可作新假设，但图像任务的结果不迁移成本项目成绩。', '',
            '状态：代码implemented；本地训练与复算reproduced；泛化和病因解释preliminary；未运行方案proposed。图见learning_to500.png，逐组证据见results.csv、run_audit.csv、comparisons.json和audit.json。']
    seconds=sum(json.loads(p.read_text(encoding='utf-8'))['seconds'] for p in (ROOT/'artifacts/gru_500_study/runs').glob('*.json'))
    lines.append(f'\n续训及在线验证累计{seconds:.1f}秒，不含数据加载与后续完整审计；不是端到端总耗时。')
    (OUT/'final_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:23]))


if __name__=='__main__':main()
