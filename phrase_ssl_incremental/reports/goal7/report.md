# Goal 7：25维安全输入下的小BiGRU与Transformer对照

日期：2026-10-13。代码、三次训练与独立复算完成（implemented / reproduced）。结论是`preliminary development diagnostic`。

## 试了什么

在Goal 6完全相同的30/8首训练/验证、25维可见音乐内容、每曲2次演奏、seed、600步、采样、损失、阈值和解码下，只把71,029参数Transformer换成单层双向GRU。主体为6005参数；为保持相同mask/missing接口增加两个58维token，总计6121参数。没有读取Goal 5测试集。

## 结果

| 模型 | Precision | Recall | F1@±1 | Exact F1 | AP |
|---|---:|---:|---:|---:|---:|
| 25维Transformer（Goal 6 Scratch） | 40.41% | 68.48% | **49.15%** | 38.69% | **39.06%** |
| 25维小BiGRU | 41.09% | 66.32% | **48.57%** | 38.11% | **35.45%** |
| BiGRU−Transformer | +0.68pp | −2.16pp | **−0.58pp** | −0.58pp | **−3.60pp** |

BiGRU的F1差值为−2.39、−1.70、+2.35个百分点，只有1/3 seed上升；AP三个seed全部下降（−3.20、−3.03、−4.57个百分点）。未达到预注册门槛，不晋级。

## 三个诊断问题

1. **架构干预是否生效？** 是。模型从Transformer变为整曲变长BiGRU，参数从71,029降至6,121；输入策略、作品、曝光量和解码保持一致。
2. **差异是否只来自阈值？** 不是。F1接近可能被阈值部分补偿，但阈值无关的AP在三个seed都更低，说明BiGRU的概率排序没有更好。
3. **什么会推翻解释，下一步是什么？** 若在新终审作品或结构化输入上小BiGRU稳定胜出，才能说明当前验证误判了架构。现有证据不支持继续堆GRU/TCN/LSTM层数；下一轮唯一变化应转为相对结构表示，例如从局部前后文生成无标签自相似/novelty线索，再与25维Transformer做同条件对照。

## 文献复盘

- [Melodic Phrase Segmentation By Deep Neural Networks](https://arxiv.org/abs/1811.05688)把稀疏标签视为主要问题，并报告CNN-CRF优于若干CNN/BiLSTM替代；这说明序列约束和标签工程可能比单纯换循环网络重要，但其任务是旋律性符号序列，不能直接当作本项目结果。
- [Self-Similarity-Based and Novelty-based Loss for Music Structure Analysis](https://arxiv.org/abs/2309.02243)把同曲相对自相似和novelty直接写入训练目标，并发现联合SSM与novelty损失优于单独任一损失。它做的是较高层音频段落边界，不是Chopin乐句句首，但“边界依赖同曲相对变化”的机制可低成本迁移测试。
- [Symbolic Segmentation: A Corpus-Based Analysis of Melodic Phrases](https://www.projects.science.uu.nl/music/rodriguez/rodriguezCMMR13.pdf)在民歌语料中发现局部时值不连续比音高不连续更常见；本项目已有时值/休止统计，但逐拍聚合未必保留跨拍相对模式。

因此下一轮不再换更深网络，而测试多尺度前后文差异/自相似novelty是否补上模型目前看不到的“这一拍相对本曲上下文发生了什么变化”。

## 核验

- 三份checkpoint SHA-256匹配。
- 从逐演奏CSV重新按作品宏平均，最大差`1.11e-16`。
- 源访问114条，测试作品和test label mapping均不存在。
- 全项目测试在修正token参数计数后通过。
