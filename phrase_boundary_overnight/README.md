# 乐句边界预测

这是当前乐句划分研究的可读代码快照。仓库保留模型、预处理、训练、推理、评测、划分和关键结果；不包含原始音频、完整数据集、特征缓存、模型权重或海量日志。

## 任务

- **输入是什么：** 每个 quarterbeat（一个四分音符时值）一行，共58维：9维对齐演奏的速度/力度曲线、16维谱面线索、9维速度层级表示、24维有序音型重复特征。它不是原始频谱；速度层级也不等于声波能量。
- **预测的是句首还是句尾：** 预测 DCML `{` 标记的**新乐句开始**。句尾、终止式和下一句开始不能互换。
- **标签来自哪里：** DCML Chopin Mazurkas 的乐句标注。首个已知句首不计入正例；未知前后缀和量化歧义位置由 `label_mask` 排除，不当作负例。
- **预测单位：** quarterbeat 网格上的逐位置概率，随后解码为离散句首事件。

## 当前采用的方案

当前候选是训练时连续时间遮挡的四种子小型 BiGRU 集成：

```text
58维输入
  → Linear(58, 32)
  → 单层双向GRU（每方向14维）
  → Dropout(0.2)
  → LayerNorm(28)
  → Linear(28, 1)
  → 每个quarterbeat的句首概率
  → 4个seed等权平均
  → 固定B10句距解码
```

每个模型6005参数。训练时对每条输入连续4个有效位置置为训练归一化后的零值；推理时不遮挡。64位置窗口、batch 8、AdamW、学习率0.001、权重衰减0.0001、正类权重10、600个optimizer step。这里的600不是600个epoch。

- 模型定义：[`src/current_gru_models.py`](src/current_gru_models.py)、[`src/mentor_sequence_models.py`](src/mentor_sequence_models.py)
- Dataset / Sampler：[`src/run_recurrence_depth_study.py`](src/run_recurrence_depth_study.py)、[`src/mentor_sequence_models.py`](src/mentor_sequence_models.py)
- 标签与基础特征：[`src/slice_energy_features.py`](src/slice_energy_features.py)、[`src/phase3_features.py`](src/phase3_features.py)、[`src/motif_recurrence_features.py`](src/motif_recurrence_features.py)
- 实际数据重建：[`score_only_export_training.py`](reports/mentor_continuation_20260916/score_only_export_training.py)
- 时间遮挡：[`temporal_mask_sampler.py`](reports/mentor_continuation_20260916/temporal_mask_sampler.py)
- 训练及四种子复核：[`temporal_mask_training.py`](reports/mentor_continuation_20260916/temporal_mask_training.py)、[`temporal_mask_replication.py`](reports/mentor_continuation_20260916/temporal_mask_replication.py)
- 推理：[`infer_masked_candidate.py`](reports/mentor_continuation_20260916/infer_masked_candidate.py)
- 解码与评测：[`src/run_halo_decoder_composition.py`](src/run_halo_decoder_composition.py)、[`src/evaluation.py`](src/evaluation.py)、[`locked_holdout_test.py`](reports/mentor_continuation_20260916/locked_holdout_test.py)

此前比较过小CNN、TCN、LSTM、Transformer、多模态/谱面分支和若干损失及特征方案。当前保留该BiGRU候选，是因为时间遮挡在内部开发池的两轮、多seed对照中通过预定门槛；这不表示Transformer在一般音乐任务上无效。

## 运行入口

在 `phrase_boundary_overnight` 自己的虚拟环境中运行。PyTorch需按本机CUDA版本安装；其余依赖见 [`requirements-base.txt`](requirements-base.txt)。

```powershell
Set-Location 'C:\path\to\Music-Synchronization\phrase_boundary_overnight'

# 标签、谱面与有序重复特征的关键生成/核验入口
& '.\.venv\Scripts\python.exe' '.\reports\mentor_continuation_20260916\score_only_features_export.py'

# 原始两seed时间遮挡实验，再做seed44/45复核和四seed集成
& '.\.venv\Scripts\python.exe' '.\reports\mentor_continuation_20260916\temporal_mask_training.py'
& '.\.venv\Scripts\python.exe' '.\reports\mentor_continuation_20260916\temporal_mask_replication.py'

# 对一个已有作品推理；输出目录必须尚不存在
& '.\.venv\Scripts\python.exe' '.\reports\mentor_continuation_20260916\infer_masked_candidate.py' `
  --piece-id chopin_op06_no1 --fold 0 --out '.\demo_output'

# 冻结历史留出检查：必须先prepare，再evaluate
& '.\.venv\Scripts\python.exe' '.\reports\mentor_continuation_20260916\locked_holdout_test.py' prepare
& '.\.venv\Scripts\python.exe' '.\reports\mentor_continuation_20260916\locked_holdout_test.py' evaluate

# 项目自身测试（不要从上级目录收集external_runtime中的第三方自测）
& '.\.venv\Scripts\python.exe' -m pytest -q '.\tests'
```

这些入口保留了当时的哈希合同和资源守卫。GitHub快照不含原始DCML/MazurkaBL数据、缓存及权重，因此另一台电脑可直接读代码和报告，但完整重训/推理需先恢复同版数据与未提交的派生产物；不要删除守卫后把不同数据跑出的分数当作复现。

## 数据划分与防泄漏

- 当前24首开发池采用两折、按opus隔离；同一作品的不同演奏不会跨训练/验证集合。
- 完整清单：[`release/current_splits.csv`](release/current_splits.csv)。
- 历史留出评估含17首、6个opus、804次演奏；它不在当前24首池内，但在更早研究中使用过，因此**不是全研究过程严格盲测**。
- 测试准入：[`cohort_locked.json`](reports/mentor_continuation_20260916/locked_holdout_test/cohort_locked.json)。

## 当前结果

主指标是每次演奏内、距离优先贪心的一对一事件匹配，容差为±1 quarterbeat；先在作品内平均，再按作品等权宏平均。不能用逐位置accuracy替代。

| 证据范围 | F1@±1 | 精确位置F1 | AP | 状态 |
|---|---:|---:|---:|---|
| 24首池内部开发、四seed集成+B10 | 83.49% | 67.43% | 69.63% | reproduced，但反复开发过 |
| 17首冻结后历史留出 | **63.70%** | 55.85% | 结果见逐曲表 | reproduced，非严格全研究盲测 |

所以当前可信结论不是“已经泛化到80%”，而是：内部开发超过80%，跨作品泛化证据仍只有63.70%，80%目标尚未达到。

- 内部实验：[`temporal_mask_replication/结论与三问.md`](reports/mentor_continuation_20260916/temporal_mask_replication/结论与三问.md)
- 历史留出总结：[`primary_summary.json`](reports/mentor_continuation_20260916/locked_holdout_test/primary_summary.json)
- 逐作品结果：[`primary_per_work.csv`](reports/mentor_continuation_20260916/locked_holdout_test/primary_per_work.csv)
- 完整测试说明：[`测试报告.md`](reports/mentor_continuation_20260916/locked_holdout_test/测试报告.md)
- 机器可读实际配置：[`release/current_config.json`](release/current_config.json)

## 小样例

[`release/example_input_label_prediction.csv`](release/example_input_label_prediction.csv)包含同一演奏连续13个quarterbeat：58维输入、标签、评分mask、四模型平均概率和最终解码结果。样例覆盖一个真实句首，便于逐列理解格式，但不能用于评价总体性能。

## 已知问题

- 当前性能主要瓶颈是未见作品的泛化，而不是内部开发分数。
- 58维表示依赖既有谱面—演奏对齐；源坐标通过不等于所有音频对齐都经人工确认。
- 内部结果受重复开发选择影响；下一次可信结论需要真正未参与方法选择的新作品和冻结协议。
- `implemented`表示代码存在；`reproduced`表示本机按保存产物复算；`preliminary`表示科学解释证据有限；`proposed`表示尚未执行。
