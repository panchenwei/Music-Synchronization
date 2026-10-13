# Goal 1：纯无标签数据与谱面对齐审计

状态：**complete / implemented and locally reproduced**。本轮训练次数：**0**。

## 结果

- 候选配对作品：45；可用：42；停止并排除：3。
- 可用演奏：1954；64-beat/32-beat窗口：19653；整曲变长序列：1954。
- 同一份58维缓存同时提供两种互不覆盖的数据视图：固定64-beat滑窗，以及整曲变长序列。变长序列训练时按批次最长序列补齐，补齐位置必须由padding mask排除损失和评价。
- 每个演奏输入严格为：9维演奏曲线 + 16维谱面线索 + 9维多尺度速度代理 + 24维有序动机关系 = 58维。
- 谱面位置统一使用 `occurrence_start_qb + 4 × mc_onset`；beat行 `i` 对应 `[i,i+1)`。
- 所有四组特征共享同一个 `quarterbeat = 0..n-1` 轴；逐缓存形状、有限值及无标签字段检查见 `cache_checks.csv`。
- 原始输入访问仅包括DCML notes/measures与MazurkaBL beat_time/beat_dyn；禁止访问数：0。完整路径和SHA-256见 `source_access.csv`。
- 没有读取 harmony、乐句标签、label mask、旧缓存、旧模型、旧预测或句距先验。

## 对齐规则

`next`列按每次访问顺序选择第1、第2……个出口；访问次数超过已列出口、指向不存在小节或超过安全步数即停止该作品。长度兼容规则沿用beat网格端点差异容差：`min(|score_qb-N|, |score_qb-(N-1)|) ≤ 2`。超过容差不回退到折叠谱，也不猜测。

音符起点按 `[i,i+1)`落入beat；持续音只要与区间相交就计为活动。动机分支保留每beat四个子位置；只合并唯一可确定的延音线，歧义延音不静默合并。

## 被停止的作品

- `chopin_op17_no3`: Cannot uniquely select next edge at mc=41, visit=3, candidates=[42, 43]
- `chopin_op33_no3`: Score/beat length mismatch: score=144.0, beat_rows=192, error=47.0
- `chopin_op50_no2`: Score/beat length mismatch: score=558.0, beat_rows=381, error=177.0

## 可视化

- `chopin_op06_no1` → `visualizations/chopin_op06_no1_alignment.png`
- `chopin_op06_no4` → `visualizations/chopin_op06_no4_alignment.png`
- `chopin_op07_no2` → `visualizations/chopin_op07_no2_alignment.png`

橙色小节块表示重复访问；音符散点、四组输入幅度与展开后的quarterbeat共用横轴。图仅用于对齐审计，不是模型结果。

## 产物

- `manifests/goal1/measure_mapping.csv`
- `manifests/goal1/event_mapping.csv.gz`
- `manifests/goal1/window_manifest.csv.gz`
- `manifests/goal1/full_sequence_manifest.csv.gz`
- `manifests/goal1/feature_schema.csv`
- `manifests/goal1/source_access.csv`
- `cache/unlabeled58/*.npz`
- `reports/goal1/completion_audit.json`
