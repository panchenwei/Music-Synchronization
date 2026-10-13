# 乐句起点：增量自监督研究

这是与旧版 `phrase_boundary_overnight` 隔离的研究目录。它复用同一份原始
DCML/MazurkaBL 数据，但自行生成清单、缓存、模型和报告；不会覆盖旧实验。

## 任务与输入

- 预测 DCML `{` 标出的**新乐句开始**，不是句尾或终止式。
- 每个 quarterbeat 一行，共58维：9维对齐演奏曲线、16维谱面线索、
  9维多尺度速度表示和24维有序动机关系。
- 标签映射为 `occurrence_start_qb + 4 * mc_onset`，首拍不评分；只在首个
  明确句首到最后明确句尾之间计算监督和指标。
- 事件F1使用每次演奏内一对一匹配，同时报告精确位置和 ±1 quarterbeat。

完整列定义见
[`manifests/goal1/feature_schema.csv`](manifests/goal1/feature_schema.csv)。与旧版
共用的最小输入/标签/预测格式样例见
[`../phrase_boundary_overnight/release/example_input_label_prediction.csv`](../phrase_boundary_overnight/release/example_input_label_prediction.csv)。

## 已完成实验

| Goal | 内容 | 状态 |
| --- | --- | --- |
| 1 | 42首作品、1954次演奏的58维无标签管线 | implemented / reproduced |
| 2 | 小型BiGRU遮挡重建smoke | implemented / reproduced smoke |
| 3 | 整曲变长Transformer重建smoke | implemented / reproduced smoke |
| 4 | 3 seed音乐输入与去内容控制的正式SSL验证 | implemented / reproduced；晋级0/3 |
| 5 | 同结构Transformer：Scratch与SSL初始化直接监督对照 | implemented / reproduced；科学结论preliminary |

Goal 5模型是58→64投影、2层Transformer Encoder、4头、FFN128、动态正弦
位置编码和逐拍二分类头，共71,029参数。30/8/4首作品按作品隔离，每首取前
2次演奏；测试只在六个模型及验证阈值冻结后读取。

测试集仅4首作品，3 seed宏平均结果：Scratch 的 ±1拍F1为 **47.23%**、
Exact F1为29.63%、AP为42.08%；SSL初始化为 **43.34%**、24.48%、36.40%。
当前证据不支持这套SSL预训练带来增益。详见
[`reports/goal5/report.md`](reports/goal5/report.md)和
[`reports/goal5/registered_protocol.json`](reports/goal5/registered_protocol.json)。

这与旧版BiGRU的内部83.49%或历史留出63.70%使用不同划分、演奏数量、模型、
标签覆盖和解码协议，不能直接作为同条件涨跌。最终模型家族仍未固定。

## 代码入口

- 数据与58维特征：[`src/goal1_unlabeled.py`](src/goal1_unlabeled.py)
- BiGRU smoke：[`src/goal2_smoke.py`](src/goal2_smoke.py)
- Transformer与变长批处理：[`src/goal3_transformer_smoke.py`](src/goal3_transformer_smoke.py)
- SSL正式对照：[`src/goal4_pretrain.py`](src/goal4_pretrain.py)
- 监督训练、标签、解码和指标：[`src/goal5_downstream.py`](src/goal5_downstream.py)
- 独立指标复算：[`src/goal5_audit.py`](src/goal5_audit.py)

## 在另一台电脑运行

Python 3.12依赖见 `requirements.txt`。先复制路径模板并修改为本机的数据目录：

```powershell
Copy-Item '.\configs\data_paths.example.toml' '.\configs\data_paths.toml'
python -m pip install -r requirements.txt
```

然后在本目录运行：

```powershell
# 重新生成无标签清单和缓存（需要原始DCML/MazurkaBL数据）
python -m src.goal1_unlabeled --config configs/data_paths.toml

# 不依赖原始数据或checkpoint的代码级测试
python -m unittest tests.test_goal1_unlabeled tests.test_goal2_smoke `
  tests.test_goal3_transformer tests.test_goal4_controls `
  tests.test_goal5_labels_metrics -v
```

`goal2_smoke`至`goal5_downstream`会写入本地缓存、checkpoint和报告。完整重训前
先阅读对应protocol；不要用已公开的4首测试作品继续选结构或阈值。完整的
`python -m src.goal5_audit` 还需要本地checkpoint、逐演奏NPZ和访问审计记录，
这些大文件/机器相关文件没有发布。

## 未提交内容

原始数据、缓存、checkpoint、逐演奏NPZ概率、日志和含本机绝对路径的访问清单
都由 `.gitignore` 排除。报告中的checkpoint哈希用于核对原实验，但仓库本身不
包含权重。
