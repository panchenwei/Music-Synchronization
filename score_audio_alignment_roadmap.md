# Score-Audio Alignment Project Roadmap

## 1. 项目目标

本项目要解决的问题是：

- 输入：`MusicXML / XML` 乐谱文件 + 一段对应演奏音频
- 输出：为乐谱中的时间单位打上与音频对应的时间戳
- 最终工具能力：
  - 能在乐谱上标记每个小节、拍点、音符在音频中对应的时间
  - 能支持播放音频时在谱面上高亮当前位置
  - 能导出结构化对齐结果，便于后续分析或可视化

建议先把“标记拍点/小节时间”作为第一阶段目标，再逐步细化到“每个音符”的时间。

---

## 2. 总体路线

推荐按照下面 5 个阶段推进：

1. 问题定义与调研
2. Baseline 复现
3. 数据与评测体系搭建
4. 自己的方法开发
5. 工具化与可视化落地

这条路线的核心思想是：

- 先搞清楚学术界和工程界怎么做
- 先复现一个能跑通的基线
- 再在基线之上迭代，而不是一开始就从零设计复杂模型

---

## 3. 第一阶段：问题定义与调研

### 3.1 先明确任务边界

首先要把“对齐”拆清楚，不然后面很容易范围失控。

建议把任务分成 3 层：

- `Measure-level alignment`
  - 每个小节在音频中从几分几秒开始
- `Beat-level alignment`
  - 每一拍在音频中的时间
- `Note-level alignment`
  - 每个音符的起始时间、结束时间

对于科研项目，推荐目标定义为：

- 第一版：`MusicXML + audio -> beat / measure timestamp`
- 第二版：`MusicXML + audio -> note-level alignment`
- 第三版：谱面可视化工具

### 3.2 调研重点

调研时不要泛泛地看“音乐信息检索”，而要重点看下面几类工作：

#### A. Score following / score-audio alignment

关键词：

- `score audio alignment`
- `musicxml audio alignment`
- `symbolic audio alignment`
- `score following`
- `offline score alignment`
- `performance alignment`

重点关注：

- 输入是什么：MIDI、MusicXML、谱面图像、还是符号乐谱
- 输出是什么：帧级、拍级、音符级
- 方法是什么：DTW、HMM、CRF、Transformer、cross-attention、CQT/chroma 对齐
- 是否支持真实演奏中的速度变化、rubato、漏音、踏板、表情

#### B. Symbolic-to-audio matching

因为你的输入是 XML，很多工作会先把 XML 转成更适合计算的符号序列，再和音频特征对齐。

重点看：

- 如何从 `MusicXML` 提取音高、时值、拍点、小节、tempo 标记
- 如何把乐谱渲染成 MIDI 或合成音频
- 如何把“符号时间轴”和“真实演奏时间轴”对应起来

#### C. 可复现的工程 baseline

优先找：

- 开源代码
- 提供数据集
- 能离线运行
- 输入接近 `MusicXML + audio`

如果论文很好但没有代码，前期可以记录，但不要作为第一复现目标。

### 3.3 调研产出

这一阶段建议输出一个 `survey.md`，至少包含：

- 论文标题 / 链接
- 任务类型
- 输入输出定义
- 核心方法
- 数据集
- 是否开源
- 复现难度
- 与本项目的相关性

建议最终筛出：

- `2~3` 篇核心论文
- `1~2` 个可直接复现的 baseline

---

## 4. 第二阶段：Baseline 复现

这一阶段的目标不是创新，而是先把“对齐”这件事做通。

### 4.1 最推荐的 baseline 思路

对于你的任务，最稳妥的第一版 baseline 通常不是端到端神经网络，而是：

1. `MusicXML -> symbolic representation`
2. `MusicXML -> MIDI / synthesized audio`
3. 从真实音频中提取特征
4. 从合成音频或符号表示中提取对应特征
5. 用 `DTW` 或其变体进行时间对齐

这是最适合先复现、先跑通、先出结果的一条路线。

### 4.2 一个可执行的 baseline pipeline

#### Step 1. 解析 MusicXML

从 XML 中提取：

- 小节号
- 拍号
- tempo 标记
- 音符 pitch
- duration
- onset 顺序
- tie / rest / chord 信息

可用工具方向：

- Python `music21`
- `partitura`
- `pretty_midi`（偏 MIDI）

推荐先试 `music21` 或 `partitura`。

#### Step 2. 将 MusicXML 转成 MIDI / 合成音频

目的不是做高质量渲染，而是获得一个“标准参考时间轴”。

输出：

- `score.mid`
- `score_synth.wav`

这一步会给你一个规则节奏、规则力度的参考版本，后续可与真实演奏做对齐。

#### Step 3. 对真实音频和合成音频提特征

常见特征：

- `CQT`
- `chroma`
- `mel spectrogram`
- onset strength envelope

第一版推荐：

- 先做 `chroma + onset`
- 如果效果一般，再试 `CQT`

#### Step 4. 做时间对齐

使用：

- `DTW`
- `FastDTW`
- segmental DTW

输出：

- 合成音频时间 `t_score_audio`
- 真实音频时间 `t_real_audio`
- 二者的 warping path

#### Step 5. 回投到乐谱事件

因为合成音频时间轴和 MusicXML 事件是一一可追踪的，所以可以把 DTW 得到的时间映射回：

- 小节
- 拍点
- 音符 onset

最终得到：

- `measure_id -> real_audio_time`
- `beat_id -> real_audio_time`
- `note_id -> real_audio_time`

这就是后面在谱面上打标记的核心中间结果。

### 4.3 复现阶段的最低成功标准

只要达到下面 3 点，就算 baseline 跑通：

- 给定一份 `MusicXML` 和对应音频，能输出每个小节的时间戳
- 在速度变化不太夸张的片段上，对齐结果基本正确
- 能导出结构化结果（如 `json/csv`）

不要一开始就追求复杂钢琴演奏上的音符级完美对齐。

---

## 5. 第三阶段：数据与评测体系

如果没有评测体系，后面模型改进会非常痛苦。

### 5.1 数据准备

需要两类数据：

#### A. 公开数据

优先找带有以下要素的数据：

- 音频
- MIDI 或符号乐谱
- 对齐标注，或者至少可构造对齐参考

适合优先考察的方向：

- 钢琴独奏数据集
- 古典乐对齐数据集
- 带 MIDI-performance 对应关系的数据

说明：

- 很多公开数据天然更接近 `MIDI <-> performance`，不一定直接是 `MusicXML <-> audio`
- 但你可以把 MIDI / XML 看成统一的“符号乐谱侧”，先完成研究验证

#### B. 自建小规模数据

建议自己做一小批高质量样例：

- `5~20` 首短曲目
- 每首有 `MusicXML`
- 每首有对应音频
- 人工校验小节/拍点时间

这批数据很重要，因为它最贴近你的最终任务。

### 5.2 评测指标

建议分层评测：

- 小节级误差：
  - 预测小节时间 vs 人工标注时间
- 拍级误差：
  - 平均绝对误差（ms）
- 音符级误差：
  - onset MAE
  - 在阈值内的准确率（如 `50ms / 100ms`)

还可以加：

- 失败样本分析
- 不同速度变化下的鲁棒性
- 不同录音质量下的鲁棒性

### 5.3 误差分析模板

每次实验建议都回答：

- 错在开头还是结尾
- 错在慢速段还是快速段
- 和弦密集处是否失效
- rubato 段是否漂移
- 休止符 / 连音处是否映射错误

---

## 6. 第四阶段：在 Baseline 上做研究改进

当 baseline 稳定后，再考虑创新点。

### 6.1 可选改进方向

#### 方向 A. 更好的特征

从简单 `chroma` 升级到：

- `CQT`
- 音高感知特征
- onset-aware 特征
- 自监督音频 embedding

#### 方向 B. 更好的对齐算法

从普通 DTW 升级到：

- constrained DTW
- multi-scale DTW
- segmental DTW
- HMM / Viterbi alignment

#### 方向 C. 符号侧建模更精细

把 XML 中更多信息用起来：

- tempo markings
- dynamics
- articulation
- fermata
- repeat / volta / da capo 等结构信息

#### 方向 D. 学习型方法

当你已经有一些对齐数据后，可以尝试：

- 音频编码器 + 乐谱编码器
- cross-modal attention
- frame-to-event matching
- differentiable alignment / monotonic attention

但这一步建议放在 baseline 之后，否则很容易训练出一个结果不稳定、也难解释的系统。

### 6.2 推荐的创新切入点

如果项目周期有限，优先考虑下面两类创新，成功率更高：

- `基于 MusicXML 结构信息约束的对齐`
  - 利用小节、拍号、tempo、段落结构提升鲁棒性
- `分层对齐`
  - 先粗对齐小节，再细对齐拍点，再细化到音符

这两个方向既有研究价值，也比较贴近最终工具落地。

---

## 7. 第五阶段：做成工具

科研结果最后要落成一个可用工具，建议按“离线工具优先”的思路做。

### 7.1 工具输入输出

#### 输入

- `score.xml` / `score.musicxml`
- `performance.wav` / `mp3`

#### 输出

建议至少导出两类结果：

1. 结构化对齐文件 `alignment.json`
2. 可视化结果

`alignment.json` 示例结构：

```json
{
  "audio_file": "performance.wav",
  "score_file": "score.musicxml",
  "measures": [
    { "measure": 1, "time_sec": 0.00 },
    { "measure": 2, "time_sec": 2.31 }
  ],
  "beats": [
    { "measure": 1, "beat": 1, "time_sec": 0.00 },
    { "measure": 1, "beat": 2, "time_sec": 0.64 }
  ],
  "notes": [
    { "note_id": "n001", "measure": 1, "time_sec": 0.02 }
  ]
}
```

### 7.2 工具架构建议

建议拆成 4 层：

#### A. Parser

负责：

- 解析 MusicXML
- 建立乐谱事件索引

#### B. Aligner

负责：

- 特征提取
- 时间对齐
- 事件时间回投

#### C. Exporter

负责：

- 输出 `json/csv`
- 生成可用于前端显示的数据

#### D. Viewer

负责：

- 显示乐谱
- 在小节/拍点/音符上打时间标记
- 音频播放时同步高亮

### 7.3 乐谱显示方案

如果后面要做可视化，可以考虑：

- 将 `MusicXML` 转成浏览器可显示的谱面
- 在前端叠加时间标签与播放指针

前端第一版不必太复杂，只要做到：

- 点击某小节，跳到音频对应时间
- 播放音频时，高亮当前小节/拍点

这已经很有展示价值。

---

## 8. 推荐的执行顺序

下面这个顺序最稳：

### Week 1: 调研与任务收敛

- 明确任务定义：小节级、拍级、音符级
- 找 `5~10` 篇核心论文
- 选出 `1~2` 个 baseline
- 确定开发语言和主要库

### Week 2: 数据打通

- 解析 `MusicXML`
- 导出音符 / 拍点 / 小节信息
- 完成 `XML -> MIDI / synth audio`

### Week 3: Baseline 跑通

- 对真实音频和合成音频提特征
- 跑 `DTW`
- 输出小节/拍点时间戳

### Week 4: 结果校验

- 做 `json/csv` 导出
- 选几首样例人工检查
- 画对齐结果图

### Week 5-6: 改进方法

- 调特征
- 加结构约束
- 做分层对齐

### Week 7+: 工具化

- 做命令行版本
- 做简单可视化界面
- 准备论文实验图和 demo

---

## 9. 第一版技术方案建议

如果现在就要定一个最现实的起步方案，我建议：

### 方案名：`MusicXML -> synth audio -> DTW -> beat/measure timestamps`

理由：

- 最容易快速打通
- 和你的输入形式契合
- 可解释性强
- 便于后续扩展到 note-level

### 技术栈建议

- Python
- `music21` / `partitura`
- `librosa`
- `numpy`
- `scipy`
- `matplotlib`
- 如需前端展示，再补一个简单 Web viewer

---

## 10. 当前最值得先做的事

建议立刻开始的任务不是写模型，而是下面这 4 件事：

1. 建一个 `survey.md`
2. 选定一个 XML 解析库（优先试 `music21`）
3. 选定一个 baseline（优先 `synth + DTW`）
4. 准备一首最小可跑通样例：`1 个 MusicXML + 1 个对应音频`

只要这 4 件事完成，项目就从“想法”进入“可执行状态”了。

---

## 11. 建议的项目目录

```text
music-synchronization/
  docs/
    survey.md
    experiment_log.md
  data/
    raw/
    processed/
    examples/
  src/
    parser/
    features/
    aligner/
    export/
    viewer/
  outputs/
    alignments/
    figures/
  notebooks/
  README.md
```

---

## 12. 交付物拆分

为了让科研和工程都能往前走，建议把最终交付物拆成这几项：

- `D1` 调研文档
- `D2` baseline 复现代码
- `D3` 小规模评测集
- `D4` 改进方法实验结果
- `D5` 可视化对齐工具
- `D6` 论文/汇报材料

---

## 13. 一句话结论

这个项目最合适的推进方式是：

**先用 MusicXML 合成参考音频，再用音频特征 + DTW 做离线对齐，先输出小节/拍点时间戳，随后逐步细化到音符级并做谱面标注工具。**

这条路线风险最低，也最适合从“调研和复现”平滑过渡到“科研改进和工具落地”。
