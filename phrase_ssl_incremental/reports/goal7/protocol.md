# Goal 7：25维安全输入下的小BiGRU与Transformer对照（运行前冻结）

状态：`registered / not yet run`。

## 假设

当前只有30首监督训练作品。若Transformer的容量与归纳偏置是主要限制，那么在与Goal 6 Scratch完全相同的可见输入、作品、采样、600步预算、损失与解码下，约6千参数的单层双向GRU应比71,029参数Transformer有更高的验证AP和F1。

## 模型与唯一主要变化

58维接口仍保留，但第25～57列和元数据列始终使用mask token，实际可见音乐内容与Goal 6一致。唯一主要变化是编码器：`Linear(58,32) → 1层双向GRU(每方向14) → Dropout(0.1) → LayerNorm(28) → Linear(28,1)`。使用整曲变长序列与padding mask，不切64拍窗口。

## 评价与停止

- 复用30/8首训练/验证、每曲2次演奏和seed17/43/89；不读取Goal 5测试集。
- Adam lr0.0003、600 optimizer steps、相同batch顺序和正类权重；按验证宏AP选checkpoint，再按验证F1选阈值。
- 主要对照是Goal 6的25维Scratch Transformer。
- 晋级：BiGRU的AP和F1平均均至少高2个百分点，且两项至少在2/3 seed提高。
- 三次BiGRU运行或15分钟上限停止，不追加宽度、层数或steps。

结果仍是重复开发验证，不是终审泛化证据。
