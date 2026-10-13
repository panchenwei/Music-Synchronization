# Project Status

## Goal 10最新状态

- `COMPLETE / implemented and locally reproduced; preliminary nested OOF decoder evidence`：38首开发作品做5轮嵌套作品级评估，每首只外评一次；旧4首测试作品未加载。
- 局部解码OOF F1/Exact/AP为45.22%/32.49%/35.38%；固定训练集句距解码为49.61%/35.99%/35.38%，F1 +4.39pp、Exact +3.50pp，4/5折同向。
- AP完全不变，确认这是可迁移的输出结构收益，不是模型表征提升；句距解码作为候选后处理晋级。
- 下一步测试最小的句距结构化训练损失，目标是同时提高AP与冻结解码后的F1；详见`reports/goal10/report.md`。

## Goal 9最新状态

- `COMPLETE / implemented and locally reproduced; preliminary development diagnostic`：冻结Goal 6三份Scratch Transformer概率，只比较局部峰值与训练集句距动态规划解码。
- 局部解码验证F1/Exact/AP为49.15%/38.69%/39.06%；句距解码为52.48%/40.29%/39.06%。F1三个seed全升，平均+3.33pp，按预注册门槛仅作为候选后处理晋级。
- AP完全不变，说明增益来自输出结构约束，不是模型表征或概率排序提高；不能称为模型F1提升。
- 未读取Goal 5测试集。下一步需作品级外层交叉验证或新终审作品，避免继续适配同一8首验证作品；详见`reports/goal9/report.md`。

## Goal 8最新状态

- `COMPLETE / implemented and locally reproduced; preliminary development diagnostic`：在同一25维Transformer上加入2/4/8拍前后文novelty。
- novelty模型验证F1/AP为48.55%/39.52%，控制为49.15%/39.06%；AP仅+0.46pp，F1−0.60pp，未晋级。
- 句首novelty均值高于负例，但单特征AP只有10.32%～12.25%，说明有弱信息但重叠很大。
- 三轮结论：输入一致性、缩小为BiGRU、固定novelty均未解决问题。下一阶段应测试标签序列约束或监督式相对结构目标，并启用新的开发/终审协议。
- 未读取Goal 5测试集；详见`reports/goal8/report.md`。

## Goal 7最新状态

- `COMPLETE / implemented and locally reproduced; preliminary development diagnostic`：同一25维安全输入下比较6,121参数BiGRU与71,029参数Transformer。
- BiGRU验证F1/AP为48.57%/35.45%，Transformer为49.15%/39.06%；BiGRU AP三个seed全降，未晋级。
- 缩小模型没有解决概率排序问题；不继续堆GRU/TCN/LSTM层数。下一步测试同曲相对自相似/novelty表示。
- 未读取Goal 5测试集；详见`reports/goal7/report.md`。

## Goal 6最新状态

- `COMPLETE / implemented and locally reproduced; preliminary development diagnostic`：修正预训练与监督输入不一致，只保留同一25维安全内容。
- 同一30/8首训练/验证、3seed、600步；没有加载Goal 5已公开的4首测试作品。
- Scratch F1/AP为49.15%/39.06%，SSL为47.81%/37.84%；SSL的F1三个seed全降，平均−1.34pp，未晋级。
- 输入一致性不是主要病因；当前逐拍统计重建SSL路线停止。下一步比较同一25维输入的小BiGRU与Transformer。
- 详见`reports/goal6/report.md`与`reports/goal6/independent_audit.json`。

## Goal 5最新状态

- `COMPLETE / implemented and locally reproduced; preliminary`：整曲Transformer Scratch与SSL初始化的下游对照完成。
- 冻结30/8/4首训练/验证/测试划分，3seed×2条件，每次600步。测试在模型/阈值冻结后才读取。
- 测试三seed平均：Scratch ±1拍F1 47.23%、Exact29.63%、AP42.08%；SSL F1 43.34%、Exact24.48%、AP36.40%。
- 本轮未显示自监督可靠增益；4首测试数量有限。独立二分图匹配复算通过，27项测试通过。
- 详见`reports/goal5/report.md`。未根据测试调参或启动第二架构。

## Goal 4最新状态

- `COMPLETE / implemented and locally reproduced; scientific conclusion preliminary`。
- 30首训练、8首开发验证、4首保留未读取；60/16个完整演奏，3seed×正常/去内容共6次、每次400步。
- 正常音乐模型最佳macro重建MSE约1.104；均值1.103929；插值1.270705；正常与去内容几乎一致，晋级0/3。
- 独立预测/基线复算通过，全项目22项测试通过。输入/目标一致性需要修正，尚未进入下游标签微调。
- 详见`reports/goal4/report.md`与`reports/goal4/independent_audit.json`。

## Goal 3最新状态

- `COMPLETE / implemented and locally reproduced smoke`：最高优先级整曲变长Transformer已跑通。
- 两层、四头、隐藏64、前馈128、四组重建头，共74,734参数；完整288/337拍训练与121拍验证，无64拍截断。
- 主训练300次tiny-overfit更新；loss 1.134811 → 0.000703，验证1.734206；全项目19项测试通过。
- 未使用乐句标签，未开展完整预训练或下游评分。原Goal 1与Goal 2产物保留。
- 详见`reports/goal3/smoke_report.md`。

## Goal 2最新状态

- `COMPLETE / implemented and locally reproduced smoke test`：小型BiGRU、连续4～8拍遮挡、独立遮挡/缺失token与四组重建头已实现。
- 4个训练窗口、2个隔离作品验证窗口；250次tiny-overfit更新，加1次恢复一致性验证；总计14项自动测试通过。
- 训练重建loss：0.937333 → 0.027781；验证loss：2.829779。没有乐句标签、没有下游准确率结果，完整预训练尚未启动。
- 此Goal按其明确要求运行64拍BiGRU smoke；整曲变长Transformer主路线仍为后续计划。
- 结果见`reports/goal2/smoke_report.md`。

## Goal 1数据状态

- 状态：`GOAL 1 COMPLETE`
- 已完成：纯无标签原始数据读取、DCML next展开、quarterbeat映射、58维输入、固定窗口与整曲变长序列清单、访问审计、3首可视化
- 可用作品：42；排除作品：3；演奏：1954；64/32窗口：19653；整曲序列：1954
- 模型训练：`0`（本Goal禁止训练）
- 下一步：Goal 2 Transformer自监督遮挡任务；尚未启动

详情见 `reports/goal1/data_audit.md`。
