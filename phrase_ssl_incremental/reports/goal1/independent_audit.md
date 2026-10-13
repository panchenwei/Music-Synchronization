# Goal 1 独立完成核验

日期：2026-10-13。状态：implemented / locally reproduced。训练次数：0。

运行命令（项目目录内）：

```powershell
& '..\phrase_boundary_overnight\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

实际结果：11 tests，全部通过。

| 要求 | 核验依据与结果 |
|---|---|
| 仅访问允许的无标签原始文件 | source_access.csv包含4种允许角色，各45个文件；访问保护器拒绝harmonies与旧缓存 |
| next展开与停止规则 | 合成反复展开测试通过；歧义出口测试会停止；op17no3实际因第3次访问mc41时出口不唯一被排除 |
| 音符位置公式 | 全量event_mapping核验local_qb=4×mc_onset，global_qb=start+local_qb，beat和subtick映射一致 |
| 缺失mc_onset停止 | 缺失坐标测试触发AlignmentError |
| 起音与持续音映射 | 半开区间测试核验跨拍持续音、恰在整数拍起音和终止的归属 |
| 58维与统一坐标 | 42份缓存全量核验9+16+9+24维形状、有限值、逐拍坐标与精确拼接；每拍4个动机子位置存在；无label字段 |
| 谱面长度兼容 | 42首均满足登记的≤2 quarterbeat容差；op33no3误差47、op50no2误差177，已排除 |
| 固定窗口 | 19653个窗口均长64、步长32；尾窗有效长度和padded标记一致 |
| 整曲变长清单 | 1954条记录覆盖全部可用演奏；起点0、终点等于整曲长度，存在多种长度 |
| 三首图 | op06no1、op06no4、op07no2图已生成并人工视觉检查；展示弱起、反复及不同结尾；非空文件检查通过 |
| 不启动训练 | 本阶段没有训练入口调用，目录无pt/pth/ckpt模型文件 |

原始数据管线每次从notes/measures与beat_time/beat_dyn重新计算，测试阶段读取本项目新生成缓存以核验结果。长度兼容仅证明坐标网格兼容，不等同于人工确认所有谱本细节。三首异常作品保留具体原因，不猜测对齐。

后续路线：Transformer使用整曲变长序列；CNN＋RNN使用64/32固定窗口。补齐掩码和模型训练属于下一阶段，当前只有数据清单和缓存。
