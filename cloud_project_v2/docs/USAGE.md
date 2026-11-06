# 清洗工程使用说明

本工程没有模型训练入口。命令在项目根目录的PowerShell执行。

```powershell
# 全量审核、确定性缓存、自动验收；未完成的任务可安全续跑。
.venv-competition\Scripts\python.exe -m cxr_clean --stage all --workers 4 --output data/clean_v1

# 仅扫描；仍完整解码原图，不依赖旧512缓存。
.venv-competition\Scripts\python.exe -m cxr_clean --stage scan --workers 4 --output data/clean_v1

# 从扫描断点继续出清单、缓存和最终报告。
.venv-competition\Scripts\python.exe -m cxr_clean --stage cache --workers 4 --output data/clean_v1

# 合成数据检查，不调用模型或真实患者图片。
.venv-competition\Scripts\python.exe -m unittest discover -s tests -v
```

原始CSV改变时必须使用新版本输出目录；已完成的清洗报告不作为不同数据版本的容器。所有阶段先核对原路径、尺寸、修改时间、扫描代码指纹；缓存阶段对实际读取的源字节重新核对SHA256。SQLite是可恢复过程证据，CSV/JSON/报告是交付件。

已通过验收的输出目录禁止再次覆写，CLI会明确拒绝；新策略使用新目录或逐图复核派生工具。清洗v1实际执行时的完整源码保存在`data/clean_v1/code_snapshot/cxr_clean`，与`environment.json`逐文件哈希对应；后加的已完成目录保护不改变本次图像处理结果。

## 如何读取交付

| 文件 | 含义 |
|---|---|
| `train_clean.csv` | 后续训练唯一推荐入口；并非提交预测文件 |
| `valid_manifest.csv` / `test_manifest.csv` | 官方逐图覆盖不变；按`study_key`聚合检查 |
| `image_audit.csv` | 每张原图的解码状态、哈希、尺寸、质量指标 |
| `study_manifest.csv` | 检查级标签、图数与复核状态，保留多视角关系 |
| `exclusions.csv` | 从新训练清单排除的原图及明确原因；原图未删 |
| `review_queue.csv` | 待复核的质量/近重复/标签疑点，未认定为错误 |
| `duplicate_groups.csv` | 精确内容重复；同组成员逐行列出，字节/像素两种依据分别保留 |
| `near_duplicate_hash_pairs.csv` + `perceptual_hash_membership.csv` | 感知哈希组对与成员；压缩表示候选、不截断，不是自动排除名单 |
| `class_distribution.csv` + `train_*_cooccurrence.csv` | 十类清洗前后计数、检查级标签共现；不改标签/不重采样 |
| `preprocess_manifest.csv` + `preprocessing.json` | 原SHA与派生SHA、300RGB缓存、回读成功/失败数 |
| `acceptance.json` / `CLEANING_REPORT.md` | 机器验收和中文结论；临床疑点限制必须同时阅读 |

CSV采用UTF-8 BOM；列表字段是JSON。`image_id=split:原CSV的零基行号`用于复核定位，ID保留字符串，禁止读成浮点数。`relative_path`相对于`cxr_first/cxr_first`，`cache_relative_path`相对于本版本输出目录。`targets`按1–10类顺序表示十维多热向量，test为空，不能拿空值当十类全阴性。

## 统一尺寸、归一化与增强

缓存保比例缩放、居中补黑到300×300，保留完整胸廓；RGB无损PNG中保存0–255像素，绝不是“已经归一化”的张量。`load_input(path)`返回ImageNet归一化后的float32 CHW；数据加载时才产生浮点张量，避免持久化三通道浮点数据浪费空间。

未来经授权训练时，显式`load_input(path, augment=True, rng=rng)`才逐次随机抽取亮度、对比度和小角度旋转；验证/测试保持默认关闭。清洗缓存不把随机增强永久烘焙到图片。不盲目左右翻转/强裁剪/噪声遮挡，疾病及左右解剖信息可能受损。当前无增强训练、无模型特征提取。

归一化参数来自预训练权重的固定参数，并非从valid/test估计；保持比例补边相对于B3默认中心裁剪的收益尚未通过模型实验验证。类别权重、过采样、损失函数、Mixup/CutMix、冻结微调和阈值搜索属于后续训练试验，不把它们混进本次清洗结论。

## 安全复核与回滚

只用本机读取原图；不要上传患者图像或带身份的清单到第三方查重服务。图像损坏和医学错标是不同问题；低质量候选需人工判断，不能拿算法分数当诊断结论。若发现错标，保留原标注与修订来源，另建版本，不能覆写本版本记录。

撤销本次处理只需停用`data/clean_v1`，原始CSV与图像仍可直接读取。旧代码、Git历史、模型与旧结果完整位于`archive/legacy_20261016_2345`，共用Python环境不是旧模型实现。

## 已交付的技术复核版本

当前应读取`data/clean_v1_reviewed/train_clean.csv`。自动父版先去重14张；随后分别查看2张异常原图、派生缓存和同检查保留视角，按绑定源SHA的配置隔离2张。复核版训练30328张、Study18513项；没有改变任何疾病标签或Study阳性数。

```powershell
# 这条命令已执行，交付目录不可再次覆写；更改决策时用新的输出名。
.venv-competition\Scripts\python.exe -X utf8 -m cxr_clean.curate --parent data/clean_v1 --output data/clean_v1_reviewed --decisions docs/manual_review_v1.json
```

复核版缓存路径以`../clean_v1/images300/`开头，必须相对于本版目录解析。保留父版目录；不要仅移动CSV后把相对路径解释成原图目录。取消技术隔离时改用父版`train_clean.csv`，取消全部清洗时改用原始CSV。

查看复核版`technical_review_evidence.jpg`可对照异常图与同检查保留图。父版`review_visuals`另有质量低值16格、近重复8对、seed43随机16格；这些是技术抽查，不是对全部2769候选的医学审核。复核版继承父版逐图解码验收，并重新核对实际引用的33968个缓存SHA，没有重复声称又解码一遍。
