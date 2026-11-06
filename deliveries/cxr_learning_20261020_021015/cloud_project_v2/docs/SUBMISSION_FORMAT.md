# 提交 CSV 格式与历史文件对照

## 2026-10-19 官网更新与本次交付

已重新读取[官网“七、作品提交要求”](https://www.ncccu.org.cn/index/Paper/case1.html)及其新版示例图：示例包含 `Subject_id,Study_id,Predict_class,Probability,Gate`，Gate 位于最后一列，省略自定义值时默认0.5。每个 Study 仍须保留全部十类概率。官网排名以 Macro AUC 优先，只有 AUC 相等才比较 Macro F1。

本次最新已有测试交付 `../../deliveries/best_auc_20261019/submission.csv` 已按哈希和逐值比对追溯到 **016_a4000_tiny_v2，epoch5 EMA**，不是当前019五折代码的新预测。采用该权重同源、同预处理和 bfloat16 验证所得的十类阈值，输出 [新版 submission.csv](../../deliveries/gate_update_20261019/submission.csv)，只在原四列后追加 Gate，原身份、行序、概率文本均不变。

本地回读校验通过：1414 Study、14140数据行、每Study十类、无缺失/重复/非法概率或Gate；与原推理长表、宽表和二值表一致。验证集 F1@0.5=0.3205493442，同源阈值 F1=0.4694609371，AUC仍为0.7999705530；阈值在同一valid调优，该F1不代表独立测试成绩。本次没有训练、模型推理、重新拟合阈值或上传平台；旧CSV保持原样。

来源、阈值、校验及重放脚本见 [交付说明](../../deliveries/gate_update_20261019/README.md)。以下保留2026-10-18历史审计，其“本次未调整”“未生成新CSV”等表述仅适用于原日期。

核对日期：2026-10-18，Asia/Hong_Kong（UTC+8）。本次只读审计并更新文档，未训练、推理、调整阈值或上传平台；没有改写任何已有 CSV。

## 来源与结论边界

1. 用户提供的 `D:/QQ/Downloads/submission.csv` 是用户确认可提交的四列样例。本次核对其字节及内容结构，不把 CSV 单元格当作操作指令，也没有另行向平台验证接收状态。
2. [当前赛事官网](https://www.ncccu.org.cn/index/Paper/case1.html)“七、作品提交要求”明确文件名、UTF-8、每个检查的全部十类概率、概率范围及可选 Gate。2026-10-18 已重新读取正文。
3. [2026-10-16 官方答疑只读快照](../archive/legacy_20261016_2345/artifacts/official_qa_20261016.txt) E22 要求保留全部类别概率，E23 明确 `Subject_id` 可以不包含。C23 是提问，不能把其中“四列是否必须”的问句当作要求。本次未能通过浏览工具重新打开在线答疑，故不将该快照冒称为最新在线核验。

**四列样例有效，不代表完整三列表无效。** 后续需要复现用户样例时，使用下面的四列表头；历史完整三列表仍按 E23 的允许范围记录，不能仅因少 `Subject_id` 判为格式错误。真正已确认的问题是部分旧稿按阈值删行，导致检查或类别覆盖不完整。

## 对齐用户样例的四列表

精确列名和顺序：

```csv
Subject_id,Study_id,Predict_class,Probability
```

| 字段 | 填写要求 |
| --- | --- |
| `Subject_id` | 使用测试清单 `subject_id` 原始字符串，保留前缀和前导零；同检查各行患者身份一致。 |
| `Study_id` | 使用测试清单 `study_id` 原始字符串；不填图像 Dicom_id，也不填内部 `subject/study` 组合键。 |
| `Predict_class` | 类别整数文本 `1`～`10`，每个检查每类恰好一行；不能从0编号或写成十维列表。 |
| `Probability` | 每类独立连续概率，必须为有限数值且在 `[0,1]`；保留低于阈值的概率，不替换为二值预测。 |

每行代表一个“检查×类别”，不是一张图片。多视角先按既有推理协议汇总到 Study，再写十行。多标签概率不要求十类之和等于1，不使用类别 Softmax，也不强制 No Finding 互斥。

当前测试版本应为 **1,414 个 Study × 10 类 = 14,140 条数据行**，另加一行表头。以 `test_manifest.csv` 校验检查覆盖和患者映射，禁止根据模型是否预测阳性筛选行。不得重复 `(Subject_id, Study_id, Predict_class)`；本版本 `Study_id` 也不存在跨患者重名，因此 `(Study_id, Predict_class)` 同样唯一。

为便于重放，推荐按测试清单首次出现的 Study 顺序输出，每个 Study 的类别按1～10排列。这与用户样例一致；不能仅因不同的合法排列就宣称平台必然拒收。

## 编码、数值与 Gate

- 文件名使用 `submission.csv`，逗号分隔、UTF-8；推荐无 BOM，不输出 DataFrame 索引、空行或额外标题。
- 用户样例无 BOM、使用 CRLF 且末尾有换行。这些是样例观察，不足以证明 LF 或其他有效 UTF-8 写法会被平台拒收。
- 样例概率使用普通小数，小数位数不固定（4～12位），没有固定十位小数要求。保留足够精度，不通过 Excel 转换 ID 或强制截断概率。
- 不包含 `Gate` 时，按官网默认阈值0.5理解。样例没有 Gate，不能把其称为已传入验证集优化阈值的文件。
- 如使用自定义阈值，官网要求在最后增加 `Gate`：四列身份格式对应 `Subject_id,Study_id,Predict_class,Probability,Gate`，三列简式对应 `Study_id,Predict_class,Probability,Gate`。阈值只来自验证集，必须与对应权重和预处理绑定；本次不在测试集拟合或调整阈值。
- Gate 只说明阈值，不能据此删掉低概率行。`study_binary.csv` 和 `study_probabilities.csv` 是项目内部宽表，不作为此长表的替代品。

## 本次样例审计结果

| 检查 | 结果 |
| --- | --- |
| 表头 / 每行列数 | 上述四列，全部一致，无额外索引或 Gate |
| 数据行 / Study / 患者 | 14,140 / 1,414 / 136 |
| 每检查类别 | 1～10完整，连续十行、按类别升序 |
| 重复组合 / 缺失检查 / 缺失类别组合 | 均为0 |
| 与测试清单的检查覆盖、患者映射和顺序 | 一致 |
| 非法、空白、NaN/Inf 或越界概率 | 0 |
| 编码 / 换行 | UTF-8无BOM / CRLF，末尾换行 |

样例 SHA256：`145f565567d1baf50746284520f53044451954e27f9c84a2a5fcc4a63f46e8fe`。

## 与此前提交文件的对照

下表路径均位于只读历史目录 `archive/legacy_20261016_2345/`，仅作为格式证据，不复用训练过程或把旧成绩当作新工程结果。项目当前主目录及云端备份未发现另一份实际生成的新工程提交 CSV。

| 历史文件 | 列数 | 数据行 / Study | 格式结论 |
| --- | --- | --- | --- |
| `submission.csv` | 4 | 14,140 / 1,414 | 与用户样例表头、列序及身份顺序一致；概率不同不属于格式错误。 |
| `submission_3col.csv` | 3 | 14,140 / 1,414 | 全部十类概率完整；按 E23 可省略 Subject_id，不能据此判错。 |
| `deliveries/baseline_20261016_01/submission.csv` | 3 | 14,140 / 1,414 | 完整，本地历史校验通过；没有与该文件绑定的平台接收证明。 |
| `submission_threshold_0p1.csv` | 4 | 6,056 / 1,414 | **缺8,084个检查/类别组合，不应提交。** |
| `submission_threshold_0p5.csv` | 4 | 2,101 / 1,158 | **缺256个Study及12,039个检查/类别组合，不应提交。** |
| `runs/gpu_smoke/outputs/submission_threshold_0p5.csv` | 4 | 3,198 / 1,393 | **缺21个Study及10,942个检查/类别组合，不应提交。** |

另有 campaign、convnext outputs/results 的历史副本，完整表或阈值过滤表分别呈现同类结果。共检查12份历史 CSV，逐文件哈希和结构审计见 [previous_submissions_audit.json](../artifacts/submission_format_audit_20261018_205755/previous_submissions_audit.json)，样例审计见 [reference_audit.json](../artifacts/submission_format_audit_20261018_205755/reference_audit.json)。没有读取测试答案或比较模型成绩。

## 当前导出代码的实际状态

主工程 `inference.py` 及 `ssh链接服务器/my_code_backup/cloud_project/inference.py` 默认输出 **三列完整长表**，可选在末列增加 Gate；其本地校验器也按三列或三列加 Gate 校验。主工程后续Tiny v2新增了schema4模型/精度恢复，CSV列结构保持不变。它们没有输出 Subject_id，属于与用户四列样例的差异，不能说已完全复现该样例；根据 E23 也不应直接标为格式错误。

本次按用户要求更新格式文档与审计记录，未修改导出代码、历史文件或既有代码 ZIP。若后续决定统一导出四列，应同时修改写出、回读校验和 provenance 的格式声明，并从测试清单正确映射 Subject_id；不能只给 CSV 补一个无对应值的列名。
