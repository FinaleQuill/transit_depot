# 当前阶段：E13教师诊断与独立BCE学生对照

2026-10-27（香港）更新。027 KD512已正常12轮早停，E7最佳.805593834、实测约86分钟；当前按[KD512验收](KD512_REVIEW_20261027.md)只导出/验证E7。下方旧命令是阶段记录，不要在已有目录重跑。BCE实测11轮约80分钟，KD每轮含验证/保存约7分06秒；短12轮实验仅候选，预计1–1.5小时。不自动串联GPU任务，每阶段结束后先复盘。

## 已完成的教师诊断，保留原命令

```bash
cd /home/featurize/work/cloud_project_v2
/home/featurize/.conda/envs/cxr-kd-v1/bin/python -u diagnose_teacher_epochs.py \
  --output-dir ../runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_epoch_logits_v2 \
  --execute
```

建议在已有tmux会话中运行。去掉`--execute`只预览，不加载检查点、不创建输出、不使用GPU。执行前会检查原五教师全部完成、患者隔离及GPU占用；已有输出目录不能覆盖。

本阶段读取原五个最终EMA（20/12/12/12/12）及fold0保留的E13 EMA，只预测官方valid的1422张图，不训练、不生成OOF、不改教师集合。原验证只保存Study概率，这次为保存逐图双视图原始logits重新前向。E13保留真实的13/20轮与未完成标志，不能直接冒充原固定轮数教师。

输出目录包含：

- `raw_logits/*.npz`及配套JSON：六组`N×2×10` FP32原始logits、原图/患者/Study身份、标签、原图SHA、检查点与视图来源。FP32描述保存格式；实际前向精度另行登记，4090默认BF16。
- `raw_index.json`：六组文件完整性及来源索引。
- `teacher_epoch_diagnostic_report.json`和Study预测：A=(20,12,12,12,12)对照B=(13,12,12,12,12)，分别比较identity/hflip2、原始概率/训练pos_weight逐图校正概率。校正先逐视图做`sigmoid(z-log(w_train))`，再平均；它是待验证候选，不保证校准改善。
- `evidence/`：本次命令、日志和源码快照。

运行结束后将结果交回分析。比较B不等于已经采用B；不自动替换E20、不拟合阈值、不用test选方案，也不会据此宣称已有严格OOF缓存。

已有raw logits以后可仅用CPU重算这些预定方案，无需再次跑教师：

```bash
CUDA_VISIBLE_DEVICES='' /home/featurize/.conda/envs/cxr-kd-v1/bin/python diagnose_teacher_epochs.py \
  --replay-dir ../runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_epoch_logits_v2 \
  --output-dir ../runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_epoch_cpu_replay_v1 \
  --execute
```

## 下一阶段：先有独立纯BCE基线

教师诊断复盘后再确定正式训练指令。新入口为`train_distill.py --training-mode supervised_bce --alpha 0`，不传`--soft-labels`，无需先提取教师缓存。默认仍是KD模式，避免旧蒸馏命令静默改变含义。

基线保留Tiny512、ImageNet-22k→1k初始化、seed43、pos_weight、Study等权、有效batch32、原增强/学习率/EMA、最多25轮和patience5、官方valid双视图选EMA。与后续同配置KD512只比较有无蒸馏，不能同时换损失或数据。S9检查点明确记录纯监督、无教师来源；导出、固定TTA/集成和核验材料保留这一事实。KD旧S7/S8来源检查仍保留。

先比较BCE512与α=.2/T=2的KD512；确认KD是否有益后，再决定KD768和额外seed。当前学生候选是完整train上的独立成员，不是学生交叉验证折。OOF提取和后续多成员入口仍见[原TTA/集成手册](KD_TTA_ENSEMBLE_USAGE.md)，须先完成本页复盘再使用。

## 各阶段大致耗时

以下按当前4090、单任务、数据已解压、预训练权重已缓存估算。人工复盘等待、首次环境安装和故障重试不计入；学生尚无本机实测，区间用于排期，训练前2轮后按日志更新。

| 阶段 | 大致耗时 | 依据或边界 |
| --- | --- | --- |
| 本次六检查点valid逐图双视图logits诊断 | 5–10分钟 | 上次五教师同1422图双视图全程3分39秒；本次多E13及文件校验/保存 |
| 已存logits的CPU复算与统计 | 普通复算1–3分钟；含1000次患者bootstrap约5–15分钟 | 无需模型/GPU，统计次数影响耗时 |
| 严格OOF双视图提取 | 20–40分钟，一次 | 30328图，每图仅对应折外教师预测；共60656次单图等价前向，并非五教师各跑全train |
| Tiny512纯BCE基线 | 约2–4小时/个 | 最多25轮、每轮验证；早停会缩短，尚未实测 |
| Tiny512蒸馏 | 约2–4小时/个 | 离线缓存，训练时不加载教师；KL额外计算较小 |
| Tiny768蒸馏 | 约4–7小时/个 | 像素数为512的2.25倍，实际还受微批/读图影响；尚未实测 |
| 后续额外seed学生 | 每个512约2–4小时；每个768约4–7小时 | 只有验证收益和T4预算支持才补训 |
| FP16导出及单候选官方valid验证 | 约5–15分钟/候选 | 单模型较快，多成员/TTA更慢；CPU导出本身通常1–3分钟 |
| T4官方口径计时与材料核验 | 环境已就绪约10–30分钟/候选 | 实际计时段为100图×3轮另加预热，环境上传/安装另计；必须真实T4验收 |

若最终依次做BCE512、KD512、KD768三次学生训练，按满预算预留约8–15小时GPU时间，另加OOF/验证；不会默认全部启动。教师已完成，近期无需重训。现有Base768训练实测约17–18分钟/轮（每折约2.4万图），学生估算结合模型规模和分辨率推算，不能当作实测性能。

部署体积和≤100ms仍以最终实际导出、T4端到端结果验收。离线训练耗时与官方单图推理时限是两个不同指标。
