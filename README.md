# 胸片多标签分类：ConvNeXt V1 离线蒸馏

当前路线：**V1 Base768五折教师 → 严格OOF逐视图TTA软标签 → 双Tiny768不同seed学生 → FP16固定hflip2/概率加权集成及官方核验**。

教师采用已冻结的混合轮数：fold0为20轮，fold1–4各12轮；集合入口为 `../runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_collection.json`。五折教师已于2026-10-26 19:26（香港）全部正常完成。
用户最新明确**双Tiny768集成**为优化目标。029 E8实际FP16验收AUC=.803983596；027 Tiny512=.805402258；二者预定50/50概率平均=.807157435、六类升四类降，患者配对区间跨0，保留为探索性精度参考。下一项只交接030 Tiny768 seed44，从同一ImageNet新训，与029仅seed/output不同；已CPU预览、未启动。完成后先复盘，再验证双768的50/50及25/75、75/25全局概率权重，不按十类大量搜权重。当前结果、下一训练及监控命令见[双Tiny768计划](docs/DUAL_TINY768_PLAN_20261027.md)；原生曲线见[Tiny768训练复盘](docs/TINY768_REVIEW_20261027.md)。共享逐图TTA输入已实现，CLAHE/DALI尚未应用或安装；4090计时代理约80ms不构成实际T4认证，参见[资源压测说明](docs/RESOURCE_STRESS_USAGE.md)。旧模型、教师与OOF保留。

**阶段验收约定：**每个大阶段完成后先核验结果、分析整体/逐类AUC与患者级不确定性、反馈是否需要修改，再确定下一阶段；029实际FP16与固定融合诊断已完成，本次只交接030新训指令，不自动连跑其它阶段。

## 操作入口

| 阶段 | 入口 | 说明 |
| --- | --- | --- |
| 当前教师计划 | `run_mixed_teachers.py`、`run_mixed_teachers.sh` | [现行混合教师操作说明](docs/KD_V1_MIXED_USAGE.md)；不要重复启动运行中的控制器 |
| 教师训练 | `train_teacher.py`、`run_teachers.py` | 固定轮数最终EMA，外折不参与训练或选模 |
| 当前教师诊断 | `diagnose_teacher_epochs.py` | E13单因素对照、逐图双视图原始logits、CPU重放；不替换教师 |
| OOF软标签 | `extract_soft_labels_tta.py`、`distill_tta_data.py`、`teacher_collection.py` | 两视图原始logits，逐图严格折外；保留旧identity入口 |
| 学生对照与蒸馏 | `train_distill.py`、`run_student_experiments.py` | 独立BCE512先行；KD的α=.2、T=2；双视图验证、EMA及完整恢复 |
| 导出与验证 | `export_student.py`、`compose_students.py`、`validate_student.py`、`validate_teacher_tta.py` | 实际FP16、固定TTA与1–4成员，官方valid诊断 |
| 最终推理与计时 | `predict.py`、`cost.py`、`deploy_kd.py` | 共享单图预测，Study概率汇总，官方T4口径 |
| 核验材料 | `build_training_evidence.py`、`package_kd_submission.py`、`audit_trail.py` | 每个实际部署成员的来源、日志、源码与checkpoint证据 |
| 概率比较 | `compare_student_candidates.py` | 十类AUC、Study切片及可选患者配对bootstrap |
| 准备检查 | `preflight_kd.py` | 只读清单/环境检查，不启动训练 |

训练环境：`/home/featurize/.conda/envs/cxr-kd-v1/bin/python`；依赖见 `requirements_kd.txt`。
官方容器推理依赖见 `requirements_inference.txt`。新训练、提取入口默认预览，显式 `--execute` 才执行。

## 公共模块

- `model_overnight.py`：合规V1 Base/Tiny模型白名单。
- `dataset.py`、`dataset_overnight.py`、`cv_splits.py`：清洗清单、原图加载与患者隔离。
- `train_overnight.py`：现行教师、学生和验证程序仍复用的训练/损失/保存工具；文件名沿用历史，不是待删旧入口。
- `kd_common.py`、`runtime_compat.py`、`planned_handoff.py`：来源登记、运行兼容与既有训练交接。

## 数据与材料

原图 `/home/featurize/data/cxr_first` 和 `data/` 清单保持只读；train为30328张、18513个Study，valid为1422张、859个Study。
模型与训练状态在项目外 `../runs/`，交付在 `../deliveries/`。

[官方规范清单](docs/OFFICIAL_RULES_CHECKLIST.md) · [材料留存说明](docs/KD_V1_COMPLIANCE.md) · [学生/部署步骤](docs/KD_V1_USAGE.md) · [环境准备说明](docs/KD_V1_START_HERE.md)

真实训练的源码快照、日志、环境和逐轮哈希仍保留在证据目录，供官方复现核验使用。

## 历史记录与清理

2026-10-26已删除废弃B3/高分辨率旧方案、旧Tiny/V2训练与推理入口、旧启动打包脚本及不再使用的测试；不修改当前训练依赖和已有训练产物。

[清理说明及删除清单](docs/history/CODE_CLEANUP_20261026.md) · [清理前README原文](docs/history/README_BEFORE_CLEANUP_20261026.md) · [完整工作记录](WORKLOG.md)

`docs/`中的旧方案说明与`artifacts/`中的文字报告保留原始时间语境。旧命令只用于复盘，不作为当前启动入口。
