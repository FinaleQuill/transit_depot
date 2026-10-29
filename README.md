# 胸片多标签分类：ConvNeXt V1 离线蒸馏

**新窗口须重建环境：**Featurize已配环境会丢失，旧“已就绪”仅是历史验证。先读[新环境恢复及接手步骤](docs/NEW_ENVIRONMENT_HANDOFF_20261029.md)，不要直接盲用旧`/tmp`解释器或硬编码驱动挂载的容器入口。

**当前默认已采用等值加速（2026-10-30）：**Pillow同系数/22位定点舍入的原生单图并行LANCZOS，运行时AVX2/generic分派；精确uint8→FP16查表、固定TTA及一次概率回传。同环境新测v1本地124.754857ms→**81.509946ms/图**；NFS三轮80.781137ms，完整项目副本离线重建后82.190887ms。原100图/另5图预热、batch1串行三轮全部保留，逐图及Study概率与v1逐字节相同；033模型/config/原CSV未变，无训练或test重生成。证据在`../deliveries/037_dual768_native_optimization_20261029/optimization_summary.json`。

**项目运行准备：**源码根没有`vendor/`，新环境须按[最新恢复交接](docs/FINAL_OPTIMIZATION_HANDOFF_20261030.md)给`prepare_inference.py`传038的归档参数；独立提交项目副本则从包根运行`python code/prepare_inference.py --force-build`。在固定比赛镜像完成依赖/native准备后，再运行共享`predict.py`/`cost.py`。完整项目在`../deliveries/038_dual768_optimized_project_20261030/`，含原生源码/许可、离线依赖、两份初始化原件和七个真实训练run证据；只剩真实T4的`cost_report.json`待办。2080Ti和固定valid样本结果不替代主办方T4/官方样本复测。历史133.44ms及旧v2/v3/soup记录保留，不重做已拒绝候选。

当前路线：**V1 Base768五折教师 → 严格OOF逐视图TTA软标签 → 双Tiny768不同seed学生 → FP16固定hflip2/概率加权集成及官方核验**。

教师采用已冻结的混合轮数：fold0为20轮，fold1–4各12轮；集合入口为 `../runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_collection.json`。五折教师已于2026-10-26 19:26（香港）全部正常完成。
用户最新明确**双Tiny768集成**为优化目标。033已按用户授权完成真实组装、完整valid、4090三轮计时及完整test CSV：旧43占25%、新44占75%，各hflip2；实际FP16 valid AUC=.808384442、模型文件128,766,272bytes、本机端到端79.280300ms。最终CSV已生成并核验，1414Study×10类共14140行；见[CSV交付](docs/SUBMISSION_DUAL768_20261028.md)及[submission.csv](../deliveries/033_dual_tiny768_seed43_25_seed44_75/test_predictions/submission.csv)。未上传平台、未用test选择或拟合，正式T4与主办方最终复测仍待。valid/本机复盘见[真实部署复盘](docs/DUAL_TINY768_DEPLOYMENT_REVIEW_20261028.md)，生成前状态见[当时验收登记](docs/DUAL_TINY768_ACCEPTANCE_20261028.json)，CSV当前状态见[交付登记](docs/DUAL_TINY768_SUBMISSION_20261028.json)。

单44实际FP16=.807512423，033点差+.000872019、患者区间跨0；原50/50诊断=.808370073，与25/75仅差.000014369，不宣称稳定收益或25/75显著更优。027及此前参考保留，[预登记](docs/DUAL_TINY768_PLAN_20261027.md)和[选择记录](docs/DUAL_TINY768_SELECTION_20261028.json)不回写历史。共享逐图TTA输入已实现，CLAHE/DALI尚未应用或安装；本次真正双seed计时取代旧同run不同epoch的精度成员代理，但不能替代T4认证。教师、OOF、原模型和执行归档均保留。

**阶段验收约定：**每个大阶段完成后先核验结果、分析整体/逐类AUC与患者级不确定性、反馈是否需要修改，再确定下一阶段；033实际组合验证与本机计时已完成复盘，固定当前候选，不自动开始新训练、CLAHE或test选参。

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
