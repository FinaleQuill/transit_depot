> 当前分支是 2026-11-03 用户授权的独立 Study 重构研究。入口：[实验协议](reconstruction/PROTOCOL.md)；运行产物：`../reconstruction_runs_20261103/`。下文历史工程状态保留供追溯。

**当前已停止训练（2026-11-04香港23:20）：** 用户明确要求安全停止并不再训练。E2局部第3轮已通过SIGTERM中断，保留第2轮完整last、最佳epoch0权重及第3轮日志/中断前进度；训练容器及控制器已退出，GPU无计算进程。所有自动交接已取消，STOP保留。**禁止依据下方历史队列或旧授权恢复训练、启动新实验或自动比较；需用户新的明确指令。** 真实保存核验见运行根 `USER_STOP_RECORD.json`。

**2026-11-04晚间提速：** E2全局已3轮早停；局部已进入第2轮，每轮约135分钟的主要原因是5倍区域计算。正在安排第2轮完整保存后进行限定600秒的激活保留探针，达标才采用，不减少图像/窗口/分辨率或重跑轮次；实际状态看 `controller/CHECKPOINT_SPEED_TRANSITION.json` 与[监督说明](docs/RECONSTRUCTION_E2_SUPERVISION_20261104.md)。这不是已取得真实整轮加速的声明，见[计算性能记录](docs/RECONSTRUCTION_E2_COMPUTE_SPEED_20261104.md)。

**2026-11-04后续用户修订：** E2从原固定6轮改为连续3轮无有效提升早停（min_delta=0.0001、最多6轮），既有轮次计入；第3轮完整保存后迁移，见[早停修订](docs/RECONSTRUCTION_E2_EARLY_STOPPING_20261104.md)与运行目录controller/EARLY_STOP_TRANSITION.json。最新答疑已明确允许同患者跨Study固定平滑；恢复为后续独立候选，当前E2不混入该后处理。

**完整规则入口：** 父目录中的 [rules.pdf](../rules.pdf)、[rules_1.pdf](../rules_1.pdf) 和[项目答疑记录](../RULES_QA_CLARIFICATION_20261104.md)共同构成当前完整规则，必须合并阅读。答疑在仓库docs中另留同步Git副本。

**2026-11-04 16:30执行提速：** E2全局臂已完整保存epoch1；新独立控制器PID36434从原断点续跑epoch2，将后续输入执行设置为6workers/prefetch2。科学配置、完整Study和训练预算保持；原11416已退出，勿重复启动。实际覆盖与测试见[性能记录](docs/RECONSTRUCTION_E2_PERFORMANCE_20261104.md)，当前进度和安全停止仍见[监督说明](docs/RECONSTRUCTION_E2_SUPERVISION_20261104.md)。

**2026-11-04 当前进展：** [E2匹配全局/局部对照](docs/RECONSTRUCTION_E2_PROTOCOL_20261104.md)于香港14:51独立后台启动，控制器PID11416、源码`18f90a8`。队列只执行全局继续训练6轮→全图＋四个原像素局部6轮→固定患者配对比较，各臂先评估epoch0；查看[监督说明](docs/RECONSTRUCTION_E2_SUPERVISION_20261104.md)及实际进程/进度，禁止重复启动。机械训练、断点恢复和chunk8显存测试通过，不代表已取得AUC收益；E3/E4不自动展开。

[E0/E1评估收尾](docs/RECONSTRUCTION_E0_E1_REVIEW_20261104.md)已完成：E0最佳第8轮0.797840，E1最佳第7轮0.799239，患者配对差值+0.001399、95%区间跨零。用户明确接受E1前12个完整epoch、第13轮不再跑；原paused/interrupted状态保留，禁止从旧后台说明恢复E1。E2从E0最佳EMA出发。

**2026-11-02最终复盘：** [当前阶段全局复盘与部署决策报告](docs/FINAL_STAGE_REVIEW_AND_DEPLOYMENT_DECISION_20261102.md)已完成。B_base 在完整 E3 后按用户 STOP 暂停，最佳 E1；全部收尾评估及 Study 计时队列均 completed，不再续训。融合 B−A 的 AP-only 差为 −0.001266786，患者95%区间 [−0.004323471,+0.001286890]，停止裁剪＋擦除分支。部署工程主候选冻结为045（Tiny768＋Base768，hflip2），043单Base唯一后备；不因本轮微小valid涨分换权重，不自动生成赛事CSV。

**新规计时实测：** 官方样例的045串行/Study batch为190.728/163.798ms每Study（2080Ti、旧镜像、10 Study、三轮），尚未满足100ms目标，亦非T4认证。下一方向只做冻结模型的Study批处理与FP16 TensorRT工程；新3.0镜像仅查manifest，未拉取/运行或构建引擎。详见[计时规则修订](docs/STUDY_COST_RULE_UPDATE_20261102.md)及最终报告。旧控制器STOP/failed和旧复盘blocked是人工终止记录，不能据此重启；实际收尾状态见 `artifacts/new_study_cost_20261102/finalization/STATUS.md`。

**2026-11-02此前修订（历史）：** 用户授权先完成A全部成员再自动B，最多25轮，AP-only耐心5轮/min_delta0.0001早停，保存最佳EMA；原两轮协议已被取代。新控制器会等待当前A_tiny首轮完成并安全迁移，不重复训练。请查 `artifacts/tight_crop_ab_20261102/earlystop25/STATUS.md` 和 [新运行说明](docs/TIGHT_CROP_AB_EARLYSTOP_RUN_20261102.md)，不要再次启动训练。

**2026-11-02此前启动（历史）：** AP优先的Tight Crop配对训练已授权并由后台控制器启动（启动PID 28152）。先查 `artifacts/tight_crop_ab_20261102/STATUS.md` 和进程，勿重复启动；计划与运行说明见 [A/B运行说明](docs/TIGHT_CROP_AB_RUN_20261102.md)。A/B对应同初始化、seed48、固定E2，B加p=.5随机擦除，自动输出AP-only详细报告；本轮不读test、不自动提交。下方冻结头和旧停止记录均是历史任务。

**2026-11-01最新运行交接：** 045最高valid CSV已交付；冻结旧A/B骨干仅训分类头的新任务已后台启动。用户因token不足要求保存进度，先读[冻结头交接](docs/FROZEN_HEAD_HANDOFF_20261101.md)及`artifacts/kd_v1/frozen_head_20261101/STATUS.md`，不要重复启动。旧bounded_finetune状态是历史已完成任务。

**2026-11-01最新任务：** 已按用户最高valid要求交付045（0.817149原解码CSV）；随后继续冲0.82，另登记冻结A/B骨干仅训练FC头的新路线。先读[045交付记录](docs/BEST_817149_SUBMISSION_20261101.md)与`docs/FROZEN_HEAD_PLAN_20261101.json`；旧“本轮不预测test”仅是前次复盘决定，已由用户新指令替代。

**2026-11-01复盘完成：** 两项固定4轮训练和实际组合核验已结束，最高valid 0.817149190/93.49ms，draft 0.816904529/87.10ms；未达0.82，增益区间跨0，原043/044保留、本轮无test预测。已保存实验模型，不重复启动后台训练；先读[最终复盘](docs/BOUNDED_FINETUNE_REVIEW_20261101.md)。

**2026-11-01最新授权与执行：** 用户已恢复提分/训练并允许最多12小时；旧停止说明是历史。当前固定4轮Tiny/Base EMA微调与Study打包，先读[本轮运行交接](docs/BOUNDED_FINETUNE_HANDOFF_20261101.md)，查实际进程再操作，不重复启动。

# 胸片多标签分类：ConvNeXt V1 离线蒸馏

**最新用户实测反馈（2026-10-31）：**用户报告044（valid 0.816677）的实际提交成绩不如043（valid 0.810570），或可视为基本持平，所述分差约0.01分。两版绝对平台成绩未提供，不换算成AUC差。**valid提分未体现为已证实的平台收益，不能默认044优于单B。** 详见[平台反馈记录](docs/PLATFORM_FEEDBACK_20261031.md)。当前只更新记录，停止优化与后续80ms目标保持不变。

**下一位agent接手入口（2026-10-31）：**先读[独立项目交接文档](NEXT_AGENT_HANDOFF.md)。当前044的valid AUC为0.8166770734、2080Ti平均84.820141ms，CSV及固定模型已交付并审计；用户已停止继续优化，80ms仍保留为后续目标。下方旧阶段说明保留历史语境，不能当作恢复实验的指令。

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
