# 内部验证患者影响与弱类诊断

`reconstruction/diagnostics.py` 是独立 CPU 诊断入口，不修改 E0/E1 冻结训练、控制器或评估实现，不读取图片、断点、H、官方 valid 或 test。

输入必须是一个已写完的内部验证 epoch 目录，包含 `predictions.npz` 和 `report.json`。入口核验固定 `split_plan.json` SHA、dev 的注册绝对路径与 SHA、3,703 Study／2,211 患者／6,070 图覆盖，再逐 Study 检查原始标签、患者、视角顺序和图数。NPZ 必须与 report 所绑定的 SHA 相同；报告 AUC 从预测重算。另一个参考目录走相同核验，再按 Study 身份对齐，任何标签或患者不一致均拒绝。

上述检查只证明预测及标签属于注册dev，**不等于上游权重血缘审计**。本工具不读取模型权重，不能独立证明训练器、教师、初始化或SSL从未接触H；这些仍须依据训练来源记录、冻结源码、断点哈希及控制器完整生命周期验收。

```bash
OPENBLAS_NUM_THREADS=1 python3 -m reconstruction.diagnostics \
  --candidate /home/featurize/work/reconstruction_runs_20261103/E0_base768_seed20261103/epoch_01 \
  --output /home/featurize/work/reconstruction_runs_20261103/diagnostics/E0_epoch01_example \
  --model-stage incomplete_training_epoch
```

示例只说明机制；不得把第1轮当完整模型效果。此前版本的真实机制产物在 `../reconstruction_runs_20261103/diagnostics/E0_epoch01_mechanics_v2/`，内含当时源码哈希；收口修订后未重跑真实数据。运行目录存在时拒绝覆盖。`--reference` 可指定另一份完整、相同 dev 的预测；`--model-stage selected_completed_run` 对候选和提供的参考均要求trainer与compliance终态匹配、完整连续history、best_epoch在历史范围内且该行AUC与报告、best值和历史最大值一致。输入status/history/compliance文件稳定读取并保存各自SHA，诊断结束再核对未变；仍不宣布外部泛化或最终部署结果。

输出包括以下并列诊断：

- 原始 Study 加权的整体／逐类 AUC，以及全部2,211患者的精确 leave-one-patient-out 结果。每次移除一个患者的全部 Study，重新计算每类AUC与十类均值。`removal_delta = AUC_without_patient − AUC_all`；缺正例或负例时对应值为 null，十类均值也为 null。
- 若提供参考，计算移除该患者前后 `candidate − reference` 差值的变化，避免把单臂高影响误当两臂差异来源。敏感性不是置信区间，不替代现有患者配对 bootstrap。
- 按绝对影响排序的患者、逐类高分负例／低分正例候选。候选不是已确诊错例或错标，不自动移除患者、修改标签、拟合阈值或开新实验。
- AP投影、单图／多图、严格PA＋LATERAL，以及显式包含LL侧位别名的PA＋侧位切片。后两项均排除其他／未知投影；PA LLD不自动映射。各切片报告阳性与阴性的Study、患者支持数。
- 患者等权敏感性：每个切片内，患者的每个Study权重取 `1 / 该患者在切片内的Study数`。保留所有Study及其原始标签，不做跨Study标签或概率平均；这不是官方主指标。

精确删除算法先将患者索引排序一次并分段建组，再按类别计算全体正负对的胜出数量（同分计半对），删除触及该患者的正样本对与负样本对，再补回被减两次的患者内部对。复杂度约 `O(C N log N + C Σ患者 n患者 log n患者)`，无需逐患者扫描全体Study或重新排序全体预测。

测试命令：`OPENBLAS_NUM_THREADS=1 python3 -m unittest reconstruction.test_diagnostics -v`。测试使用独立正负pair枚举验证ties、多Study混合标签患者及每次删除结果，并覆盖患者配对重排、标签／SHA／清单／覆盖拒绝与切片定义。全部采用CPU合成数据；真实示例仅读取已完成epoch的dev预测和元数据。
