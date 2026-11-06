# A测速、候选CSV交付与B恢复

2026-10-30，Asia/Hong_Kong。用户本轮明确授权：先测A、生成A的submission.csv，结束后继续B。这取代此前“本阶段不生成新test预测”的范围限制；033原CSV及038安全基准仍保留。没有上传平台，没有依据test预测或榜单反馈调参。

## 实际结果

| 项目 | 结果 |
|---|---|
| 固定学生 | A：039 Tiny768/P2，E9最佳EMA，原始合规Tiny初始化 |
| 实际推理 | 单模型、FP16、768、identity+hflip平均概率，Study多图概率平均 |
| 官方valid Macro-AUC | 0.809115940852（033同机0.808385242530） |
| 2080Ti端到端 | **73.935760687ms/图** |
| 三轮均值 | 73.182002150 / 74.373183450 / 74.252096460ms |
| 每图三轮均值P95 / max | 91.400486116 / 97.613490332ms |
| 实际推理资产 | 66,996,559bytes（权重55,712,511＋config11,284,048） |
| CSV | 2218图、1414Study、14140数据行，另有表头 |
| 输出大小 | submission.csv 691,900bytes；逐图sidecar 1,352,956bytes |

CSV：`/home/featurize/work/deliveries/041_tiny768_p2_A_submission_20261030/submission.csv`。

CSV SHA256：`44dde95be64f42c151af7b23ee4a81ffb2158822eabb9f6249a59a73f795f66e`。
逐图sidecar SHA256：`3cb7e6bd1d936702b3487ec8d06c0f69e4f615ec25823d12f881c07ddb98f82a`。

仅验证点分略升，不是稳定提分结论。A的类1/4/8低于033，患者配对区间跨0，类1后期退化未修复；完整对比保留在`artifacts/kd_v1/student_upper_bound_20261030/reports/A/REPORT.md`。本次生成使用用户选定的固定A，不通过测试输出重新选择checkpoint、TTA、融合、阈值或预处理。

## 测速与部署身份

从A真实best.pt用未修改的`export_student.py`导出正式schema2单Tiny部署，182个FP16张量、27,827,818元素，与此前已完成valid的A实验导出逐字节相同；权重文件SHA也相同：`fea2350a479fc65883f687ce038289666345c5b0d5fc7da44576a5f995742b5f`。源checkpoint SHA为`65cdb72f63a5ec0d36486d4d8dcb34dd2fde5ed85aa52d3a10be2b0dbf633f42`。

测速调用未修改的`benchmark_local_resources.py`和正式Predictor；预测调用未修改的`predict.py`。同一固定镜像、AVX2/native、seed43，原034固定100张valid图及另5预热、batch1串行三轮，从读图到实际写CSV全部计时，初始化排除。实际SM75/68SM，8GiB allocator上限，无MPS/功耗或时钟调整；所有轮次均保留。

三轮概率、固定二值、排序及并列相同。每轮53个完整Study的530个概率，还与A原完整valid保存概率的FP64字节全部相同，最大误差0。

计时时B容器被整体冻结，GPU队列先排空并连续两次利用率0。B的CUDA上下文及约4.5GiB显存仍驻留，**没有B计算重叠，不能称完全空卡独占**。计时输入来自NFS，输出本地根盘。该数字为2080Ti开发诊断，不替代真实T4的cost_report.json，也未声称最终官方核验通过。

模型/config、全部三轮CSV及执行日志在`/tmp/cloud_student_runs_20261030/041_A_submission_20261030/`；CSV、sidecar、生成报告、独立审核和测速报告已复制到持久work的041目录。模型不再额外复制进空间吃紧的work；A最佳精确FP32 EMA及来源已有`artifacts/kd_v1/student_upper_bound_20261030/models/A/`备份，完整可续训断点仍在本地原run及完整滚动副本，换机前迁移。

## 独立完整性核验

预测前2218张原始test图全部与冻结manifest的file_sha256匹配。只做完整性读取，不计算测试类别先验、概率分布或测试指标，不改原CSV、原图或标签。

新独立`audit_student_submission.py`重新解析四列表头、原始身份、每图/Study的十类唯一覆盖、概率有限且位于[0,1]，再按原manifest顺序用FP64重算Study均值。14140提交行、22180逐图行、全部1414个Study的均值FP64字节相同，最大误差0；8项CPU测试通过。

## B暂停与恢复

没有修改任何冻结的训练、控制器或推理源码。新host协调器`run_a_submission_interlude.py`先整体冻结B所在容器，再只读取B MainThread栈；必须正面确认在forward/backward/optimizer计算中，并排除DataLoader取数帧，才保持冻结。这避开已有主线程180秒取数deadline；模糊栈立即恢复重试。边界判定3项CPU测试通过。

实际第一次门控即通过。B于**19:06:22.806**冻结，A测速与CSV/审核全部结束后于**19:09:53.537**恢复，暂停约210.73秒。训练内存、优化器、RNG和loader现场保留，没有将冻结冒称磁盘断点；没有重启B或回放已完成批次。finally恢复及独立15分钟host恢复watchdog均已配置，正常恢复后watchdog已结束。

恢复后仍是原host PID **171397**，日志从E1 6100推进至6200/18513、loss有限；原控制器PID13107继续正常运行。训练原始walltime将包含这段人为暂停，不改历史计时值。后续B仍仅使用官方train/valid，沿原批准配方继续训练和评估。

总执行/暂停栈/恢复证明：`artifacts/kd_v1/student_A_submission_20261030/execution.json`。
原图及模型逐bit核验：`artifacts/kd_v1/student_upper_bound_20261030/A_test_original_integrity_20261030.json`、`A_export_tensor_parity_20261030.json`。
测速对A完整valid字节复核：`artifacts/kd_v1/student_A_submission_20261030/timing_vs_frozen_A_valid.json`。
