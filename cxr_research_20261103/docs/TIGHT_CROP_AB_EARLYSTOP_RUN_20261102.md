# AP 优先早停与 A 完成后衔接 B

用户在原 A_tiny 首轮进行期间明确修改了实验预算：两轮不足以充分观察适应效果，
改为最多 25 轮，连续 5 轮没有超过 0.0001 的 AP-only Macro-AUC 提升就早停。
这一修改登记在任何新一轮完整 valid 指标产生之前，取代原来的固定 E2 选择规则。
机器配置见 [早停修订计划](TIGHT_CROP_AB_EARLYSTOP_PLAN_20261102.json)。

## 调度与选择

固定顺序为 **A_tiny → A_base → B_tiny → B_base → 自动汇总评估**。
A 的两个成员都完成或早停后，才允许启动 B。每个成员监控自己每轮的 AP-only
Study Macro-AUC；最终组间对照仍使用固定 Tiny/Base 融合，不改变融合权重。

- 最多 25 轮；`patience=5`；`min_delta=0.0001`；方向为越大越好。
- 超过早停参照分数加 min_delta 时重置计数，否则计数加一；第一轮建立参照。
- `best.pt` 保存观测到 AP 指标最高的 EMA，平分保留较早轮；min_delta 只作用于
  耐心计数，不能丢弃虽较小但实际更高的最佳 checkpoint。
- `last.pt` 保存完整模型、优化器、scaler 和随机状态，用于故障恢复。
- A/B 对应成员保持同初始化、seed、数据、共同增强和最大预算；早停后实际轮数
  可以不同，报告须同时列出训练轮数、最佳轮数、停止原因及 AP 分数。
- B 仍使用已审阅的主体 Tight Crop 和 p=0.5 的随机擦除，valid 不擦除。

## 保留正在进行的首轮

原 A_tiny 的冻结源码、两轮配置及原 checkpoint 不改写。已在首轮进行中写入 STOP，
旧训练程序会完成整轮训练、valid 评估和完整保存，再在下一轮开始前暂停。
旧控制器遇到这个 STOP 后不再启动 B；这是已登记的调度迁移，不是训练算法故障。

新控制器已设计为先等待旧任务与旧调度器自然退出，无需给任何训练进程发终止信号。
随后核对原 E1 checkpoint 的 SHA、完整进度和源配置，将
模型、EMA、优化器、scaler、RNG 和进度迁移到独立的 `earlystop25/A_tiny`。
迁移只写新产物，保留原始证据及迁移来源 SHA。

第一轮 warmup 的学习率只依赖当前更新数和 warmup 长度，不依赖总轮数，所以
这轮的实际更新与同配置的 25 轮计划一致。第二轮起采用 25 轮 cosine 计划。
这是明确记录的计划迁移，不冒称原两轮计划的原样继续。

E1 的 AP 分数使用与后续训练验证相同的 FP32 EMA、FP16 autocast、batch4、hflip2
重新计算；不能拿最终部署的全 FP16/batch1 分数混入早停曲线。最终对照再用固定的
FP16/batch1 推理，以报告真实候选输出。

## 结果解释与运行入口

AP-only 是主要终点：完整十类 AUC、每类阳性/阴性数、单图与多图 AP 切片，
患者配对 Bootstrap 1000 次。总体分数不涨不自动否决 AP 的改进。
AP-valid 同时参与了早停和最佳轮选择，因此最终分数及 Bootstrap 是探索性结果；
它们不校正选择偏差，也不能单独证明线上泛化。保持 valid 独立于梯度训练，不读 test。

- 新控制器：`run_tight_crop_ab_earlystop_pipeline.py`
- 新训练：`tight_crop_ab_earlystop_training.py`
- 新评估：`tight_crop_ab_earlystop_evaluate.py`
- 本地运行根：`/tmp/cloud_student_runs_20261030/tight_crop_ab_20261102/earlystop25/`
- 持久化根：`artifacts/tight_crop_ab_20261102/earlystop25/`
- 实时状态：该持久化根下的 `STATUS.md`、`status.json`。

控制器应后台独立会话运行，防重复启动；每轮保存并核验断点，故障恢复至多一次。
达到早停或 25 轮上限后自动衔接下一成员，不需要用户再次输入命令。
由于预算增加，不能沿用两轮实验的用时估计或承诺明早全部结束。
