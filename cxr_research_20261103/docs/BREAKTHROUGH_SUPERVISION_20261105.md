# DINOv2 独立训练监督与恢复

用户2026-11-05明确授权自主接管、新训练、先保存后清理；该授权覆盖历史暂停，旧队列未恢复。

正式源码：`d2217a6`；完整策略见[突破方案](BREAKTHROUGH_PLAN_20261105.md)。香港时间2026-11-05 20:04:09启动独立控制器PID218075；容器 `cxr-breakthrough-dinov2-20261105`。关闭对话不会终止这个独立会话，主机或容器停止仍会中断计算。勿仅看PID存在判断有效训练。

运行根：`/home/featurize/work/reconstruction_runs_20261103/BREAKTHROUGH_20261105/DINOv2_B518_fold0123/`。

| 文件 | 用途 |
|---|---|
| `progress.json` | 当前epoch、覆盖Study/图片、loss、实际optimizer_updates与skipped_updates、单轮估时。评估期间会切换为evaluating_development。 |
| `training.log` | 训练器stdout/stderr，10次更新一次常规记录。 |
| `controller_status.json` | 独立进程身份、完整Docker命令、退出码；不自动重试失败实验。 |
| `history.jsonl` | 完整epoch训练/开发结果；首轮未完成前不存在是正常的。 |
| `epoch_XX/report.json` / `predictions.npz` | 开发AUC、患者等权诊断、逐类与切片及完整Study预测。 |
| `last.pt` | 每轮完整可恢复断点的持久化副本，原子替换；本地state另有一份。 |
| `best_fp16.pt` | 最高开发AUC的候选，不是已经验证的最终提交模型。 |
| `status.json` / `failure.json` | 轮次/最后状态或异常；不能忽略controller非零退出。 |
| `compliance/` | 本次真实训练事件、源码和环境证据；每轮/正常退出从本地同步。 |

本地完整状态：`/home/featurize/data/cxr_breakthrough_20261105/state/DINOv2_B518_fold0123/`。数据原件只读，容器无网络。

正式实验最多18轮、至少8轮、patience5/min_delta0.0002，有效32 Study；选择只看fold4开发Macro-AUC。fold4已不再作为独立H；不逐轮打开官方valid/test。全部14,811训练Study每轮一次，包含全部图片；末批按实际Study数归一化。最后部署体积预算500MB，当前DINO FP16约173.2MB。

本次已经完成的机械验证：官方容器15项CPU检查通过；64Study/101图的518 GPU前反向、EMA、完整保存通过；pack16/chunk8另外通过；真实GPU从last继续到第二epoch成功，总有效更新2→4、跳步0。首次pack4/chunk2约24秒、pack16/chunk8约26秒（均很短，非可靠整轮benchmark），因此正式先选pack4/chunk2，保留约3.3GB实测峰值和完整数据覆盖。resume的较长耗时含本次文件系统等待，未用作正式吞吐推算。

安全停止：在运行根创建 `STOP`，训练器在当轮训练、验证和完整断点保存后退出；不需要中断GPU计算。恢复前检查实际进程/锁、failure和最后完整checkpoint，明确保留源码/配置/数据SHA；再用 `python -m breakthrough.launch --config breakthrough/plan.json --execute --resume`。原有STOP需在明确恢复决定后处理。禁止同时启动第二个同名run。若本地盘丢失，先把工作区`last.pt`恢复到配置规定的checkpoint_dir，并恢复已归档的依赖与公开初始化；恢复验证会检查证据一致。

本轮完成后输出是开发结果和可恢复模型；下一步依照方案审查证据，再执行患者折复验或固定全train重训。后台不自动扫描新融合比例，也不上传比赛平台。


2026-11-05 20:37补充：独立CPU配对诊断已运行，PID225021，状态在运行根上级`reassessment/development_comparisons/monitor_status.json`。每个完整epoch后与已核验fold4 CNN参考比较；36小时有界等待、无GPU/无训练改动/无自动提交。原训练源码与配置SHA已复核未变。真实平台反馈和更严格采用原则见[平台泛化复审](PLATFORM_GENERALIZATION_REASSESSMENT_20261105.md)。


**2026-11-05 21:25最新状态：用户明确立即停止且不要再训练。** 当前DINO已收到SIGTERM退出，CPU比较监控亦停止；STOP保留。第2轮中断，只有第1轮完整断点/评估。禁止依据上方历史启动/恢复命令继续执行。未来需新的用户训练授权，并改用官方原始未清洗数据；本次没有启动新训练。
