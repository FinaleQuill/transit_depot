# 原始数据DINO：用户停止后的独立保存核验

2026-11-06，香港时间。用户最新指令为“已经收敛了。停止训练，分析效果”。本次只读核验进程、已有日志/预测身份、源清单和断点；仅使用无GPU、无网络、输入只读挂载的官方CPU容器加载权重，没有执行训练、模型前向、恢复或新预测。

**结论：训练已停止；完整第1–21轮均已保存，最佳为第20轮，开发Macro-AUC 0.8049053322。第22轮只处理5,920/14,811 Study，不能计作完成。完整恢复点与最佳权重均完整可读、哈希一致，历史提交和原始CSV未改变。**

运行目录：`reconstruction_runs_20261103/BREAKTHROUGH_20261105/RAW_REBUILD/DINOv2_B518_fold0123/`。完整断点目录：`/home/featurize/data/cxr_breakthrough_20261105/state/RAW_REBUILD/DINOv2_B518_fold0123/`。

## 停止事实与状态解释

- `USER_STOP_ANALYSIS_20261106.json`登记12:55:05.604收到用户停止指令，随后`docker kill --signal=SIGTERM cxr-raw-dinov2-20261105`成功返回0。`STOP`文件保留。
- 训练证据`compliance/status.json`为`interrupted`，最后事件12:55:06.942记录`KeyboardInterrupt: Training interrupted by signal 15`。控制器12:55:09退出130，原`controller_status.json`机械记作`failed`。
- 本次独立检查时，`docker ps`无运行容器，`nvidia-smi --query-compute-apps`无GPU计算进程，原控制器314314、docker客户端314316均已退出。
- 日志中的DataLoader worker `Aborted`出现在信号中断后的worker关闭路径。结合明确的SIGTERM、KeyboardInterrupt、0跳步和有限模型权重，这是主动停止过程中的清理异常，不能据此说训练数值崩溃。
- 最后`progress.json`仍写`training/epoch22`，`status.json`仍写`epoch_completed/epoch21`，两者是被中断前的最后快照，**不能据此断言当前仍在训练**。

建议保留原失败/中断日志，另用本次停止收尾记录声明`user_stopped`、`last_complete_epoch=21`、`selected_epoch=20`、`partial_epoch=22`、`restart_authorized=false`；不要把原控制器失败状态改成训练成功，也不要因它为`failed`而自动重启。

## 完整轮次与部分轮次

`history.jsonl`逐项核验为1–21，无缺轮、重复或伪造第22轮。每轮完整优化14,811 Study/24,277图，每轮463次有效更新、0跳步。

| 项目 | 已保存事实 |
|---|---|
| 最后完整轮次 | 21，12:39:18报告完成 |
| 第21轮Macro-AUC | 0.8048998595020104 |
| 最佳轮次/分数 | 20 / 0.8049053321892679 |
| 第21轮完整断点累计更新 | 9,723，跳步0 |
| 第21轮早停计数 | stale=1，meaningful=0.8049053321892679 |
| 第22轮停止前快照 | 5,920/14,811 Study，9,736图，累计9,908次更新，跳步0 |
| 第22轮完整结果 | 没有epoch22预测/报告/完整checkpoint，不计入效果比较 |

第22轮快照的9,908是当时内存中已执行的累计更新；可恢复的完整断点保留9,723。中途185次更新没有单独恢复断点，不得把第21轮last描述成包含第22轮部分结果。用户已要求停止，不恢复该部分训练。

## 断点和最佳权重的独立核验

使用固定官方镜像`sha256:125d8bcb4ce5a2a9acec6d9838b37e2ae1e2014c09b848700f9530f2f7856a0b`，`torch.load(..., map_location='cpu')`逐一加载；容器没有`--gpus`、没有可写数据挂载。

| 文件 | 大小（字节） | SHA256 |
|---|---:|---|
| 本地last.pt | 1,385,946,557 | `ae16479adb4b5c2c363fd032edf641d5bd437b57644d85c8f0d30b4b97ef8dda` |
| 本地best.pt | 346,479,833 | `3692e1a28a672fee7c2dee957147d35491cf786803122e3ae7fde269928d3e9c` |
| 本地best_fp16.pt | 173,258,457 | `f296418d25051bc2c7aff8425673be034938879a67c00a819439f4a140425172` |
| 工作区best_fp16.pt | 173,258,457 | `f296418d25051bc2c7aff8425673be034938879a67c00a819439f4a140425172` |

last的哈希与`recovery.json`及第21轮compliance保存事件一致；其内嵌epoch21、best_epoch20、完整model/EMA、179个Adam参数状态、scaler、Python/NumPy/Torch/CUDA随机状态及更新计数均存在。model和EMA的全部浮点tensor均有限，内嵌evidence与当前`configuration.json`完全相等。

best和FP16均内嵌epoch20及同一Macro-AUC。逐tensor检查FP16恰为best EMA的half转换，非浮点tensor逐值相同；不是只验证文件名或元数据。单模型权重约173.26MB，最终包仍须另计配置/任何其他推理资产和正式时延，当前不是最终交付验收。

完整last保存在本地`/home/featurize/data`，未复制到持久工作区；`recovery.json`也明确写`full_checkpoint_persisted_to_workspace=false`。持久工作区有最佳FP16及证据/预测。不能将本地完整状态描述成远端持久备份。

## 原始数据覆盖、配置和血缘

重新读取并交叉检查三份清单：

| 用途 | 图片 | Study | 患者 |
|---|---:|---:|---:|
| 官方原始train全量 | 30,344 | 18,513 | 11,055 |
| 优化fold0–3 | 24,277 | 14,811 | 8,844 |
| 开发fold4 | 6,067 | 3,702 | 2,211 |

训练/开发image_id并集恰为全部30,344张原始清单；患者集合不相交；所有source行`exclusion_reasons=[]`、`training_eligible=True`。没有沿用历史16张排除，没有把开发划分冒充删样本。fold4为开发，不是独立最终确认。

内嵌配置仍为上限23轮、原18轮学习率日程、min4/patience4/min_delta0.0002，518输入、有效batch32、pack4/chunk2。last保存的学习率最大2e-5、最小6.8719476736e-8，与原低LR末端/层衰减一致。26份登记源码哈希全部未变，六项输入（source/train/dev/folds/公开初始化/provenance）哈希全部与冻结配置一致。

内嵌evidence继续声明原始图30,344、official valid未用、test未用；最终保存事件正确关联新raw训练器。18→23授权记录、父18轮备份身份和新旧evidence保留在`extension_23_20261106/authorization.json`；本次不修改原血缘或历史终态。

本轮原始CSV均重新计算SHA，与`/home/featurize/data/raw_rebuild_20261105/audit.json`原始登记一致：

- train：`3a07b4bb5c81758718d4ccb69885c9160103aeb55ab8206c71fef152de334f17`。
- valid：`7e1930138718eee4595f48820c26b69d95fb021bc9d48b46006c0aacd2415df6`。
- test：`63e107d840cbf27aee7c1c0ced1157aef454baa1141fdc3851e6c3011471529e`。

本轮没有重新逐图解码/重哈希全部原图，原图完整性结论依赖既有全量准备审计、冻结清单、训练时逐图校验和只读挂载；不冒称新增全量图像审计。

## 旧晋级文件与历史提交

运行根`stage_decision.json`仍为**18轮正常结束时的旧文件**：best_epoch18、training_complete=true、两个变体eligible_for_second_fold。延长过程在第22轮中断，没有运行新的正常终态晋级写入。因此该文件不能代表第20轮最佳的最新诊断，更不构成恢复训练/第二折训练的授权。应保留原文件并在本次效果复盘中明确“历史18轮晋级判断”；若读取第20轮已有comparison作停止后分析，应另写分析结果及用户截断范围，不冒称完成原23轮预算。

重新核验五份根提交文件，与前次审计完全相同：

| 文件 | SHA256 |
|---|---|
| submission.csv、submission4.csv | `3ff38f2c4c248ce4913c712f30f89ad53d54b98a2e82acb1ac458517833e99d8` |
| submission_best.csv | `44dde95be64f42c151af7b23ee4a81ffb2158822eabb9f6249a59a73f795f66e` |
| submission_robust_0818992.csv | `fedbbbef4fd45763a8f88bc2a7bfab36e6bc3ecd33bce0487ad54eaa7da29d5f` |
| submission (2).csv | `ea83de14afb8a8438428dc071b29438b2201a7113cad03aa4181de09284b3661` |

没有新测试CSV或平台成绩，本次不能将开发0.804905称为测试80.49分；已知平台最佳041仍在`submission_best.csv`。本报告仅确认停止与保存事实，效果差值/置信区间应引用对应同患者开发比较，不能据旧stage文件自动继续实验。
