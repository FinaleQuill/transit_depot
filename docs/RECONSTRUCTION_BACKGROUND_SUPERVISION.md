# E0 / E1 独立后台训练监督说明

交接时间：2026-11-04，香港时间。当前任务是等待既定后台结果，整个重构研究尚未完成。对话自动续行目标已暂停，不需要 AI 持续轮询；E2/E3/E4、多分辨率、水印干扰及弱类后续训练均等待用户回来结合结果决定。

## 实际运行与范围

复用了原有独立控制器，没有重复启动训练。实际训练队列严格为 **E0 → E1 → CPU 指标汇总与 1,000 次患者配对 bootstrap → 退出**。每臂最多25轮，至少8轮，按内部验证 Macro-AUC 及原登记耐心值早停；后臂仅在前臂训练、断点、生命周期和预测完整性核验通过后开始。

2026-11-04 00:58:54 香港时间现场核验：

- 控制器 PID 331369，PPID=1，自有 SID/PGID=331369，无 TTY；stdin 为 `/dev/null`，stdout/stderr 写入 `controller.launch.log`。关闭对话或终端连接不影响该进程，不依赖当前工具会话。
- E0 Docker 容器 `cxr-reconstruct-e0-1793717705_331369` 实际仍运行。第3轮完整 Macro-AUC=0.7774432876；第4轮已完成9,728/11,108 Study、15,921张图，累计1,347次有效更新、1次AMP跳步。E1尚未开始。
- 第4轮进度从00:44:25的3,616 Study增加至00:58:40的9,728，日志和状态持续更新；不是只验证PID存在。
- 全部冻结源码、配置和注册train/dev输入SHA核验一致；未写任何STOP标记。Docker未配置故障自动重启；宿主机重启、断电或进程错误不会自动无限重试。

现场证据位于运行目录 `recovery/LIVE_RECOVERY_20261103.json`、`recovery/E0_epoch02_checkpoint_verification.json`、`recovery/BACKGROUND_VERIFIED_20261104.json`。第2轮best/last实际文件SHA已与生命周期事件核对；后续训练会正常替换best/last，历史哈希仍留在ledger。

## 路径与保留的证据

| 内容 | 路径 |
| --- | --- |
| 独立代码工作树 | `/home/featurize/work/cxr_research_20261103` |
| 实际运行根目录 | `/home/featurize/work/reconstruction_runs_20261103` |
| 控制器和实时状态 | 运行根目录下 `controller/STATUS.json`、`controller/PROGRESS.md` |
| 固定队列/配置/源码哈希 | `controller/frozen_plan.json`；源码核心提交 `2cc037b`，启动记录 `ec0c7cc` |
| 当前逐臂配置 | 工作树下 `reconstruction/configs/E0.json`、`reconstruction/configs/E1.json` |
| 控制器启动与异常输出 | `controller.launch.log`、`controller_launch.json` |
| 训练及汇总日志 | `controller/logs/E0_*.log`、`E1_*.log`、`comparison_*.log` |
| E0产物 | `E0_base768_seed20261103/` |
| E1产物 | `E1_base768_seed20261103/`，启动后创建 |
| 每臂逐轮记录 | `history.jsonl`、`progress.json`、`epoch_XX/report.json`及`predictions.npz` |
| 每臂恢复与择优资产 | `last.pt`完整raw/EMA/optimizer/scaler/RNG/更新进度；`best.pt`最佳EMA；`configuration.json` |
| 每臂生命周期/源码快照 | `compliance/events.jsonl`、`compliance/status.json`及会话证据子目录 |
| 每臂完成核验回执 | `controller/E0_verified.json`、`controller/E1_verified.json` |
| 最终人类可读报告 | `controller/comparison/REPORT.md`，两臂完成后才生成 |
| 最终完整数值 | `controller/comparison/report.json`，含整体/十类、AP投影、单/多图、配对区间和最佳轮来源 |

两臂配对的是各自在fold0所选best，区间是探索性证据，不校正反复选模偏差。PA＋侧位及患者影响的新诊断工具已保存，但未加入后台新决策队列；用户返回后可对完整结果补做。局部模块已准备但没有排入训练。任何阶段都不自动读取H、官方valid或test，不自动部署或提交。

训练核心源码仍是冻结版本。恢复期间新增水印诊断、局部准备与患者诊断分别保存为 `ee963a3`、`1dd5387`、`5ee18a2`，没有热改运行中的训练模块。

## 查看进度和日志

以下均是手动只读命令，不启动训练：

```sh
CXR_RUN=/home/featurize/work/reconstruction_runs_20261103
cat "$CXR_RUN/controller/PROGRESS.md"
cat "$CXR_RUN/controller/STATUS.json"
docker ps --no-trunc
tail -n 40 "$CXR_RUN"/controller/logs/*.log
tail -n 40 "$CXR_RUN/controller.launch.log"
```

只看逐轮指标：

```sh
cat /home/featurize/work/reconstruction_runs_20261103/E0_base768_seed20261103/history.jsonl
cat /home/featurize/work/reconstruction_runs_20261103/E1_base768_seed20261103/history.jsonl
```

E1启动或首轮完成前，其目录/history不存在是正常现象。`STATUS.json` 的 `log_path` 指向当前任务日志；`trainer_progress.state` 区分训练与内部验证。ETA只外推当前训练轮，不是整个队列完成时间。

两臂完成后查看最终报告：

```sh
cat /home/featurize/work/reconstruction_runs_20261103/controller/comparison/REPORT.md
cat /home/featurize/work/reconstruction_runs_20261103/controller/comparison/report.json
```

## 完成、失败与安全停止

正常结束须同时看到控制器 `state: completed` 和最终数值报告 `complete: true`。控制器会先核对两臂真实完整history、生命周期结束记录、best/last及预测SHA；退出码0本身不算实验完成。汇总失败也会留下 `failed` 状态及日志。已成功完成的证据不因后臂失败而删除。

若需要提前安全停止整个队列，只执行：

```sh
touch /home/featurize/work/reconstruction_runs_20261103/controller/STOP
```

**交接时没有执行此命令。** 控制器约30秒内转交STOP，当前训练继续完成完整epoch、开发集评估、断点和history写入；等待实际容器退出后标为 `paused`，后续E1或汇总不再启动。STOP可能需要等待一整轮，不能按立即停止理解。若已在CPU汇总阶段，控制器会停止汇总；训练断点仍保留。

应写控制器STOP，不直接写单臂STOP；单臂暂停会被控制器认定为未成功而记为failed。不使用 `kill -9` 或 `docker stop` 代替完整轮停止。停止后保留STOP及所有证据，用户回来后先核对实际进程、终态和恢复断点，再决定显式恢复；不要复制启动命令或只凭旧PID重跑。

进程错误会保存现场并终止队列，没有无限重试、自动扩展实验或重启机制。正在运行时也不要编辑 `controller/frozen_plan.json` 登记的源码、输入或配置，否则哈希保护会请求安全停止。工作盘交接时约剩8.8 GiB；现E0/E1在无其他大写入时有余量，后续多臂存储另作安排。

最终研究仍需额外开发折/seed、有限H确认及实际加载资产≤500,000,000字节、100ms/Study验收。本次仅移交既定后台队列，不能把其启动或中间AUC当作整个目标完成。
