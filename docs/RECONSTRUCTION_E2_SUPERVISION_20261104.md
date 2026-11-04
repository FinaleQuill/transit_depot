# E2 后台队列监督说明

2026-11-04香港16:30执行提速更新：原控制器在全局臂完整epoch1及评估/断点后暂停，新独立控制器PID36434从原last.pt续跑epoch2；后续两臂实际采用6个读图进程、每进程prefetch2。注册配置中的workers2保留历史值，实际覆盖见 `performance_20261104/loader_workers6_prefetch2.json`、`controller/performance_plan.json` 和新会话的 `execution_profile` / `data_loader_created`。源码、科学配方和原history不改。启动记录改看 `performance_launch.json`，最终实测见 `performance_20261104/LIVE_SPEEDUP.json`；详细说明见仓库 `docs/RECONSTRUCTION_E2_PERFORMANCE_20261104.md`。原11416只作历史定位，不要复活旧入口。

16:36核验已完成：连续640Study的稳定速度2.945→4.350Study/s，约1.48倍；等待占比57.45%→0.61%。这是跨轮观测区间，局部正式臂尚未启动，不冒称全部队列同倍率。后台独立性与实际新loader已验，记录在 `PERFORMANCE_BACKGROUND_VERIFIED.json` 和 `performance_20261104/RESULT.json`；无需AI继续监控。

2026-11-04香港14:51已启动，源码`18f90a8`、控制器PID11416。14:53现场核验：PPID1，自有SID/PGID，无TTY，stdin=/dev/null，输出落盘；全局臂epoch0完整开发集评估从129推进到385/3703 Study。此时尚无正式训练后AUC，不把机械验证或进程存活当作提分结果。PID只作历史定位，当前状态必须查看下列文件及真实进程。

队列只执行全局继续训练6轮、全图＋原图局部6轮，然后汇总三项预定义患者配对比较。两臂各自先做完整epoch0基线评估，尚未优化时的模型也可保持最佳。E1第13轮不会恢复，E3/E4不会自动启动。

代码工作树：`/home/featurize/work/cxr_research_20261103`。
正式运行根目录：`/home/featurize/work/reconstruction_runs_20261103/E2_20261104`。
独立后台启动与现场核验以根目录 `launch.json`、`BACKGROUND_VERIFIED.json` 为准；仅有本说明或旧PID不证明训练已启动。

查看当前进度与日志（只读，不启动训练）：

```bash
CXR_E2=/home/featurize/work/reconstruction_runs_20261103/E2_20261104
cat "$CXR_E2/controller/PROGRESS.md"
cat "$CXR_E2/controller/STATUS.json"
tail -n 30 "$CXR_E2"/controller/logs/*.log
tail -n 30 "$CXR_E2/controller.launch.log"
tail -n 30 "$CXR_E2/controller.performance.launch.log"
docker ps --format '{{.Names}} {{.Status}}'
```

每臂结果分别在 `global_control_seed20261104/`、`global_local_seed20261104/`：`history.jsonl`仅记录真正训练完成的1–6轮，`epoch_00/`是优化前基线，`epoch_01/`至`epoch_06/`是逐轮完整预测与指标；`best.pt`保留所选EMA，`last.pt`保留模型/EMA/优化器/scaler/RNG/更新计数。`configuration.json`与`compliance/`绑定准确源码、父模型、依赖、数据和生命周期。

局部臂开始前，全局臂应已完成；局部臂的 `epoch_zero_equivalence.json` 记录两臂初始概率差是否通过2e-6门槛。若未通过，队列保留失败并停，不放宽门槛或自动重试。

完成后查看：

```bash
cat /home/featurize/work/reconstruction_runs_20261103/E2_20261104/controller/comparison/REPORT.md
```

必须同时看到控制器 `state: completed`、两臂核验回执以及 `comparison/report.json` 中 `complete: true`。没有容器或退出码0本身不证明科学实验完成。报告包含局部−全局、局部−初始、全局−初始；区间仍是同开发集选模后的探索证据，不自动选择部署模型。

安全停止整个新队列：

```bash
touch /home/featurize/work/reconstruction_runs_20261103/E2_20261104/controller/STOP
```

控制器会请求当前臂在完整训练轮、评估和断点后暂停；若正处epoch0，则等完整基线评估及断点后暂停。**不是立即停机。** 不要用旧E0/E1目录下的STOP控制E2，不要直接强杀容器。若在CPU总结阶段，STOP终止总结；已完成的模型证据保留。暂停或失败后保留现场；恢复必须先核实状态与断点，再显式指定原配置，不能复制旧启动命令造成重复训练。

正式模型和证据在工作盘NFS。机械冒烟的完整权重在 `/home/featurize/data/cxr_reconstruction_smokes_20261104/` 的本机磁盘，换实例可能不保留；对应配置、日志、SHA和核验摘要位于正式根目录 `preparation/`。冒烟只用训练图，不是提分证据。

后台进程使用独立session、stdin=/dev/null且日志落盘；核验通过后可关闭对话或终端。普通控制器不会调用LLM、不消耗AI对话token；宿主机重启、断电或故障不会自动恢复。当前阶段不创建AI自动轮询目标，训练结束后再结合完整结果做下一轮研究判断。
