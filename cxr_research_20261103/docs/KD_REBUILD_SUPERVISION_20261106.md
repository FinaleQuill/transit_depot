# 新蒸馏训练监督入口

## 当前状态：已停止（2026-11-06 18:11，香港）

用户因比赛仅剩约一周要求停止当前蒸馏思路。KD_REBUILD_20261106根目录`STOP`保留；当前DINOv3-L内折0于第3轮5,408/9,879 Study处收到SIGTERM，最近完整断点为第2轮。退出码130/`KeyboardInterrupt`是用户停止，不是数值失败。controller状态文件保留其`failed`历史字段；单run状态为`interrupted`。容器、控制器、GPU训练进程均已退出，六教师无一完成，未启动学生或生成submission。last和EMA断点SHA已复核，见[USER_STOP_RECORD.json](/home/featurize/work/reconstruction_runs_20261103/KD_REBUILD_20261106/USER_STOP_RECORD.json)。不得恢复该队列或另开训练，直到用户对新方案另行明确授权。

一周赛期的候选转向方案见[ONE_WEEK_PIVOT_PLAN_20261106.md](ONE_WEEK_PIVOT_PLAN_20261106.md)。以下段落为准备、启动及运行中的历史记录，不代表当前仍在训练。

2026-11-06，香港时间。用户已授权第一阶段；启动状态以以下实时文件和实际进程为准。准备、机械测试与正式训练分别登记，不把GPU试跑当正式完整轮次。

- 实验根：`/home/featurize/work/reconstruction_runs_20261103/KD_REBUILD_20261106/`
- 队列：`teacher_queue.json`，六个固定教师，完成后等待教师复盘。
- 后台控制器：`launch.json`、`controller_status.json`、`controller.log`。
- 当前/首个run：`dinov3_l_inner0/progress.json`、`status.json`、`training.log`。
- 每轮完整记录：各run的`history.jsonl`与`checkpoint_manifest.json`。
- 原始数据分层证据：`/home/featurize/data/kd_rebuild_20261106/plan.json`、`_SUCCESS.json`。
- 完整恢复与EMA：`/home/featurize/data/kd_rebuild_20261106/state/<run_id>/last.pt`、`latest_ema_fp16.pt`。
- 来源/环境：数据盘`assets/teacher_source_manifest.json`、`environment/manifest.json`。
- 机械测试：实验根`mechanics/`，仅验证512/768前反向、保存及恢复；不用于选模。

运行中以`optimizer_updates`增加、有限loss/梯度、真实GPU计算和日志连续推进共同核验。首epoch完整保存前，不声称已完成一轮或已有有效AUC提升。外层H不用于训练期选轮。

本次启动入口：`python -m teacher_distill_20261106.launch --execute`。该命令有独立锁和一次启动回执，**已有launch.json后禁止再次执行来重复启动**。不依赖本对话持续在线，控制器使用独立session和空stdin。

需要停止时，实验根`STOP`会让训练在当前完整epoch保存后停止，并阻止队列下一个教师；若用户要求立即停止，应另对当前容器SIGTERM，最近完整epoch断点保留，中途更新不冒称完整保存。旧实验STOP保留且不读取为新实验授权。

本次没有自动学生训练、第二外层/全量重训或submission生成。最新的正式启动核验与实际PID将写入实验根`BACKGROUND_VERIFIED.json`。

## 正式启动核验（2026-11-06 15:47，香港）

15:45:13启动，源码提交`cf0ed55`；控制器PID12157/PPID1/SID12157，stdin为`/dev/null`；容器`cxr-kd-20261106-dinov3-l-inner0`，容器宿主PID12194。两次观测真实optimizer_updates为4→7，0跳步，224 Study/367图已处理，loss=0.734500、梯度范数=3.3738，峰值分配约7.48GB。首轮未完整结束、尚无正式整轮checkpoint或AUC结果；当前粗略ETA不作为完整轮实测。

核验回执：[BACKGROUND_VERIFIED.json](/home/featurize/work/reconstruction_runs_20261103/KD_REBUILD_20261106/BACKGROUND_VERIFIED.json)；数据/配置/队列/环境/训练源码SHA均绑定，受保护的规则、原CSV和旧提交七文件SHA未变。实时进展以后续`progress.json`为准；本节是启动快照。
