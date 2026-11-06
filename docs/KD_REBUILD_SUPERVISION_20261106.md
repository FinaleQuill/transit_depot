# 新蒸馏训练监督入口

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
