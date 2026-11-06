# KD V1留存修补记录

检查时间：2026-10-25T16:30:46.036192+08:00

[材料核对与操作说明](../../../docs/KD_V1_COMPLIANCE.md)

- repair_report.json：当前训练健康、保护输入、63项测试与验收边界。
- rules/：官网、PDF、答疑及用户转发官方通知原文和来源记录。
- source_before_patch/、source_after_patch/：修补前后源码；不冒充运行中内存已被更新。
- environment_record.json、environment_lock.txt：实际训练解释器和66项依赖。
- retention_probe_summary.json：对真实留存材料的快照检查；训练未完成，结果正确保持pending。
- 当前连续日志、逐轮账本和后台心跳见 /home/featurize/work/runs/021_convnext_v1_base768_teacher_cv5/evidence/。

第1轮精确文件mtime未留存；保留的是原始只读记录里的哈希和保存完成后的状态提交时间，已明确标注来源，不生成旧权重、不伪造时间。最终模型、CSV与T4验收尚未完成。
