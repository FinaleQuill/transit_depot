# 项目交接（2026-11-03）

项目：`/home/featurize/work/cloud_project_v2`  
当前提交：`0ffbf64`（`Add living compliance package and training lifecycle evidence`）

## 当前状态

- 合规主包已升级为可持续维护的 living package：
  `artifacts/compliance_closeout_20261102/package/`。
- 原审查草案 ZIP 已保留为只读快照，快照证据在
  `artifacts/compliance_closeout_20261102/snapshots/closeout_c8bd868/`。
- 训练生命周期记录已接入源码：新 run 应记录启动、epoch、checkpoint SHA 和结束状态；相关入口为
  `training_compliance.py`、`training_hooks.py`、`manage_living_compliance.py`。
- `artifacts/compliance_demo_20261103/` 是未执行训练的合规机制示例，只含合成小文件，不能当作模型或实验结果。
- 七项冻结条件已登记于
  `docs/REVIEW2_FREEZE_CHECKLIST_20261103.md`，尚未执行冻结。

## 未完成事项

最终推理成员尚未确定；官方指定样本、T4 和逐 Study 严格计时验收尚未完成；队伍真实名称也未提供。因此不能把当前材料称为最终提交包，也不能把本地 2080Ti 结果当作官方 Gate 证据。

## 下一位 Agent 规则

先读 `AGENTS.md`、`README.md`、`docs/COMPETITION_RULES.md`、`docs/CLEANING_POLICY.md`、`WORKLOG.md` 和本文件。除非用户明确授权，不训练、不改权重、不租机器、不重新生成提交 CSV、不冻结最终成员。新 run 必须在启动前创建 compliance 记录；失败或中止也要记录为 abandoned。A1–A5 申报材料保持冻结，valid 仍未并入训练。

当前工作区另有三个未跟踪文件：`fixed_student_candidate.py`、`review_student_upper_bound.py`、`student_probability_fusion.py`。接手时不要擅自删除或纳入本次合规包。
