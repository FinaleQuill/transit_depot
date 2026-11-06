# 当前执行方案：20/12/12/12/12五折教师

## 当前后续入口

五教师已于2026-10-26 19:26（香港）全部完成。用户现已批准TTA及多学生升级，后续以[当前TTA/集成手册](KD_TTA_ENSEMBLE_USAGE.md)执行：先教师valid诊断，再用新`extract_soft_labels_tta.py`提取逐视图OOF缓存，随后手动启动学生阶段。本页其余内容保留此前教师调度与单学生交接记录。


## 2026-10-26恢复记录

2026-10-26 13:18（香港），用户授权恢复因存储I/O异常退出的训练。存储写入、fsync、重命名及读回检查通过，冻结计划、源码和数据清单校验一致；fold3已从第4轮完整断点恢复第5轮，随后继续fold4的12轮。fold0–2不重跑训练。
当前任务在独立后台会话运行，`cxr-kd-mixed`已切换到新日志`../runs/024_convnext_v1_base768_mixed20_12_cv5/evidence/controller_background_20261026T051826Z.log`。五教师完成后仍停住，OOF及学生不自动启动。恢复检查与回执见`artifacts/kd_v1/teacher_resume_20261026T051809Z/`。


用户已确认保持本方案，不改15轮。fold0已于2026-10-25 21:26（香港时间）在原021目录正常完成20轮。
第13轮恢复断点已用硬链接保存在`fold_0/retained/epoch_013_last.pt`，并保留已有SHA256、时间和原始状态回执。
接力控制器先前收到SIGINT中断，已于21:38在独立后台会话恢复；tmux会话`cxr-kd-mixed`现在只显示日志，查看窗口退出不会向训练任务发送信号。

## 现在无需重新启动训练

自动顺序为：

1. 原fold0已完成20轮，保留最终权重，不再重跑。
2. 旧调度器已由精确交接保护退出，未开始旧20轮的fold1。
3. 等旧GPU训练退出，依次用官方valid评估第13轮和第20轮，记录AUC差。中途权重仅供诊断，不变造为已完成教师，也不自动选择OOF教师。
4. 新023目录依次训练fold1–4，各固定12轮，原五折患者划分、768、有效batch32、seed43不变。
5. 五个教师全部完成后，控制器停住；不会自动提取软标签或启动学生。

查看自动控制器：

```bash
tmux attach -t cxr-kd-mixed
```

按Ctrl+B，松开后按D离开查看窗口，任务继续。另一个终端可查询总状态：

```bash
cd /home/featurize/work/cloud_project_v2
bash run_mixed_teachers.sh --status
```

状态`validating_epoch13_and_epoch20`表示验证正在进行；随后是`training_folds_1_to_4_twelve_epochs`。
`controller_alive: true`表示接力控制器正在运行；`controller_not_running`表示控制器已退出，同时保留`last_recorded_phase`，不能把旧阶段当成仍在执行。
只有`phase: completed`且`all_teachers_complete: true`才进入下一步。若是`failed`，先保留错误并检查；阶段失败不会自动继续后续GPU任务。

## 全部教师完成后：手动提取折外软标签

```bash
cd /home/featurize/work/cloud_project_v2
KD_PYTHON=/home/featurize/.conda/envs/cxr-kd-v1/bin/python
"$KD_PYTHON" -u extract_soft_labels.py --execute \
  --fold-plan data/cv5_overnight/folds.json \
  --teacher-collection ../runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_collection.json \
  --image-root /home/featurize/data/cxr_first \
  --output ../runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits
```

看到`Strict OOF cache complete:`且成功退出后，再安排现有单Tiny512学生。当前部署保持FP16、无TTA、无集成；等现有路线结果与交付验收完成后，再讨论纯监督对照、768或多学生扩展，暂不新增这些实验。
后续学生命令中的`--soft-labels`应使用上面的024路径，原021目录不再是完整五教师集合入口。

## 路径、来源与官方材料

| 内容 | 路径 |
|---|---|
| 原fold0、20轮 | `../runs/021_convnext_v1_base768_teacher_cv5/fold_0/` |
| 新fold1–4、各12轮 | `../runs/023_convnext_v1_base768_teacher12_cv5/` |
| 冻结调度计划 | `../runs/024_convnext_v1_base768_mixed20_12_cv5/plan.json` |
| 五教师集合声明 | `../runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_collection.json` |
| 当前阶段 | `../runs/024_convnext_v1_base768_mixed20_12_cv5/transition_status.json` |
| 13轮/20轮官方valid比较 | `../runs/024_convnext_v1_base768_mixed20_12_cv5/validation_epoch13_vs20/comparison_report.json`，重试可能有新后缀，以状态文件记录为准 |
| 合并的训练留存材料 | `../runs/024_convnext_v1_base768_mixed20_12_cv5/evidence/` |

官网、PDF、答疑与用户转发官方通知均保存在`artifacts/kd_v1/compliance_repair_20261025/rules/`，见[官方规范清单](OFFICIAL_RULES_CHECKLIST.md)。
新计划只调整未开始教师的固定预算，初始化、患者隔离、seed、日志、逐轮时间及哈希、源码和依赖继续留存。不会把13轮中途权重改成完成标志，也不修改原021的20轮元数据。

最终打包前刷新024材料索引，并将`--training-evidence`指向024/evidence：

```bash
"$KD_PYTHON" training_evidence_monitor.py refresh \
  --run-dir ../runs/024_convnext_v1_base768_mixed20_12_cv5
```

控制器源码、参数、后台启动回执和SIGINT中断日志均已归档；本次恢复没有修改冻结计划或旧教师权重。恢复说明见`artifacts/kd_v1/mixed_teacher_recovery_20261025/recovery_report.json`。
后台任务已经运行，现在不要重复启动`--execute`。阶段子进程继承独占锁，即使控制器意外退出，也不会允许另一控制器同时启动GPU任务。
