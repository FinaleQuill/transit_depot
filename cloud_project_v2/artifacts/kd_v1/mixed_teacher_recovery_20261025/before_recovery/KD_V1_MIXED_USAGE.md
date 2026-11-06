# 当前执行方案：20/12/12/12/12五折教师

用户已批准本方案，自动控制器已在tmux会话`cxr-kd-mixed`启动。
当前fold0保持原021目录和20轮计划；第13轮恢复断点已用硬链接保存在`fold_0/retained/epoch_013_last.pt`，并保留已有SHA256、时间和原始状态回执。

## 现在无需重新启动训练

自动顺序为：

1. 原fold0继续完成20轮，原GPU训练进程不重启。
2. 旧调度器尝试启动下一折前，由精确交接保护正常退出计划流程；不会再开始旧20轮的fold1。
3. 等旧GPU训练退出，依次用官方valid评估第13轮和第20轮，记录AUC差。中途权重仅供诊断，不变造为已完成教师，也不自动选择OOF教师。
4. 新023目录依次训练fold1–4，各固定12轮，原五折患者划分、768、有效batch32、seed43不变。
5. 五个教师全部完成后，控制器停住；不会自动提取软标签或启动学生。

查看自动控制器：

```bash
tmux attach -t cxr-kd-mixed
```

查看当前fold0训练输出仍可使用：

```bash
tmux attach -t cxr-kd-v1
```

按Ctrl+B，松开后按D离开查看窗口，任务继续。另一个终端可查询总状态：

```bash
cd /home/featurize/work/cloud_project_v2
bash run_mixed_teachers.sh --status
```

状态`waiting_for_fold0_twenty_epochs`表示正常等当前折完成；随后是验证和四折12轮训练。
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

看到`Strict OOF cache complete:`且成功退出后，再安排学生及纯监督对照。学生仍为单Tiny512；是否改768，需要后续验证与T4实测。
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

控制器源码与参数已有自动归档；如果其进程意外结束，先检查状态和现有训练锁，再用同一计划恢复。此命令现在无需重复运行：

```bash
cd /home/featurize/work/cloud_project_v2
bash run_mixed_teachers.sh --execute
```

阶段子进程继承独占锁，即使控制器意外退出，也不会允许另一控制器同时启动GPU任务。
