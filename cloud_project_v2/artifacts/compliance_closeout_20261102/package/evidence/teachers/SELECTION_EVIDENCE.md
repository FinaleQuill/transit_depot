# 教师固定终点证据

本文件由包内可打开的源码与每折审计组成。教师 `best.pt` 是固定训练轮数完成时写入的最终 EMA 快照；“best”不是按外折标签或 valid AUC 选择。

| 证据 | 内容 |
|---|---|
| `evidence/teachers/fold0_source/train_teacher.py:27` | `SELECTION_PROTOCOL = "fixed_epoch_final_ema"` |
| `evidence/teachers/fold0_source/train_teacher.py:266,282` | `complete = epoch == config.epochs`；仅 complete 时写入 `best.pt` |
| `evidence/teachers/fold0_source/train_teacher.py:70` | `outer_fold_used_for_selection: False` |
| `evidence/teachers/fold1_4_source/train_teacher.py:30` | `SELECTION_PROTOCOL = "fixed_epoch_final_ema"` |
| `evidence/teachers/fold1_4_source/train_teacher.py:275,291` | 仅固定终点 complete 时写入 `best.pt` |
| `evidence/teachers/fold1_4_source/train_teacher.py:73` | `outer_fold_used_for_selection: False` |
| `evidence/teachers/teacher_collection.json` | fold0=20 epochs，fold1–4=12 epochs，固定集合映射 |
| `evidence/teachers/fold_0/history.jsonl` 至 `fold_4/history.jsonl` | 每折实际 epoch 完成记录及 `selection_protocol` / `outer_fold_evaluated=false` |
| `evidence/teachers/fold_0/checkpoint_audit.jsonl` 至 `fold_4/checkpoint_audit.jsonl` | 每折 best checkpoint 的实际文件 SHA 与 epoch |

## 不能写成的事实

现有证据不支持“按哪个 valid AUC 选出 best.pt”；应申报为固定 epoch 最终 EMA。held-out fold 只用于其余训练流程的折外预测，未用于 checkpoint 选择。
