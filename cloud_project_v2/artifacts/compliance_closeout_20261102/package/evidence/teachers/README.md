# 教师与软标签证据索引

- `teacher_collection.json`：五折路径/固定轮数集合声明。
- `teacher_run_fold0.json`、`teacher_run_fold1_4.json`：训练级固定终点声明。
- `fold_0`–`fold_4`：每折 config/history/checkpoint audit/status。
- `fold0_source`、`fold1_4_source`：教师源码和来源 manifest。
- `SELECTION_EVIDENCE.md`：固定 epoch EMA 与 outer-fold 选择禁用的逐行证据。
- `oof_metadata.json`：软标签缓存元数据。
