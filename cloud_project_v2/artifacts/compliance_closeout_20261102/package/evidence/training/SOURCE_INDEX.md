# 历史执行源码索引

本目录区分“当前源代码快照”与“当时执行快照”。缺失的历史源码不通过当前副本冒充；原manifest可校核保留下来的具体文件。包内索引如下。

| 训练阶段 | 当时实际源码 | 对应配置/日志 |
|---|---|---|
| student_A | `evidence/training/student_A/source/study_student.py`；`evidence/training/student_A/source/train_distill.py` | `evidence/training/student_A/config.json`；`evidence/training/WARMUP_BY_WEIGHT.md`指定日志 |
| student_B | `evidence/training/student_B/source/study_student.py`；`evidence/training/student_B/source/train_distill.py` | `evidence/training/student_B/config.json`；`evidence/training/WARMUP_BY_WEIGHT.md`指定日志 |
| X1 | `evidence/training/X1/source/run_bounded_finetune.py`；`evidence/training/X1/source/study_student.py`；`evidence/training/X1/source/train_distill.py` | `evidence/training/X1/config.json`；`evidence/training/WARMUP_BY_WEIGHT.md`指定日志 |
| crop_A_tiny | `evidence/training/crop_A_tiny/source/run_bounded_finetune.py`；`evidence/training/crop_A_tiny/source/study_student.py`；`evidence/training/crop_A_tiny/source/tight_crop_ab_earlystop_training.py`；`evidence/training/crop_A_tiny/source/tight_crop_ab_training.py`；`evidence/training/crop_A_tiny/source/train_distill.py` | `evidence/training/crop_A_tiny/config.json`；`evidence/training/WARMUP_BY_WEIGHT.md`指定日志 |
| crop_A_base | `evidence/training/crop_A_base/source/run_bounded_finetune.py`；`evidence/training/crop_A_base/source/study_student.py`；`evidence/training/crop_A_base/source/tight_crop_ab_earlystop_training.py`；`evidence/training/crop_A_base/source/tight_crop_ab_training.py`；`evidence/training/crop_A_base/source/train_distill.py` | `evidence/training/crop_A_base/config.json`；`evidence/training/WARMUP_BY_WEIGHT.md`指定日志 |
| crop_B_tiny | `evidence/training/crop_B_tiny/source/run_bounded_finetune.py`；`evidence/training/crop_B_tiny/source/study_student.py`；`evidence/training/crop_B_tiny/source/tight_crop_ab_earlystop_training.py`；`evidence/training/crop_B_tiny/source/tight_crop_ab_training.py`；`evidence/training/crop_B_tiny/source/train_distill.py` | `evidence/training/crop_B_tiny/config.json`；`evidence/training/WARMUP_BY_WEIGHT.md`指定日志 |
| crop_B_base | `evidence/training/crop_B_base/source/run_bounded_finetune.py`；`evidence/training/crop_B_base/source/study_student.py`；`evidence/training/crop_B_base/source/tight_crop_ab_earlystop_training.py`；`evidence/training/crop_B_base/source/tight_crop_ab_training.py`；`evidence/training/crop_B_base/source/train_distill.py` | `evidence/training/crop_B_base/config.json`；`evidence/training/WARMUP_BY_WEIGHT.md`指定日志 |

教师实际源码：`evidence/teachers/fold0_source/train_teacher.py` 与 `evidence/teachers/fold1_4_source/train_teacher.py`；固定终点证据见 `evidence/teachers/SELECTION_EVIDENCE.md`。

当前版本副本：`evidence/training/source_current/`。XLSX G列为这些“当前源码”文件哈希，不声称与所有历史run的执行文件相同。
