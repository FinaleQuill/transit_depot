# 代码阅读地图

行号对应本学习副本，中文注释插入后与原项目行号不同。先看 LEARNING_GUIDE.md，再按函数追踪。

| 文件 | 函数/类 | 副本行号 |
| --- | --- | ---: |
| [dataset.py](dataset.py) | `read_manifest` | 78 |
| [dataset.py](dataset.py) | `letterbox_rgb` | 152 |
| [dataset_overnight.py](dataset_overnight.py) | `_transforms` | 31 |
| [dataset_overnight.py](dataset_overnight.py) | `OvernightImageDataset` | 61 |
| [dataset_overnight.py](dataset_overnight.py) | `collate_overnight` | 189 |
| [dataset_overnight.py](dataset_overnight.py) | `make_overnight_loader` | 221 |
| [dataset_overnight.py](dataset_overnight.py) | `aggregate_study_predictions` | 251 |
| [model_overnight.py](model_overnight.py) | `build_overnight_model` | 60 |
| [train_overnight.py](train_overnight.py) | `compute_loss_config` | 119 |
| [train_overnight.py](train_overnight.py) | `build_weighted_bce` | 172 |
| [train_overnight.py](train_overnight.py) | `build_optimizer` | 247 |
| [train_overnight.py](train_overnight.py) | `set_learning_rates` | 269 |
| [train_overnight.py](train_overnight.py) | `train_epoch` | 306 |
| [train_overnight.py](train_overnight.py) | `evaluate` | 396 |
| [train_overnight.py](train_overnight.py) | `run_training` | 616 |
| [cv_splits.py](cv_splits.py) | `_content_groups` | 73 |
| [cv_splits.py](cv_splits.py) | `build_fold_plan` | 121 |
| [cv_splits.py](cv_splits.py) | `load_fold_records` | 234 |
| [run_all_folds.py](run_all_folds.py) | `training_command` | 91 |
| [run_all_folds.py](run_all_folds.py) | `frozen_config` | 143 |
| [run_all_folds.py](run_all_folds.py) | `orchestrate` | 184 |
| [run_all_folds.py](run_all_folds.py) | `main` | 338 |
| [inference_overnight.py](inference_overnight.py) | `_predict_model` | 260 |
| [inference_overnight.py](inference_overnight.py) | `_ensemble_phase` | 319 |
| [inference_overnight.py](inference_overnight.py) | `run_ensemble_inference` | 381 |
| [train.py](train.py) | `search_class_thresholds` | 245 |
| [train.py](train.py) | `calculate_metrics` | 270 |

训练入口读取顺序：`run_overnight.sh → run_all_folds.py → train_overnight.py`。

最终推理入口读取顺序：`inference.py --ensemble-plan → inference_overnight.py → train.py` 的阈值/指标函数。此处是调用关系说明，本次没有执行入口。
