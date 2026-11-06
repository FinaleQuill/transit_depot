# 数据边界与独立复算证据

这里仅有官方 train/valid 元数据、派生清单和五折计划，不包含图像或 test 元数据。

| 清单 | 患者 | Study | 图片 | 文件 |
|---|---:|---:|---:|---|
| 官方原始 train | 11055 | 18513 | 30344 | `official_train.csv` |
| 实际训练 train | 11055 | 18513 | 30328 | `train_clean.csv` |
| 官方原始 valid | 228 | 859 | 1422 | `official_valid.csv` |
| 实际独立 valid | 228 | 859 | 1422 | `valid_manifest.csv` |

`recomputed_stats.json` 是本次只读复算输出；脚本为 `recompute_train_valid_stats.py`。运行 `python evidence/data/recompute_train_valid_stats.py` 可从包内证据重算，不需要服务器绝对路径、第三方库、原图或 test。脚本只接受固定的五个文件名，检查清单 split 和路径前缀均为 train/valid；只向 stdout 输出，不修改输入文件。

`recomputed_stats.json` 的 `checks` 记录逐图身份/标签与官方 CSV 一致；实际 train 保留全部原 Study、患者和 Study 标签；valid 图像覆盖完全相同；train/valid 患者重叠 0；比较清单已有文件/像素摘要的重叠 0。这里比较的是历史记录的摘要，不表示本轮重新打开图像复算摘要。

五折由官方 train 的患者固定分组，seed 43。每折 held-out 恰有 2211 患者；每折训练 8844 患者。held-out 图数依次为 6070、6062、6053、6079、6064；每张实际 train 图恰覆盖一次。脚本从 `study_to_fold`、`patient_to_group` 和实际训练清单重算，核对每个患者只有一个 held-out fold、每折训练/held-out 患者与已有内容摘要无交叉，并与原 `folds.json` 声明计数一致。该结果证明划分隔离，不单独证明 checkpoint 选择过程没有使用 held-out 标签；选择过程须另查训练证据。

`metadata_validation_report.json`、`preflight_data.json`、`preflight_ready.json` 是历史证据原件；`sources.json` 保存原路径、原文件 SHA-256 与字节数。原图缺失等历史检查不可冒充本轮新图像扫描。

原始 train 的 16 张排除图全部在 `recomputed_stats.json.training_excluded_images` 中列出。`WORKLOG_cleaning_excerpts.md` 原记录说明 14 张同 Study/同标签/同像素重复、2 张技术废片隔离。原清洗审阅文件 `docs/manual_review_v1.json` 在当前工作区未找到，因此本包不重新鉴定这 16 张图的视觉质量，也不把缺失的逐图审阅原件伪称为已收齐。当前原始 CSV 对照可独立证明排除数和 Study/标签守恒。
