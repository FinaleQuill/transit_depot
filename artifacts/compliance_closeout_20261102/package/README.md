# 合规收口审查包（未最终冻结）

本目录是 2026-11-02 的合规材料收口包，针对 V1 训练血缘与历史最高分 041 `submission.csv`。本轮没有训练、优化、租机、改权重或新模型实验。

## 状态边界

- A1–A5 的模型名、固定 revision、许可证全文、HF model card 快照和预训练 SHA 已由提交人独立核验，本包原样保留。
- 041 CSV 已按指定 SHA 放入根目录；它标为历史最高分 CSV，不自动表示当前最终推理成员已冻结。
- 五折教师不参与最终推理，申报其 checkpoint SHA 和来源；教师二进制不放入 `model/`。
- 当前 C 区最终推理成员仍待权重冻结。C 区的 A6=FP16 声明可以记录当前候选精度，但不得把候选写成最终成员。
- `cost_consistency_report.md`已附实际100样本三轮核验：逐图及54 Study最大差0，二值/排序差异0；仅本地完整性核验，正式T4/指定样本验收仍待。

## 阅读顺序

1. `compliance_matrix.md`
2. `compliance_selfcheck.md`
3. `declarations/`
4. `模型核验材料与数据声明清单.xlsx`
5. `model/file_inventory.md` 和 `cost_consistency_report.md`
6. `code/`、`licenses/`、`evidence/`

附录 A 的目录树是提交示例，`pretrain_model/` 不是规则正文明确的必放二进制目录；本包保留空目录说明文件，完整来源、许可证、model card 和预训练 SHA 在 `licenses/` 与 XLSX 中提供。当前按附录A字段示例整理；最终上传前需填真实队伍名并冻结C区成员，不另设必须获得空白模板的闸门。

## 代码与复现入口

`code/train.py` 是本轮打包时添加的显式分派入口（teacher/distill/study），真实历史程序为 `train_teacher.py`、`train_distill.py`、`study_student.py`；执行快照、配置和日志在 `evidence/training/`、`evidence/teachers/`。`model.py` 和 `utils.py` 是命名别名，未冒充历史执行源码。完整源文件身份见 `evidence/runtime/runtime_source_manifest.json`。

`code/predict.py` 与 `code/cost.py` 使用同一加载器，根 `cost.py` 仅转交到 `code/cost.py`。041历史原配置已按完整SHA恢复至 `model/historical_041/config.json`，证据见 `evidence/weights/041_config_recovery.json`；不是 `evidence/weights/A_config.json`（044实验配置）。最终身份仍未冻结。

`code/requirements.txt` 引用推理依赖；训练依赖在 `code/requirements_kd.txt`。官方3.0版本与历史训练/推理版本分别保存在 `evidence/runtime/`，不混同。native_lanczos 源码、许可和构建文件在 `code/native_lanczos/`；预编译扩展只是CPython3.12历史副本，不证明新环境运行结果相同。

依赖图和源码语法检查为静态核对，见 `evidence/runtime/static_code_closure.json`；没有代替实际运行。剩余阻塞和命名状态见 `scope_and_blockers.md`。包内文档引用以ZIP根目录为基准，`文件:行号` 为文本行号，`XLSX::工作表!单元格` 为工作簿定位。

## Living package 维护入口

2026-11-03 起本目录作为持续维护的主包；见 `LIVING_PACKAGE.md`、`LIVING_PACKAGE.json` 和 `REVIEW2_FREEZE_CHECKLIST_20261103.md`。原 ZIP 为历史只读快照，不再用旧交付回执冒充主包当前哈希。上文 2026-11-02 执行记录保留历史时间语境。最终成员尚未冻结。
