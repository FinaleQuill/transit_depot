# 合规对照表

审查范围为当前V1血缘、041历史最高分材料与待冻结的最终推理申报；不是已过闸最终包。A1–A5原样冻结。所有证据以ZIP根为相对路径；冒号后是行号，双冒号后是工作表与单元格。

状态只用已满足/待补/不适用。已满足表示本行所需材料在包内可定位，不能由此推导待补的运行认证已通过。条款原文在evidence/rules两份PDF及文本。

| 条款 | 要求原文摘要 | 我方证据（文件+行号或单元格） | 状态 |
|---|---|---|---|
| 规则第 1 条 | 仅非医学开源预训练；申报名称、公开来源、许可、上游数据和SHA；本条按本次V1血缘评价。 | `模型核验材料与数据声明清单.xlsx::申报清单草案!E4`；`模型核验材料与数据声明清单.xlsx::申报清单草案!E11`；`模型核验材料与数据声明清单.xlsx::申报清单草案!F9`；`模型核验材料与数据声明清单.xlsx::申报清单草案!F16`；`A1_A5_FREEZE_RECORD.md:3`；`declarations/negative_declaration.md:3` | 已满足 |
| 规则第 2 条 | 仅官方train用于监督微调，不引入外部医学数据。 | `declarations/data_declaration.md:7`；`evidence/data/recomputed_stats.json:23`；`evidence/data/recomputed_stats.json:249` | 已满足 |
| 规则第 3 条 | valid用途须披露；并入训练后禁止再早停/选checkpoint/调参，相关选择须并入前完成。 | `declarations/data_declaration.md:11`；`evidence/training/student_A/source/train_distill.py:513`；`evidence/training/student_A/source/train_distill.py:684` | 已满足 |
| 规则第 4 条 | 训练增强不能引入test信息；推理关闭随机增强。 | `declarations/data_declaration.md:23`；`code/inference_reproducibility.py:36`；`code/deploy_kd.py:482`；`evidence/training/WARMUP_BY_WEIGHT.md:17` | 已满足 |
| 规则第 5 条 | test不得训练/伪标签/估计统计/阈值/选模；仅最终预测与脚本完整性核验。 | `declarations/data_declaration.md:15`；`evidence/runtime/041_historical/README.md:3`；`evidence/teachers/oof_metadata.json:15` | 已满足 |
| 规则第 6 条 | 固定权重前向、eval、无跨样本自适应；固定TTA/集成允许，参数须预先冻结。 | `declarations/inference_postprocessing_declaration.md:5`；`declarations/inference_postprocessing_declaration.md:9`；`code/deploy_kd.py:448` | 已满足 |
| 规则第 7 条 | 书面规则100测量/5预热/3轮、含读图到写出、不含初始化；cost和predict共用加载与前向且结果一致。 | `cost_consistency_report.md:3`；`evidence/runtime/cost100/cost_report.json:779`；`code/submission_runtime.py:8`；`scope_and_blockers.md:3`；`scope_and_blockers.md:7`；`scope_and_blockers.md:15` | 待补 |
| 规则第 8 条 | 实际加载去重权重+必要config/统计/类别/引擎≤500MB；附逐文件清单。 | `model/file_inventory.md:3`；`evidence/weights/inventory.md:9`；`model/runtime_files.json:23` | 待补 |
| 规则第 9 条 | 账号仅一次撤销及截止前操作；本轮没有平台动作，不能证明账号历史操作次数。；不适用理由：本轮仅本地材料整理，不涉及平台撤销或提交操作 | `procedure_and_scope.md:5` | 不适用 |
| 规则第 10 条 | 来源材料完整、环境与结果可复现；候选材料不替代最终运行包。 | `README.md:24`；`evidence/weights/041_config_recovery.json:6`；`scope_and_blockers.md:11`；`evidence/runtime/ENVIRONMENT_VERIFICATION.md:19` | 待补 |
| 规则第 11 条 | 申报真实无遗漏；valid用途/推理精度必须写明，泄漏与伪造禁用。当前仍有实质缺口，不能声称全部材料齐全。 | `declarations.md:26`；`模型核验材料与数据声明清单.xlsx::申报清单草案!E51`；`scope_and_blockers.md:23` | 待补 |
| 附录 A1 | 所用开源模型名称。 | `模型核验材料与数据声明清单.xlsx::申报清单草案!E4`；`模型核验材料与数据声明清单.xlsx::申报清单草案!E11` | 已满足 |
| 附录 A2 | 权重公开下载链接；固定revision。 | `模型核验材料与数据声明清单.xlsx::申报清单草案!E5`；`模型核验材料与数据声明清单.xlsx::申报清单草案!E12`；`licenses/README.md:11` | 已满足 |
| 附录 A3 | 许可全文，无NC/gated；A1–A5按提交人已核验冻结。 | `模型核验材料与数据声明清单.xlsx::申报清单草案!E7`；`模型核验材料与数据声明清单.xlsx::申报清单草案!E14`；`licenses/convnext_base/LICENSE-Apache-2.0.txt:2`；`licenses/convnext_tiny/LICENSE-Apache-2.0.txt:2`；`A1_A5_FREEZE_RECORD.md:7` | 已满足 |
| 附录 A4 | 上游训练主数据来源（模型卡或论文）。 | `模型核验材料与数据声明清单.xlsx::申报清单草案!E8`；`模型核验材料与数据声明清单.xlsx::申报清单草案!E15`；`licenses/convnext_base/HF_MODEL_CARD.md:14`；`licenses/convnext_tiny/HF_MODEL_CARD.md:14` | 已满足 |
| 附录 A5 | 预训练文件SHA-256；本轮不重新检验上游已冻结项。 | `模型核验材料与数据声明清单.xlsx::申报清单草案!F9`；`模型核验材料与数据声明清单.xlsx::申报清单草案!F16`；`evidence/XLSX_frozen_cell_comparison.json:8` | 已满足 |
| 附录 A6 | 推理加速精度必须申报；C区写FP16，身份未冻结仍单列缺口。 | `模型核验材料与数据声明清单.xlsx::申报清单草案!E51`；`model/runtime_files.json:3` | 已满足 |
| 附录 A 总则/示例 | 中文真实材料；示例code/model/pretrain_model/licenses/cost/XLSX，CSV须历史最高分；pretrain_model目录树不是强制二进制要求。 | `submission_record.md:5`；`evidence/rules/interpretation.md:7`；`README.md:24`；`scope_and_blockers.md:19` | 待补 |
| 补充说明第一节 | 书面规则及正式解释优先，群答疑不自动变更规则。 | `evidence/rules/interpretation.md:3`；`scope_and_blockers.md:17` | 已满足 |
| 补充说明第二节 | 非医学开源预训练，禁止外部医学二次预训练；提交来源。 | `declarations/data_declaration.md:19`；`模型核验材料与数据声明清单.xlsx::申报清单草案!E4`；`模型核验材料与数据声明清单.xlsx::申报清单草案!E11`；`declarations/negative_declaration.md:9` | 已满足 |
| 补充说明第三节 | 蒸馏允许；教师学生均须合规，申报SHA/来源/许可/血缘/脚本/软标签；离线教师不强制附二进制。最终学生仍未冻结。 | `模型核验材料与数据声明清单.xlsx::申报清单草案!F21`；`模型核验材料与数据声明清单.xlsx::申报清单草案!F22`；`模型核验材料与数据声明清单.xlsx::申报清单草案!F23`；`模型核验材料与数据声明清单.xlsx::申报清单草案!F24`；`模型核验材料与数据声明清单.xlsx::申报清单草案!F25`；`evidence/teachers/SELECTION_EVIDENCE.md:3`；`evidence/training/WARMUP_BY_WEIGHT.md:20`；`模型核验材料与数据声明清单.xlsx::申报清单草案!E52` | 待补 |
| 补充说明第四节 | 官方train自监督允许并需申报，外部/test禁用。本次V1链路为监督教师与蒸馏学生。；不适用理由：本申报V1链路没有自监督阶段 | `procedure_and_scope.md:9` | 不适用 |
| 补充说明第五节 | 外部无标签医学数据禁用；外部数据也不得用于早停、超参或选模。 | `declarations/data_declaration.md:19`；`declarations/negative_declaration.md:5` | 已满足 |
| 补充说明第六节 | 训练增强允许；推理不随机，不汇总测试信息拟合/自适应。 | `declarations/data_declaration.md:23`；`declarations/inference_postprocessing_declaration.md:13` | 已满足 |
| 补充说明第七节 | 固定、无参数拟合的患者确定性聚合允许，禁统计自适应；本包实际仅同Study均值。 | `declarations/inference_postprocessing_declaration.md:9`；`code/deploy_kd.py:583` | 已满足 |
| 补充说明第八节 | eval推理，冻结BN/LN/GN；不得用test/独立valid重估统计。LN不伪称有训练集running stats。 | `declarations/inference_postprocessing_declaration.md:5`；`code/deploy_kd.py:448`；`code/model_overnight.py:68` | 已满足 |
| 补充说明第九节 | 实际模型体积、正式时限、100样本二值与排序0差异且绝对概率差≤1e-3；超时整改不许换权重或预测。 | `cost_consistency_report.md:3`；`model/file_inventory.md:3`；`scope_and_blockers.md:15` | 待补 |
| 补充说明第十节 | 平台主指标取4位→百分制2位；同分副指标/时间排序，由主办方执行。；不适用理由：本地包不执行平台评分或排序 | `procedure_and_scope.md:7` | 不适用 |
| 补充说明第十一节 | 正式规则与补充解释高于页面FAQ与群内答疑。 | `evidence/rules/interpretation.md:3`；`scope_and_blockers.md:17` | 已满足 |

当前待补行：7。对应真实阻塞见 scope_and_blockers.md；不以“候选已申报”替代最终成员冻结，不以静态代码检查替代实际cost结果。
