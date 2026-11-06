# 第二轮陌生人审查（仅 ZIP）

审查输入仅为 `compliance_round1_review.zip`，实算 SHA-256：
`28d4d2fd93997a2929ecf132d82587a51443ee2c13a3a34fee8e8c7d51d904e5`。
下列路径全部是该 ZIP 内位置，不借助服务器原件或聊天解释。没有运行包内模块、模型、训练、推理或读取 test 图像；仅解析 ZIP、文本、XML 工作簿和现存 CSV。

结论：**这是有明确缺口的审查包，不能作为零缺口最终提交包。** 当前 29 行矩阵包括规则 1–11、A1–A6、附录总则及补充 I–XI，计 19 项已满足、7 项待补、3 项不适用。7 项待补应继续保留。除下述文档问题外，没有发现需要把这些待补项提升为“已满足”的依据。

## 已逐项打开的证据

| 核验范围 | ZIP 内位置与实际结果 |
|---|---|
| 矩阵覆盖和引用 | `compliance_matrix.md:9` 至第37行共29条；90个文本行/单元格引用全部存在且非空。存在引用不代表运行合规通过。逐条实际内容存于本报告同名 JSON 的 `matrix_references`。 |
| A1–A5 冻结 | 工作簿 `申报清单草案!A4:G9`、`A11:G16` 的62个原有单元格与包内 `evidence/weights/input_frozen_workbook.xlsx` 值、XML属性及单元格XML一致。未访问上游或重新评价已冻结项目。 |
| A6 | `模型核验材料与数据声明清单.xlsx::申报清单草案!E51` 明确 FP16；`E50/E52` 同时说明最终成员未冻结。精度声明存在与最终身份/体积待补相容。 |
| 五教师 SHA 和选点 | `申报清单草案!F21:F25` 均与 `evidence/weights/inventory.json` 及各 `evidence/teachers/fold_N/checkpoint_audit.jsonl` 的 best 记录相同，epoch依次20/12/12/12/12。`fold0_source/train_teacher.py:266,282` 和 `fold1_4_source/train_teacher.py:275,291` 仅在固定终点保存 best；包内证据支持固定终点 EMA，而非 held-out AUC 最佳。 |
| XLSX 可打开证据 | 工作簿三个表全部按XML逐单元格读取；所有明确 H 列证据及逐权重配置的实际配置/日志引用可定位。源码 G 列 SHA 与相应包内源文件逐字节相同。泛称路径另列为文档问题 D2。 |
| 数据与补充 II–VIII | `declarations/data_declaration.md:7,11,15,19,23`、`evidence/data/README.md:7` 至第20行、三节声明与分折/训练证据提供 train/valid、增强、确定性聚合和冻结规范。原始30344图与实训30328图区分，未把历史清洗审阅缺失说成新扫描完成。 |
| 推理共享入口 | 根 `cost.py:12`–18转交 `code/cost.py`；`code/cost.py:279`–280及`code/predict.py:88`–89均使用 `submission_runtime.load_submission_predictor`，逐图调用 `predict_image` 并用相同 StudyAccumulator。这里是源码结构证据，不是数值一致性运行证据。 |
| code/闭包 | `code/` 所需5个命名入口齐全；42个Python文件可AST解析。`evidence/runtime/runtime_source_manifest.json` 内所列原样源码/运行材料副本的包内SHA相符。新wrapper的身份在 `README.md:26` 说明，没有冒充历史训练入口。 |
| 历史CSV | 根 `submission.csv` 实算691900 bytes、14140数据行、1414 Study；SHA为`44dde95be64f42c151af7b23ee4a81ffb2158822eabb9f6249a59a73f795f66e`，与包内041回执相同。`submission_record.md:8`把76.92归为提交人确认，未声称有平台原始凭证。 |
| 附录示例及补充 I/X/XI | `evidence/rules/official_rules.pdf.txt:122`、129、134–146将目录树及命名称为建议/示例；没有依据把缺外发模板认证本身变成硬性违规。`pretrain_model/README.md`披露未附二进制；补充 III:49明确离线教师二进制不必附。平台舍入/撤销未执行，各“不适用”行附了本轮范围理由。 |

## 可直接依包内证据补正文档

| ID | 问题与定位 | 直接补正方式及影响 |
|---|---|---|
| D1 | `lineage_and_distillation.md:23`写“硬损失系数0.8”，没有限定KD warmup结束；同一文件第27行及逐权重表有KD warmup2。包内 `evidence/training/student_A/source/train_distill.py:455`–456 与 `evidence/training/student_B/source/study_student.py:203`–204实际为 `(1-alpha)*hard + alpha*soft`。 | 改为硬损失系数为`1-alpha`，只有alpha到0.20后为0.80；不改源码或日志。现稿会让审查员误解warmup阶段的实际损失，属于声明精确性问题。 |
| D2 | XLSX `申报清单草案!E27:E34`、`E40`、`E43`泛称`evidence/training/各run/source/`，该字面路径不存在。H列指向当前源码，不直接回答“本行所述历史版本究竟是哪份”。 | 增加包内历史源码索引或给出实际run目录，例如`student_A/source/`、`student_B/source/`、`X1/source/`、`crop_…/source/`及教师源码目录；不用修改A1–A5。否则陌生审查员需自行猜run和版本。 |
| D3 | `cost_consistency_report.md:5`–15把精确config缺失列为阻塞，但未完整列出此实现还需要的外部输入。`code/deploy_kd.py:655`–658在无原始完整manifest时返回`complete_studies_proven=False`；ZIP没有`test_manifest.csv`。`code/cost.py:196`–198默认根sidecar，实际sidecar在`evidence/runtime/041_historical/`。 | 文档补齐未来核验所需`--manifest`与`--image-submission`的来源/哈希和显式路径。可以保留不附test元数据的选择，但不能让审查员误以为找回config便能用默认参数完整核验；manifest来源/实际恢复若缺则继续待补。 |
| D4 | `evidence/runtime/official_sample_INSPECTION_20261102.md:34`引用`SOURCE_RECEIPT.txt`，ZIP无此文件；该记录第5行的绝对源tar不在包内。 | 若现存该收据则归档并加包内链接；否则加历史原样副本说明，明确收据未附及样例数量依赖已有检查记录，不冒称陌生人能从ZIP重算原样例成员。18/10图结论已在记录第19–20行可读，不需重新读图。 |
| D5 | `evidence/XLSX_CELLS.md:17`标`自查!A1:C11`，工作簿实际自查表止于第10行；源码清单把`申报清单草案!C34`写“KD公共损失实现”，但对应`kd_common.py`是来源/公共协议，损失在train_distill。 | 索引范围改为A1:C10；C34标题与实际职责一致。均为文档精确性修正，不涉及权重或实验。 |

自动引用抽取将 XLSX E7/E14 中公网 URL 的 `/licenses/LICENSE-2.0.txt` 误识为包内相对路径；已人工排除，不将这两项报告为缺文件。A1–A5里的旧NOTICE/429文字已由`A1_A5_FREEZE_RECORD.md:7`解释为历史状态，未以此要求重开上游核验。

## 实际缺数据或决定，不能靠改文档补齐

| ID | 需作者/后续实际材料回答的问题 | ZIP 中证据与合规影响 |
|---|---|---|
| G1 | 原始041的精确config原件在哪里？ | `model/historical_041/STATUS.json:7`–12及`cost_consistency_report.md:7`指明SHA `348901151c0ec1aa42b09015bf69c6be3b642173f58c9beb8b0161b271fbeb39`、11284048 bytes，文件在ZIP确实缺失。`041_config_archive_search.json`报告未找到且保留014归档EOF；不能据此构造替代config。阻塞041完整复现。 |
| G2 | 真正100样本逐张cost对照的输出与报告在哪里？100图、5预热图和完整身份映射如何供复测？ | ZIP无`cost_report.json`。`cost_consistency_report.md:3`、`scope_and_blockers.md:7`–9均承认未运行；官方样例记录只含18张测量图和10张预热图（`official_sample_INSPECTION_20261102.md:19`–20）。无实测不能填写概率≤1e-3且二值/排序0差异。保持规则7/补充IX待补。 |
| G3 | 最终实际加载哪份权重/配置；如何与包内041 CSV对应？ | XLSX `申报清单草案!E50:E52`和`model/runtime_files.json:23`为未冻结/null；候选哈希没有等价替代最终加载集合。体积清单/最终学生SHA、可运行命令和T4运行认证尚不能成立。保持规则8/10/11及补充III待补。 |
| G4 | 逐Study公告与正式逐图平均规则的最终书面口径、正式完整样本和运行结果是什么？ | `evidence/rules/official_rules.pdf.txt:47`–63、补充文本第105–123行仍写逐图平均；样例检查第11行转发Study公告。`scope_and_blockers.md:17`已有更严内部约束但没有认证；静态旧cost代码不能证明逐Study运行。继续保留计时缺口，不按旧数字冒称过闸。 |

命名尚无队伍名和当前模板版本确认，见`scope_and_blockers.md:21`。这影响最后填名及交付识别；附录只给建议名/示例表，不能额外声称“官方未认证该模板”就是必然违规。历史V2分支与平台历史未知已在`evidence/legacy/README.md:7`、21如实披露；当前否定声明限定V1血缘，不应扩大为账号所有历史行为已被证明。

## 状态结论

原矩阵7条待补分别为规则7、8、10、11、附录总则、补充III、IX；其保留有文件依据。
已满足条目只能按矩阵第3–5行界定的“当前V1血缘和材料可定位”理解，不能延伸为最终包已经获认证。
完成上述D类文档修正后，G1–G4仍需真实原件、运行证据或冻结决定；本轮不能把它们改成零缺口。

机器可读逐引用、逐单元格、SHA和语法检查在`round2_stranger_review.json`，它是本审查的输出，不是包内推理运行报告。审查中未修改原ZIP或package。
