# 指定 ZIP 有限回归记录

输入：`compliance_corrected_review.zip`，SHA-256 `b9b96f7ea432caf822437ada44460b599b389dc9f739609479c1770f6055837d`。仅读取该 ZIP；未执行模型、读取测试图像、重复数值比较或上游核验。

| 检查 | 结果与可打开位置 |
|---|---|
| 全量包哈希 | `PACKAGE_SHA256SUMS.json` 的 319 个文件逐一比对字节数及 SHA-256，错误 0；成员差异/重复见 `final_zip_review.json::membership`。 |
| A1–A5 | `模型核验材料与数据声明清单.xlsx` 第4–9、11–16行与 `evidence/weights/input_frozen_workbook.xlsx` 比较 62 个已存在单元格，差异 0。 |
| D1 | `lineage_and_distillation.md:23` 已写实际 hard=`1-alpha`、KD=`alpha`，仅 alpha=0.2 时为0.8/0.2。 |
| D2 | `evidence/training/SOURCE_INDEX.md` 的 38 个具体文件均在包内；XLSX E27:E34/E40/E43 已指向该索引。 |
| D3 | `cost_consistency_report.md:24` 及后续命令明确原身份清单、历史逐图sidecar和实际100/5输入来源。 |
| D4 | `evidence/runtime/SOURCE_RECEIPT.txt` 已入包。 |
| D5 | `evidence/XLSX_CELLS.md:17` 改为 A1:C10；XLSX C34 改为“KD来源登记与公共文件协议”。 |
| G1 | `model/historical_041/config.json` 为11,284,048字节，SHA-256 `348901151c0ec1aa42b09015bf69c6be3b642173f58c9beb8b0161b271fbeb39`；`evidence/weights/041_config_recovery.json` 记录原字节恢复。 |
| G2 | `evidence/runtime/cost100/`、`cost_inputs/receipt.json`、`cost100_independent_recheck.md/json` 均入包且哈希匹配清单。已有结果为3轮100图、54完整Study、概率/二值/排序差异0；本轮不重复比较。它仅为官方3.0镜像+RTX2080Ti+本地固定样本的一致性证据，不是T4/官方固定样本/逐Study时延认证。 |
| G3/G4 | `scope_and_blockers.md:11` 后保留最终成员冻结和正式运行验收缺口；矩阵 19 已满足/7 待补/3 不适用。不能宣布零缺口。 |

本指定 ZIP 另发现 D6：`model/runtime_files.json:10` 仍写“本轮未执行”，`:18` 写“核验待执行”；`model/file_inventory.md:7` 仍写“本轮未执行”；`model/historical_041/STATUS.json:2` 仍为 `weight_only`。这些与本包已有实际 cost 证据冲突。root 已告知修正工作目录文案；修正发生在指定 ZIP 之外，本记录不宣称已核对重封包后的字节。

首次 ZIP 审查 `evidence/selfcheck/round2_stranger_review.md/json` 保留原发现，属于历史记录，不改写成事后已修复状态。有限回归至此结束。
