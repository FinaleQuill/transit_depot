# XLSX 精确单元格索引

工作簿：`模型核验材料与数据声明清单.xlsx`；下列证据全部位于包内。

| 工作表/单元格 | 内容 | 包内证据 |
|---|---|---|
| 申报清单草案!A4:G9、A11:G16 | 冻结A1–A5及model card来源（62个已有单元格内容/属性逐项相同） | evidence/XLSX_frozen_cell_comparison.json；evidence/weights/input_frozen_workbook.xlsx |
| 申报清单草案!F21:F25 | 五折教师实际SHA-256 | evidence/weights/inventory.json；evidence/teachers/fold_0至fold_4/checkpoint_audit.jsonl |
| 申报清单草案!F38:F39 | A/B best_ema_fp32实际SHA（已纠正草稿误填last EMA的错误） | evidence/weights/inventory.json |
| 申报清单草案!F41:F42 | X1与045 Base FP16成员候选SHA | evidence/weights/inventory.json；evidence/weights/045_config.json |
| 申报清单草案!F44:F47 | 弃用A/B完整best.pt断点SHA | evidence/weights/inventory.json |
| 申报清单草案!E10、E17、E51 | A6=FP16；成员未冻结 | evidence/weights/A_config.json、B_config.json、X1_config.json |
| 申报清单草案!E52 | 最终成员/哈希/体积未冻结 | evidence/weights/inventory.json |
| 申报清单草案!E53、H53 | 三节声明 | declarations.md |
| 逐权重配置!A2:J6 | 五折教师：无KD，LR warmup1、seed43 | evidence/teachers/fold_0至fold_4/config.json |
| 逐权重配置!A7:J13 | A/B/X1/弃用A/B：逐权重KD与LR warmup、seed、起点和日志 | evidence/training/WARMUP_BY_WEIGHT.md |
| 自查!A1:C10 | 原稿失效措辞、错误修正及仍有的最终冻结缺口 | 本工作簿及上述包内证据 |
