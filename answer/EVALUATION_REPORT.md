# 020：停止训练、五折评估与最终提交

完成时间：2026-10-20T13:14:46.034738+08:00

实际运行目录：`/home/featurize/work/runs/020_convnextv2_base768_posbce_cv5_4090`。用户请求中的目录名称未带 `_4090`，本次已核对实际目录。

训练已按用户要求在轮次边界安全退出并保存。没有一折到达原计划20轮或早停终点；五折均完成了完整训练及验证轮次，均有已核验的最佳检查点。

| Fold | 已完成完整轮数 | 最佳 epoch | val_macro_auc | Macro F1 @ 0.5 |
|---|---:|---:|---:|---:|
| 0 | 6 | 6 | 0.8067351887 | 0.4328396775 |
| 1 | 5 | 5 | 0.8013960260 | 0.4269616289 |
| 2 | 5 | 5 | 0.7947949065 | 0.4214838283 |
| 3 | 5 | 5 | 0.8032683722 | 0.4301939654 |
| 4 | 5 | 5 | 0.7996021371 | 0.4265177362 |

AUC 读取各折 best_metrics.json 的 validation.macro_auc。JSON 未保存 F1，表中 F1 从对应最佳 best_oof.npz 按概率 >= 0.5 独立补算；这是各自内部验证折的指标。

本次集成固定使用上述五折 best.pt 中的 EMA 权重，每折权重0.2；原图等比缩放并Pad到768，原图与水平翻转两种TTA。每图先sigmoid，再按Study内视角、TTA及Fold分别求概率均值。

在全部五折训练都未见过的官方valid（859 Study）上，集成 Macro AUC = **0.8002477168**；固定0.5阈值 Macro F1 = **0.4242437496**。这是本地验证成绩，未上传比赛官网。

thresholds.json 另存官方valid拟合的十类阈值；它们不改变本次四列概率提交。按同一valid拟合阈值后的F1见 ensemble_assessment.json，不当作独立评测成绩。

## 下载及上传

- 用户指定文件：[submission_final.csv](submission_final.csv)，绝对路径 `/home/featurize/work/answer/submission_final.csv`。
- 官网文件名别名：[submission.csv](submission.csv)，与前者逐字节相同，上传时可直接选此文件。
- SHA256：`b092d26196c5586af55a9173cc5c8582c003bd2e9135aa94ce17f2f760a0620b`。

独立格式检查已通过：UTF-8；表头 Subject_id,Study_id,Predict_class,Probability；14,140条数据行及1条表头；1,414 Study各10类；没有缺失、重复、非法数值；身份和行序与提交样例一致。
概率范围：[0.00537364812335, 0.984866702557]。长表与Study宽表的最大概率差为 0.0。

## 复核证据（均位于 audit/）

- final_summary.json：完整轮次、OOF、检查点SHA及跨折隔离审计。
- ensemble_plan_768_equal.json、inference/provenance.json：冻结集成配置及实际执行来源。
- ensemble_assessment.json：独立重算官方valid集成指标。
- independent_submission_validation.json：推理完成时的独立检查；archived_submission_validation.json：归档到answer后对实际下载文件再次独立检查，均通过。
- protected_inputs_verified.json：原始源码、清单、样例哈希保护。
- stop_request.json、stop_confirmed.json：停止训练及释放GPU的证据。
- inference.log、driver_result.json：本次独立推理记录。

官网格式来源：https://www.ncccu.org.cn/index/Paper/case1.html 。本次未执行上传。

本次提交文件、说明及审计副本统一收在此 answer 目录；删除 answer 不会删除原始代码、数据或训练检查点。审计记录中的旧临时输出路径为实际执行时的路径，归档位置映射见 OUTPUT_MANIFEST.json。
