# 2026-10-19 Gate 更新交付

推荐提交本目录的 `submission.csv`。文件名和UTF-8编码符合[官网要求](https://www.ncccu.org.cn/index/Paper/case1.html)，表头为：

```csv
Subject_id,Study_id,Predict_class,Probability,Gate
```

Gate 在格式上可选，但建议为这份预测填写。对应权重在官方859个Study验证集上，默认0.5的 Macro F1 为0.3205493442，保存的十类阈值对应 F1 为0.4694609371；Macro AUC 都是0.7999705530。阈值由同一验证集调优，F1提升不是独立测试成绩或平台收益保证。

## CSV与阈值来源

- 最新已有测试交付：`../best_auc_20261019/submission.csv`，与 `../016_a4000_tiny_v2_final/submission.csv` 字节一致。
- 原四列表SHA256：`c985cf8736ebb342af6c8662acbbb277042fa8ef779524b18751220333969650`。
- 权重：`../../runs/016_a4000_tiny_v2/best.pt`，第5轮EMA；实算SHA256：`2c3f88b25e04ea86af20eaf2e41d42db241cb5013657283afa3eabdf54893b4b`。
- 阈值：`../../runs/016_a4000_tiny_v2/thresholds.json`，SHA256：`9ff807c2109af7ea31876135d3942220a994ad042074719aea4fdffb09da7da1`，本目录保存带来源字段的副本。
- 协议：原图384、identity TTA、bfloat16、EMA、先平均多视角图像概率得到Study概率。原推理记录的二值比较为 `Probability >= Gate`，本次沿用原阈值。官网正文未提供评测端比较符实现，尚未在线提交验证。
- 当前019五折实现与本CSV来源不同；没有将019或其他epoch的阈值套到016预测上。根目录 `../../submission.csv` 是此前用于对齐身份和格式的样例，概率不同，不是本次最新016预测。

| 类别 | Gate |
| --- | ---: |
| 1 | 0.06606084108352661 |
| 2 | 0.0818309634923935 |
| 3 | 0.2133771777153015 |
| 4 | 0.10196448862552643 |
| 5 | 0.2568320035934448 |
| 6 | 0.2523840367794037 |
| 7 | 0.2193019688129425 |
| 8 | 0.23350277543067932 |
| 9 | 0.38331663608551025 |
| 10 | 0.35870736837387085 |

同一类别在所有Study行填写相同Gate。完整保留低于阈值的概率，不删行、不二值化概率、不改变原值。

## 当前 runs 的比较

按官方规则优先比较 Macro AUC，只有 AUC 相等才比较 Macro F1。以下均为官方859个Study验证集；F1为同轮在该valid调优后的值，不是平台测试分数。

| run / 权重 | Macro AUC | 自定义阈值 Macro F1 | 结论 |
| --- | ---: | ---: | --- |
| **016，第5轮EMA** | **0.7999705530** | 0.4694609371 | 最佳；已有完整测试预测，本交付来源 |
| 017，继承的best，第5轮 | 0.7999705530 | 0.4694609371 | best.pt与016字节一致，是同一基线 |
| 017，新增最佳第6轮 | 0.7998773790 | 0.4703623729 | F1略高，AUC未提升；没有该权重的测试CSV |
| 019，五折384/512翻转TTA集成 | 0.7934358232 | 0.4635856495 | 官方valid完整；测试推理已取消 |
| 018 | 无有效指标 | 无有效指标 | 仅启动记录，无完成轮或最佳权重 |

016共完成6轮，最佳仍第5轮；017从第5轮热重启后新增完成第6、7轮。019的fold0内部验证AUC为0.8024026369，但来自训练集内部的3703个Study，不能与上述官方859个Study直接比较。

对照证据：[016指标](../../runs/016_a4000_tiny_v2/best_metrics.json)、[017新增F1最佳指标](../../runs/017_a4000_tiny_v2_warm/best_f1/best_metrics.json)、[019官方valid指标](../019_convnextv2_base512_cv5_4090/valid_metrics.json)、[019测试推理状态](../019_convnextv2_base512_cv5_4090/status.json)。补充封存的014第6轮：AUC=0.7977187851、调阈值F1=0.4861241884，仍因AUC低于016而不优先选择。

## 验收与重放

1414个Study、每Study恰好十类，共14140数据行；UTF-8无BOM、CRLF。已完整回读并验证身份、行序、无重复和缺类、概率与Gate在[0,1]。所有原四列文本原样保留，去掉Gate后可重建原CSV的完全相同SHA256。逐值核对原推理长表与宽表，并确认新Gate生成的二值结果与原保存二值表一致。源CSV、模型、阈值及train/valid/test清单哈希未变。

新CSV SHA256：`b2b16d56d5129871e65e914d9e268936b2471cc83799dcfa9b6424024a9878bf`。

详情见 [validation_report.json](validation_report.json) 与 [provenance.json](provenance.json)；官网示例图见 [evidence/official_submission_example.png](evidence/official_submission_example.png)。本次训练0次、模型推理0次、阈值重新拟合0次，未上传平台。

可从工作根目录使用标准库脚本重放到一个全新目录，不覆盖已有交付：

```sh
python deliveries/gate_update_20261019/add_gate.py --output deliveries/gate_update_replay
```
