# 本次提交格式独立检查记录

2026-10-20（香港时间）在线只读核对[赛事官网](https://www.ncccu.org.cn/index/Paper/case1.html)“五、赛题和数据”和“七、作品提交要求”。没有上传比赛数据或派生产物，没有访问测试答案，没有操作训练进程或 GPU。

- 官方测试集为 1,414 次检查，每个 Study 必须包含 1～10 全部十类连续概率，因此应有 14,140 条数据行，另加表头。
- Probability 必须为 [0,1] 的有限数值。低于分类阈值的行也必须保留。
- Gate 是可选的最后一列；不提供自定义 Gate 时默认阈值为 0.5。四列表头 `Subject_id,Study_id,Predict_class,Probability` 与用户给出的可提交样例一致。
- 官网要求文件名为 `submission.csv`，UTF-8 编码。为同时满足用户本次指定的 `submission_final.csv` 文件名，可在独立交付目录准备同字节的 `submission.csv` 上传别名，保留原历史样例不动。

本地测试清单核对：2,218 张图像、1,414 个 `(subject_id, study_id)`、136 位患者。每个 Study 先汇总图像概率，再按十类写十行；输出行数不能按图像数计算。

`independent_submission_check.py` 仅使用 Python 标准库，独立于项目的 CSV 写出/检查函数。它检查准确表头、行宽、UTF-8、Study/患者/类别键覆盖、重复、行序、有限概率和 [0,1] 范围；可与同次推理的 `study_probabilities.csv` 逐值比对，并输出 JSON 报告和 SHA256。

```bash
python /home/featurize/work/deliveries/020_assessment_drafts/independent_submission_check.py /path/to/submission_final.csv --wide /path/to/study_probabilities.csv --json-out /path/to/independent_validation_report.json
```

已用历史样例对校验器做只读自检：14,140 数据行、1,414 Study、136 患者、无缺失/重复/异常概率，身份顺序与测试清单一致。此项自检不代表本次新模型预测已完成；本次实际产物必须另行检查。

主线推理代码审查：`inference_overnight.py` 先用检查点固定的精度和原图预处理计算 sigmoid 概率，再对同一 Study 的原始多视角取均值，对 TTA 变换取均值，最后对冻结计划中的 Fold 权重归一化后取均值。设置 `tta={"scales":[768],"hflip":true}` 且每折权重为 1 即为本次要求。校验和测试采用同一冻结协议，测试不参与阈值拟合。实际推理先运行官方 valid 作为诊断和阈值来源，再运行 test；默认四列概率提交不携自定义 Gate。

本地格式通过表示已通过这些明确规则的校验，不等于比赛平台已接收，也不代表官方测试成绩。
