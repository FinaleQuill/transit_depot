# 固定双Tiny768测试CSV交付

2026-10-28（香港）。按用户“按照这版生成一个”的授权，已用033固定部署生成测试集`submission.csv`，命令正常exit0，实际推理约3分06秒。没有重新训练、按test调参或上传平台。

提交文件：`../deliveries/033_dual_tiny768_seed43_25_seed44_75/test_predictions/submission.csv`。

| 项目 | 结果 |
| --- | --- |
| 模型 | 两个ConvNeXt V1 Tiny768，seed43/44各自E8 EMA |
| 固定融合 | 全十类概率权重.25/.75，Study内固定均值 |
| 精度 / TTA | FP16 / 各原图+hflip，共四次前向 |
| 编码 / 列 | UTF-8，Subject_id,Study_id,Predict_class,Probability |
| 测试图 / Study / 患者 | 2218 / 1414 / 136 |
| 数据行 | 14140，每Study类别1～10各一次，另有一行表头 |
| 文件大小 | 686443bytes |
| SHA256 | `ea83de14afb8a8438428dc071b29438b2201a7113cad03aa4181de09284b3661` |
| Gate | 未添加自定义Gate，保持默认.5；没有阈值筛行 |

独立CPU审计通过：完整身份/类别覆盖无缺失重复，概率均有限且在[0,1]；逐图22180行按固定Study均值独立重算，与提交概率误差0。2218张原图SHA与冻结清单一致，53个受保护模型/源码/清单/材料文件前后SHA不变。未统计疾病概率分布、推测测试标签、评估测试AUC或据此选择模型/参数。

同目录的`submission.image_predictions.csv`和`submission.csv.report.json`是逐图一致性及复现材料，不是额外候选；提交预测文件使用`submission.csv`。逐图文件应保留，供将来官方cost对最终CSV的一致性核验使用。

实际config SHA为`9363149e143dec2871ac5fb4fbfdee4885f9f62add6a9f2acf4704635311b590`，与生成前已经验证的033一致；教师、OOF、推理实现及原训练权重不改。预测通过`predict.py:main`及原共享Predictor执行，入口固定Python/NumPy/Torch/CUDA seed43，没有随机增强或新增患者跨Study平滑。

已重新核对[官网格式要求](https://www.ncccu.org.cn/index/Paper/case1.html)和[10月27日正式补充](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261027/b23d6a1f19602da2a9bd4048108175d0.pdf)。原件/获取时间/SHA在`artifacts/kd_v1/submission_rules_recheck_20261028/`，补充适用记录见[规则记录](COMPETITION_RULES.md)。新说明不要求改变本次冻结版本，本次未因test结果修改任何实现。

CSV已完成本地格式与完整性核验，未上传平台，尚无平台接收或测试成绩。T4正式计时及最终双学生/教师来源、训练证据等核验材料仍待完成；4090此前79.28ms不能替代T4认证。此前[部署验收登记](DUAL_TINY768_ACCEPTANCE_20261028.json)保留生成CSV前的历史状态，本次状态另记[CSV交付登记](DUAL_TINY768_SUBMISSION_20261028.json)。

执行和审计证据：`artifacts/kd_v1/dual768_submission_20261028/{execution.json,prediction.log,protected_before.json,audit.json,source/}`。CPU审计中的`no_gpu_work_by_this_audit`仅描述审计，本次主代理确实执行了完整测试集GPU前向；本次新训练0、测试拟合0、平台上传0。
