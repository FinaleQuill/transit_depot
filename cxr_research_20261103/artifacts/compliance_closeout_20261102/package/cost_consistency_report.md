# cost.py / predict.py 一致性报告

状态：本地100样本三轮数值一致性已满足；官方指定样本/T4/逐Study时限验收待补。

## 实际执行与结果

用户要求的核验已使用原041恢复配置、原FP16权重和包内既有cost.py执行，没有更换模型、调整参数、统计AUC或选择模型。官方3.0镜像digest `sha256:125d8bcb4ce5a2a9acec6d9838b37e2ae1e2014c09b848700f9530f2f7856a0b`；本机为2080Ti，不能冒充T4。

| 项目 | 三轮结果 | 证据 |
|---|---|---|
| 测量图/完整Study | 100图 / 54 Study；另5图预热 / 4 Study；两组不重叠 | evidence/runtime/cost_inputs/receipt.json |
| 逐图概率最大绝对差 | 0 / 0 / 0 | evidence/runtime/cost100/cost_report.json 的 comparison.image_rounds |
| 54 Study概率最大绝对差 | 0 / 0 / 0 | 同报告 comparison.rounds |
| 二值/Gate差异 | 每轮0图、0 Study；固定0.5审计阈值，无拟合 | 同报告 binary_mismatched_studies / gate_mismatched_studies |
| 类内排序与并列差异 | 每轮0图、0 Study（样本内类排序及样本间逐类排序均核对） | 同报告 sample_class_order_mismatched_studies / cross_study_class_order_mismatched_studies |
| CSV原件 | SHA 44dde95be64f42c151af7b23ee4a81ffb2158822eabb9f6249a59a73f795f66e，未覆盖 | submission.csv、submission_record.md |

比较代码原容差为1e-7，比要求1e-3更严；实际差为0，无需放宽容差。报告退出码2源于official_compliance_pass=false（硬件非T4），不是数值比对失败。

环境实际记录：Python3.12.3、Torch2.10.0+cu128、CUDA12.8、cuDNN91002、driver610.57.04、FP16、native AVX2；见cost_report.json.environment。原模型+config共66,996,559bytes。

该既有cost顺带记录66.296177ms/图，单位是旧逐图均值，仅参考，不能转成新规每Study≤100ms或T4验收结论。本轮没有速度优化或新时延实验。

## 输入来源与复现参数

现有官方示例只有18图测量/10图预热，不足书面100/5，见evidence/runtime/official_sample_INSPECTION_20261102.md和SOURCE_RECEIPT.txt。本次另按原manifest的(Subject_id,Study_id)字典序选完整Study，恰100图，再从剩余Study取5图；不按标签、概率、视角、质量选样。它是自选固定完整性子集，不是官方指定样本。

`evidence/runtime/cost_inputs/receipt.json`记录105图ID、原文件SHA、复制前后原SHA与副本SHA；跨设备硬链接不可用且原发现函数拒绝目录外软链接，因此仅复制这105图，原图未改。ZIP不附测试图片；复测者需用官方授权图片按receipt恢复相同目录。

覆盖与映射文件：`evidence/runtime/cost_inputs/original_test_identity_manifest.csv`（只含身份、相对路径和既有文件SHA，无标签/质量字段），SHA `13cdb61730471747963d21d5190b49f67483904a5364a2300c39a4711027c623`。完整Study覆盖true；原manifest源SHA见receipt。

核验依赖通过显式参数提供：
```sh
python cost.py submission.csv /inputs/pictures_100 /inputs/pictures_5 \
  --model-dir model/historical_041 \
  --manifest evidence/runtime/cost_inputs/original_test_identity_manifest.csv \
  --image-submission evidence/runtime/041_historical/submission.image_predictions.csv \
  --output-dir /new-output/cost100
```
实际Docker命令原件在evidence/runtime/run_cost_check.sh；`/inputs`、`/out`为挂载路径，输出目录必须不存在。环境版本固定于上述digest，公共timm依赖使用已保存addons，根cost委托code/cost。

## 共用推理与配置恢复证据

`code/predict.py`和`code/cost.py`命令行都用`submission_runtime.load_submission_predictor`，都调用固定`predict_image`、StudyAccumulator和相同CSV写出函数；根cost仅为打包launcher。seed43、eval、固定hflip2、关闭随机增强。静态来源见evidence/runtime/static_code_closure.json，实际输出见evidence/runtime/cost100/。

原041 config初查丢失，随后从A最佳EMA保留metadata和历史导出函数恢复；完整SHA `348901151c0ec1aa42b09015bf69c6be3b642173f58c9beb8b0161b271fbeb39`、11,284,048bytes与历史回执相同。见model/historical_041/config.json及evidence/weights/041_config_recovery.json，未使用044配置替换、未重导权重。

## 尚未闭合的正式验收

最终成员未冻结，本次只核验历史041；主办方指定完整样本、T4实测与逐Study严格时限仍没有通过证据。规则第7条/补充第九节因此仍为待补，不能以本地0差异冒充全部过闸。

独立复算结果及输入SHA见 `evidence/runtime/cost100_independent_recheck.md` 和 `.json`，用标准库直接重算概率、二值、排序与Study均值，不调用cost比较函数。
