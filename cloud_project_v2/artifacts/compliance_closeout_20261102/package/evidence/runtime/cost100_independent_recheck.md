# 100 图既有 cost 输出独立复核

结果：三轮的100图逐图概率、54个完整Study概率、二值判定及排序全部与历史041参考一致，最大概率差为0。这里是**官方3.0镜像内、2080Ti、本地固定自选样本的功能一致性证据**；不是T4验收，不是官方指定100样本认证，也不是逐Study时延过闸。

本次独立复核仅读取包内已生成CSV、身份清单与JSON报告。使用独立Python标准库实现逐类读取、浮点双精度字节比较、阈值0.5判定和完整排序/并列分组比较，没有导入 `cost.py` 或其比较函数；没有新增模型前向、训练、改权重或读取原图。

## 逐轮独立结果

| 轮次 | 逐图数 / 概率数 | Study数 / 概率数 | 逐图 max\|Δp\| | Study max\|Δp\| | 二值不一致 | 图内类排序/并列不一致 | 同类跨图/Study排序/并列不一致 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 / 1000 | 54 / 540 | 0 | 0 | 0 | 0 | 0 |
| 2 | 100 / 1000 | 54 / 540 | 0 | 0 | 0 | 0 | 0 |
| 3 | 100 / 1000 | 54 / 540 | 0 | 0 | 0 | 0 | 0 |

每轮所有比较值的FP64字节亦一致。另按完整身份manifest原始顺序对逐图结果重新累加并除以Study图数，独立复算的54个Study均值与该轮cost Study CSV逐字节相同。

三轮Study CSV的文件SHA均为 `e41aaed99a5f92f00361fda0d08a820da585dba880e6b2180af399b89862b072`；三轮逐图CSV的文件SHA均为 `883c89496eadeac299e592db060940143ace323ca6ba18df8f37d23833c581b1`。

## 身份覆盖和参考文件

- measured：100张图，54个完整Study；19个单图、24个双图、11个三图Study。
- warmup：5张，与测量图身份交集为0。
- 仅以Subject_id、Study_id和文件名查 `cost_inputs/original_test_identity_manifest.csv`，映射到历史sidecar的Image_id；没有按概率匹配身份。
- 对测量54个Study，所选图像ID集合与完整身份清单中的该Study集合完全相同；没有漏视图。
- 根 `submission.csv` 仍为 `44dde95be64f42c151af7b23ee4a81ffb2158822eabb9f6249a59a73f795f66e`，检查前后未改动。逐图参考是 `041_historical/submission.image_predictions.csv`。
- 读取的全部输入文件SHA、逐项身份和各轮统计见 `cost100_independent_recheck.json`。图像文件自身没有被本次复核重读；selection manifest记录的源/复制文件SHA相符。

## 精度、native和认证边界

实际 `cost100/cost_report.json` 记录FP16、seed43、native AVX2、8 CPU线程上限、关闭TF32及随机增强。上述输出与历史041 FP16参考在本次100图上没有发现native或FP16带来的概率、二值或排序变化。

本次没有单独隔离native与FP16因素，没有与FP32参考重新比较，也不推断其余未测图像完全等值。报告的66.29617746982451 ms/图只作为该次旧逐图口径记录，不转换成逐Study硬上限结论。

实际运行GPU是RTX2080Ti，`official_compliance_pass=false`、`status=not_certified`。三轮数值匹配不能把此状态改为T4已通过；官方指定样本及正式计时口径仍需对应验收材料。
