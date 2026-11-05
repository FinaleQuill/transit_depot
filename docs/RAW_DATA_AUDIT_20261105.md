# 官方原始数据、Study结构与未清洗清单重建

审计日期：2026-11-05，香港时间。依据本轮用户明确重新授权训练、并要求官方原始未清洗数据；不恢复任何已停止的历史训练进程。

## 原始CSV独立复算

唯一行和标签来源为 `/home/featurize/data/cxr_first/{train,valid,test}.csv`；不是 `clean_v1_reviewed`。所有原图只读。测试部分只作完整性和结构描述，不使用测试标签、预测或这些统计拟合权重、阈值、采样比例、模型选择。

| 项目 | 原始train | official valid | official test |
|---|---:|---:|---:|
| 图片 | 30,344 | 1,422 | 2,218 |
| Study | 18,513 | 859 | 1,414 |
| 患者 | 11,055 | 228 | 136 |
| 单图Study | 8,507 (45.951%) | 391 (45.518%) | 775 (54.809%) |
| AP-only Study | 50.402% | 51.455% | 59.264% |
| 患者最大Study数 | 21 | 89 | 62 |
| 来自≥5次检查患者的Study比例 | 18.506% | 66.473% | 92.008% |
| 缺失ViewPosition的图比例 | 3.961% | 2.743% | 6.267% |

原始CSV没有真实检查日期或时间字段。数值Study_id不能直接冒充已验证的时间顺序。不同Study确有不同标签，不能将某患者一次阳性标签广播到全部检查。

## 训练集本身已经证明病例构成差异

仅使用原始train标签：

| train患者分组 | Study数 | AP-only | No Finding | 每Study平均疾病数（排除类10） |
|---|---:|---:|---:|---:|
| 只有1次检查 | 7,526 | 33.125% | 52.432% | 0.838 |
| ≥5次检查 | 3,426 | 74.197% | 17.951% | 1.779 |

因此，检查频次与视角、疾病负担有关，不只是患者编号的差异。普通train随机患者划分的大验证集可以很精确地衡量train平均病例，却仍不能自动代表official valid的复诊病例。应同时检查独立开发集整体、AP-only、频繁复诊、单/多图、逐类表现；新分数不能仅由整体AUC上涨判为平台突破。这里的模型判断依据是train/valid，不将test分布用作采样或融合拟合目标。

train共有3,007位患者在不同Study间出现不同的完整标签向量，valid为86位。这使全患者硬性均值或标签恒定假设存在真实代价；即使使用被规则允许的固定患者平滑，也需要在允许的开发数据上证明收益和病程变化鲁棒性。

## 原始清单重建

可重放脚本：`raw_rebuild/prepare_data.py`。它逐行解析原CSV、保留全部标签，逐个读取/解码原图并计算文件SHA256与EXIF方向校正后的RGB像素SHA256。没有质量阈值、排除清单或重复去除逻辑；任何缺失/不可读图片都会写明失败并停止发布清单，不静默跳过。

输出目录：`/home/featurize/data/raw_rebuild_20261105/`。

- `source_train.csv`：全部30,344张原train图、18,513 Study。
- `train.csv`：原患者fold0–3，24,277张图、14,811 Study、8,844患者。
- `dev.csv`：原患者fold4，6,067张图、3,702 Study、2,211患者。明确为开发集，不能再称从未接触过的独立H。
- `valid.csv`：官方valid保持全部1,422图、859 Study、228患者，用途由主训练方案冻结。
- `folds.json`：保留历史Study→fold分配，仅将来源重新绑定到原始全量清单；对患者和精确内容跨fold再次检查。
- `audit.json`：计数、原CSV SHA、原始划分交集、图像解码结果、cost样例核验和各输出SHA。

历史清洗版保留30,328图，原始版恢复16图：13张进入train、3张进入dev。Study和患者分配没有改变。恢复16图约占原始图片0.0527%；这是用户要求的数据来源纠正，不能据此解释全部平台退分，也不能承诺仅恢复这16图就有大幅收益。

接口保持 `reconstruction.data.read_records` / `StudyDataset` 所需字段，包括原始身份、相对路径、10类targets、`status=ok`、所有train行 `training_eligible=True`、原文件及像素SHA。训练直接读原始JPEG，不读取历史缓存路径。

完整性核验已于香港21:50完成：train/valid/test全部33,984张图逐个成功解码，缺失/不可读0；全部为8位灰度L，实际尺寸与原CSV尺寸不一致0。train原图宽1067–4280、高1409–4280；valid宽1266–3520、高1716–4280；test宽1286–4280、高1554–4280。原CSV前后SHA一致；33,984张源图从未写入。

`read_records`和现有`validate_mapping`对真实清单验证通过。train/dev的患者、Study、文件SHA、像素SHA交集均0；原始train/official valid的对应四种交集也均0。所有历史保留的30,328张图，新的逐图文件SHA和原始标签与历史记录完全一致；同时新增恢复全部16张，不删图、不改标签。

输出SHA256：source_train=`4df97bb5f30ec733e5c78b66a63418c26c53396e5e49fb1f494f70ba11e89d7a`；train=`898c4096282ba765ed8f5f234200cff31f93ffc6ac5bdaf6970616d2f41b9f6e`；dev=`6f35f5272011e5e5f811d31097004e8176f29e845ad5223c14c7234bc9f81d62`；folds=`efbe91213c28fa55f82cb7ee72d069e9b93acaa72cb45f0aa60b33159a0b4777`。完整SHA见`audit.json`。此数据审计任务没有启动训练。

## cost_data_test.tar的边界

tar实际不是完整100-Study正式计时集：`pictures_100`只有10 Study/18图/9患者，`pictures_5`为5 Study/10图/5患者。

`pictures_100`中9患者均缺少其官方test中的其他Study：每位只提供1–2 Study，而同患者在test.csv有6–52 Study。`pictures_5`也没有一位患者拥有完整检查上下文。因此跨Study患者平滑即使规则允许，也不能假设本样例提供完整病程；最终必须解决cost与CSV上下文一致性。

准备脚本已逐个核对tar内28张图片与官方原test同名图的完整字节SHA，28/28一致。不展开目录、不改test图、不从样例调参。

## 验证

`python -m unittest raw_rebuild.test_prepare_data -v`：3项针对性测试通过，覆盖原始多标签（含No Finding）保留、完全重复图仍纳入、缺失文件显式报告、跨fold精确像素重复拒绝而不删图。
