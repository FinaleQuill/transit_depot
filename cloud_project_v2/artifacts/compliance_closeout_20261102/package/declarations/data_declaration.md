# 数据声明

本声明针对本审查包登记的 V1 训练血缘和 041 历史提交材料，按官方规则与补充说明填写。

## 训练数据

监督训练只读取赛方官方 train 派生清单 `evidence/data/train_clean.csv`。该清单包含 11,055 名患者、18,513 个 Study、30,328 张图；官方 train 原始清单为 30,344 张，历史清洗日志登记 14 个同 Study 同标签同像素重复和 2 个技术废片被隔离（本轮从元数据复核 16 个排除身份与标签/Study 守恒，不重新鉴定图片；原逐图人工审阅文件当前缺失，见 evidence/data/README.md）。原始图像和原始标签没有被覆盖。五折计划 `evidence/data/folds.json` 按 `Subject_id` 隔离；各折的交集检查与计数见 `evidence/data/preflight_data.json`。

## 验证数据

只读官方 valid：228 名患者、859 个 Study、1,422 张图，哈希登记在 `evidence/data/MANIFEST_HASHES.md`。valid 未并入训练，用于早停、checkpoint/候选比较和诊断。若未来决定将 valid 并入训练，必须在并入前完成全部早停、checkpoint 选择、阈值和超参数决定；本审查包没有进行该并入。

## 测试数据

test 只用于最终固定模型预测和提交格式/身份完整性检查。没有用于训练、二次预训练、自监督、伪标签、统计量估计、归一化统计量估计、质量或置信度阈值拟合、checkpoint 选择或候选选模。测试集没有反向改变模型或后处理。`submission.csv` 是历史最高分 041 CSV，文件哈希见 `submission_record.md`。

## 外部数据和预训练

没有使用外部医学数据、医学文献衍生数据、医院数据或外部医学无标签数据。模型初始化使用已申报的通用自然图像 ImageNet 预训练 ConvNeXt V1 Base/Tiny 权重；没有使用医学预训练权重，也没有用任何外部医学数据继续预训练或自监督。

## 增强

几何、像素、遮挡和其他随机增强仅在训练期启用；推理和提交生成关闭全部随机增强算子。推理只使用固定的 EXIF/RGB/letterbox、固定归一化、冻结模型和声明的确定性 horizontal-flip TTA（如该成员配置启用）。
