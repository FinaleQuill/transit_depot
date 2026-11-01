# 实验追加登记

本表从本次新增条目开始，既有实验历史仍见 WORKLOG 与各阶段文档。

| 日期（香港） | 实验 | 数据范围 | 结果与状态 |
| --- | --- | --- | --- |
| 2026-11-02 | 主体包络 tight-crop 原型 + p=0.5 擦除演示 | 仅 train:006758、train:005642；合成测试 | 去除面积43.2783%/1.9672%，768线性倍率1.0000/1.0201；文字设备仍在，未达去字目标。六项合成检查通过，无训练或全量执行。见 docs/TIGHT_CROP_TWO_IMAGE_REVIEW_20261102.md。 |
| 2026-11-02 | 固定96样本主体裁剪对比 | 旧审阅集train/valid各48、4视角各12；非全量 | 固定参数，52裁/44全，40张倍率>1.01、最大1.2159；文字设备保留。身份/几何/全框像素核验通过，未训练。见 docs/TIGHT_CROP_GALLERY_REVIEW_20261102.md。 |
| 2026-11-02 | Tight Crop后二次纯黑收缩预览 | 冻结96图中的5张(AP/PA/Lateral)，仅出图 | 非零丢失全部0；相对一次裁剪最大放大5.28%，2/5无倍率增益；保留等比补边，无训练或正式切换。图见 artifacts/exact_black_trim_5case_20261102/index.md。 |

| 2026-11-02 | tight_crop_ab_20261102：配对Tiny/Base seed48固定E2 | train训练、valid评估；A完整letterbox、B主体crop+p=.5擦除 | AP-only主要终点、患者bootstrap1000；控制器PID 28152于03:21（香港）启动。见 docs/TIGHT_CROP_AB_PLAN_20261102.json，实时状态 artifacts/tight_crop_ab_20261102/STATUS.md。 |

| 2026-11-02 | tight_crop_ab_20261102/earlystop25：用户修订A后B、最多25轮 | train优化、valid AP-only早停；B原crop+p=.5擦除 | patience5/min_delta0.0001，保留AP最佳EMA；安全迁移原A_tiny E1，见 docs/TIGHT_CROP_AB_EARLYSTOP_PLAN_20261102.json。 |
