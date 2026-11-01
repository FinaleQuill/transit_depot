# 初步裁剪框肉眼复核入口

2026-11-02（香港）。本轮仅重绘已保存的 train/valid 候选框并准备严格版本框架，
未重跑几何检测、未训练、未运行模型、未读取 test。候选来自 `8cff076` 的旧检测器；
它们不是经过医学确认的安全框。源图、标签、正式预处理与旧候选 CSV 保持原样。

## 怎么看图

- 左图：完整原图等比例缩小；中图：仅供显示的 Pillow FIND_EDGES；右图：原图上的候选框。
- **青色为原图边界，红色为旧 CSV 归一化候选裁剪框。** 两框重合处用红色虚线和外置青色括线区分。
- `NO CUT` 表示保留完整画面；最大黑边样本可能属于此类，不应误读为裁剪已获准。
- 每个实际裁边附有 1:1 原像素条带：`raw_1to1.png` 无遮罩；`1to1.png` 用透明红色标出拟删除部分。
  查看细节时在图片查看器中使用原始大小，聊天缩略图可能缩小条带。
- 视角沿用 manifest 字段，不根据本次观感改写；文字/不规则/边缘评分只用于挑选复核样本。
- 中图是完整解码后重绘的显示边缘，**不是**原审计 JPEG draft 路径的逐像素重放；裁剪框没有重算。
  图上同时列出旧 CSV 原像素坐标及其与归一化框映射的误差，避免隐藏旧坐标问题。

## 优先查看这十二张

这些均来自 train，valid 的对应分层样本见完整索引。

| 视角 | 典型候选（裁剪面积中位） | 最大裁剪候选 | 最大黑边（可能不裁） | 需重点检查的候选 |
| --- | --- | --- | --- | --- |
| AP | [024667](../artifacts/geometry_review_20261102/train/AP/train_AP_02_train_024667.jpg) | [019013](../artifacts/geometry_review_20261102/train/AP/train_AP_03_train_019013.jpg) | [006758](../artifacts/geometry_review_20261102/train/AP/train_AP_04_train_006758.jpg) | [005642：文字及设备](../artifacts/geometry_review_20261102/train/AP/train_AP_05_train_005642.jpg) |
| PA | [027181](../artifacts/geometry_review_20261102/train/PA/train_PA_02_train_027181.jpg) | [015295](../artifacts/geometry_review_20261102/train/PA/train_PA_03_train_015295.jpg) | [010083](../artifacts/geometry_review_20261102/train/PA/train_PA_04_train_010083.jpg) | [029718：贴近下边缘](../artifacts/geometry_review_20261102/train/PA/train_PA_05_train_029718.jpg) |
| Lateral | [007760](../artifacts/geometry_review_20261102/train/Lateral/train_Lateral_02_train_007760.jpg) | [026622](../artifacts/geometry_review_20261102/train/Lateral/train_Lateral_03_train_026622.jpg) | [009682](../artifacts/geometry_review_20261102/train/Lateral/train_Lateral_04_train_009682.jpg) | [007023：不规则前景](../artifacts/geometry_review_20261102/train/Lateral/train_Lateral_06_train_007023.jpg) |

[完整 96 张索引和边缘条带](../artifacts/geometry_review_20261102/index.md) ·
[逐图坐标、选择原因与源 SHA](../artifacts/geometry_review_20261102/index.json)

每个 train/valid × AP/PA/Lateral/Missing 层各 12 张，96 张源图互不重复。
所有选图的源 SHA 与冻结清单一致，渲染前后匹配；索引绑定两份候选 CSV 和渲染脚本 SHA。
这只证明图像来源和框的展示可追溯，不证明自动裁剪安全。

## 已知问题与下一步判断

代码复核发现的问题比“边缘算子较弱”更具体：

1. 旧 Otsu 分子缺少 `total * cumulative_weight` 中的 `total`；train 的 28,653/30,328
   张阈值为 254，valid 为 1,348/1,422。全局分割及逐边 Otsu 投影均受影响。
2. `safe_crop_frame()` 接收的前景组件并未实际参与裁框约束。旧 `safe_crop=True`
   只代表旧程序条件通过，不能称已保护肺尖、肋膈角或胸壁。
3. 名义缩放因子与整数分析尺寸不完全一致，导致旧原像素框的右/下边界出现最多约 3 px
   偏差，甚至在全框回退时发生。展示保持旧证据，未静默修正检测结果。

因此，肉眼审图用于发现明显误切和确定下一版重点。即使这批样例未见误切，也应先修复
数值与坐标错误，再考虑任何裁剪实验；不能把换用 Canny 单独当成解决方案。
旧黑边计数仍是固定灰度阈值下的诊断统计，不能解释为可安全删除的面积或图片数量。

[严格版本代码框架](STRICT_GEOMETRY_FRAMEWORK_20261102.md) 已登记：train 冻结阈值、
`cv2.Canny`、中心连通分量、`findContours`/凸包、逐边否决、原图 margin 和保守坐标映射。
本轮仅提供设计，等待用户看图后的决定，不安装依赖、不切换检测器或启动 A/B。

## 保存与复现

- 可视化脚本：[render_geometry_review.py](../render_geometry_review.py)。仅从冻结候选中选图，拒绝越过 train/valid 根的源路径。
- 图片及索引：`artifacts/geometry_review_20261102/`，留在服务器，不加入 Git。
- 旧审计 CSV、summary 和预登记文件的原样持久副本：`artifacts/geometry_audit_preliminary_20261102/`；
  其内容仍有上述已知局限，复制不等于重新审计或批准实验。
- 本文、渲染脚本和严格框架用 Git 保存；模型、原图和候选数据不入库。
