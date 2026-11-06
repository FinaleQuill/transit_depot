# 主体裁剪固定96张样例对比

2026-11-02（香港）。用户授权将新主体裁剪扩展至此前展示的其他图片。本轮对冻结
索引中的96张样例执行同一算法，每个train/valid × AP/PA/Lateral/Missing层各12张。
没有依据valid结果调整参数，没有读取test、模型推理或训练，也没有全量切换。

## 快速看图

总览中每个样本**左边是完整原图直接等比补边到768，右边是主体裁剪后等比补边到768**。
两边都不拉伸，也没有随机擦除。总览供快速比较，细小边界请打开单图和1:1原像素条带。

| 视角 | train 12图总览 | valid 12图总览 |
| --- | --- | --- |
| AP | [打开](../artifacts/tight_crop_gallery_20261102/contact_train_AP.jpg) | [打开](../artifacts/tight_crop_gallery_20261102/contact_valid_AP.jpg) |
| PA | [打开](../artifacts/tight_crop_gallery_20261102/contact_train_PA.jpg) | [打开](../artifacts/tight_crop_gallery_20261102/contact_valid_PA.jpg) |
| Lateral | [打开](../artifacts/tight_crop_gallery_20261102/contact_train_Lateral.jpg) | [打开](../artifacts/tight_crop_gallery_20261102/contact_valid_Lateral.jpg) |
| 缺失视角 | [打开](../artifacts/tight_crop_gallery_20261102/contact_train_Missing.jpg) | [打开](../artifacts/tight_crop_gallery_20261102/contact_valid_Missing.jpg) |

[完整96图逐张索引](../artifacts/tight_crop_gallery_20261102/index.md) ·
[逐图指标与来源](../artifacts/tight_crop_gallery_20261102/metrics.json)

每张六联图上排为原图、亮组织种子、主体支持掩膜；下排为原图上的框、完整画面768、
新裁剪768。青框表示原图范围，红框是候选裁框，绿框是主体支持范围。

## 本组样例结果

这是此前按极端黑边、裁剪量、文字及不规则性挑选的**有偏复核集**，统计不能推断
完整train/valid上的比例，也不是模型效果评估。

| 视角 | 样例数 | 产生裁剪候选 | 保留完整画面 | 768线性倍率增加超过1% | 最大线性倍率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AP | 24 | 15 | 9 | 14 | 1.2013× |
| PA | 24 | 11 | 13 | 10 | 1.2013× |
| Lateral | 24 | 13 | 11 | 7 | 1.1718× |
| 缺失视角 | 24 | 13 | 11 | 9 | 1.2159× |
| 合计 | 96 | 52 | 44 | 40 | 1.2159× |

52个裁剪候选中9个的线性倍率仍精确为1：裁掉了短轴空白，但决定缩放的最长边未变。
另3个倍率增加不超过1%。44张完整画面中43张是外扩后本来就覆盖全图，只有1张
`train:013922` 因候选框外仍有较大支持区而拒绝裁剪。不能把44张全部称为安全闸门拒绝。
全部NO CUT病例的新旧768结果已逐像素核对一致。

优先复核以下病例：

- [AP / valid:001034](../artifacts/tight_crop_gallery_20261102/valid/AP/valid_001034/comparison.jpg)：
  裁去顶部空白，768线性倍率1.2013×；PORTABLE等文字仍在。
- [PA / train:010083](../artifacts/tight_crop_gallery_20261102/train/PA/train_010083/comparison.jpg)：
  去除面积27.59%，768线性倍率1.2013×；边缘字母仍在，不能称去字成功。
- [Lateral / valid:000196](../artifacts/tight_crop_gallery_20261102/valid/Lateral/valid_000196/comparison.jpg)：
  768线性倍率1.1718×，需检查上下肺野与后侧肋膈角保留情况。
- [Lateral / train:009682](../artifacts/tight_crop_gallery_20261102/train/Lateral/train_009682/comparison.jpg)：
  去除面积33.92%，但线性倍率1.0000×；原图本身已不完整，不能将已有截断归因于新裁剪。
- [缺失视角 / train:013922](../artifacts/tight_crop_gallery_20261102/train/Missing/train_013922/comparison.jpg)：
  检测到框外较大支持区，保持原图，不强行套用候选。

本轮说明主体裁剪对部分带大空白的图像有几何收益，但仍不能清除胸廓范围内的文字、
电极和导管，也没有证明泛化增益、解剖安全或时延合格。抽看没有误切不等于整组或全量
安全。算法参数保持两图原型版本，不在此复核集上继续调参。

## 执行与身份

- 检测器：[tight_crop.py](../tight_crop.py)，来自提交`7638765`，SHA256
  `85ec065e2b4a9bb23a0defcafc3681cd474a579563affd3501ddcb4d2426cd8d`，本轮未改。
- 输入索引：`artifacts/geometry_review_20261102/index.json`，SHA256
  `4dccd0d13893068470ea2a5ef98ab24cfc1ad4880fab1f62b374a3ef8afacbeb`；96个ID固定。
- 新入口：[preview_tight_crop_gallery.py](../preview_tight_crop_gallery.py)，只接受固定索引，
  校验canonical manifest、各split根路径、原图尺寸和SHA，阻止覆盖旧输出。
- [render_gallery_contacts.py](../render_gallery_contacts.py) 仅从已有六联图生成总览，
  不访问数据集或重新检测。
- 96张源SHA执行前后匹配索引及清单，96个EXIF Orientation均缺失，无方向标记导致的
  转置差异。源图只读；输入索引和检测器SHA执行前后不变，全部输出可追溯。
- 共96张六联图、96张768结果、98条原像素边缘条带、8张总览及指标，约56MB，图片不入Git。
- 本轮独立CPU可视化，不重做模型测试，不作端到端90ms验收；下一步仍以人工复核和
  用户指令为准。旧safe-crop继续废弃。
