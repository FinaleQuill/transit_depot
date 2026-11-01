# 严格图像几何候选框架（待批准，不是执行结果）

日期：2026-11-02（Asia/Hong_Kong）。本文件只设计将几何候选检测改为
`cv2.Canny` + `cv2.findContours` 的后续实现；没有安装 OpenCV、修改检测器、
扫描图像、运行模型或重新生成任何清单。生产路径仍以完整原图的现有预处理为准。
轮廓、凸包和连通分量只是图像亮度的**机械信号**，均不等于肺、胸廓边界或医学安全证明。

## 现有实现的静态核查

- `audit_image_geometry.py:74-85` 的 Otsu 分子现在是
  `(total_weight * cumulative - cumulative_weight) ** 2`。正确的、允许约去公共
  常数后的类间方差分子应是
  `(total_weight * cumulative - total * cumulative_weight) ** 2`，分母是
  `cumulative * (total - cumulative)`；两类均非空时才参与最大值。这个缺失的
  `total` 与 Pillow 无关，是 NumPy 公式的数值错误。`cv2.threshold(...,
  THRESH_BINARY | THRESH_OTSU)` 可作为独立对照，但不应靠换库掩盖公式错误。
- 可手算反例：三个灰度值 `[0, 128, 255]` 各一个像素。当前公式在阈值 0/128
  的分数分别为 `383^2/2 = 73344.5`、`638^2/2 = 203522`，选择 128；正确
  公式分别为 `383^2/2 = 73344.5`、`382^2/2 = 72962`，选择 0。若 0 到
  255 各有一个像素，当前式会选 254，而正确 Otsu 在 127 附近分开；错误式
  偏向高阈值，**254 不是所有输入的必然结果**。
  此处只是代数核查，尚未跑数据分布或回归测试。全局前景分割与逐边投影
  `otsu_boundary_offsets()` 都受这一错误影响，已有候选框不能视为已验证。
- `audit_image_geometry.py:322-325,336-339,386-391` 的分析尺寸各轴经过整数
  舍入，但映回原图时仍用名义 `scale=512/max(W,H)`，四坐标还用 `round()`。
  短轴实际比例是 `analysis_width/W` 或 `analysis_height/H`，可能不同于名义比例；
  `round()` 对左/上边可能向内推进，对右/下边可能向内收缩。现有
  `crop_aspect_ratio` 也在分析坐标中计算，不能代替原图整数框的长宽比闸门。
- `safe_crop_frame()` 当前没有用传入的 `component` 实际约束裁边，边缘来自
  Pillow `FIND_EDGES`，没有 Canny/真实外轮廓/凸包。因此旧 CSV 的 `safe_crop`
  是旧规则的诊断字段，不能解释为通过下述严格框架。仅把 Pillow 边缘算子
  换成 Canny 也不会修复 Otsu、连通分量未约束、原坐标映射及保护区问题。
- `analyze_row()` 对 JPEG 调用 `draft("L", (512,512))`，而 `load_analysis_pixels()`
  绘制示例时重新完整解码并缩到 512。两条路径的像素及边缘可能不一致，
  红/绿/蓝框即使源于同一候选行，也不能证明展示的纹理就是算法所见纹理。
  后续应让检测与可视化共用相同解码/缩放结果，并同时另附原图供审查。

## 冻结输入与配置

只允许规范的 `train_clean.csv` 和 `valid_manifest.csv`，以及两者各自行内
`relative_path` 指向的官方 split 根。逐行 `resolve(strict=True)` 后须确认位于
`image_root / split` 内；拒绝 `test`、任意替换 manifest、跨 split、软链接逃逸、
重复 ID、尺寸或源 SHA 不符。`valid` 仅用冻结 train 配置做诊断；不能重拟合
阈值，也不准根据 valid 框重新挑配置。审计只写新目录，不改原图、缓存、标签、
manifest 或部署路径。解码失败要登记失败，不以伪造的成功框继续处理。

配置对象建议明确以下字段并在读 valid 前冻结：

```python
@dataclass(frozen=True)
class GeometryConfig:
    schema_version: int
    analysis_long_edge: int = 512
    blur_kernel: tuple[int, int] = (3, 3)
    blur_sigma: float = 0.0
    dark_threshold: int = ...          # train 边缘像素分位数，受固定上限约束
    dark_min_coverage: float = 0.97
    dark_min_fraction: float = 0.015
    canny_low: int = ...               # train 拟合后冻结的整数
    canny_high: int = ...              # high > low；不得逐张自适应
    contour_min_area_fraction: float = ...
    edge_min_span_fraction: float = ...
    edge_max_gap_fraction: float = ...
    agreement_fraction: float = 0.02
    max_crop_side_fraction: float = 0.08
    safe_margin_fraction: float = 0.02
    safe_margin_original_px: int = 32
    central_roi_fractions: tuple[float, float, float, float] = ...
    protected_roi_fractions: dict[str, tuple[float, float, float, float]] = ...
    aspect_p01: float = ...            # train 原图可解码尺寸，P1
    aspect_p99: float = ...            # train 原图可解码尺寸，P99
    train_manifest_sha256: str = ...
    train_source_set_sha256: str = ...
    opencv_version: str = ...
    detector_version: str = ...
```

`...` 是待按 **train** 分布或事先固定的值，不能被解释为已拟合常数。
拟合记录须含 train 纳入 ID、视角分层、边缘像素分位数定义、固定上限、分位数
方法、Canny 阈值与连通性/覆盖的选取办法、train P1/P99、样本数与缺失数；
若 train 图像或元数据不可用，停止冻结，不能用 valid 补齐。默认沿用 512
最长边、等比例缩放；256 只是**另列提案**，必须先做框一致性、人工复核和时延
验证，不能静默替换 512。JPEG draft、灰度转换、插值法及 OpenCV 版本均进入
配置/执行身份，避免训练统计与运行图像预处理不一致。

将配置按固定字段顺序序列化为 canonical JSON（UTF-8、无 NaN、固定数字表示）
得到 `config_sha256`；保存 manifest 内容 SHA、train 源文件 `(image_id, SHA)`
有序列表 SHA、实现源码 SHA、库版本。每个输出行保存源图字节 SHA、manifest
中的 SHA、匹配结果、配置 SHA、train manifest SHA、检测器版本、原图与分析图
尺寸、各边原始信号/否决理由、最终整数框和回退理由。源 SHA 验证须覆盖**每张**
输入图，不仅是抽样示例；源 SHA 不符时本行失败且不可作为候选。

## 检测函数框架

以下是设计用伪代码，不是可运行脚本。所有框使用半开区间 `[x0,y0,x1,y1)`。

```python
def fit_train_geometry_config(train_manifest, image_root, fixed_limits):
    rows = verified_rows(train_manifest, split="train", image_root=image_root)
    # 仅 train：可解码原图宽高、边缘灰度分布、预登记的视角分层统计。
    stats = collect_train_stats(rows, fixed_limits)
    config = freeze_config(stats, fixed_limits, train_manifest_sha256=sha256_file(train_manifest))
    return config, canonical_sha256(config)

def detect_geometry(row, split, config, image_root):
    source = verified_source(row, split, image_root)  # 只准 train/valid；校验源 SHA/尺寸
    W, H, gray = decode_gray_once(source)
    aw, ah = aspect_preserving_size(W, H, longest=512)
    small = resize_gray(gray, (aw, ah), config)  # 记录精确 aw, ah；不使用名义 scale 回映
    blur = cv2.GaussianBlur(small, config.blur_kernel, config.blur_sigma)
    threshold, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if constant_or_degenerate(binary):
        return fallback(W, H, "degenerate_otsu")

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    central = components_intersecting(labels, config.central_roi_fractions)
    component = largest_valid_central_component(central, stats, config.contour_min_area_fraction)
    if component is None:
        return fallback(W, H, "no_central_component")
    component_mask = np.uint8(labels == component) * 255
    contours, hierarchy = cv2.findContours(component_mask.copy(),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = largest_non_degenerate(contours)
    if contour is None:
        return fallback(W, H, "no_valid_contour")
    hull = cv2.convexHull(contour)
    if hull_is_unstable_or_touches_frame(hull, (aw, ah)):
        return fallback(W, H, "ambiguous_hull")
    edges = cv2.Canny(blur, config.canny_low, config.canny_high, L2gradient=True)

    side_results = {}
    for side in ("top", "bottom", "left", "right"):
        dark = contiguous_dark_band(small, side, config.dark_threshold,
                                    min_coverage=config.dark_min_coverage)
        boundary = supported_long_canny_boundary(edges, side, dark, config)
        side_results[side] = vet_side(side, dark, boundary, contour, hull,
                                       small, edges, config, (W, H), (aw, ah))
    box = outward_original_box(side_results, (W, H), (aw, ah), config)
    return whole_box_gate_or_fallback(box, side_results, config, (W, H))
```

先在中心 ROI 中找**与中心区域相交且面积足够**的连通分量，再取其中最大者；
不能先取全图最大亮区、再假定其是胸廓。轮廓和凸包均来自这个分量；它们只为
候选裁边提供保守外包络。凸包可能跨越真实凹陷、贴住标记或设备；接触画面边缘、
面积异常、多个中心分量相近、轮廓不稳定时回退，不靠凸包修补缺失组织。
512 缩图及 JPEG draft 可能抹去原图仅一像素宽的组织、细管或细字；这些结构
在缩图上消失时，任何 Canny/轮廓判据都无法可靠检出。无法获得充分信号时
一律不裁。高分辨率原图边缘条带可用于离线安全复核；若部署也依赖它，完整
解码和额外检测的时间必须计入端到端预算，不能沿用 draft 解码的 84.82ms
基线却省略这部分成本。

## 逐边否决与原图整数框

对每条边独立计算候选。`vet_side` 仅在以下条件**全部满足**时才接受该边，
任何缺失、冲突或数值非有限都保留该边原样，并输出机器可读的否决代码：

1. 从外边起连续近黑带，采样位置覆盖至少 97%，原图对应厚度至少该边长的
   1.5%；阈值来自冻结 train 配置，不能逐图提高以制造暗带。暗带宽度和拟裁
   宽度都不得越过单边 8% 上限；反常宽暗带标记复核。
2. Otsu 中心连通分量的实际外轮廓与其凸包均位于候选裁切线**内侧**。暗带内
   若存在与边缘相连的亮组织、明显梯度、突出的轮廓、强 Canny 结构或多个
   不相容轮廓，该边否决。使用整条边的投影及局部分段，避免全边均值掩盖
   一角的组织。
3. Canny 在暗带与保留组织交界附近形成足够长、间隙受限的支撑边，位置与
   Otsu 外轮廓/凸包边界的**同边投影**相差不超过该轴分析尺寸的 2%。
   检查支撑边连续性、靠近暗带内缘及边外噪声；内部肋骨/肺纹理边不能单独
   当成图像边界。没有稳定长边缘即否决。
4. 将保留的组织轮廓、凸包及受支持的 Canny 边界都作为不得越过的界限。
   对左/上边，裁切线位于所有受保护边界之外至少 `max(0.02*原图对应边长,
   32px)`；右/下边反向。margin 在原图轴分别计算，**先映射再做整数约束**。
5. 保留中央胸廓与预登记的左右下角、上部角落等保护区视觉 proxy；这些只
   是保护皮肤/肺尖/肋膈角的宽松位置闸门，绝不冒称自动解剖识别。边缘
   文字、左右标记、引流管、电极、植入器械等高频结构被发现或难以排除时，
   该边否决并送人工复核；不得以“黑底”删除可能有临床意义的标志。

每条边输出 `dark_width`, `dark_coverage`, `otsu_boundary`, `hull_boundary`,
`canny_boundary`, `edge_span`, `edge_max_gap`, `protected_signal`, `margin_px`,
`candidate_original_px`, `accepted`, `veto_codes`。多信号的阈值、投影办法与
保护区必须冻结在配置中。`accepted=False` 的边始终取原图边界，不得因另一边
通过而顺带移动。

坐标变换必须使用**实际** `(aw, ah)` 与 `(W, H)`，不使用名义 512 缩放因子。
先在原图连续坐标中施加安全 margin，再把保留矩形向外保守取整：

```python
# a_* 是分析图半开坐标；其边界先减/加对应的原图安全 margin。
x0 = max(0, math.floor(a_x0 * W / aw - margin_x))
y0 = max(0, math.floor(a_y0 * H / ah - margin_y))
x1 = min(W, math.ceil(a_x1 * W / aw + margin_x))
y1 = min(H, math.ceil(a_y1 * H / ah + margin_y))
# 未获准的边直接覆写为 0、0、W、H；必要时加一分析像素的插值不确定带。
```

若候选本身已包含 margin，不能重复加；实现中只保留一个明确的 margin
施加点。检查最终**原图整数框**的每边裁去比例 `<= 0.08`、有效宽高、原图
长宽比落入 train P1-P99、中心及保护区包含性。整框任一硬闸门失败，直接
返回精确 `(0, 0, W, H)`，`crop_applied=False`，记录如
`aspect_outside_train`, `crop_limit`, `protected_region`, `signal_disagreement`,
`mapping_uncertain` 的原因；不得返回 `(0,0,aw,ah)` 或把分析坐标误作原图框。
对全框回退仍保留每边审计细节及原图 SHA。完整图原本超出 train P1-P99 时，
不能借裁边伪造“通过”；记 `original_aspect_outside_train` 并回退。

## 人工确认与后续验证门槛

机器输出只能称 `geometry_candidate` 或 `review_required`。按 AP、PA、侧位、
缺失视角及裁剪比例，从 train/valid 各层抽取至少 12 张，含最大/中位/最小
比例，展示原图、Otsu 中心分量与凸包、Canny、暗带、原图最终框和各边原因。
人工确认没有切掉皮肤、肺尖、肋膈角、设备及文字后，才可另行申请离线 A/B；
医学安全仍需有资格的人员判断。少数已选样例肉眼通过不能推出算法整体可靠，
更不能推出全量医学安全。确认前不得把候选框用于训练或提交。

合成单测方案（仅设计，本次不编写/运行）：

- Otsu 的三灰度 `[0,128,255]` 手算例、均匀图、两峰图、复制像素后的
  不变性；将修正的纯 NumPy 公式与 OpenCV 的阈值在非并列样本上对照。
- 非方形尺寸（含短轴舍入误差）、四边各一条黑带、仅部分边有黑带：核查
  `floor`/`ceil` 向外、整数框每边 margin、精确 full fallback。
- 黑带内插入一像素亮皮肤突起、角落文字/标志、跨边导管、断裂 Canny 边、
  不相交的中央亮块、凸包接触边框：相关边应否决，其他边按各自证据处理。
  一像素合成例主要用于**暴露缩图盲区**；若细线被 512/draft 消去，不能把
  检测器未报警当成安全通过，也不承诺缩图检测能找回所有细小结构。
- 7.9%/8.1% 单边、train P1/P99 边界、退化 Otsu、源 SHA 不符、跨 split/
  test 路径、软链接逃逸、valid 配置不匹配：验证拒绝或明确回退原因。
- 固定 512 与独立 256 提案的框一致性及合成延迟测量；256 不通过时保留
  512，不能用 256 指标冒充 512 实测。

若后续获准 A/B，仍按现有计划用同一模型/输入和完整计时口径比较，先做
人工框审查、train/valid 预登记，之后才检验 Study Macro-AUC、逐类、患者
bootstrap、视角/图数切片与结果一致性。已交付另一配置的 2080Ti 基线均值
84.82ms 对应新增检测均值预算 **5.18ms**，单图 P95 硬门 **7.77ms**；
当前冻结头 95.41ms 版本不能借用该预算。上述检测器尚未实现或计时，
**不承诺达门槛或性能收益**；真实 T4 仍须单独验收。
