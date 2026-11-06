# 一周重构的独立证据复核

2026-11-06，香港时间。本报告只读取历史日志、权重元数据、数据清单和已有预测缓存并执行 CPU 核验，没有训练、GPU 前向、测试集分析或提交修改。用户当前已明确授权学生训练并禁止重训教师；历史 STOP 继续保留，不能恢复旧队列。

## 冠军与失败提交的真实身份

| 候选 | 真实模型与训练范围 | 本地评价口径 | 用户报告平台分 |
|---|---|---|---:|
| submission_best / 041 | 单 ConvNeXt V1 Tiny768，039/P2、seed44、E9 EMA；全 train 18,513 Study、历史 30,328 图；严格患者 OOF KD | official valid 0.809115941，859 Study/228患者 | 76.92 |
| submission4 / 044 | Tiny+Base、逐类融合 | official valid 0.816677 | 76.91 |
| Robust | Base768+Tiny768+Opacity512 | official valid 0.818992 | 76.75 |
| 033 / submission (2) | seed43/44 Tiny768 原配方，固定 25/75 | official valid 0.808384 | 76.72 |
| DINO20/CNN50 | DINOv2 Base518 E20 + 旧 fold4 CNN Base768 E12；两者均只见 fold0–3，约80% train；identity-only | train 内 fold4 0.808671877，3,702 Study/2,211患者 | 75.6 |

当前 SHA256 再次核验：冠军 `44dde95be64f42c151af7b23ee4a81ffb2158822eabb9f6249a59a73f795f66e`；DINO/CNN `a5055b9c329a698c2e85fa2b1a6a41280b1961a958d96d455fb688960ad6f02f`。

DINO 的内部开发比较证明其相对同折 CNN 有提升，但同折 CNN 并非冠军 Tiny。开发集不同、训练数据少20%、无 hflip、目标及模型均不同，不能将两份约0.809的本地数字视为相等泛化能力。已有导出部分复算未见足以单独解释退分的全面数值崩溃，完整部署一致性仍需由另一合规审计闭合；本报告不将失败唯一归因为架构、蒸馏或数据量。

更高 local valid 并未稳定对应更高平台成绩。Robust 相对041约58%的本地增量来自类1，而该类 valid 只有24阳性 Study/16阳性患者。不能继续把在同一228患者集合中选择出的局部最优当成可靠线上预测。

## 041 配方必须保留的细节

原始冻结配置：`cloud_project_v2/artifacts/kd_v1/student_upper_bound_20261030/models/A/metadata/config.json`。其关键设置为：

- 通用原版 `convnext_tiny.fb_in22k_ft_in1k_384` ImageNet 初始化，768完整胸廓 letterbox；seed44，FP16+GradScaler。
- **平方根正类权重**，按18,513 Study计数：`sqrt(negative_studies/positive_studies)`。类1权重约5.363，而非旧全比例28.764。
- 硬目标是逐图加权 BCE，乘 `N_images/(N_studies*Study图数)`；shuffle遍历全图。逻辑 batch 是 **8图×累积4=32图**，并非32 Study或完整Study BCE。
- KD 使用原30328×2×10 logits：先 `z_teacher-log(w_teacher)+log(w_student_sqrt)`，除 T=2，分别 sigmoid 后对 identity/hflip 两视图求均值。逐类 Bernoulli KL 含正负项并乘 T²；alpha 两轮从0渐进至0.2。稳定目标为0.8硬BCE+0.2KD。
- AdamW，backbone/head 峰值 LR 为5e-5/5e-4；WD .05，dropout .2，droppath .1，grad clip1；EMA .999，原实现 `ModelEmaV3(use_warmup=True,warmup_power=.75)`。
- **原学习率 horizon=25轮，实际最佳是第9轮。** 固定训练9轮时必须将截止轮数与衰减horizon分开；直接把 epochs 改9并缩短余弦不是复现。原短12日程对照曾从0.805594降到0.803891，不能假定快衰减天然更好。
- 推理为 identity+hflip 的概率均值，再按 Study 内全部图概率均值。旧041 FP16权重55,712,511 bytes，含旧大config资产66,996,559 bytes。

## 原始轮数、更新和耗时核验

历史A每轮恰好30,328图、3,791微批，尾窗保留，计划948个优化器窗口。完整14轮 `backward_batches=53074`、真实 `optimizer_updates=ema_updates=13266`、AMP跳步6；13272计划窗口减6恰好13266。不能将其写为0跳步。

原history未逐轮保存更新计数。按冻结调度公式反演保存的最后LR，并结合相邻轮数上限及最终计数，可核验E9已有约8,528次成功更新；这一数值是从LR重建，不能冒称原checkpoint直接保存的逐轮计数。新run应直接记录逐轮成功更新与AMP跳步。

| 完整轮数 | backbone最后LR | train秒数 | 原生验证AUC |
|---:|---:|---:|---:|
| 1 | 4.98575949e-5 | 1195.22 | 0.731854 |
| 2 | 4.97980799e-5 | 1219.77 | 0.767031 |
| 3 | 4.91932122e-5 | 1221.62 | 0.788381 |
| 4 | 4.81957454e-5 | 1202.38 | 0.798924 |
| 5 | 4.68227478e-5 | 1234.84 | 0.804872 |
| 6 | 4.50997092e-5 | 1161.35 | 0.807135 |
| 7 | 4.30524800e-5 | 1206.31 | 0.807244 |
| 8 | 4.07177305e-5 | 1193.72 | 0.808884 |
| 9 | 3.81382516e-5 | 1182.23 | 0.809129 |

前9轮纯训练合计10,817.45秒（3小时00分17秒），平均每轮1201.94秒（20分02秒）。原命令香港13:29:52至第9轮last保存16:43:03约3小时13分11秒；第1轮last至第9轮last的8个完整间隔平均21分25秒，含验证/保存。完整14轮至早停约5小时。新环境图像读取、768吞吐及额外哈希/断点开销仍须首轮实测，不能把历史时间当保证。

## 现有教师可以直接复用

现存024五教师为 ConvNeXt V1 Base768，fold0固定20轮、fold1–4固定12轮。逐项读取OOF metadata和NPZ：各折图数6070/6062/6053/6079/6064，各折均2,211患者，教师各训练8,844患者；每折缓存患者与其教师训练患者交集均为0。五份checkpoint实际均存在，每份约351.37MB。缓存logits为30328×2×10 FP32，原始前向精度BF16，不能称其FP32前向。

与raw完整`source_train.csv`对照，缓存缺16图、涉及11 Study/10患者；16图各自原fold教师均明确排除了相应患者。若需补齐，只需冻结现有教师做推理，不需要任何教师优化器更新。也允许这16图保留完整硬BCE、KD系数为0；必须使其硬标签系数为1，不能因为缺缓存连硬损失一起乘0.8。

已有 `teacher_epoch_logits_v2/raw_logits/fold{0_e20,1_e12,2_e12,3_e12,4_e12}.npz` 保存官方valid的1422×2×10原始logits与图像身份。教师未见valid，最终并训时可用于固定软目标，但当前最小实现允许valid只用系数1的硬BCE，更容易审计。无需为了最终并训重建教师。

**泄漏边界：** 当前全train教师OOF可合法用于全train学生、official-valid评价；但不能把同缓存用于train内部某一外折学生并声称全链路独立OOF，因为其它折教师见过其验证患者。冠军041见过全部train患者，也不能在某train折作折外基准。若现有教师不重训，一周内优先保留official valid作历史开发验收，而非伪造嵌套独立性。

## 多seed证据及最终建议

029/030是原全比例pos_weight的seed43/44，原生BF16最佳AUC分别0.804177/0.807570，真实FP16单44为0.807512；033固定25/75为0.808384，患者配对区间[-0.000714,+0.002738]跨零。033平台76.72低于04176.92，不能据此认为加seed43必胜。041是平方根pos_weight、2080Ti FP16的新配方，历史seed比较还混有精度与硬件差异。

最有依据的一周动作是：

1. 先以完整raw30,344图/18,513Study训练一个Tiny P2 seed44，固定9轮，保留25轮LR horizon；末次仅一次official-valid机械/质量验收。它是必要的可靠复现，不将恢复16图本身宣传为突破。
2. 主要突破来自**配方冻结后加入官方valid的真实标签**：最终31,766图/19,372Study/11,283患者，从通用初始化固定日程重训最终学生，结束取最终EMA。valid里有更多AP及反复检查困难病例，新增859Study比仅恢复16图更有收益动机。合并后不得继续以原valid早停、挑seed或融合权重。
3. 现存30328图OOF沿用alpha.2，缺缓存raw16图和新增valid1422图可用硬BCE系数1。统计新pos_weight必须仅从最终训练清单的Study计算，保留教师旧权重并在KD中重新对齐；不要改写原标签或假造完整OOF。
4. 多seed只在同配方全train阶段获得足够训练内/开发证据后作为受限候选，固定有限全类别权重。最终refit之后不以已参训valid挑成员。不要继续堆DINO、逐类补丁、未经证实的loss/裁剪/采样或昂贵新教师。
5. 原冠军保留并冻结；候选先完成实际FP16导出和predict/cost同链路复现，后续每日一次提交按明确不同假设安排，避免消耗在同一valid上的微小融合差。

该路线有数据利用与已知强配方支持，但没有证据能保证超过76.92。最终仍须按规则实际加载的权重及必要资产≤500,000,000 bytes；研究暂不筛时延不等于最终免除时延规则。cost样例是不完整运行样例，不能用于训练或选择模型策略。

证据：A的`metadata/{config,history,status,best_metrics}.json*`、`checkpoint_audit.jsonl`、`A.command.json`；`docs/{A_SUBMISSION_AND_SPEED,PLATFORM_GENERALIZATION_REASSESSMENT,OOF_CACHE_REVIEW,TINY768_SEED44_REVIEW,DUAL_TINY768_DEPLOYMENT_REVIEW}_*.md`；现有024 OOF及valid原始logits；`/home/featurize/data/raw_rebuild_20261105/source_train.csv`。
