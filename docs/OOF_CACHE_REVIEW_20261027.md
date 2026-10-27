# 严格OOF缓存验收与首个KD512对照

2026-10-27（香港）。结论：缓存完整、患者排除与教师来源通过，无需重新提取；可以执行首个Tiny512 KD对照。当前仅完成CPU验收和参数预览，未启动学生训练。

## 完整性与来源

缓存`../runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2`包含30328图、18513 Study、11055患者，`logits`为30328×2×10 FP32；实际教师前向为BF16，保存原图/水平翻转两个视图的原始logits。保存格式和前向精度不混称。

| 项目 | 验收结果 |
| --- | --- |
| 生产缓存读取器 | 实际`load_soft_labels`通过 |
| 原train身份/顺序/路径/原图SHA字段 | 与冻结清单逐项一致 |
| fold0–4图数 | 6070 / 6062 / 6053 / 6079 / 6064 |
| 每折患者数 | 各2211，患者完整分折 |
| 对应教师训练患者与OOF患者交集 | 五折均0 |
| 教师 | 原集合20/12/12/12/12、最终EMA、实际检查点SHA一致 |
| 初始化来源 | 既定V1 ImageNet-22k→1k来源与许可声明一致 |
| 类别权重 | 各教师训练Study重算的pos_weight与逐图映射一致 |
| valid/test混入 | 无 |
| 预先温度化/校正 | 无 |
| 文件临时残留/非有限值 | 无 |

NPZ SHA256：`573d0bf0f846f7f802cbb536a67ba640d209fae9d448aa72e3e8a31e2894b6c5`。
metadata SHA256：`39b8fd6014b989e7eb8905303410ac963ece1bb985816e29a52253a6b2c9e2bd`。
缓存合计8,352,953bytes。此次原图SHA核验是缓存身份字段与冻结清单绑定，未重新读取全部原图计算哈希，也未重放教师前向。

每图仅由排除过该患者的对应教师预测，未使用另外四个见过该患者的教师平均；官方valid未参与教师训练/缓存，test未参与任何拟合。当前仍在已复核的规则边界内；严格患者OOF是本项目协议，不意味着官方要求教师统一epoch。

## 描述性质量

raw logits全局范围[-17.375,12.625]，十类均非恒定。先逐视图sigmoid再平均、再按Study平均的pooled OOF Macro-AUC=0.789509541，identity为0.787652776；五折双视图分别为0.787381/0.793217/0.792056/0.795078/0.798371。

| 类别 | OOF Study AUC | 阳性Study数 |
| --- | ---: | ---: |
| 1 | 0.645890 | 622 |
| 2 | 0.830850 | 900 |
| 3 | 0.755034 | 942 |
| 4 | 0.694003 | 1436 |
| 5 | 0.869736 | 2404 |
| 6 | 0.788503 | 3885 |
| 7 | 0.794193 | 3968 |
| 8 | 0.741121 | 4508 |
| 9 | 0.899967 | 4654 |
| 10 | 0.875799 | 6683 |

这些是在原train的不同外折、由不同教师产生的描述性结果；pooled分数受跨折分布和教师分数尺度影响，并不是折分数的简单平均。不能把它与官方valid上的BCE学生0.803635直接比较来判定谁更强，也不把它当学生成绩预测。此处不按外折标签挑教师epoch、拟合逐类alpha或校准参数；没有新增超参数搜索。

实际学生目标为：

`q = mean_views(sigmoid((z_teacher − log(w_teacher_train) + log(w_student_train))/2))`。

全局q范围[0.000174,0.998004]，各类标准差约0.166–0.279，无塌缩；教师→学生log权重偏移最大仅0.0012205，映射符合预期。纯CPU重算与原生PyTorch目标最大误差1.02e−7，学生恰好匹配目标时KL最大1.52e−7，符合浮点误差量级。

类别1正/负Study的平均q为0.3169/0.2124，类别4为0.4861/0.3604，类别8为0.5623/0.4305，区分相对较弱。温度化、权重对齐后的q是训练软目标，不是已校准的患病概率，不要求阳性均值接近1。全局数值正常也不能证明教师对每类都足够好。

当前完整性结论来自确定性身份/哈希/数值检查；以上OOF AUC只做质量描述，未新增患者bootstrap推断，不将其用于跨数据集显著性比较。此前official valid上的学生/教师配对区间仍跨0，KD能否提升必须由学生对照检验。

## 当前决策

1. 不重提缓存、不重训教师、不切E13、不改ASL或逐类KD权重。保留原集合A及E13候选的真实来源；没有分布证据要求在首次KD前改T或alpha。
2. 先做已批准的Tiny512、α=.2/T=2对照。损失为两项均按类均值/Study等权，稳定后`0.8×加权BCE + 0.2×Bernoulli KL×T²`；前2轮alpha渐进到.2，KL包含阳性与阴性项，不将十类做softmax。
3. 硬标签仍占主要系数；重点复盘类别7/8/9是否弥补相对教师的缺口，以及1/4/8是否被较弱教师拖低。不能只看Macro-AUC，也不能预先保证优于BCE。
4. 从同一ImageNet权重、seed43重新训练，不从BCE E6续训。独立预览与026配置逐项比较，除训练模式、alpha、软标签路径、输出目录外，其余设置全部相同，包括512、有效batch32、学习率、增强、EMA、25轮预算、patience5及hflip2 valid选模。
5. 本次新run为`../runs/027_tiny512_kd_seed43`，只训练这一项；026纯BCE和已验收的55.78MB FP16候选保持原位。教师是离线来源，学生训练/最终推理不同时加载五教师。

## 现在执行的单阶段命令

最多25轮。按BCE实测每轮含验证/保存约7分14秒，满预算约3小时，若接近11轮早停约80–100分钟；KD有额外小量目标计算，先按约1.5–3.5小时排期，前两轮后更新。

```bash
cd /home/featurize/work/cloud_project_v2
NO_ALBUMENTATIONS_UPDATE=1 /home/featurize/.conda/envs/cxr-kd-v1/bin/python -u train_distill.py \
  --fold-plan data/cv5_overnight/folds.json \
  --soft-labels ../runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2 \
  --training-mode distillation --alpha 0.2 --temperature 2 \
  --image-size 512 --seed 43 --validation-tta hflip2 \
  --batch-size 8 --accumulation-steps 4 \
  --epochs 25 --patience 5 \
  --output ../runs/027_tiny512_kd_seed43 \
  --execute
```

训练完成后先看最佳/末轮、完整日志/AMP、硬BCE与KL、学习曲线和十类AUC，再导出最佳为同样FP16/hflip2复核并与BCE实际预测做患者配对比较；不因一次较高训练精度分数就自动推进768/多seed。若KD不改善，应保留BCE路线，而不是继续叠加复杂度。

学生仍是同一Tiny骨干；最终体积须实际导出计所有调用成员与必要配置，≤500MB。T4官方端到端≤100ms、完整test CSV与核验材料仍待验收，本阶段不能证明最终提交全部合规。

证据：`artifacts/kd_v1/oof_cache_review_20261027/{integrity.json,quality.json,quality_script.py,kd_preflight.json}`。无原生提取命令回执时不凭文件mtime推算真实提取耗时；此前20–40分钟是排期估算。本次只有CPU缓存/元数据/哈希/数学检查与参数预览，没有加载模型、运行GPU或启动训练。
