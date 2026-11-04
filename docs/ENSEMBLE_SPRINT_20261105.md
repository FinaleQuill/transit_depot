# 固定三模型冲刺

2026-11-05 03:38（香港）。用户已明确授权保存0.817963成果并生成CSV，同时立即启动多模型训练/推理，不跑对照或消融、不需要过程汇报。此前暂缓CSV的限制已由这次明确生成指令解除。

已完成成果：`/home/featurize/work/submission_sprint_0817963.csv`，SHA256 `099e82fa107278d6ff35a358bb2cbee2a38fb5b94b1a60758e3fd4fd9cd059a3`；完整模型/源码/逐图预测在SPRINT_20261104/deployment_candidate。1414Study×10行、2218图覆盖与原序FP64均值核验通过，旧submission_best不改。

## 阵容与训练骨架

| 成员 | 初始化 | 训练 | 分工 |
|---|---|---|---|
| Sprint ConvNeXt-Base768 | 已完成四轮最终EMA | 冻结 | 完整胸廓与类1稳定锚点 |
| ConvNeXt-Tiny512 context | 041 A E9精确FP32 EMA＋A_F2 FP32分类头 | seed57，固定4轮 | 全图GAP、尺度适配与稳定上下文 |
| ConvNeXt-Tiny512 opacity | 029 KD seed43 E8精确FP32 EMA | seed73，固定6轮 | 同一次编码的类特异空间attention残差，重点类8 |

两Tiny独立训练，继承不同历史run后重置AdamW/优化器状态。使用原官方train全部18513Study/30328图；完整Study内所有视角均值监督，不向局部区域硬贴Study标签、不裁掉胸廓。平方根正类权重BCE＋训练专用排名队列；context类8权重1.5、rank0.15；opacity类8权重3、类4权重1.5、类3权重1.25、rank0.25。512作为真实训练分辨率，不直接把旧768推理缩小。

温和翻转、旋转、曝光及gamma增强（context0.85–1.15；opacity0.75–1.25），EMA、warmup+cosine、AMP FP16、梯度裁剪；最终固定轮次EMA，无中间valid、不选epoch。不追加教师前向或新蒸馏，不用测试图调模型。原祖先KD来源与OOF证据保留，不声称本轮从通用初始化重训。

## 冻结融合

所有成员固定identity+hflip，先各自平均TTA概率，再按下表概率加权，最后完整Study图像均值。融合系数在新模型训练前冻结，不连续搜索或基于test拟合。

| 类别 | Base768 | context512 | opacity512 |
|---|---:|---:|---:|
| 类1 | 1.00 | 0.00 | 0.00 |
| 类8 | 0.40 | 0.20 | 0.40 |
| 其余 | 0.50 | 0.25 | 0.25 |

采用用户已验证可优化的1×768＋2×512组合范围。实际推理资产预计约289MB，最后逐文件复核≤500MB；正式100Study平均≤100ms仍待最终测量，不能把结构预算当作时间合规结论。

## 后台任务

`python -m ensemble_sprint.launch --execute`以独立session串行运行context4轮→opacity6轮→唯一固定组合完整valid推理。对比保存的0.817962878结果，同时保留041参考，不跑新对照；报告十类/切片及患者区间。新Macro严格超过原sprint且类8不退化才选新集成，类1固定保持；若赢则按本次授权生成独立测试CSV，若输保留已交付sprint成果，不自动补实验。

运行根：`/home/featurize/work/reconstruction_runs_20261103/ENSEMBLE_SPRINT_20261105`。`controller_status.json`看当前阶段，`context512/progress.json`和`opacity512/progress.json`看训练，最终`final_comparison/report.json`。完整断点写本机`/home/featurize/data/cxr_ensemble_sprint_20261105`，每轮FP16 EMA和合规证据持久写work。

训练先估2–3.5小时，末次验证与胜出CSV另估15–30分钟；以真实稳定吞吐更新。运行不依赖对话，无自动重试，不覆盖旧模型。根STOP在轮次保存后暂停，需明确续训才重开。

## 核验

模型/损失4项官方镜像CPU机械检查通过，真实timm Tiny纯FP16合成前向有限；两个父模型SHA/FP32 EMA/原始split/类序与完整train-valid隔离预检通过。这些是运行正确性检查，不是AUC消融或泛化证明。

历史512模型较弱、A与029有共享教师知识，所以互补收益并非已证实。选择此方案是训练新的尺度/关注方式分工；只在唯一最终比较后判断结果。无平台提交或撤销操作。
