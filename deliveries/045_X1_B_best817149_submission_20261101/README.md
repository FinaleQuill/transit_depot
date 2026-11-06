# 045：用户选定当前最高valid方案

submission.csv为本轮明确按最高valid AUC生成的正式格式候选；未上传平台。

- valid Macro-AUC：0.8171491901128627，不是平台test成绩，未达0.82。
- 2080Ti真实端到端三轮平均93.490616ms，真实官方T4复测仍待。
- 新Tiny X1 E4与旧Base B E10，各768/FP16/hflip2；原始完整JPEG解码。第1类仅B，其余九类各50%，原manifest顺序FP64 Study均值。不是87ms的draft候选。
- UTF-8四列Subject_id,Study_id,Predict_class,Probability，1414Study×10类=14140行。
- SHA256：7379ac54cf4a8c66dd021e0f7c36a5fdaa69f32f6af2efd7ce3a4cf9bc3920da。
- 原2218张test图SHA匹配；独立核验身份、十类覆盖、有限概率和逐图→Study均值逐bit一致，误差0。无test统计拟合、指标或选模。
- 固定模型持久目录：../../cloud_project_v2/artifacts/kd_v1/bounded_finetune_20261101/reviewed_models/full_decode/。权重已持久保存，采用不可原地修改的hardlink；本目录不重复拷贝模型以节约工作区空间。
- SELECTION_BEFORE_TEST.json绑定最终预测前的模型/权重/配置；原始predict报告保留实际/tmp生成路径，本目录CSV与原始产物逐字节相同。

相对旧AB的valid点差仅0.000319，患者区间跨0；用户明确指定按最高点分交付，不将其解释为已证明平台提分。043/044保持原样。后续训练只能利用train/valid，不能从本CSV概率拟合或选择。
