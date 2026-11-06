# 探索轮暂停交接（2026-11-03）

用户最新指令：**“先保存进度，停止吧”**。已经停止，恢复必须有新的用户授权。不得因旧计划存在而启动ROI/MIL或其他训练。

## 真实执行状态

- 三臂640/768/896各完成从共同041/E9 EMA起点的3轮微调；不是仅拿一个权重扫尺寸。seed44，共同768+hflip2验证。
- E3评估及比较完成；E1/best（全部真实epoch1）的三次完整FP16重评、train gap、单模计时和2000患者配对比较也已完成。
- 当前没有本轮训练/评估/队列/等待watchdog进程。ROI和MIL的run目录均不存在，**没有启动**。ROI-only队列准备被用户中止，不能说已安排后台训练。
- 固定状态 `artifacts/resolution_explore_20261103/STATUS.json` 已写 `STOPPED_BY_USER`，合并保留三臂完成指标和compliance引用。原FAILED状态完整归档到 `queue/attempts/user_stop_20261103T135642Z/`；没有把旧失败改写为成功。
- STOP标记保留在本轮根目录、queue、roi_queue_20261103。E1独立状态 `artifacts/resolution_e1_analysis_20261103/STATUS.json` 维持COMPLETED历史事实。

## 最新结果与决策

完整中文分析：`artifacts/resolution_e1_analysis_20261103/RESOLUTION_DECISION.md`。
敏感性地图：同目录 `SENSITIVITY_MAP.png` / `.pdf`。
E1比较原件：同目录 `comparison/comparison.json`；四份回执：`queue/receipts/`。

| train尺寸 | E1 Macro-AUC | AP-only AUC | 抽样train−valid gap | 三图P50/P95 ms |
|---|---:|---:|---:|---:|
| 640 | 0.803244 | 0.734243 | 0.105433 | 593.6/652.8 |
| 768 | 0.807556 | 0.738682 | 0.105912 | 596.8/670.2 |
| 896 | 0.807891 | 0.737578 | 0.100023 | 635.9/711.2 |

- 896−768 Macro +0.000334，患者配对95% CI [−0.002916,+0.004208]。类1/4/8点涨幅>0.003，但逐类CI全部跨零；0.003仅筛查线，不能叫可信线。类1/2仅16/13阳性患者。640损失主要包括类3实变；类9的640−768区间落在±0.003内。完整逐类表与E3辅助比较已写报告。
- 当前不建设按病分辨率通路。共同768推理只测试训练尺寸效应，不能宣称高分推理无用。E1和E3来自同次训练及同valid，非独立重复；CI不含训练随机性/选模偏差。
- E1固定50:50的768+896+hflip2 Macro 0.808187，相对较好896仅+0.000296、CI跨0，低于用户+0.004，放弃此组双Tiny。E1融合计时未测；不得拿E3时间当E1实测。E3双Tiny+hflip2 P50/P95为651.2/674.1ms。
- 类1=心纵隔影增大，24阳性Study/16患者；类4=肺炎，68/37。工作假设优先为投影/诊断定义与影像表征重叠，不能判定标签错误或统一归因早期/小病灶。类4 E1共同FP 18个Study中10个来自一名拥有89个valid Study的患者，肺部阴影共标签12/18；共同FN 47个Study跨30患者。
- 规则补充第七节允许同患者多图的固定确定性聚合，未限于同Study，不能把旧执行单保守禁令当官方规则。仅train的患者关系损失未见明文禁止，但不建议强迫不同时期Study十类预测一致。
- E1固定跨Study患者均值使Macro 0.807556→0.777570，差−0.029986，95% CI [−0.050144,−0.012057]；AP-only差−0.033592。类1/4单独CI跨0。放弃该无条件患者均值，保留同Study平均。E3此前未单独预注册；E1已预注册，但只是已见E3后的非独立复核。

## 已准备但未执行的下一步

建议恢复后优先纯ROI单包训练，与res768同起点/seed44/E3/预算/验证尺寸；保留MIL为随后独立候选。不捆绑擦除/关系损失/难例采样，不直接选最终成员。
ROI审图及源码/配置SHA均已核对，方法v2.2纯放大，不再用85%固定占比。图册：`artifacts/roi_magnify_explore_20261103_v2_2/gallery/index.md`、`SELF_REVIEW.md`，四视角各12张，9/48应用、39 NO CUT、0缩小，倍率1.0136–1.2152；有偏样本不能代表全量覆盖。源片基底遮挡不作完整解剖保证。
ROI-only新队列未完成也未启动；旧queue_plan包含更多训练，恢复时不能盲跑旧计划。需重新冻结仅本次授权的后续任务。

## 复现、代码与合规

- 路径故障已修：640 retry结果位于evaluations/res640_retry1，比较读取注册映射。回归测试commit `530ed14`；修复commit `b849ad9`；对应测试已通过。
- `artifacts/resolution_e1_analysis_20261103/render_resolution_review.py`生成报告/图；`resolution_report_receipt.json`记录输入及输出SHA。
- `analysis/class14_diagnostic.py/.json/.md`、`class14_diagnostic_inputs.json`包含逐病例引用、完整分母、绝对AUC患者CI、输入SHA与预注册；`analysis/replay_patient_mean_e1.py`、`patient_mean_e1*.json/.md`为患者均值可复放证据。
- 旧两张名含shared_errors的图册实际是896/E3高置信FP/FN，并非E1共同错误；不能混用。
- 正式实验仅官方train/valid；无test图像推理、新CSV、平台提交或申报变更，valid未并入训练。但必须披露：辅助规则审查一次过宽rg命中submission派生文本，结果丢弃且未用于判断，详见 `analysis/ACCESS_INCIDENT.md`；不能声称整个会话绝未接触test派生文本。
- 三个原有未跟踪文件未动：fixed_student_candidate.py、review_student_upper_bound.py、student_probability_fusion.py。

监测固定入口：

```sh
cat /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/STATUS.json
```

最终成员仍未冻结；线上提交由用户操作。当前停止是用户要求，不是模型或合规失败。
