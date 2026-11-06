# 2026-11-05 历史与退分独立审计

范围：读取主仓库 AGENTS/README/规则摘录/关键 WORKLOG、主要 E0/E1/E2、教师、KD、041/043/044、Sprint/Ensemble/Robust 评估材料与相关源代码；核对当前 Git 147 条提交及近期变更。对四份已保存 official-valid 预测做独立 CPU 正负样本对枚举 AUC 复算；没有新 GPU 推理、训练或测试集预测拟合，没有更改共享源码。不冒称逐行审完全部历史源码。

## 事实

1. `submission_best.csv` 对应041单Tiny768（A_P2_E9），本地valid Macro .8091159409。A_F2_B_F2为 .8176145465，Sprint为 .8179628779，Robust为 .8189923726；四者独立复算全部一致。文件名数字不是官方平台分数。用户这次报告Robust平台退分，但未提供绝对分数；不能推算跌幅。历史 `docs/PLATFORM_FEEDBACK_20261031.md` 已记载044本地 .816677在线不优于043本地 .810570。
2. official-valid 859 Study来自228患者，前10患者295 Study（34.34%）。十类阳性Study为24/37/46/68/115/177/193/214/260/306，阳性患者16/13/24/37/34/66/64/72/52/166。源清单 `cloud_project_v2/data/clean_v1_reviewed/valid_manifest.csv`。
3. Robust相对041十类点AUC都上涨，但净Macro增益 .0098764317 的58.05%来自类1（.6902195609→.7475548902），该类仅16阳性患者。2000次患者bootstrap所有逐类差值95%CI均跨0；Macro区间[.001853,.021037]，未校正长期同valid选择。新的重采样不能解除旧选择偏差。
4. Robust=保留Sprint Base768 + Tiny A_F2 768 + Opacity Tiny512；类1仅Base，其余40/40/20概率融合，hflip2，完整Study图像概率平均，无患者平滑。本轮新增训练0次。`robust_ensemble/predictor.py` 与 `docs/ROBUST_ENSEMBLE_RESULTS_20261105.md`。
5. Robust对Sprint增益 .001029495，原10000次患者配对区间[-.000600,+.002792]跨0。官方valid单图/多图Macro分别 .759024/.854708；病谱不同，不能因差异断定图数为因果。
6. 先前context512+opacity512替换原Tiny768的三模型 .817043865，比Sprint低 .000919013。给类8加权与空间attention尚无可归因稳定收益证据。类8显著低于十类均值只说明任务间难度差异，不证明它是最容易获得增量的病种。
7. 新隔离Study研究 E0/E1 从合规通用初始化、fold1–3训练、fold0开发：最佳 .797840/.799239，差 .001399 的患者CI[-.001001,+.003703]。dev为3703 Study/2211患者；其最低类1/4，与官方valid‘类8唯一显著低均值’判断不同。不能跨验证集把 .799解释为模型比 .819弱。
8. E2从E0最佳出发，全局3轮开发 .796436/.794982/.793816；局部2轮 .796341/.795613，均不及epoch0 .797838。局部每图5次768骨干，约5倍计算；尚无收益，完整第3轮未完成。来源 `reconstruction_runs_20261103/E2_20261104/*/history.jsonl`。
9. 无条件同患者全部Study均值已有负结果：.807556→.777570，差-.029986，患者CI[-.050144,-.012057]。来源 `cloud_project_v2/artifacts/resolution_e1_analysis_20261103/analysis/patient_mean_e1.md`。规则允许此候选，不等于数据支持。
10. 长期路线同质化：ImageNet ConvNeXt V1 Base/Tiny、五折旧教师OOF、多个后续学生与冻结头、空间/排序微调。旧全train父权重见过原H4，不能用于H4独立确认。E0/E1没有该血缘问题。已有教师、BCE/KD、Tiny/Base和E2多条学习曲线体现后期训练loss降而开发AUC降。

## 推测与结论边界

最有支持的泛化风险是同一228患者被重复用于epoch、逐类融合、冻结头、分辨率和弱类目标选择，叠加患者/病谱分布差异。现有事实不能证明退分的唯一原因，不能声称测试标签错误、具体域偏移或CSV实现故障。最近CSV完整性和模型→CSV一致性已有逐项证据，本次未发现足以解释退分的计算错误。

最值得立即投入的是新合规通用异构表征（DINOv2 reg4 Base）的患者隔离Study训练，配方在新开发集确定后用H4有限确认，再固定完整train训练。应保留已有E0作为同划分基准，避免继续按官方valid类1/8微小涨分手调十类融合。不要优先恢复E2五区域或无条件患者均值。

## 独立复算产物

- `/tmp/history_audit_20261105.py`：仅official-valid CSV/标签的独立正负样本对枚举和患者bootstrap。
- `/tmp/history_audit_20261105.json`：四套十类AUC、患者等权敏感性、文件SHA、阳性支持数、2000次bootstrap。
- 核心官方valid预测分别位于 `cloud_project_v2/artifacts/kd_v1/student_upper_bound_20261030/reports/A/valid_hflip2/`、`cloud_project_v2/artifacts/kd_v1/frozen_head_20261101/actual/A_F2_B_F2/valid/`、`reconstruction_runs_20261103/SPRINT_20261104/final_comparison/` 和 `reconstruction_runs_20261103/ROBUST_ENSEMBLE_20261105/final_comparison/`。
