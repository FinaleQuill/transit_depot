# 保留双768的单候选互补验证

登记：2026-11-05 13:00（香港），看到新候选结果前冻结。

阵容为已完成的 Sprint Base768、原 A_F2 Tiny768、Opacity Tiny512，全部固定 FP16/hflip2、完整 Study 图像概率均值。类1完全沿用 Base；其余九类为 40% Base＋40%原Tiny＋20%Opacity，即原 Sprint 的80%加Opacity的20%。只有这一个配方，不搜索比例或改类路由。

上一轮用两个 Tiny512 替换原 Tiny768，损失集中在单图类2/4/6。独立缓存审计发现 Opacity 的患者等权类8为0.772457，高于Sprint的0.765057，类8预测相关0.923233；存在互补，但单模也会破坏已有排序，因此仅低权重补充。Context不加入。患者等权是同一valid的敏感性检查，不是新的独立验证。

本次新增训练0次。已有多轮全骨干续训呈现loss下降、验证退化，当前缺少足够证据支持继续同源训练。保留此前全部权重、原始清单与两份交付CSV，不恢复任何旧训练控制器。

固定采用门槛：Macro严格提高、类8严格提高、单图Study Macro不降、患者等权Macro不降；后两项仅1e-12数值容限，另检查类1逐值不变。类2/4/6及其单图变化必须逐项报告。不用事后修改门槛让候选过关，不以患者bootstrap跨零冒称已证实线上收益。

后台执行一次完整实际FP16验证（859Study/1422图），逐图认证两父预测与历史缓存一致，保存预测后进行10000次患者整组配对重采样。通过门槛才自动构建可搬移推理包、核验固定三Study一致性并生成完整test CSV；test不用于选择。未通过则保留Sprint，无自动加训/新融合。预计20–35分钟，正式T4时延按用户要求暂缓，500MB限制仍核验。

执行：`python -m robust_ensemble.launch --config robust_ensemble/plan.json --execute`。

运行根：`../reconstruction_runs_20261103/ROBUST_ENSEMBLE_20261105/`。`controller_status.json`、`final_compare.log`和`final_comparison/progress.json`为实际状态。独立会话控制器不依赖对话连接，`STOP`只阻止尚未开始的后续阶段。

资产与所有规则/原清单/旧CSV身份绑定在[固定方案](../robust_ensemble/plan.json)。父成员审计见[审计记录](../../reconstruction_runs_20261103/ROBUST_ENSEMBLE_20261105/evidence/EXISTING_MEMBER_AUDIT.md)。


2026-11-05 13:11（香港）接手复核：采用既有未启动草稿，官方3.0镜像15项CPU测试通过。切片患者等权统一为在该切片内按每患者Study数量倒数赋权，总体采用门槛不变。实际加载权重及原配置294095910字节，运行配置另计；启动器冻结40个本地依赖源文件。

监督入口：[后台监督](../../reconstruction_runs_20261103/ROBUST_ENSEMBLE_20261105/SUPERVISION.md)。已确认实际验证100→200→300张持续推进，用户要求提供轮数及监督后关闭对话；本次训练0轮，后台继续唯一最终验证和有条件CSV导出。
