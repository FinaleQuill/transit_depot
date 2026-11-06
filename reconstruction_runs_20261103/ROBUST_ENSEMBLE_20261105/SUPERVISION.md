# 本轮集成后台监督

**最新：本轮已完成并退出（13:35香港）。Macro0.818992373，五项门槛通过；完整test CSV已生成并审计。最终状态见[completion_status.json](completion_status.json)，交付CSV为[submission_robust_0818992.csv](../../submission_robust_0818992.csv)。原controller_status.json保留报告JSON写入失败历史；该故障已CPU恢复，无重跑推理。**

更新时间：2026-11-05T13:14:11.384079+08:00（香港）。

本轮新增训练 **0轮/0次**，使用已完成的Sprint Base768＋原A_F2 Tiny768＋Opacity512；正在执行一次完整实际FP16验证。Base此前完成4轮Sprint微调，Opacity此前完成6轮，原Tiny768及其A_F2分类头沿用既有成果。本次不重训、不续训、不启动旧控制器。

固定融合：类1只取Base；其余40% Base＋40% Tiny768＋20% Opacity512。全图hflip2，完整Study均值。

## 实际运行

- 控制器PID：201971，PPID1、独立SID、stdin=/dev/null，关闭对话不影响执行。
- 启动时间：2026-11-05T13:11:16.632817+08:00。
- 当前状态：running / final_compare；快照进度：500/1422张。该行是写入时快照，实时状态以下列文件为准。
- 已观察100→200→300张连续推进，逐图父模型输出认证未报错；15项官方镜像CPU测试通过。
- 预计完整验证及胜出CSV总计20–35分钟；实际耗时以最终状态文件为准。

## 实时文件

- [控制器状态](controller_status.json)：当前阶段、完成阶段、失败原因或最终结果。
- [验证进度](final_comparison/progress.json)：已完成图数和耗时。
- [验证日志](final_compare.log)：逐100张进度及最终分数。
- [控制器日志](controller.log)：控制器异常输出。
- 胜出后会新增winning_export.log与deployment_status.json。

```bash
cat /home/featurize/work/reconstruction_runs_20261103/ROBUST_ENSEMBLE_20261105/controller_status.json
cat /home/featurize/work/reconstruction_runs_20261103/ROBUST_ENSEMBLE_20261105/final_comparison/progress.json
tail -n 20 /home/featurize/work/reconstruction_runs_20261103/ROBUST_ENSEMBLE_20261105/final_compare.log
```

## 自动完成条件与输出

完整valid为859 Study/1422图/228患者，只评估这一个候选。验证结束做10000次患者整组配对统计。Macro与类8必须严格提高，单图Macro与患者等权Macro不降，类1预测完全保持；全部通过才生成新包和完整test CSV，test不参与选择。

- 最终指标与置信区间：`final_comparison/report.json`。
- 是否采用：`final_comparison/selection.json`。
- 胜出时新CSV：`deployment_candidate/submission.csv`，交付状态`deployment_status.json`。
- 未胜出时保留`/home/featurize/work/submission_sprint_0817963.csv`，不会额外训练或搜索融合。
- `submission_best.csv`和已交付Sprint CSV均不覆盖。
- 运行异常时保存failure/status并停止，不自动重试。`STOP`文件只拦截尚未开始的后续阶段，不立即中断正在运行的验证。
- 这是精度候选，正式T4时间验收按本轮授权暂缓；本轮没有平台上传。

源码35ee40d；启动记录4a6383a。plan.json、40份源码快照、父预测和模型身份记录均在本目录。


最终交付：1414Study/2218图/14140行，逐图→Study均值误差0；新CSV SHA256 `fedbbbef4fd45763a8f88bc2a7bfab36e6bc3ecd33bce0487ad54eaa7da29d5f`。两份原CSV及全部冻结输入/模型SHA保持，GPU无计算进程，后台退出，正式T4时延尚待。
