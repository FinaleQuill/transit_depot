# 可直接复制给新agent

请接手 `/home/featurize/work/cloud_project_v2`。先读 `AGENTS.md`、`AGENT_HANDOFF.md`、`docs/FINAL_OPTIMIZATION_HANDOFF_20261030.md` 和 `docs/NEW_ENVIRONMENT_HANDOFF_20261029.md`，按 `/home/featurize/work/rules.pdf`（用户称rule.pdf）及 `/home/featurize/work/rules_1.pdf` 执行。

当前优化已经完成，不是旧v1待实施阶段：2080Ti同环境本地端到端从124.754857降到81.509946ms；NFS80.781137ms；完整项目副本取消外部PYTHONPATH、离线恢复依赖并从源码重建后82.190887ms。均为原034固定100图/另5图预热、batch1串行三轮、读图至实际写CSV。默认是Pillow等值原生AVX2/generic单图并行缩放、精确uint8→FP16 LUT、单次D2H。105图输入及100图/53Study概率逐字节一致，033权重/config、FP16/TTA/25-75融合与原test CSV未改；本轮未重跑完整valid，不能冒称已跑。未训练、未重生成test、未上传。

成果证据在 `../deliveries/037_dual768_native_optimization_20261029/optimization_summary.json`；完整项目在 `../deliveries/038_dual768_optimized_project_20261030/`，旁边有ZIP和 `.zip.sha256.json`。初始化原件、离线依赖、七个训练run真实证据及逐图CSV已入包；package唯一待办是真实T4的cost_report.json，不能宣称T4已达标。最终按规则提交项目由比赛环境运行，不要自动让我租卡或迁服务器。

Featurize新窗口环境可能全部丢失：先只读核验Git/资产/挂载/GPU/空间，再自动恢复固定digest比赛镜像、依赖和原生算子；不要让我手工执行。恢复/必要核验无需再次确认。不要假定/tmp、Docker层、驱动挂载或全量原图仍在，旧helper硬编码610.57.04不可盲用。源码根没有vendor，按最新交接给038归档参数；038独立包自带code/vendor，应在本地副本prepare，原038/ZIP保持只读。work仅约1.2GiB余量，大文件/镜像/解压放本地盘；无需下载全量数据或重复下载包内初始化原件。

保住033/034/037/038及全部慢轮，不回滚旧v1、不重做已拒绝v2/v3/soup，不训练、不改预测语义、不重生成test、不上传或删资产。cost的seed43、逐图概率/二值/排序检查与训练证据打包已经修复；官方100图如不闭合Study仍保持pending，不能偷偷补齐。默认先完成必要恢复和状态核验，不自动重跑已完成的GPU实验；后续范围以我的新消息为准。简单重复核验子agent用high，正式计时不并行其他GPU/下载/安装/重CPU任务。先简要报告当前状态与确有必要的下一步。
