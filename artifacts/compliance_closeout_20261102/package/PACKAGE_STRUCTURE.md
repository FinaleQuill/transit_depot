# 包结构及复测范围

当前是待验收审查草案，真实队伍名未提供。ZIP根包含code/、model/、pretrain_model/、licenses/、cost.py、XLSX和041 submission.csv；额外evidence/与声明方便脱离聊天独立审阅。

- code/：命名入口、完整静态依赖闭包、native来源与许可、依赖清单/离线addons。
- model/historical_041/：FP16原权重和SHA完全匹配的恢复config；仅历史041，不擅自冻结为最终。
- pretrain_model/：说明目录；示例结构不强制预训练二进制，来源/许可证/哈希齐备。
- licenses/：已冻结A1–A5原件副本；本轮无改动。
- evidence/teachers/与training/：真实源码、配置、日志、哈希、选择协议、逐权重warmup。
- evidence/runtime/cost100/：实际100图三轮完整输出与report。
- evidence/runtime/cost_inputs/：确定性选择与身份映射、文件SHA；不含图像文件。
- evidence/selfcheck/：逐条打开引用的记录、首ZIP陌生人审查及补正记录。

测试图只为用户要求的100样本一致性核验读取；本ZIP不上传原始胸片。模型权重+config历史041合计66,996,559bytes，整个ZIP大小不是规则500MB的模型加载体积。

当前机器上的105张字节副本留在cost_inputs/（包外），用于本地复查；不加入Git、不对外上传。

全部成员路径由最终PACKAGE_SHA256SUMS.json列举，可逐字节核对。最终成员冻结及T4/正式计时证据未完，见compliance_matrix.md待补项。
