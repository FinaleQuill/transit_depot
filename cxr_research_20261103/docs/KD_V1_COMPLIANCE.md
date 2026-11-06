# KD V1 材料留存修补与验收边界

本次同时依据官网、官网链接的PDF，以及用户转发的官方备查通知核对。
通知原文已另存，来源如实标为用户转发；公开网页的文字不能替代该补充通知。
规则原件、抓取时间与SHA256见[规则证据](../artifacts/kd_v1/compliance_repair_20261025/rules/README.md)。

## 当前训练和本次操作

检查期间，教师调度器PID12048、fold0训练PID12060持续运行。
已观测完成第1、2、3轮，训练BCE分别约1.090881、0.980209、0.933159；之后继续第4轮。
累计完成2277次优化器和EMA更新，AMP跳步0。损失下降不能证明验证AUC已经提高。

未给训练进程发送停止信号，未重启训练，未更改种子、折计划、轮数、损失或数据。
当前fold0继续执行其已加载的代码，由独立后台观察器补记；后续新进程会使用新增的原生留存机制。
本次没有加载真实检查点、运行额外模型前向或使用GPU做验证，也未重新读取大权重计算哈希。

## 官方补充通知的五项材料

| 材料 | 已落实的位置与机制 |
|---|---|
| 训练代码、依赖及版本 | `artifacts/kd_v1/compliance_repair_20261025/source_before_patch/`、`source_after_patch/`；实际Python3.12.15及66项依赖记录在同目录的`environment_record.json`、`environment_lock.txt` |
| 固定随机种子 | 各折真实`config.json`保留seed43；原有Python/NumPy/PyTorch/CUDA/数据加载及恢复随机状态逻辑保持原样 |
| 完整训练日志 | 当前run的`evidence/console_recovered_before_attachment.txt`和`console_live.raw.log`联合覆盖；后续CLI从启动起自动持久保存stdout/stderr及命令回执 |
| 初始化权重声明及SHA256 | 各折真实`config.json`保留精确模型、固定revision、原始权重路径、公开链接、SHA256、许可证全文与上游数据说明；后续学生同样记录 |
| 每轮checkpoint时间戳及哈希 | 各训练输出目录的`checkpoint_audit.jsonl`追加保存；新进程在保存后立即记录原文件mtime、已有SHA256、epoch和源码版本，追加fsync并校验摘要链 |

当前教师证据根目录：
`/home/featurize/work/runs/021_convnext_v1_base768_teacher_cv5/evidence/`。
`evidence_manifest.json`索引日志、实际配置、源码版本、依赖及账本；`monitor_status.json`记录观察器心跳和已完成轮次覆盖。

### 历史恢复的真实边界

- 第1轮原始`last.pt`已被原代码滚动保存覆盖。其哈希和保存完成后的原状态提交时间，已从另一对话保留的原始只读工具输出恢复。原工具记录、原记录SHA256、会话及行号均已留存。
- 第1轮原文件的精确mtime无法追溯，账本明确以`observed_status_at`保存原状态时间，**不会冒称该时间就是原文件mtime**，也不重新生成旧权重。
- 接管前日志来自仍完整保留的tmux可见历史。前两轮各123个应打印的批次标记全部存在，并有启动与轮末输出；这是恢复的文本日志，不冒称原始字节流。16:06:17（香港）之后是连续原始终端管道记录。
- 根据通知文字保留的是每轮时间戳和文件哈希；不会把所有约1.4GB训练断点复制成独立文件。账本如实声明权重文件会滚动覆盖，最终教师、恢复断点及部署模型仍按流程保存。

这些来源和限制随材料保留，不隐瞒也不伪造。补齐材料不能替代赛方最终审查。

## 官网和PDF的其余验收项

| 要求 | 当前核对与后续验收 |
|---|---|
| 非医学、非NC、公开可下载且许可允许使用的预训练来源 | 实际Base/Tiny为ImageNet-22k→1k的V1精确权重，登记Apache-2.0及上游MIT材料；正式路径拒绝旧V2来源 |
| 训练、验证与测试数据边界 | 教师只使用官方train的折内患者；不使用外折选模；官方valid保留给学生早停与比较；test仅最终固定参数推理，不用于伪标签或拟合 |
| 固定推理与完整CSV | 后续使用同一单模型512/FP16预测接口，完整输出每Study十类概率；1414个Study应为14140数据行，UTF-8，文件名submission.csv |
| 推理调用文件合计≤500MB | 训练完成后导出仅EMA的FP16权重及必要配置，再按实际字节验收。约1.4GB恢复断点不是最终推理文件 |
| T4端到端平均≤100ms | 使用指定环境、100张图、5张预热图、batch1串行连续3轮；读图到写结果计时，且与最终CSV一致。当前4090训练不能证明此项通过 |
| 核验材料和可复现 | 打包时归档来源许可证、初始化原权重、代码、依赖、数据声明、训练证据和实际验证/计时报告；缺项必须保持pending |

官网、PDF与补充通知分别保存并逐项核对。在线答疑若与较新的明确PDF限制冲突，按较严格的PDF边界执行；不会拿旧答疑放宽医学数据或预训练来源限制。

## 现在需要做什么

当前训练可继续，不需要为了留存改动重启。仍按原顺序完成五折教师，再提取OOF；学生训练前可再检查一次证据覆盖。
新教师和学生CLI会自动记录日志与启动前源码版本，原有启动命令继续适用。

只读查看当前观察器及训练进度：

```bash
cat /home/featurize/work/runs/021_convnext_v1_base768_teacher_cv5/evidence/monitor_status.json
tmux attach -t cxr-kd-v1
```

教师结束后，观察器正常退出；学生自己的原生账本和日志包装继续留存。
全部训练结束、最终打包之前，刷新材料索引：

```bash
cd /home/featurize/work/cloud_project_v2
KD_PYTHON=/home/featurize/.conda/envs/cxr-kd-v1/bin/python
"$KD_PYTHON" training_evidence_monitor.py refresh \
  --run-dir ../runs/021_convnext_v1_base768_teacher_cv5
```

最后的`package_kd_submission.py`命令必须带：

```bash
--training-evidence ../runs/021_convnext_v1_base768_teacher_cv5/evidence
```

完整命令已更新到[分步手册](KD_V1_START_HERE.md)。留存检查会快照并验证小文件，不扫描训练断点二进制；缺日志、逐轮记录、初始化、种子或源码关联时不会宣称材料齐全。
