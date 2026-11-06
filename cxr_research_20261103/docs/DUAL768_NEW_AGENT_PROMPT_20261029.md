# 可直接转发：只测033双768，不优化

> 历史任务文本：2080Ti测速已完成。用户最新要求直接留在2080Ti优化、本轮不训练，见[当前优化记录](DUAL768_OPTIMIZATION_2080TI_20261029.md)。下文“只测速/返回4090”不再是当前任务范围。

下面分隔线内的文字可直接发给已连接新2080Ti服务器的agent。若Git工程不在目标机，但测速包已迁移，则优先阅读包内README_TIMING_ONLY.md；任务和操作说明都不依赖全量数据集。

---

你现在接手的是**仅测速任务**，请自动完成，不要停在给命令或等待我手动执行。

我同一时间只租一台服务器，已从4090切到当前代理卡。当前优先应为Featurize的RTX2080Ti 22G；如果实际是RTX3060，请记录具体型号和显存。本次只测冻结的033双Tiny768：seed43/44占25%/75%、FP16、各identity+hflip、四次串行前向、batch=1。测完我会返回4090，再由原agent决定是否及如何优化。

请先读取：

1. /home/featurize/work/cloud_project_v2/AGENT_HANDOFF.md（若工程存在）。
2. /home/featurize/work/cloud_project_v2/docs/DUAL768_PROXY_TIMING_HANDOFF_20261029.md，或测速包内README_TIMING_ONLY.md。

材料优先在：

/home/featurize/work/deliveries/034_dual768_proxy_timing_20261029/

其中dual768_timing_only.tar.gz是完整的小测速包，bundle目录若已存在可直接用；packaging_verification.json记录压缩包SHA。包内已含033真实双seed权重/config、原样推理/测速源码、100张valid图和另5张预热图、完整valid manifest与4090参考结果。**无需重新下载或解压全量数据集，不需要教师、OOF或训练断点。** 若原路径不在，先在本人工作区查找这份测速包，再按包内说明解压；找不到就报告缺失，不能换成旧031便携包或别的模型。

我已授权你自动完成以下工作，不必逐项再问确认：

- 核验包SHA和内部SHA256SUMS、真实GPU/CPU/磁盘及空闲状态。
- 核实Python3.12、torch2.10.0+cu128、torchvision0.25.0+cu128、timm1.0.30、numpy1.26.4、Pillow10.4.0；缺环境时在目标机空间足够的本地盘准备隔离环境，不覆盖全局驱动/CUDA，不盲用旧conda绝对路径。
- 仅派生selection的manifest_path指向包内manifest，保留原件与所有图片/标签/哈希。
- 先CPU元数据预览，再按操作文档的seed43包装器，用现有benchmark_local_resources.py自动执行唯一主基准：5图预热、100图batch1串行三轮，读图到写结果全部计时。2080Ti22G/306012G用8GiB allocator上限；若实际为8GB卡按文档用6GiB。查询真实SM，不用默认128，不启用MPS。
- 保存全部三轮Study及逐图CSV、local_resource_report.json、stdout/stderr、退出码、执行起止/墙钟、环境/CPU/GPU/存储记录。保留全部异常和慢轮，不重跑择优。
- 测后用CPU比较同机三轮和4090参考，报告覆盖、最大/平均概率差、固定0.5二值差异及排序/并列变化；不改概率或放宽现有检查来制造通过。
- 主基准完成后，若环境允许，在独立进程做一轮短分段诊断，记录CPU预处理、四次前向CUDA事件跨度及端到端耗时；只用包外诊断包装器，保存脚本，不能改生产源码，也不能用有插桩时间替换主基准。事件跨度不能冒称纯kernel时间。无法稳定做分段时，先保住主基准，明确该诊断未执行。
- 将结果和核对脚本保存在/home/featurize/work/dual768_proxy_timing_results/的独立本次目录。结束前复核输入和源码SHA未变，给我简短报告：三轮均值、总体均值、P95/max、预测差异、实际环境、结果目录。最后停止任务，等我返回4090；不要自动开始优化。

边界必须遵守：不训练、不改Loss/增强/预处理/精度/TTA/权重/融合、不加sleep或调功耗/时钟、不执行旧breakdown脚本、不重新生成test CSV、不访问test分布、不上传、不购买或归还实例、不删除训练资产。代理卡结果不是T4认证；现有benchmark的官方通过字段应保持false。当前基线4090为79.280300ms/图，AUC0.808384来自历史完整valid，不要把它写成这次重新测得的指标。

简单重复核对用medium，复杂诊断用high，不用ultra。若遇到确实无法完成的环境或材料缺失，保留日志并说明具体缺项，不得更换测速对象或编造结果。

---
