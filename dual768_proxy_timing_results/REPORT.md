# 033 双 Tiny768 原样代理测速结果

日期：2026-10-29，Asia/Hong_Kong（UTC+8）。本次已完成测速并停止，无训练、优化、换权重或 test CSV 重新生成，无上传、购买、归还实例或功耗/时钟调整。

## 无插桩主基准

- 唯一对象：033 真实两成员，seed43/44，逐类固定 25%/75%；FP16、channels_last、768、每模型原图+hflip，四次串行前向，batch=1。生产源码与权重未改。
- 原 valid 固定 100 图，53 个完整 Study、40 名患者；另 5 图预热（4 Study、4 患者），与测量患者互斥。5 图预热一次后连续三轮；全部 300 次观测保留。
- 口径：读图片至逐图/Study CSV 写入并 flush，含预处理、四次前向、回传、后处理及同步；模型初始化与预热排除。输入 SHA 检查提前读图，属于自然缓存状态，非冷盘/冷启动测试。
- seed43 包装器仅设种子及查询真实 68 SM；8GiB PyTorch allocator 上限。4090 参考为 16GiB，差异如实保留；显存上限不表示模拟 T4。
- 主进程：2026-10-29 02:04:59 至 02:06:02（香港），完整脚本墙钟 62.737753 秒，退出码 **0**。只执行一次主基准，未重跑择优。

| 主基准 | 均值 ms/图 | P95 ms | max ms |
| --- | ---: | ---: | ---: |
| 第1轮 | 139.518903 | 154.471698 | 157.624477 |
| 第2轮 | 138.196810 | 152.706592 | 155.577926 |
| 第3轮 | 137.900679 | 151.800595 | 158.837344 |
| 三轮每图均值后统计（100点） | **138.538797** | **152.527627** | **156.821626** |
| 全部原始观测（300点） | 138.538797 | 153.759707 | 158.837344 |

全部 300 次观测均超过 100ms；没有删除慢样本或慢轮。峰值 allocated/reserved 为 212289536 / 249561088 bytes。未出现主基准执行异常。

4090 参考均值 79.280300 ms/图；本机/4090 实测比值 1.747456，包含 CPU 与存储差异，只作描述。`latency_pass_on_current_hardware=false`、`T4_latency_certified=false`、`official_compliance_pass=false`。**代理卡结果不是 T4 认证。**

## 预测一致性

- 本机三轮逐图与 Study 概率逐项完全一致（最大绝对差 0），固定 0.5 二值、样本内十类排序/并列、每类跨 Study 排序/并列均一致；benchmark 原有 1e-7 容差未修改。
- 对 4090 每一对应轮：逐图覆盖 100×10=1000 项，Study 覆盖 53×10=530 项，身份/类别覆盖完全一致。
- 逐图最大/平均绝对概率差：**0.00024498999118804932 / 2.4192805275561112e-05**。
- Study 最大/平均绝对概率差：**0.00024498999118804932 / 1.9770578710048648e-05**。
- 三轮均无绝对差 >1e-3，无固定 `p>=0.5` 二值变化，无样本内类别排序或并列变化。
- 每类跨 Study 排序：**第3类有一对 Study 严格反转，每轮均为同一对**；不是三对不同 Study。其余类无变化，无并列关系变化。具体身份及全部概率差保存于核对 CSV；没有修改概率、扩大容差或据此宣称语义完全一致。
- 本次是 valid 代理诊断，未执行官方 test 样本一致性认证、未重新计算/拟合 test 指标。

## 实际环境与存储

- GPU：NVIDIA GeForce RTX 2080 Ti，22528 MiB（CUDA 可见 23078109184 bytes），68 SM，计算能力 7.5；驱动 610.57.04，无 MPS。
- CPU：Intel Xeon E5-2686 v4 @ 2.30GHz，8 个可用逻辑 CPU，affinity 0–7；主进程 OMP/MKL/OpenBLAS 各1线程。平台常驻服务保留，未杀其他进程；基准启动前 GPU 无其他计算进程。
- 软件：Python 3.12.15，torch 2.10.0+cu128，torchvision 0.25.0+cu128，timm 1.0.30，NumPy 1.26.4，Pillow 10.4.0；CUDA 12.8，cuDNN 9.10.2（91002）。完整依赖锁与 torch 构建信息已存档。
- 输入、输出：均为 `/home/featurize/work` 的 **NFS v3 持久工作盘**，没有改为本地 SSD 测量。NFS 后端物理介质不可由本机确认，不标称官方 SSD。
- 隔离 Python/依赖位于 `/tmp/dual768_timing_env`，底层 `/dev/vda3` ext4 的本地虚拟磁盘；lsblk 的 ROTA=1 仅为虚拟块设备报告，不能证明宿主物理盘型号。未覆盖平台全局 Python/CUDA/驱动。隔离环境本身不作为持久资产，版本与安装日志已保存在 work。
- GPU 空闲前/主基准后快照：温度 26/32°C，功耗 18.82/16.66W，功耗上限 250W，graphics/SM 300MHz，显存时钟 405MHz。这些是空闲快照，**不代表负载时钟/功耗**；未在无插桩主基准内插入监测、sleep 或调节命令。

## 独立可选分段诊断

主基准完成后另进程运行，目录 `diagnostic_20261029_020648/`，独立 5 图预热+1轮100图，退出码 0，不替代上面的主结果。

- CPU `load_image_arrays` 均值（含读取/解码/预处理）：98.709547 ms/图。
- 四次根模型前向 CUDA Event 跨度均值：9.244753, 9.044435, 9.040167, 9.027039 ms；合计 36.356393 ms/图。
- 插桩端到端均值/P95/max：138.300553 / 153.665374 / 155.610263 ms/图。
- CUDA Event 跨度可能含主机提交间隙，不是纯 kernel 忙时；各段不视为严格互斥分解。未运行 Profiler trace。全部100点、预测、包装器、console 和退出码保留；不据此执行优化。
- 诊断的逐图及 Study CSV 与无插桩主基准第1轮逐字节一致。

## 证据与完整性

- 主结果：[local_resource_report.json](dual768_033_native_20261029_020459/local_resource_report.json)、三轮 `round_*.csv` / `round_*.image_predictions.csv`。
- 主日志/退出码/起止/执行参数：根目录 `dual768_033_native_20261029_020459.console.log`、`.exit_code.txt`、`.execution_receipt.json` / `.txt`、`.command.json`。
- 一致性与全部300点：[comparison_report.json](comparison_with_4090/comparison_report.json)、`all_probability_differences.csv`、`all_order_or_tie_changes.csv`、`all_300_timing_observations.csv`。
- 包 SHA/原始与 portable selection/manifest/config、CPU 元数据预览、所有环境/硬件/磁盘快照、依赖锁和安装日志：`preparation/` 与 `metadata_preview.json`。
- 计算脚本：`analysis/compare_with_4090.py`、`analysis/diagnose_segments.py`、`analysis/write_summary.py`；seed/资源包装器与启动器在 `preparation/`。
- tar 实测 266803659 bytes，SHA256 `ae2703c82d5d704b4fe7ed5a6d2702c14b2cca2e14673c4365ee2f0097b5b7a4`。包内 SHA256SUMS 的131项在执行前、主基准后、诊断后均通过；包括源码、真实两权重/config、全部105图及原始 manifest/selection。只新增 portable selection，且仅 `manifest_path` 重定位。
- [独立 CPU 终审](analysis/final_independent_audit.json)的34项检查全部通过；审计脚本及执行说明同目录保留。
- 结束 GPU 计算进程为空：True。工程 git status 无改动；未删除任何训练资产。

本次工作到此停止。等待用户返回4090，由原 agent 再决定是否以及如何优化。
