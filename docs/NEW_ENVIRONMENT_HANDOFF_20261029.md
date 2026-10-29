# 2026-10-30恢复入口更新：默认81ms实现，勿回滚v1

**先读[最新完整交接](FINAL_OPTIMIZATION_HANDOFF_20261030.md)。** 当前默认已是AVX2/generic并行LANCZOS＋精确LUT＋一次D2H：本地81.509946ms、NFS80.781137ms，完整038项目副本独立离线重建82.190887ms。当前代码/逐图一致性/seed43/7run证据及完整包均已完成，唯一项目待办是真实T4 cost_report.json。

新环境仍要自动核查挂载、GPU、空间及固定digest镜像、依赖/native。work最后约余1.2GiB，镜像/解压/副本放有空间本地盘；不要重复下载全量数据或两份包内初始化原件。旧tmp、Docker层与610.57.04驱动挂载不可假定保留/适用。

038目录与同级ZIP/SHA回执保持只读。独立包副本有code/vendor，可在固定镜像内从副本根运行`python code/prepare_inference.py --force-build`；源码根没有vendor，必须按最新交接显式传入038的archive/manifest或先恢复旧tmp addon/PYTHONPATH。`--check-only`也不是host Python命令。Git只含原生源码，不含.so；native fallback能运行不等于加速已启用。

034仍仅是105原图/旧源码，不能覆盖当前Git。旧“默认保留v1、先重跑NFS与本地、cost/seed和最终证据未完成”已失效；下文保留10月29日历史恢复命令、镜像/归档SHA及旧实验语境，不作为当前默认版本或自动实验安排。

---

## 以下为2026-10-29历史原文

# 新窗口交接：先重建比赛环境，再继续2080Ti优化

2026-10-29，Asia/Hong_Kong。用户明确：**Featurize新窗口会是新环境，已安装环境会丢失。** 接手必须按环境未配置处理，不能直接照抄“环境已就绪”的旧状态。本文件优先于旧“仅测速/返回4090”和已完成环境快照。

目标仍为在2080Ti降低耗时、研究提分；**先不要训练**。环境恢复及必要校验可自动执行，不必逐项询问；不要重新生成test CSV、上传、购买/归还实例或删除训练资产。重复读取和核验的子agent使用`high`推理强度，避免过度思考。

## 1. 保住哪些文件，哪些必须重建

先确认`/home/featurize/work`的实际挂载及材料存在；以下是预期持久化资产，不是新环境存在性保证。

| 路径 | 内容 |
|---|---|
| `work/cloud_project_v2/` | Git代码及交接；本交接前HEAD为`d580c13`，最新HEAD包含本次文档提交 |
| `work/deliveries/033_dual_tiny768_seed43_25_seed44_75/` | 冻结两权重/config、原valid及最终test CSV |
| `work/deliveries/034_dual768_proxy_timing_20261029/bundle/` | 现成105张valid图、selection、manifest、冻结旧源码及4090参考；不是当前v1代码 |
| `work/deliveries/035_dual768_2080ti_optimization_20261029/` | 本轮全部对照结果、日志、回执、拒绝候选及中止记录 |
| `work/cloud_project_v2/artifacts/kd_v1/optimization_2080ti_20261029/` | 源码快照、审计、环境包及恢复资料；Git克隆不包含此目录 |
| `work/runs/` | 教师/学生训练断点及证据，不能为省空间删除 |

`/tmp/dual768_competition_addons`、`/tmp/dual768_timing_env`、Docker镜像层及daemon配置、系统Python、驱动路径、原图和数据集zip均**不能假定仍在**。旧原图在`/home/featurize/data/cxr_first`，不在work；之前1422张valid已核过SHA，但这不是新窗口的数据恢复证明。

测速及第一步本地盘I/O对照只用034的105张图，不需要全量数据、教师、OOF或预训练下载。需完整valid时才另核实/恢复官方原图；不要重复下载已有文件，不把约52GB数据或镜像层塞入仅剩约2.4GiB的work。

## 2. 环境恢复顺序

以下命令只用于新环境准备，不执行训练或模型推理。先选择实际存在的Python处理额外JSON校验；系统Python不能冒充比赛解释器。

```bash
cd /home/featurize/work/cloud_project_v2
git status --short
git log -3 --oneline
df -h /home/featurize/work /tmp
findmnt -T /home/featurize/work
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
docker info --format '{{.DockerRootDir}}'
```

确认实际GPU/CPU/磁盘、空闲状态及Docker能力。镜像层必须落在空间足够的本地盘；镜像压缩约4.607GB、旧机解压约8.5GB，还需下载临时空间。若Docker不可用，先记录具体原因，再准备隔离Python3.12环境；不要覆盖全局驱动/CUDA，也不要擅自改用旧torch2.2或CPU推理。

固定推荐镜像；已有则核验digest，不必重复下载。若没有再拉取：

```bash
docker pull crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6
```

恢复附加包，不需要重新解析/下载torch等核心依赖：

```bash
cd /home/featurize/work/cloud_project_v2/artifacts/kd_v1/optimization_2080ti_20261029/environment
sha256sum -c competition_addons_archive.sha256
# 仅在目标不存在时解压；已存在则核对清单，不覆盖。
if [ ! -e /tmp/dual768_competition_addons ]; then
  tar --no-same-owner --no-same-permissions -xzf competition_addons_cp312_linux_x86_64.tar.gz -C /tmp
fi
```

包为8,554,684bytes，SHA256为`eb7ecd612df5568f9f87b9f3bc3ec71cf775ed3d00c4b072467f308e2f32fe10`，802个普通文件、无链接，仅13个公开附加包。它**不是完整环境备份**。逐文件清单、精确版本和缺包时的`--no-deps`重建命令见同目录[RESTORE_COMPETITION_ADDONS.md](../artifacts/kd_v1/optimization_2080ti_20261029/environment/RESTORE_COMPETITION_ADDONS.md)。核心torch/torchvision/NumPy/Pillow/CUDA保持来自原镜像。

**先核对GPU透传，再运行项目入口。** `run_in_competition.sh`是旧2080Ti主机适配脚本：旧机缺`nvidia-container-runtime`，因此硬编码4个设备和`610.57.04`驱动库只读挂载。新机不能盲用；检查实际设备、库文件和runtime。优先使用新机正常可用的NVIDIA容器透传；否则按新机实际驱动路径适配脚本，记录差异并提交Git，不安装/替换全局驱动。同样检查脚本中的work/data绑定源存在；没有全量数据不妨碍105图任务。新增本地I/O目录须单独挂载，不能因原data挂载为只读而写进原图目录。

适配完成后做版本/import/CUDA能力探测，**不做模型前向**：

```bash
cd /home/featurize/work/cloud_project_v2
./run_in_competition.sh --version
./run_in_competition.sh artifacts/kd_v1/optimization_2080ti_20261029/environment/probe_container.py
```

固定镜像旧机实际版本：Python3.12.3、torch2.10.0+cu128、torchvision0.25.0+cu128、CUDA12.8、cuDNN91002、NumPy1.26.4、Pillow10.4.0；附加timm1.0.30；镜像sklearn1.5.2、SciPy1.17.1。记录新机实际版本、GPU/SM、驱动、CPU及挂载。`competition_ready_probe.json`是旧环境证据，新探测另存新文件，不能覆盖后宣称是原记录。

若只能用隔离Python，旧回退环境版本锁见`environment/fallback_environment_lock.txt`，核心与上述一致、Python为3.12.15；它不等于实际推荐镜像。环境或材料缺失时先报告具体缺项，不能伪造已恢复。环境准备/下载/解压全部结束后才能正式计时。

## 3. 当前采用结果，不要重做失败实验

默认代码为`e0cf3e7`实现的v1：灰度直接写连续CHW、保留float32减除顺序，原logits有限值检查合并同步、成员均值一次D2H；权重、TTA及概率融合不变。`d580c13`只补充high约定，本次后续提交仅完善交接。

| 旧2080Ti推荐镜像、NFS同环境对照 | 均值ms/图 | 结论 |
|---|---:|---|
| 原033源码 | 149.807432 | 对照 |
| 默认v1 | 133.443058 | 保留，约降10.92% |
| v2并行缩放 | 144.974469 | 更慢，拒绝；完整valid27秒后主动中止，不称完成 |
| v3 CUDA Graph | 131.120745 | 仅实验；额外synthetic初始化口径待核验，不默认启用 |

上述各测速三轮逐图及Study概率与同镜像原版字节相同；45项镜像CPU测试通过。完整valid在同核心版本的隔离环境对照1422图/859Study，8590概率字节相同，AUC=.8083852425302762、F1=.4510899186168643。4090历史AUC=.8083844422198005，差约8e-7是跨卡微差，不是提分。

唯一25/75参数平均候选AUC=.8036116016664003，十类点差全负，按预登记拒绝；患者1000次配对差值95%区间[-.008452727,+.000341116]跨0，不宣称统计确认退化。不要继续扫比例。CLAHE/DALI未采用，不自动安装或训练。

旧非容器2080Ti的138.538797ms、4090的79.280300ms保留原环境语境。新窗口硬件/软件/挂载可能变化，不能用旧133.44ms直接推算新机收益，也不能把任何2080Ti结果写成T4通过。

## 4. 恢复后优先做什么

先确认v1源代码、033权重/config、034包SHA及105图选择未变。033与CSV身份SHA在主交接；环境归档内的源码是历史材料，不要覆盖Git默认v1。

环境恢复后先做一次**独立本地盘I/O对照**：同一v1、同一新环境、同一原105图、seed43、真实SM、8GiB allocator、FP16/768/两成员25/75/各hflip2，保留5图预热和100图串行三轮。从读图到实际写CSV全部计时，模型初始化排除。GPU容量变化先核实8GiB上限是否合法。

新机先跑一次NFS输入输出参照，再将相同105张图原字节复制到有空间的本地盘，另存selection派生文件（只重定位manifest_path，不改图片/标签/选择），做本地输入输出对照。另存输出目录，核对复制SHA、三轮覆盖/概率/固定.5/排序并保留慢轮；记录实际挂载。虚拟盘ROTA标志不足以证明物理SSD。**本地结果不能覆盖NFS参照，不能称为代码提速或T4认证。** 不并行解压、安装、下载、其他GPU任务，不清系统页缓存、不加sleep、不调功耗/时钟。

先复盘I/O贡献，再决定下一项优化；无须重跑历史v2/v3/soup，也无须先重跑完整valid。若改动预测，再单独用完整valid做总体/十类AUC、F1、患者不确定性和切片，不能用test选参。正式T4、cost语义一致性/seed、训练证据及最终包仍待，见主交接P0项。

每项独立修改验证后Git提交，不推送。原图、权重、原CSV、旧报告不改；新日志/安装回执/环境探测/结果保存在work，避免再次随环境消失。最新阶段先反馈结果，再进入下一阶段；本轮不训练。
