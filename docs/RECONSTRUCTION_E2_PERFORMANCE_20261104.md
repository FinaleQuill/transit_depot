# E2输入流水线提速记录

用户2026-11-04要求压榨训练性能。本阶段仅调整数据加载执行参数，依据实际吞吐和内存选择；不依据开发AUC选择执行参数。

## 实测瓶颈

原E2全局对照epoch1在7,488/11,108 Study时，记录elapsed 2,654.9秒、data_wait 1,462.7秒（55.1%）、compute 1,192.2秒。data_wait是主进程取下一完整Study pack的等待，包含启动、IPC、pinning及打包前瞻；不是精确CUDA空闲时间。独立20秒GPU采样平均利用率39.05%（范围0–83%）、平均功耗127.9W。

机器为8 vCPU、约25GiB RAM、22GiB RTX2080Ti。容器无CPU quota/cpuset限制和节流；5秒观测两名worker各约82/84% CPU，主进程约50%，总计约2.15核。原图位于本地ext4，磁盘IO压力低；源码和运行记录位于NFS。训练原图中位约7.77MP，完整解码、RGB转换和Pillow LANCZOS占用CPU。当前workers2/prefetch1的输入供给不足是第一优先优化项。

## 比较与迁移边界

- 独立CPU探针仅使用固定train角色中按视图数/像素工作量选取的完整Study，比较workers2/4/6与prefetch1/2。逐tensor哈希等值与不做大tensor哈希的吞吐测试分开，避免验证本身成为消费者瓶颈。活跃训练期间的测试明确记录并发负载；不能当作独占吞吐或正式整轮加速。
- 保持完整Study、原像素/增强、标签、shuffle顺序、pack/chunk、FP16、学习率、Adam/EMA/scaler/RNG及6轮预算。进程数不参与每图增强seed；仍保持有序DataLoader输出、spawn与每轮新worker，不能简单启用会保留旧epoch的persistent_workers。
- 等当前epoch训练、完整dev评估、best/last和生命周期闭合后迁移。用户要求提速不构成丢弃已训练轮次的理由；不强杀，不从epoch0重跑。
- 新独立执行入口只替换数据加载器，并在新会话的配置和源码快照中准确登记实际执行profile；原科学配置和checkpoint evidence不改。原workers2属于注册默认值，实际覆盖以绑定的新profile及会话证据为准，不隐瞒二者区别。
- 新控制器保持原队列和frozen_plan，用补充冻结计划绑定新增入口、profile和迁移前证据；每次检查仍核对原源码/配置/输入。global_control原epoch1使用2worker，后续和global_local均使用选定profile；历史不会追认成新设置。
- 正式恢复后采集足够连续更新的实际Study吞吐、等待占比、GPU利用率和内存。只能把观测到的阶段加速称为实测；各轮样本排序和缓存状态不同，不冒称严格同轨迹计时对照。

运行产物统一放在 `../reconstruction_runs_20261103/E2_20261104/performance_20261104/`。最终profile、速度和监督入口以该目录的真实完成回执和本文件后续登记为准。

## 已完成的输入与内存测试

64个按训练元数据工作量分位选取的完整Study，共106张原图。测试以nice10与旧训练并行，使用官方无GPU容器；各设置在独立的fresh-worker张量校验和吞吐测试中均保持完全相同的Study顺序、图像、标签及局部几何。

| 输入 | 2workers/prefetch1 | 6workers/prefetch2 | 输入供给比值 |
| --- | ---: | ---: | ---: |
| 全图 | 4.066 Study/s | 11.116 Study/s | 2.73 |
| 全图＋四局部 | 1.436 Study/s | 3.909 Study/s | 2.72 |

全图同时测4/1、4/2、6/1、2/2，局部同时测6/1、2/2；6/2为本次候选中最快。典型局部测试6/2进程PSS峰2.739GiB。另取最高工作量32个不同Study（含唯一7图与四个5图Study），共135图/675区域，每次消费人为等待2秒填充队列；两遍均通过，采样共享内存峰1.747GiB/4GiB容量、进程PSS最高3.266GiB。CPU测量不包含GPU pinning/模型，不能冒称全部正式内存峰值；真实队列仍在恢复后观察。

选择 `performance_20261104/loader_workers6_prefetch2.json`，两臂后续均6workers/prefetch2，保持原4GiB SHM。官方CPU适配器5项与控制器15项测试通过，预检通过。当前E2global原完整epoch1仍按原2worker执行，随后才迁移；新配置不追认历史。输入性能不是模型效果指标，也不是正式整轮2.7倍提速的承诺。

## 已完成的正式恢复与速度观察

香港16:29原global epoch1完整闭合：11,108 Study/18,194图、348次有效更新/0跳步，纯训练3860.27秒（64.34分钟）；完整开发评估及best/last保存后正常paused。一次性后台交接保留STOP和历史证据，新controller_performance于16:30启动，PID36434，自有SID/PGID、PPID1、无TTY、stdin=/dev/null，继续原last.pt的epoch2，未重跑epoch1。

香港16:35完成启动后连续640Study实测（epoch2的128→768）：147.14秒、4.350Study/s。迁移前epoch1的update100→230稳定区间为2.945Study/s，观测速度比1.477倍；等待占比从57.45%降至0.61%。按新稳定区间外推纯训练约42.6分钟/轮，完整轮次还包括验证/保存及启动成本，不能宣称新整轮已经在42.6分钟完成。GPU抽样平均50.5%、进程显存最大16508MiB；原20秒样本平均39.05%，窗口不同仅作运行诊断。

截至16:36正式训练继续推进、0 AMP跳步，旧日志前缀、原源码/config/input和新增执行冻结均核验通过。新session实际记录6workers/prefetch2、augmentation_epoch10、sampler seed20261106；注册config的workers2不是当前实际值。当前剩余瓶颈转为模型计算与主进程调度，数据供给已基本覆盖；不把CPU探针2.7倍冒充训练倍率，也不声称局部正式臂已取得同样速度。

证据：运行根目录 `performance_launch.json`、`PERFORMANCE_BACKGROUND_VERIFIED.json`，性能目录 `RESULT.json`、`LIVE_SPEEDUP.json`、`formal_resource_observation.json`。控制器/日志/结果仍在原E2目录，安全停止仍是 `touch E2_20261104/controller/STOP`。有界测速进程已结束；后台仅继续原6+6和固定比较，不新增AI自动轮询目标。用户要求处理完结束对话，本次性能处理已完成，整个科学研究仍等待后台实验结果。
