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
