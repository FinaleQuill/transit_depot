# 4090 性能与显存压测记录

本轮由用户于 2026-10-18 明确授权：停止 `013_cloud_4090_fixed`，执行训练性能压测并调整启动参数。时间均为 Asia/Hong_Kong（UTC+8）。容量、后端诊断、CPU worker 矩阵、16/64/128 Study 的真实端到端窗口及最大真实视角袋检查均已完成。最终推荐 128 Study / chunk 128 / 12 worker / pad 16，prefetch 1、累积 1；本轮性能验收通过，014 只提供启动命令，尚未启动。

## 范围与既有训练事实

- 013 已停止；已完成 2 个完整训练/验证轮，第 3 轮至少完成日志中 200 批。原 checkpoint 与失败/停止记录保留，新任务不续训、不复用旧训练进程。
- 截至本报告阶段，正式训练成功启动 1 次（013），另有 012 初始化失败、反传 0 批。下表中的性能探针包含实际反传、AdamW 与 EMA 更新，不能再称“训练次数/反传次数均为 0”；探针不是新的完整比赛训练，也不保存权重或正式 epoch 指标。
- 已逐份核算 14 个 GPU 配置 JSON，合计完成 144 次反传、140 次优化及 140 次 EMA 更新，4 次 AMP 跳步；失败和中止配置一并计入事实，不伪装成全部成功。
- 性能选参仅访问 `data/clean_v1_reviewed/train_clean.csv` 白名单；不构造 valid/test Dataset，不使用验证指标选择性能配置。原图、标签、白名单只读，原图不搬入 work。
- 013 第 1 轮为冻结骨干阶段，GPU allocated 峰值 0.899 GiB；第 2 轮解冻后为 2.393 GiB，reserved 为 2.760 GiB。因此第 1 轮的 0.9 GiB 和低 CPU 总利用率，不能单独证明 DataLoader 是主瓶颈。

## 压测方法及计量口径

脚本为 `tools/perf_probe.py`，原始逐步记录与异常栈保存在 `/home/featurize/work/perf/20261018_1616/`，每个 GPU 配置使用独立 Python/CUDA 进程。容量测试、最初 w8 端到端测试及 v7 短诊断的探针 SHA256 为 `4c907b8d28c3574d08ed0f94af3bb72b8ed1b79de4aec59dcb6e1353ee7e3fee`；加入可选尾块补齐后，当前探针 SHA256 为 `4224d17be07da33b48418b0ffbbac7037b288abbbdc99619bf1c81cf4148ceb0`，不能把新旧方法的结果合成同一配置。GPU 实测为 RTX 4090，可用总容量 23.545 GiB。

所有 GPU 测试均采用 384 输入、全骨干解冻（90210836 个可训练参数）、float16 AMP、模型内部逐 block checkpoint、外层 CPU 输入整视角 checkpoint、Study BCE + Pairwise Logistic Rank（权重 0.2）、AdamW、CPU EMA，以及与正式入口相同的确定性算法和关闭 cudnn benchmark 设置。容量测试的成功配置先完成 2 次预热更新，再计时 2 次更新，Adam 状态在计时前已分配；容量表中的成功行共完成 4 次反传、AdamW 和 EMA 更新，未发生 AMP 跳步。

`BATCH_SIZE` 是 Study 数，`VIEW_CHUNK_SIZE` 是同时进入 GPU 的视角/图像数，二者独立。前 8 项 `gpu` 容量扫描固定重放一个真实 train batch（16 个 Study、26 个原始视角），视角不足 chunk 时在内存中显式重复视角。重复只用于制造容量负载，不增加独立样本、不用于正式训练。容量模式没有 DataLoader 等待，也没有持续读取新样本；其中 `images_per_second` 是固定 batch/重复视角计算速率，**不能当作比赛数据的端到端样本吞吐**。另行完成的最大袋检查也重放固定 batch，但其 526 个真实视角无需重复扩充。

`peak_allocated_gib` 是 PyTorch 张量分配峰值；`peak_reserved_gib` 还包括分配器缓存。二者均不等于 nvidia-smi 的整进程占用。OOM 行记录的是失败前观测峰值，不能因为数值较小而认为该配置安全。

## 已完成的 GPU 容量测试

| chunk 图像数 | 本批处理视角 / 额外重复 | allocated GiB | reserved GiB | 实际更新次数 | 结果 |
|---:|---:|---:|---:|---:|---|
| 16 | 26 / 0 | 3.340 | 4.031 | 4 | 通过 |
| 32 | 32 / 6 | 5.263 | 5.916 | 4 | 通过 |
| 64 | 64 / 38 | 9.111 | 10.846 | 4 | 通过 |
| 128 | 128 / 102 | 16.815 | 19.020 | 4 | 通过 |
| 144 | 144 / 118 | 18.743 | 21.346 | 4 | 通过 |
| 152 | 152 / 126 | 19.704 | 22.510 | 4 | 通过 |
| 160 | 160 / 134 | 19.997 | 22.969 | 1 | 第 2 次反传 OOM |
| 256 | 256 / 230 | 15.449 | 19.561 | 0 | 第 1 次反传 OOM |

160 视角首步能更新，第二步在 Adam 状态已存在时 OOM，说明只验收首步会低估常驻显存。其真实异常要求新增分配 2.81 GiB，当时仅余约 770.56 MiB；256 视角首步反传要求新增分配 4.50 GiB 时失败。完整 Traceback 分别在 `capacity_c160.json` 和 `capacity_c256.json` 保留。

已确认“152 为已测通过的最大 chunk、160 为已测失败的较小上界”，不能称已找到精确临界整数。144 和 152 的 reserved 分别已约占总容量 90.7% 和 95.6%，还没有计入 CUDA 非 PyTorch 开销，不适合仅凭 4 步就宣布长期安全。最终选择 128，其真实端到端与最大视角袋检查见后文。上述 8 个容量探针已累计完成 25 次实际反传、AdamW 与 EMA 更新。

本表所有配置的 train 清单测试前后 SHA256 均为 `f9a06316cfa8ab10ad4488e60a2dcf934cbc942727edd9ad2c31689823db6aef`，记录均显示未改变。该项不能替代全量原图字节哈希重新验收。

## 真实端到端阶段诊断：新 shape 开销

`e2e_b64_c64_w8.json` 使用 64 Study、chunk 64、8 worker、prefetch 1。该测试已被主动中止，状态为 `error`，异常类型为 `KeyboardInterrupt`，成功完成 9 次反传、AdamW 和 EMA 更新且无 AMP 跳步；只保留 3 个预热批及 6 个后续批，未生成完整 summary，不能作为已完成吞吐测试或公平 worker 对比。

中止原因是不同真实视角数量会产生不同尾分块 shape，首次遇到 shape 的 compute 时间远高于再次出现，固定重复 batch 的容量测试没有暴露这一开销。已完成批记录如下：

| 实际图像数 | 首次 compute 秒 | 再次出现 compute 秒 |
|---:|---:|---:|
| 107 | 37.939 | 2.370 |
| 95 | 25.000 | 2.124 |

首批 DataLoader 等待为 8.571 秒；其后完成批的等待仅约 0.0003–0.0020 秒。观察事实表明，该配置已经能提前准备 batch，当前主要时间消耗位于 compute 阶段，增加 worker 不能解决所见的十几秒至几十秒停顿。这是“新 shape 首次执行开销”的有力证据；具体是否由 cuDNN v8 执行计划创建造成，仍是待验证推测。

`TORCH_CUDNN_V8_API_DISABLED=1` 的诊断候选 `e2e_v7_b64_c64_w12_probe.json` 已完成：3 个预热批、8 个计时批，计时阶段处理 512 Study / 821 张真实图，耗时 28.115 秒，即 29.201 张真实图/秒。全程 11 次反传、AdamW 与 EMA 更新，无 AMP 跳步；计时阶段 loader 等待仅占 0.0053%，allocated/reserved 峰值分别为 9.120/12.768 GiB。该后端诊断不作为最终配置。不能把该轮与上述中止轮直接比较成“12 worker 比 8 worker 更快”，因为后端设置也发生了变化，且前一轮未完成。

## 方法修订：可选尾块补齐

`train_highres.py` 新增 `view_pad_multiple`，默认值 0 保留原行为；非零值须整除 `view_chunk_size`。只对单个尾分块的 CPU 工作张量补齐至指定倍数，模型输出立即截断回真实视角数量；原始图片、真实 Study indices/counts、targets 及后续聚合不增加虚构有效视角。当前骨干与分类头的样本计算相互独立，补齐输出不进入 loss；随机训练的位级轨迹不保证与未补齐时相同。

launcher 已接收并传递 `VIEW_PAD_MULTIPLE`、`PREFETCH_FACTOR`。训练 API 的补齐默认仍为 0；launcher 最终默认改为 pad 16，并可显式覆盖。5 项 CPU 语义与 cleanup 测试通过，覆盖参数约束、工作张量/输出数量和清理等行为；这些测试本身不构成 GPU 吞吐或长期显存验收。GPU 实测及最大袋检查另行记录于后文。

## 生效版本与启动器验收

以下真实 16/64/128 Study 窗口使用同一代码版本，并在结果 JSON 中记录了 torch 2.11.0+cu126、CUDA 12.6、cuDNN 91002，后端环境 `TORCH_CUDNN_V8_API_DISABLED=0`，未启用 v7 回退。运行时记录及本次 launcher 验收对应哈希如下：

| 文件 | SHA256 |
|---|---|
| `train_highres.py` | `91f37f29d1cb81c7fa2451dd2ed09bab9d8f6f1c1e166b029a793abd23ebe3c7` |
| `model_highres.py` | `f3433953ce63cfacb3b80b538bfa15d116a102a56ec969f4c5162d696e6531a4` |
| `dataset_highres.py` | `36d115a264715f23b3b1d6d03141aa1022cefd92354f4fd4cfbccfc3fc3fdd9f` |
| `tools/perf_probe.py` | `4224d17be07da33b48418b0ffbbac7037b288abbbdc99619bf1c81cf4148ceb0` |
| `run_cloud.sh`（本次轻量回归副本） | `d5d67693f3e740194900653c48cbc54a9572acd8cf72e99f0bb72d75e3593a58` |

交付前 `train_highres.py` 仅修订 4 处过期说明/注释，运行语义与上表测试版相同，最终 SHA256 为 `64ce9ba70cb978e9dbf1959aeacb8e430dc602ab6f4e7b94d6dbe1335bf40ce0`。`run_cloud.sh` 的默认 batch/worker/chunk/pad 改为 128/12/128/16，最终 SHA256 为 `40c8cec2a28bbb051068d73124383d86b2ba9038b2ca76c651928dc99a0fda30`；最终命令 dry-run 已通过。探针 SHA256 不变。上表保留测试当时的哈希，没有用交付版哈希改写历史实测。

新增 launcher 隔离回归 4/4 通过，使用真实 cxr-cloud Python 3.12 和只解析参数的假训练入口，不导入模型、不使用 CUDA。前台/后台分别验证成功 0 与失败 7；均正确传递 batch 128、worker 12、prefetch 2、chunk 128、pad 16，项目、原图根与工作目录均含空格。成功及失败日志、`.exitcode` 正确，后台 pane 分别保留状态 `1:0`、`1:7`，失败后台 launcher 返回 1 并提示原因。

专用 `TMUX_TMPDIR` 的旧 server 环境预设 backend=1，launcher 显式 backend=0 后，两种模式的子进程均收到 0，server 全局变量仍为 1，证实参数能覆盖旧 tmux 环境。测试只清理私有 server，未触碰用户任务；证据保存在 `/tmp/cxr_launcher_regression_vuupobrh/results.json`，同目录含 driver、终端日志、训练假入口日志与私有 server 清理记录。

## 纯 CPU DataLoader 对比

`loader_workers_v2.json` 记录相同 seed、64 Study/batch、prefetch 1、8 个预热批和 32 个计时批。每个配置的计时样本均为同一采样序列中的 3271 张真实 train 图；`pin_memory=False`，不构建模型、不使用 CUDA。结果衡量 CPU 读取/解码/增强管线的上限，**不是完整训练吞吐**。

| worker 数 | 首批 ready 秒 | 计时图数 | 计时秒 | 真实图/秒 |
|---:|---:|---:|---:|---:|
| 8 | 32.712 | 3271 | 38.257 | 85.502 |
| 12 | 48.746 | 3271 | 26.240 | 124.658 |
| 16 | 68.234 | 3271 | 20.170 | 162.171 |

三组均完整成功，16 worker 的 CPU 管线吞吐最高，但 spawn 首批准备时间也最长；只有固定模型、后端、补齐方法后的端到端比较才能确定最终训练 worker 数。

初版 `loader_workers.log` 在样本读取完成后，提前关闭仍有预取任务的 worker 时收到 SIGABRT，没有产生完整 result。该失败记录保留；v2 改为让迭代器自然耗尽再清理。此处是测试收尾失败，不代表原图解码故障，也不应丢弃失败记录后声称初版成功。

## 已完成的真实端到端窗口

下表是三次完整结束的短窗口，均为 8 个预热批加 24 个计时批，prefetch 1、全骨干解冻、AMP/内外 checkpoint/CPU EMA/Rank 0.2。吞吐分子只统计真实图像，不计补齐输出；时间包含计时批中的 loader 等待、前反向、优化器和 EMA。比较 baseline 与新配置时，batch、chunk、worker 及补齐策略同时不同，不能据此分解单项参数的收益。

| 配置（Study/chunk/worker/pad） | 计时真实图数 | 计时秒 | 真实图/秒 | 计时优化 / AMP 跳步 | allocated / reserved GiB | loader 等待占比 |
|---|---:|---:|---:|---:|---:|---:|
| 16 / 8 / 8 / 0 | 611 | 44.361 | 13.773 | 21 / 3 | 2.395 / 2.795 | 0.0310% |
| 64 / 64 / 12 / 16 | 2440 | 61.011 | 39.992 | 24 / 0 | 9.115 / 11.992 | 0.0398% |
| 128 / 128 / 12 / 16 | 4899 | 116.945 | 41.892 | 24 / 0 | 16.821 / 21.008 | 0.0281% |

来源分别为 `real_b16_c8_w8_p0.json`、`real_b64_c64_w12_p16.json`、`real_b128_c128_w12_p16.json`。三次均自然耗尽 loader、cleanup 成功，train 清单哈希保持不变。64 配置全程 32 次反传，31 次优化和 EMA 更新，唯一 AMP 跳步发生在 8 个预热批内；24 个计时批全部完成优化及 EMA。计时阶段为 1536 Study，CPU EMA 合计 2.442 秒，平均每步 0.102 秒；真实 loader 等待合计仅 0.0243 秒。

64 配置计时阶段额外计算 136 个补齐视角，模型实际处理 2576 个工作视角，额外部分占实际处理量 5.2795%，均在聚合前丢弃。该配置 startup 为 62.055 秒，首批 loader 等待为 8.473 秒，预热内还记录到 26.761 秒与 15.584 秒的首次 shape compute；这些初始化/预热成本未计入 39.992 图/秒。这个短窗口通过尚不等于完整轮或任意视角数下的长期稳定性验收。

128 配置全程 32 次反传、32 次优化及 EMA 更新，AMP 跳步为 0；计时阶段为 3072 Study / 4899 张真实图。额外计算 189 个补齐视角，实际处理 5088 个工作视角，补齐占处理量 3.7146%；loader 等待总计 0.0329 秒、CPU EMA 总计 1.7675 秒。startup 50.590 秒、首批等待 18.769 秒，均不计入 41.892 图/秒。运行中 GPU 单次采样为约 22076 MiB、利用率 100%、约 340 W；这是采样值，不能代替整轮峰值记录。12 worker 在该短窗口已足够供数，尚无完整端到端证据表明增加至 16 会继续提速。

### baseline 冷 shape 与诊断子集

baseline 的 13.773 图/秒包含 8 批预热之后仍首次出现的尾 shape：第 10 批为尾 3、compute 8.441 秒，第 11 批为尾 1、compute 16.562 秒。该完整窗口不是稳定后端缓存下的长期稳态值，不能据它宣传 64 配置“长期稳态提升近三倍”。

诊断方法按 warmup、计时批顺序累计已出现的 GPU 输入 batch 维度；full chunk 为 8，非零尾数为 `images % 8`，尾数 0 不构造新 shape。只有本批所有 GPU batch 维度都在先前批出现过，才标记为“已见 shape”；这是可重放的观测分类，不直接证明 cuDNN 内部缓存状态。8/2 首见于第 1 批，4/7/6 分别首见于第 2/3/4 批，5 首见于第 6 批，3/1 直到第 10/11 批才出现。

从 24 个计时批中，仅取已见 shape 且 `optimizer_updated=true` 的行，排除第 10、11 批和 AMP 跳步的第 23、27、28 批，得到 **19 批、304 Study、484 张真实图、16.926882 秒，即 28.593570 图/秒**。该时间仍含所选行的 loader/优化器/EMA；原始完整窗口及所有跳步记录均保留。此数仅是诊断子集，不能当成完整训练结果、完整窗口或长期稳态实测。逐行 shape 与选择索引证据为 `/tmp/cxr_baseline_shape_diagnosis_20261018.json`。

### 最终参数与最大袋验收

最终推荐 `BATCH_SIZE=128`、`VIEW_CHUNK_SIZE=128`、`NUM_WORKERS=12`、`VIEW_PAD_MULTIPLE=16`、`PREFETCH_FACTOR=1`、`ACCUMULATION_STEPS=1`，继续启用原 AMP、内外 checkpoint 和 CPU EMA。真实窗口吞吐略高于 64 配置，并完成全部实际优化；选择 chunk 128 而非已通过容量测试的 152，是为真实可变视角袋和分配器保留余量。152/160 的重复视角容量边界不等同于真实数据长期运行边界。

`worst_b128_c128_p16.json` 已通过：取 train 中视角最多的 128 个完整 Study 构成一个 batch，共 526 张真实视角，`min_views=0`，不重复扩充视角。1 个预热批和 2 个计时批共完成 3 次反传、优化及 EMA，AMP 跳步 0；allocated/reserved 峰值为 **16.841/19.039 GiB**，train 清单未变、cleanup 成功。每步仅增加 2 个工作补齐视角并截断其输出。固定袋重放速率为 44.590 图/秒，只作容量/计算记录，不作为随机采样的比赛数据端到端吞吐。

后台启动命令如下；**014 尚未执行，本次只交付命令**：

```bash
cd /home/featurize/work/cloud_project

env -u OUTPUT_DIR -u LOG_FILE -u SESSION_NAME \
  PYTHON_BIN=/environment/miniconda3/envs/cxr-cloud/bin/python \
  IMAGE_ROOT=/home/featurize/data/cxr_first \
  WORK_ROOT=/home/featurize/work \
  RUN_NAME=014_cloud_4090_max \
  TORCH_CUDNN_V8_API_DISABLED=0 \
  BATCH_SIZE=128 NUM_WORKERS=12 PREFETCH_FACTOR=1 \
  ACCUMULATION_STEPS=1 VIEW_CHUNK_SIZE=128 VIEW_PAD_MULTIPLE=16 \
  bash run_cloud.sh
```

显式清除旧 `OUTPUT_DIR`、`LOG_FILE`、`SESSION_NAME`，使新 run 名确定输出路径和 tmux 名；显式传入后端 0 避免继承旧 server 的 v7 设置。launcher 拒绝覆写已有日志或输出目录，启动后仍应确认实际 batch/更新进度。

## 适用范围与限制

真实 `e2e` 模式使用现有类别感知 Study sampler，持续读取不同的真实 train batch，不重复扩充有效视角，保持全网解冻、AMP、checkpoint、AdamW、CPU EMA 和 rank 权重一致。本轮 CPU 读取矩阵覆盖 8/12/16 worker，真实 GPU 窗口覆盖上述 3 个整体配置；没有把 CPU 矩阵冒称同模型条件下的 GPU worker 对比。本轮验收依据包括：

- 预热以外的真实 Study/s、原图/s、总时长及实际处理图像数量；
- 首批等待、稳定阶段 DataLoader 等待比例；
- 前向/反向、优化器和 CPU EMA 的分项时间；
- allocated/reserved 峰值、成功更新、AMP 跳步和异常；
- train 中大视角袋容量与连续多步稳定性。

仍可能影响整轮吞吐的因素包括：逐图 SHA256/原图解码/LANCZOS，spawn 与 `persistent_workers=False` 的每轮启动开销，两层 checkpoint 重算，逐参数同步拷贝到 CPU 的 EMA，以及每步标量同步。12 worker 在已测训练步骤中等待很少，不据此承诺 16 worker 会更快。

安全参数应在吞吐与余量之间选择：以真实端到端吞吐为主，在全网解冻、Adam 状态常驻、大视角袋和连续多步下通过，并给分配器及非 PyTorch 开销保留空间。达到 OOM 之前最后一个成功值仅是容量边界证据，不自动成为最终生产配置。

增大 Study batch 会减少每轮优化步数、改变 EMA 的步数时间尺度及 rank 正负配对分布；梯度累积也不等价于在一个更大微批中计算 rank。性能选参没有自动缩放学习率，也未用 valid/test 指标挑选参数。

本次优化和吞吐统计针对 train 步骤。验证仍按原有单图 batch 流程完整执行，首轮/既定周期的完整 train 诊断流程未改；41.892 图/秒不能外推为包含验证、诊断、保存和 worker 启动的整轮速度。短时吞吐/显存验收不保证长期无异常，也不构成 AUC 或其他比赛指标提升保证。
