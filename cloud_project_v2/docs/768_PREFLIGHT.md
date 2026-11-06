# 768 原图、正样本加权 BCE 与 RTX 4090 预检

本次仅安装环境、修改配置、CPU 检查及随机输入显存探针。真实训练次数为 0，优化器更新次数为 0；下方手动启动命令不会由代理执行。

## 输入与目标

- 固定训练/验证输入 768×768；EXIF 校正后从原图等比例 LANCZOS 缩放，居中黑色 Pad，再归一化。不拉伸，不使用旧 300/512 缓存放大。保留温和翻转、亮度/对比度与 ±5° 旋转。
- ConvNeXt V2-Base 标准全卷积骨干、GAP/LayerNorm/Dropout/十类线性头可直接接受 768 输入。AMP 使用 BF16；所有四级激活检查点启用，channels_last、TF32、cuDNN benchmark 和 GPU EMA 保留。
- `BCEWithLogitsLoss(pos_weight, reduction="none")` 在 FP32 logits 上计算。按照 [PyTorch BCE 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)，各类正项权重取训练 Study 的阴性数/阳性数，不截断。
- 同 Study 多视图只在类别统计中计一次；图像损失另乘 `N_images/(N_studies*n_views)`，保持各 Study 等权。这两种权重分别解决类别失衡和视图数量差异。
- 现有 overnight 路径本来没有 Rank/Focal，继续采用纯 BCE。强正权重已提供稀有类梯度；1–4 微批的稀有阳性很少，叠加批内 Rank 缺乏稳定成对样本与已验证收益。尚未进行新训练或声称泛化改善。
- 每折仅使用该折 train 子集统计，内部验证折及官方 valid/test 均不参与权重计算。每个 run 保存 `loss_statistics.json`，恢复必须匹配完整目标配置；使用新 run 名，保留历史权重。历史 512 checkpoint 的推理协议仍可识别。

## 全 train_clean.csv 统计

30328 图、18513 Study、11055 患者，原清单和标签只读。

| 类别 ID | 阳性 Study | 阴性 Study | pos_weight |
| --- | ---: | ---: | ---: |
| 1 | 622 | 17891 | 28.763666 |
| 2 | 900 | 17613 | 19.570000 |
| 3 | 942 | 17571 | 18.652866 |
| 4 | 1436 | 17077 | 11.892061 |
| 5 | 2404 | 16109 | 6.700915 |
| 6 | 3885 | 14628 | 3.765251 |
| 7 | 3968 | 14545 | 3.665575 |
| 8 | 4508 | 14005 | 3.106699 |
| 9 | 4654 | 13859 | 2.977869 |
| 10 | 6683 | 11830 | 1.770163 |

这是全量审计值。实际五折启动自动重算每折训练部分，数值会稍有差异；完整全量及五折统计见 `artifacts/upgrade_768/fold_loss_statistics.json`。

## 数据完整性

解压已完成，实际图片根为 `/home/featurize/data/cxr_first/cxr_first`，启动仍可传外层 `/home/featurize/data/cxr_first`。train/valid/test 清单所引用的 30328/1422/2218 张图片均存在，文件大小与原 ZIP central directory 一致。Dataset 每次只读原图并校验清单 SHA256。完整性证据为 `artifacts/upgrade_768/data_extraction_audit.json`；本次不改变清单、原图、标签或划分。

## 环境

独立环境 `/environment/miniconda3/envs/cxr4090/bin/python`：Python 3.11、torch 2.11.0+cu126、torchvision 0.26.0+cu126、timm 1.0.30、Albumentations 2.0.8。`pip check`、18 个运行模块导入、CUDA/BF16 只读检测均通过。官方预训练权重已下载并核对 SHA256，`HF_HUB_OFFLINE=1` 下 CPU 构模和 380 张量离线加载成功，无模型前向。

原 requirements 中无法从镜像获取的 NumPy 2.5.2 / SciPy 1.18.1 改为实装 2.4.6 / 1.17.1，其余精确版本保留。完整复现使用 `artifacts/requirements_768_4090.txt`；环境证据为 `artifacts/environment_768_preflight.json`。CUDA wheel 对应关系参考 [PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/#v2110)。未覆盖系统默认 Python。

## 探针语义与复现

`probe_gpu_memory.py` 对 batch 1、2、4 各开启一个独立进程；同一真实架构随机初始化，使用从 train_clean.csv 计算的加权 BCE。驻留 GPU EMA、两份 FP32 Adam moments 与已有梯度，先执行 2 次冷启动/预热微批，再执行 5 次累积前后向，并检查有限梯度。额外用 EMA 验证相应的 2/4/8 合成图像批量。无 optimizer.step、无真实图像模型输入、无训练循环和模型 checkpoint。

报告同时记录 PyTorch allocated/reserved 峰值及 nvidia-smi 采样的整卡占用；后者包含 CUDA 上下文与库开销但采样可能漏掉瞬时峰值。冷启动峰值包含 cuDNN 算法选择；吞吐是合成前后向性能，不代表包含磁盘/增强的完整训练速度。Adam moments 是等尺寸常驻模拟，没有执行真实 optimizer 更新；启动保留显存余量并沿用 OOM 降微批逻辑。

```bash
cd /home/featurize/work/cloud_project_v2
NO_ALBUMENTATIONS_UPDATE=1 /environment/miniconda3/envs/cxr4090/bin/python -u probe_gpu_memory.py
```

## 显存与性能实测

RTX 4090 24564 MiB，BF16、全阶段检查点、GPU EMA、FP32 Adam moments 与已有梯度全部启用。三档均无 OOM、加权 BCE/梯度及合成 EMA 验证输出均有限。下表峰值覆盖冷启动、累积前后向和对应验证批量。

| 微批 | allocated 峰值 GiB | reserved 峰值 GiB | 整卡采样峰值 GiB | 合成吞吐 图/秒 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.196 | 2.369 | 2.844 | 9.94 |
| 2 | 2.693 | 3.178 | 3.664 | 15.97 |
| 4 | 3.734 | 4.699 | 5.186 | 15.53 |

首轮每档 2 次预热+5 次测量；CPU loader 测试全部退出后，又对 2/4 各执行 2 次预热+20 次测量，得到表中 2/4 吞吐。两轮结果一致，batch4 安全但未提高吞吐；在要求的 1/2/4 档内，选实测最快的 batch2，累积 16，保持有效批量 32。原启动器是 8×4=32，因此不同时改变有效批量/学习率日程。GPU 采样利用率在 2/4 档均达到 100%，无需为占满显存而扩大批量。

探针无 optimizer 更新，不能等同于完整训练 step 的峰值或实际训练速度。AdamW step/EMA update 的瞬时开销未实际执行；当前配置保留充足余量，原有 OOM 降微批重试仍有效。完整原始测量分别见 `artifacts/upgrade_768/gpu_probe_768.json` 与 `gpu_probe_768_steady.json`。

CPU loader 对真实 train 前 256 张原图进行 SHA 检查、768 变换与增强，无模型计算。4/8/12 workers 分别约 34.40/64.22/90.08 图/秒；短测包含末尾进程回收，且不是端到端训练吞吐。默认 8 workers/prefetch2 已有充分供给余量，并留 CPU 给主进程。8 workers/prefetch2/batch2 的预取图像张量约 216 MiB，主机 `/dev/shm` 28 GiB；使用 pinned memory 和异步 H2D。允许根据实际 I/O 以 `NUM_WORKERS=12` 手动覆盖。证据为 `artifacts/upgrade_768/loader_benchmark_summary.json`。

## 安全启动

`run_overnight.sh` 默认 batch2/accumulation16/effective32、workers8/prefetch2，离线预训练缓存启用，使用全新 `020_convnextv2_base768_posbce_cv5_4090`，不会复用旧无正权重的训练状态。保留既有患者隔离五折、20 轮上限、早停 4 轮和完整断点恢复。

用户确认后手动运行下列一行。`--skip-inference` 让本命令完成五折后保存权重及集成清单，推理另行运行。

```bash
tmux new-session -d -s cxr768 'cd /home/featurize/work/cloud_project_v2 && bash run_overnight.sh --skip-inference'
```

查看会话：`tmux attach -t cxr768`。代理未执行启动命令，新 run 目录尚未创建。

检查已完成：49 项 CPU 测试（44 项 overnight、5 项 768 图像几何/SHA）、Bash/Python 语法、最终入口 `--dry-run`、全部环境与数据完整性检查；最终调整 batch 自动选择后，9 项 runner 测试再次通过。6 个保护清单/划分/样例文件 SHA256 与开始时完全一致。本次训练、真实图片模型推理、优化器更新均为 0。
