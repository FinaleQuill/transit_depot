# 2026-11-02 新比赛镜像与 Study 推理环境核对

本报告只做宿主只读检查、远端 manifest 查询和源码阅读。未下载镜像层、运行新容器、安装 TensorRT/ONNX Runtime、占用 GPU 或执行模型。公告给出的 PyTorch 2.10.0 / CUDA 12.8 / cuDNN 9、TensorRT 10.9.0.34、ORT GPU 1.26.0 是**待镜像内实测的声明**，本次不能验证其实际包、执行 provider 或 CUDA 加载状态。

## 已确认的环境事实

- 新地址：`crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0`。`docker manifest inspect` 在空 `DOCKER_CONFIG` 凭据上下文中成功，约 1.1 秒返回 OCI schema 2 manifest，说明此处网络可达且匿名可读 manifest。config digest 是 `sha256:ffe75e4756ccc7a1d9c2e7e8e7726ea86ae704eabf5089b0d079c2c82a39dc3d`；13 层标称压缩大小合计 18,749,268,045 B（约 17.46 GiB）。这不验证全部 blob 下载权限、下载速度、磁盘实际展开量或运行成功。
- Docker client/server 均为 26.1.0，daemon 可访问；Docker root `/var/lib/docker` 在本地根盘，根盘尚余 541G。项目 NFS 工作盘尚余 18G，不宜作为镜像层或解压临时空间。未对 daemon 作配置变更。
- 本地仅有同仓库旧无标签镜像，repo digest `sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6`、显示 8.5GB；项目 `run_in_competition.sh` 和 `handoff_a/run_in_image.py` 仍锁定此旧 digest。它不等于新 `3.0`，不能把旧容器的 import/CUDA 结果当作新镜像验证。
- 宿主 RTX 2080 Ti，compute capability 7.5，驱动 610.57.04；查询时显存已占用 5550MiB，当前训练仍可能运行。该驱动版本高于 CUDA 12.8 的常规最低驱动要求，但实际兼容仍取决于镜像内库、驱动透传及容器启动。`docker info` 列有 `nvidia` runtime，未运行容器验证其可用性，更未验证 TRT engine 构建。

## 现有实现与 1-2 条执行路线

当前 `deploy_kd.py` 是 PyTorch eager FP16 / channels-last，单图解码后共享输入和 hflip，再串行运行成员；`tight_crop_ab_evaluate.py` 同样是 Tiny/Base 双模型两视图。仓库没有 ONNX 导出、TensorRT 构建或 ORT provider 接口，`requirements_inference.txt` 也未列这些包。因此“自己装 TRT”不是单个安装命令：需要匹配镜像 CUDA/cuDNN/Python ABI、导出 ConvNeXt、确认算子及动态形状支持、建立推理封装，并做完整数值与计时回归。新镜像声称自带 TRT/ORT 时，应先验证镜像内容，再决定是否需要额外安装；不要覆盖官方核心运行时。

1. **Study 推理吞吐路线。**新公告明确最小计时单元是 Study，`batch=1` 按 Study 计算，允许同 Study 多图同时处理。可在独立候选入口对同一 Study 的所有原图作一次 batch 前向，保留原始逐图身份、完整视图、固定 hflip2/融合，并测试完整 valid 逐图/Study 概率、AUC、二值及排序。样例 `pictures_100` 实为 10 Study/18 图（各 Study 图数 1:5、2:2、3:3），`pictures_5` 为 5 Study/10 图；目录名不能充当正式 100 Study 数量证明。跨 Study 合 batch 未获本公告允许；本地三轮加 5 Study 预热只是预览约定，不能称官方计时协议。新建 `benchmark_study_batch_preview.py` 只在显式 `--execute` 后运行实际模型，默认只核目录和配置。
2. **Tiny+Base TensorRT 路线。**在隔离候选中从固定 FP16 权重导出 ONNX，先检查 ORT CUDA/TensorRT providers、TensorRT Python/runtime 版本与 SM75 支持，再分别构建 Tiny/Base FP16 engine（目标 T4 上重建）。测试 768 输入、两个 TTA 视图、输出 logits、sigmoid/融合顺序及 FP32/FP64 累积；对照当前 PyTorch 的完整 valid 概率、AUC、二值与排序。引擎及必要运行文件合计还要满足 500MB 权重限制；不要同时交付冗余原权重。2080 Ti 的 engine 不能直接视为 T4 可移植成品。

任一路线都应在训练结束且 GPU 释放后，先拉取并固定新镜像 digest，再做不加载模型的版本、CUDA、TRT/ORT provider 探测；之后才做导出、数值回归和同机计时。真正能否拉完整镜像、启动 GPU 容器以及 TRT 带来多少收益，仍待实测。若基线总时延中不可被 TRT 加速的解码/I/O/预处理占比为 `f`，前向加速 `s` 倍时总加速上限为 `1 / (f + (1-f)/s)`；即使前向无限快，总加速也不超过 `1/f`。所以不能预承诺 2x/3x，更不能用 2080 Ti 的结果替代正式 T4 cost 验收。
