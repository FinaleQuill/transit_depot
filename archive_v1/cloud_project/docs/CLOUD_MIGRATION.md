# Featurize Linux / RTX 4090 云端迁移

## 2026-10-18 性能验收后的启动配置

013已按用户要求停止，已完成两轮权重保留。当前已验收的云端默认参数为128 Study、128视角分块、12 worker、prefetch=1、尾块补齐倍数16、累积1；保持384、AMP、激活检查点和CPU EMA。真实端到端训练短测41.89图/秒，allocated/reserved峰值16.82/21.01GiB；128个最大视角Study的526张真实图也通过完整反传。原始记录、容量边界及限制见 `PERFORMANCE_20261018.md`。

014尚未启动；新后台任务使用以下命令，保留旧任务和旧日志：

```bash
cd /home/featurize/work/cloud_project
env -u OUTPUT_DIR -u LOG_FILE -u SESSION_NAME \
PYTHON_BIN=/environment/miniconda3/envs/cxr-cloud/bin/python \
IMAGE_ROOT=/home/featurize/data/cxr_first WORK_ROOT=/home/featurize/work \
RUN_NAME=014_cloud_4090_max BATCH_SIZE=128 NUM_WORKERS=12 \
PREFETCH_FACTOR=1 ACCUMULATION_STEPS=1 VIEW_CHUNK_SIZE=128 \
VIEW_PAD_MULTIPLE=16 TORCH_CUDNN_V8_API_DISABLED=0 bash run_cloud.sh
```

`VIEW_PAD_MULTIPLE`必须为0或整除`VIEW_CHUNK_SIZE`的正整数。若改回1/2/8等小分块，同时指定`VIEW_PAD_MULTIPLE=0`。补位只在内存工作张量中执行，输出截断后才进行真实Study聚合。增大Study batch改变每轮优化次数，本次没有依据验证集分数选择性能参数或宣称比赛指标提升。

## 2026-10-18 云端启动修复

下文“本次训练为0”描述最初迁移交付时点；用户此后已授权前台排错和训练验证，实际状态以 WORKLOG.md、工作盘 runs/ 和日志为准。

当前可用解释器为 `/environment/miniconda3/envs/cxr-cloud/bin/python`（Python 3.12），终端默认 `python` 为 3.11，不能用于本脚本。`PYTHON_BIN` 会传递给后台子进程，没有写死 Python。

前台复现发现预训练权重下载报 `ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`。已将 `socksio==1.0.0` 纳入 requirements 并在专用环境安装，官方骨干权重已缓存；没有禁用预训练或替换模型。

`--foreground` 现在同时输出终端和日志；启动前的 Python/路径错误也写入日志。后台失败保留 tmux pane，并在 `<LOG_FILE>.exitcode` 保存真实退出码。保留的 pane 可能已退出，必须结合日志 batch 进度和 `tmux list-panes -a -F '#S #{pane_dead} #{pane_dead_status}'` 判断，不能只看 `tmux ls`。

用户重试前先检查是否已有训练运行，选择全新的 RUN_NAME（旧日志与失败输出均不覆盖）：

```bash
cd /home/featurize/work/cloud_project
PYTHON_BIN=/environment/miniconda3/envs/cxr-cloud/bin/python \
IMAGE_ROOT=/home/featurize/data/cxr_first WORK_ROOT=/home/featurize/work \
RUN_NAME=014_cloud_4090_manual BATCH_SIZE=16 NUM_WORKERS=8 \
ACCUMULATION_STEPS=1 VIEW_CHUNK_SIZE=8 VIEW_PAD_MULTIPLE=0 bash run_cloud.sh --foreground
```

去掉 `--foreground` 则使用 tmux 后台模式。脚本现在带用户执行权限，LF 换行；`bash run_cloud.sh` 原先无需执行权限。第一轮冻结骨干，需观察第二轮解冻后的显存峰值。

最初迁移交付记录：当时只改代码、做静态检查并生成小型代码包，新增训练、模型前向、反向和推理均为0。以下保留迁移与16 Study基线说明；当前性能配置以本文顶部为准。不要复制 Windows 的 `.venv`。

## 路径与数据边界

`train_highres.py` 与 `inference.py` 支持 `--image-root`（兼容 `--raw-root`、`--data-root` 别名）和 `--work-root`。`dataset_highres.py` 的 `raw_root` 接口接收该参数，不改 `dataset.py`。默认清单相对源码目录定位；物理原图根不进入预处理指纹。

只从 `train_clean.csv`、`valid_manifest.csv`、`test_manifest.csv` 的 `relative_path` 构建图像路径，逐次读取核对 `file_sha256`。训练只加载 train/valid 清单，test 仅供另行授权的推理使用；原始 CSV 不作为训练入口。上传全量原图不会让已剔除图片重新入选，不扫描图片补样本，不重写标签/划分/图片/缓存。

**上传完整原图可以保留既有清洗效果**，前提是同时使用代码包中的清洗清单、保持原图子目录结构及字节不变。无需再次清洗或上传旧300/512缓存。清洗不是删除磁盘脏图，而是用训练白名单排除：当前 train 保留30328图/18513 Study，valid 1422图/859 Study，test 2218图/1414 Study。若只有原图而遗漏三份清单，应补传本地 `data/clean_v1_reviewed/` 中这三个CSV，不能改用原始train.csv重建训练集。

建议目录：

```text
/home/featurize/
  data/cxr_first/                 # --image-root 指向这里
    cxr_first/                   # 本地外层目录整体打包时保留这一层
      train/ valid/ test/
      train.csv valid.csv test.csv   # 上传后仍不用于训练入口
  work/
    cloud_project/
      train_highres.py dataset_highres.py model_highres.py inference.py
      dataset.py model.py train.py runtime_compat.py
      requirements.txt run_cloud.sh
      data/clean_v1_reviewed/
        train_clean.csv valid_manifest.csv test_manifest.csv
    runs/009_cloud_4090/          # 用户启动后才创建
    logs/009_cloud_4090.log
    deliveries/                  # 后续推理输出
```

如果上传包内直接是 `cxr_first/train`，也使用相同 `--image-root`：加载器只检查 `root/train`、`root/cxr_first/train` 两个明确位置，优先直接结构，不遍历纳入额外图片。测试推理同理检查 `test`。

## 本地 PowerShell 打包

在项目目录执行，生成只含代码、说明及三份清洗 CSV 的 ZIP，显式白名单排除原图、缓存、环境、权重、runs、archive、artifacts。目标已存在则拒绝覆盖：

```powershell
Set-Location -LiteralPath 'E:\PycharmCodes\计挑人工智能赛道'
powershell -NoProfile -ExecutionPolicy Bypass -File .\pack_cloud.ps1
```

如果本次交付已提供 `cloud_code.zip`，直接上传即可；重新生成时使用新文件名：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\pack_cloud.ps1 -Destination '.\cloud_code_v2.zip'
```

64GB 全量原图另行打包。Windows 自带 `tar.exe` 可生成大文件 `tar.gz`，保留整个外层 `cxr_first`（包含内层目录和原始 CSV），不写入原图目录。不要与代码 ZIP 混在一起：

```powershell
tar.exe -czf '.\cxr_first.tar.gz' -C 'E:\PycharmCodes\计挑人工智能赛道' cxr_first
if ($LASTEXITCODE -ne 0) { throw '原图打包失败，不要上传不完整的归档' }
Get-FileHash -LiteralPath '.\cxr_first.tar.gz' -Algorithm SHA256
```

JPG 的 gzip 收益通常有限；请为压缩包预留约原数据大小的额外磁盘空间。无压缩的 `tar.exe -cf '.\cxr_first.tar' -C 'E:\PycharmCodes\计挑人工智能赛道' cxr_first` 也可用，云端对应 `tar -xf`。本次没有实际打包64GB原图。

## 云端解压与安装

假设两个归档已上传到 `/home/featurize/`，在空的新工程目录解压。原图用 `--keep-old-files` 防止意外覆盖已有文件：

```bash
mkdir -p /home/featurize/data /home/featurize/work/cloud_project
tar --keep-old-files -xzf /home/featurize/cxr_first.tar.gz -C /home/featurize/data
unzip -n /home/featurize/cloud_code.zip -d /home/featurize/work/cloud_project
cd /home/featurize/work/cloud_project
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

若当前平台未提供 `python3.12`，可改用 `conda create -n cxr-cloud python=3.12 -y`、`conda activate cxr-cloud`，再执行 pip 安装。脚本固定检查 Linux/Python 3.12。`tmux` 和 `unzip` 是系统工具；缺少时执行 `sudo apt-get update`、`sudo apt-get install -y tmux unzip`。

依赖固定为 PyTorch **2.11.0+cu126** / torchvision **0.26.0+cu126**，timm **1.0.30**、Albumentations **2.0.8**，其余版本见 requirements。CUDA轮子从官方 cu126 源获取；不要用无 CUDA 的 torch 覆盖。该 torch/vision/CUDA 组合来自 [PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/#v2110)，其他发布及 Python 版本约束已核对 PyPI 元数据。NVIDIA 运行库和 Triton 随所选 torch 精确构建解析，不复制本地 Windows 库。

用户可先运行 `nvidia-smi` 和下面只检查环境的命令；它们不训练模型。如果驱动不支持所选 CUDA 构建，请在 Featurize 切换合适镜像/驱动：

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

首次正式训练需要联网下载 timm 的预训练骨干。工作目录和输出位于 `/home/featurize/work`，请预留多份大模型权重的空间。

## tmux 后台启动

在已激活 Python 3.12 环境的工程目录执行这一行即可；脚本自己创建后台 tmux，会显式传递当前环境的 Python 绝对路径，避免复用 tmux 服务时拿错解释器：

```bash
IMAGE_ROOT=/home/featurize/data/cxr_first BATCH_SIZE=16 NUM_WORKERS=8 ACCUMULATION_STEPS=1 VIEW_CHUNK_SIZE=8 VIEW_PAD_MULTIPLE=0 bash run_cloud.sh
```

输出 `/home/featurize/work/runs/009_cloud_4090`，日志 `/home/featurize/work/logs/009_cloud_4090.log`。脚本拒绝复用已有 run、日志或同名会话，不自动续训。只看参数可先 `bash run_cloud.sh --dry-run`，不会运行 Python或创建 run。

本轮固定384、ConvNeXt V2-Base、AMP、激活检查点、CPU EMA。`batch-size=16` 是 **16个完整 Study**，每个 Study 的所有清单视角都保留；`accumulation-steps=1` 即有效批量16，与之前2×16=32不同。默认 `VIEW_CHUNK_SIZE=8` 表示GPU一次处理最多8张图，与Study批量独立，不丢视角。类感知批次含8组正负对并与均匀采样混合，整批Study不重复，采样预算和损失统计随batch同步调整。没有承诺未经实测的显存峰值或AUC。

```bash
tail -f /home/featurize/work/logs/009_cloud_4090.log
tmux ls
tmux attach -t 009_cloud_4090
# Ctrl+B 然后 D 可脱离会话，训练保持运行。
```

若出现 OOM，优先减少GPU分块，不必先改Study批量。由于失败run保留证据，重试必须换名：

```bash
RUN_NAME=010_cloud_4090_chunk2 VIEW_CHUNK_SIZE=2 VIEW_PAD_MULTIPLE=0 bash run_cloud.sh
# 如仍不足：batch8/accum2保留有效批量16，chunk1进一步降低显存。
RUN_NAME=011_cloud_4090_bs8 BATCH_SIZE=8 ACCUMULATION_STEPS=2 VIEW_CHUNK_SIZE=1 VIEW_PAD_MULTIPLE=0 bash run_cloud.sh
```

`NUM_WORKERS=4` 可降低CPU/RAM压力。单纯减少累积次数不能降低单个GPU分块的前向峰值。输入维持384，不在本次自动提高到512。后续手动实验要用新run，不能将失败轮声称为可精确续训。

## 后续推理入口（需用户另行授权运行）

训练完成后，schema3推理严格加载与验证、阈值同源的EMA `model_state_dict`，不加载raw备用权重，不在测试集调阈值。以下仅是命令示例，本次未执行：

```bash
python inference.py \
  --checkpoint /home/featurize/work/runs/009_cloud_4090/best.pt \
  --thresholds /home/featurize/work/runs/009_cloud_4090/thresholds.json \
  --test-manifest data/clean_v1_reviewed/test_manifest.csv \
  --image-root /home/featurize/data/cxr_first \
  --work-root /home/featurize/work \
  --output /home/featurize/work/deliveries/009_cloud_4090 \
  --num-workers 8
```

迁移没有改变checkpoint schema或预处理指纹，保留历史B3 schema1/2兼容。`runtime_compat.prepare_runtime()` 在Linux立即返回；只有Windows CPython3.14才启用已有本地WMI回避，旧入口保留为兼容别名。旧300缓存训练不包含在此次原图云端配置内。
