# ConvNeXt V2-Base 五折通宵方案

2026-10-19 在新机完成依赖修复、五折元数据检查和随机张量显存探针；只对随机初始化模型执行前后向，没有读取胸片参与模型计算、没有优化器更新、没有启动真实训练或测试集推理。只有用户手动运行下方命令后才会训练。旧 Tiny 和 A4000 实验记录保留。

## 实际环境与入口

检查到的 GPU 是 **RTX 4090，24564 MiB**，CPU 16 线程、RAM 54 GiB。`run_overnight.sh` 现显式使用微批 **8**、梯度累积 **4**、有效批量 **32**，验证/推理微批 8。统一 4 workers、prefetch=1、每个计算库线程数 1。BF16 AMP、channels_last 和激活检查点启用；原有 OOM 降微批及完整轮恢复逻辑保留。尚未实测完整训练轮耗时或通宵稳定性。

新独立环境：`/environment/miniconda3/envs/cxr4090/bin/python`，Python 3.11.8、torch 2.5.1+cu121、torchvision 0.20.1+cu121、timm 1.0.30、albumentations 2.0.8；`pip check` 与所有运行模块导入通过。原 base 环境不改，旧 `cxr-cloud` 路径在新机不存在。解释器使用实际可执行文件副本，避免调度器的 `Path.resolve()` 将 venv 符号链接解析回缺依赖的 base。当前精确依赖见 `artifacts/requirements_4090_probe.txt`；原 `requirements.txt` 为旧 Python 3.12 环境，不能代表本次实测环境。

原图：`/home/featurize/data/cxr_first`（代码会解析内层 `cxr_first`）。官方预训练 `model.safetensors` 已缓存，SHA256 为 `ec152f1e375edc2b3dfac7a81155a449b4c5cbb7c5cf0b9494838f6c87518d73`，通过 timm 在 CPU 离线读取全部 380 个权重张量。启动脚本默认 `HF_HUB_OFFLINE=1` 和 `NO_ALBUMENTATIONS_UPDATE=1`，防止逐折/逐 worker 网络检查；原图和标签均未修改。

## 4090 显存探针结果

`probe_gpu_memory.py` 对 `[batch,3,512,512]` 随机张量分别在独立进程中完成一次 forward、BCE 和 backward。模型为项目实际的 `convnextv2_base.fcmae_ft_in1k` 十类头，共 87,703,050 参数；使用随机权重、BF16 AMP、channels_last 和各 stage 激活检查点。未执行 `optimizer.step()`。

| batch | 纯前后向峰值 allocated / reserved | 加驻留状态后峰值 allocated / reserved | 结果 |
| --- | --- | --- | --- |
| 4 | 1.59 / 1.87 GiB | 2.57 / 2.80 GiB | 全部梯度有限，无 OOM |
| 8 | 2.52 / 3.27 GiB | 3.50 / 4.10 GiB | 全部梯度有限，无 OOM |

驻留模式额外保留 GPU EMA、两份 FP32 AdamW 动量缓冲和已有梯度，模拟后续梯度累积微批的固定显存；未执行优化器/EMA更新，更新临时开销与真实数据流水线不在测量范围内。allocated 是 PyTorch 张量峰值，reserved 是分配器保留峰值，均不含全部 CUDA 驱动开销。探针进程退出后，GPU 恢复 1 MiB、无计算进程。完整原始结果见 `artifacts/gpu_probe_4090.json`。

已连接服务器 SSH/Linux 终端时，手动执行这一行：

```bash
pkill -9 -f '[r]un_all_folds.py' 2>/dev/null || true; pkill -9 -f train_overnight.py 2>/dev/null || true; tmux new-session -d -s cxr_cv5_4090 -c /home/featurize/work/cloud_project_v2 'bash run_overnight.sh' \; set-option -w -t cxr_cv5_4090:0 remain-on-exit on
```

默认新实验名 `019_convnextv2_base512_cv5_4090`。旧 `018_convnextv2_base512_cv5` 的 A4000 启动记录将微批绑定为 2，因此不复用；旧 tmux `cxr_cv5_night` 已死亡，新会话使用 `cxr_cv5_4090`。防重命令先停止旧调度器，再清理训练子进程，避免调度器重启子进程；在交互终端顶层粘贴，勿再套进含完整命令文本的 `bash -c`，以免 `pkill -f` 匹配包装 shell。

每折每次完成 1 轮，依次推进五折，保存完整检查点后再切折；每折最多 20 轮、AUC 早停 patience=4，已完成/早停的折跳过。每次续训保留优化器、EMA、Scaler、随机状态、学习率进度，不会重新 warmup。任务失败的折保留日志与已有 best，其余折继续；最后对 1–5 个可用 best 进行集成，明确记录是否只有部分折。一个 run 只允许一个调度器运行。

默认不设总时限，直到各折完成或早停，之后自动推理。完整五折的总时长尚未实测，不保证一夜收敛或达到更高 AUC。若需要约 8 小时后转入推理，可在启动行的 `bash run_overnight.sh` 后加 `--hours 8`：到时等当前轮结束再推理，推理耗时额外计算。`--skip-inference` 仅保留权重和集成清单。

看日志：`tmux attach -t cxr_cv5_4090`；按 Ctrl+B 再按 D 脱离会话。完整日志在 `work/logs/019_convnextv2_base512_cv5_4090_*.log`，分折日志在 run 的 `logs/`。程序退出后会保留 tmux 窗口。断开 SSH 不影响 tmux。

同一代码/参数下重新执行脚本会自动续训；已有交付文件保留。若改变绑定代码、训练周期或数据，需要换 `RUN_NAME`，避免将不兼容实验混为同一 run。若要在当前轮结束后停止继续训练并使用现有 best 推理，可手动创建 `work/runs/019_convnextv2_base512_cv5_4090/STOP`；恢复训练前移走该文件。

## 模型和防过拟合

- `model_overnight.py`：`convnextv2_base.fcmae_ft_in1k`，标准 GAP/LayerNorm/Dropout/Linear 十类头。使用 [timm 官方模型卡所列的 FCMAE/ImageNet-1k 预训练](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in1k)。Tiny 与 Base 结构不同，这个新实验从 Base 预训练初始化，不能直接加载旧 Tiny `best.pt`。
- `train_overnight.py`：训练分辨率按 512/448/512/384/512 逐轮循环；折内验证固定 512。等比例缩放并补黑边，保留完整胸片；温和翻转、亮度/对比度与 ±5° 旋转。
- 学习率 backbone=2e-5、head=2e-4，1 轮 warmup 后余弦衰减；AdamW weight decay=0.05、Dropout=0.2、DropPath=0.15、EMA=0.999、梯度裁剪=1。
- 保留按 Study 等权的 BCE，每个 Study 的多张图像不增加其总权重。没有自动启用 `pos_weight`，也没有加重 Mixup/CutMix；类别权重并不保证提升 AUC，强混合会额外改变本任务的小病灶与多视图目标。通过预训练、温和正则化、患者隔离验证、早停和概率平均控制方差，实际增益须由独立验证结果确认。

## 数据隔离与五折

`cv_splits.py` 仅基于原 `train_clean.csv` 分折；同患者及文件/像素精确重复的连通组不得跨折，按十类多标签分布平衡。`data/cv5_overnight/folds.json` 固定源清单哈希和每个 Study 的 fold，训练记录保留原始 split/path/labels。

30328 图、18513 Study、11055 患者全部覆盖一次 OOF；每折 2211 患者、3702–3703 Study，每类均有正负样本。验证折仅选该折的 best；原始官方 valid（859 Study）完全不进入五折训练，最终评估统一集成。test 仅用于最终推理，不能选轮数、阈值或模型权重。元数据校验见 `data/cv5_overnight/metadata_validation_report.json`。

## TTA、集成与输出

`inference.py --ensemble-plan ...` 进入 `inference_overnight.py`，旧单模型接口继续可用。逐个加载 EMA 模型，每个模型对原图执行 384/512 × 原图/水平翻转，共 4 种 TTA；先 sigmoid 概率，再多视图 Study 平均、TTA 平均和 fold 平均。

默认五折等权软投票；集成清单允许显式非负权重，但本方案不在 test 上调权，也不按噪声较大的各折 AUC 自动加权。先用完全相同的集成评估官方 valid 并拟合辅助 F1 阈值，再在 test 上输出连续概率。阈值不删提交行。

主要文件：

- `work/runs/019_convnextv2_base512_cv5_4090/fold_0/` 至 `fold_4/`：`best.pt`、完整续训 `last.pt`、`best_oof.npz`、指标和训练状态。
- `work/runs/019_convnextv2_base512_cv5_4090/ensemble_plan.json`：冻结 fold、权重、checkpoint SHA 和 TTA。
- `work/deliveries/019_convnextv2_base512_cv5_4090/submission_final.csv`：首次成功交付的位置。若该目录已有文件，改用带时间戳的新目录；实际路径写入 run 的 `latest_submission.json` 并显示在最终日志。
- 交付目录还包含 `valid_metrics.json`、`thresholds.json`、逐 Study 概率、`validation_report.json` 和 `provenance.json`。

提交严格复用 `/home/featurize/work/submission.csv` 的四列表头 `Subject_id,Study_id,Predict_class,Probability` 和身份行顺序：14140 行、1414 Study、每个 Study 十类，无遗漏/重复、概率有限且落在 [0,1]。不会覆盖样例或旧提交。

## 只检查不训练

```bash
/environment/miniconda3/envs/cxr4090/bin/python -m py_compile model_overnight.py dataset_overnight.py cv_splits.py train_overnight.py inference_overnight.py inference.py run_all_folds.py
bash run_overnight.sh --dry-run
bash -n run_overnight.sh
```

`--dry-run` 校验清单/样例并只打印将执行的命令，不创建模型、不启动训练或推理子进程，也不写 run 目录。纯逻辑测试通过不能替代真实 GPU 运行验证。
