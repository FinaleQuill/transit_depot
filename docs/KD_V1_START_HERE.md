# KD V1 从环境修复到训练、导出的操作顺序

**当前计划已变更：fold0原20轮、其余四折12轮，自动交接和新的OOF入口见[当前操作手册](KD_V1_MIXED_USAGE.md)。下面五折全20轮及021软标签路径是旧方案记录。**

本手册接续 2026-10-25 的蒸馏方案。教师为 ConvNeXt V1 Base768 五折，学生为单个 V1 Tiny512；损失为 α=0.2、T=2 的 Bernoulli KL+BCE，部署为 FP16、无 TTA、无集成。

2026-10-25已按官方补充备查通知增加完整日志、逐轮checkpoint时间戳/哈希和源码归档。当前已运行的fold0由后台观察器补记，后续教师及学生CLI自动留存；无需为此重启训练。官网/PDF/补充通知的核对、历史恢复边界及留存位置见[材料修补说明](KD_V1_COMPLIANCE.md)。

本次报错发生在 Hugging Face 下载前的网络连接阶段：环境使用 SOCKS 代理，而 HTTPX 缺少 `socksio`。独立环境已补装 `socksio==1.0.0`，并已写入 `requirements_kd.txt`；Base/Tiny 的 Hugging Face 元数据、完整权重及许可材料下载已实际通过，权重已缓存。你现在不需要再安装一次。

安装、导入检查和预检不会开始训练。以下需要训练的步骤均由你手动输入 `--execute` 启动；教师完成后不会自动继续提取 OOF 或训练学生。每一阶段结束并满足成功条件后，再输入下一阶段命令。

## 1. 现在输入：进入持续运行的终端

在当前服务器终端输入：

```bash
tmux new -s cxr-kd-v1
```

进入 tmux 后输入：

```bash
cd /home/featurize/work/cloud_project_v2
export KD_PYTHON=/home/featurize/.conda/envs/cxr-kd-v1/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
"$KD_PYTHON" -m pip check
"$KD_PYTHON" preflight_kd.py --check-files
```

成功条件：`pip check` 显示 `No broken requirements found`，预检成功退出并报告训练图 30328 张、验证图 1422 张、五折患者无交叉。预检不会解码图片或运行网络。

本机所有步骤使用上面的独立 Python。新开普通终端后，要重新输入 `cd` 和 `export KD_PYTHON=...`；重新连接同一个 tmux 会话时，原终端变量仍然保留。

如果提示 tmux 会话已经存在，用下面的命令连接，避免重复开训练：

```bash
tmux attach -t cxr-kd-v1
```

训练运行期间，按 `Ctrl+B`，松开后再按 `D`，可以离开会话，训练会继续。不要在运行训练的终端直接输入下一步命令。要查看显卡状态，可另开终端输入：

```bash
nvidia-smi
```

## 2. 预检成功后输入：启动五折教师

可先预览，不开始训练：

```bash
bash run_overnight.sh
```

真正开始训练：

```bash
bash run_overnight.sh --execute
```

该入口固定使用独立环境。会顺序训练 `fold_0` 至 `fold_4`，每折固定 20 轮，微批量 2、梯度累积 16。本次已缓存教师和学生的完整预训练权重，启动时仍会校验来源并可能请求网络；新机器未缓存时，会先下载和登记，再出现训练批次。准备期间暂时没有 epoch 输出不代表正在训练。

正常训练输出包含 `teacher fold=0 epoch=1/20` 等进度；每轮完成会保存 `last.pt`。只有每折完成全部 20 轮才生成可用于 OOF 的最终 EMA `best.pt`。

如果刚才因缺依赖退出，修复后直接重新输入同一条启动命令。当前失败的空 `fold_0` 不需要删除。调度器会对已有 `last.pt` 自动恢复；已完成的折不会重新训练。不允许同时启动两个调度器。

**何时输入下一步：** 教师命令返回提示符后，在同一个终端输入：

```bash
cat ../runs/021_convnext_v1_base768_teacher_cv5/runner_status.json
```

仅当 `all_requested_complete` 为 `true`、五折 `training_complete` 均为 `true`，并且五个 `fold_N/best.pt` 都已生成时，才能进入第 3 步。命令退出码为 0 也可能只是正常暂停，不能单凭“命令结束”判断五折已完成。

## 3. 五折全部完成后输入：提取严格折外软标签

```bash
"$KD_PYTHON" -u extract_soft_labels.py --execute \
  --fold-plan data/cv5_overnight/folds.json \
  --teachers-root ../runs/021_convnext_v1_base768_teacher_cv5 \
  --image-root /home/featurize/data/cxr_first \
  --output ../runs/021_convnext_v1_base768_teacher_cv5/oof_logits
```

成功条件：先看到各个 `OOF fold=N completed`，最后看到 `Strict OOF cache complete:`，命令成功退出。输出包含 `soft_labels.npz` 和 `metadata.json`，覆盖全部训练图，每张只由未训练过该患者的那一折教师预测。

提取需要真实执行教师前向，但不更新权重。输出目录必须是新目录，已有缓存会被拒绝覆盖。失败时先保留报错和输出现状，不要盲目删除既有结果。

## 4. OOF 完成后输入：训练学生

```bash
"$KD_PYTHON" -u train_distill.py --execute \
  --fold-plan data/cv5_overnight/folds.json \
  --soft-labels ../runs/021_convnext_v1_base768_teacher_cv5/oof_logits \
  --image-root /home/featurize/data/cxr_first \
  --output ../runs/022_convnext_v1_tiny512_kd
```

默认微批量 8、梯度累积 4，最多 25 轮；官方 valid 用于早停，patience=5。学生完成后查看：

```bash
cat ../runs/022_convnext_v1_tiny512_kd/status.json
```

成功条件：状态为 `completed` 或 `early_stopped`，且 `best.pt` 已生成。`paused` 表示需要继续恢复，`failed` 表示先排查报错。保留 `best.pt` 和 `last.pt`，恢复时两者都有用途。

## 5. 学生完成后输入：导出 FP16 模型并验证

```bash
"$KD_PYTHON" export_student.py \
  --checkpoint ../runs/022_convnext_v1_tiny512_kd/best.pt \
  --output ../deliveries/kd_v1_tiny512/model
```

成功条件：导出报告中 `size_pass` 为 `true`；部署目录有 `student_fp16.pt`、`config.json` 和 `export_report.json`。实际体积按权重及配置计，不使用训练检查点的文件大小代替。

导出成功后再输入：

```bash
"$KD_PYTHON" validate_student.py \
  --model-dir ../deliveries/kd_v1_tiny512/model \
  --image-root /home/featurize/data/cxr_first \
  --manifest data/clean_v1_reviewed/valid_manifest.csv \
  --output-dir ../deliveries/kd_v1_tiny512/fp16_validation
```

成功条件：`fp16_validation/validation_report.json` 中 `complete` 为 `true`，验证图数 1422、Study 数 859，并输出真实 FP16 的 Macro-AUC。训练时的验证分数不能代替这一步。

导出、验证、计时和打包都要求新的输出目录。已经成功执行过的步骤不需要重复执行；需要重做时，应使用新的版本目录并同步后续路径。

## 6. 可选：先在当前 4090 生成一份预测

只有第 5 步完成后才输入。这份文件独立保存在 `prediction_4090`，便于以后保留 T4 最终结果：

```bash
"$KD_PYTHON" predict.py \
  --model-dir ../deliveries/kd_v1_tiny512/model \
  --image-root /home/featurize/data/cxr_first \
  --manifest data/clean_v1_reviewed/test_manifest.csv \
  --output ../deliveries/kd_v1_tiny512/prediction_4090/submission.csv
```

成功条件：报告显示 `studies=1414`、`images=2218`、`data_rows=14140`。文件固定命名为 `submission.csv`，保留全部十类连续概率。

这一步不证明 T4 延迟达标。正式配套的 CSV 与计时建议按下一步在同一 T4 环境生成，避免不同 GPU 的浮点差异影响预测一致性检查。

## 7. 到了匹配的 T4 环境后再输入：最终预测和计时

现在不需要执行这一节。先准备符合要求的 T4 环境，把推理代码、`requirements_inference.txt`、`data/clean_v1_reviewed/test_manifest.csv`、导出的 `model` 目录以及官方图片传到 T4。不要把当前服务器的 Conda 环境目录直接复制过去，也不要把训练用的 CUDA wheels 安装到主办方已有容器里。

以下示例假设 T4 上仍采用 `cloud_project_v2/` 与 `deliveries/` 并列的目录结构。先在 T4 终端进入实际上传的 `cloud_project_v2` 目录；本节 `python` 指 T4 上已配置好的 Python，不是训练机的绝对路径。

先检查容器运行时：

```bash
python -c 'import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); assert sys.version_info[:2] == (3, 12); assert torch.__version__.split("+")[0] == "2.10.0"; assert torch.version.cuda == "12.8"; assert "T4" in torch.cuda.get_device_name(0)'
```

检查通过后，对齐推理依赖并检查一致性：

```bash
python -m pip install -r requirements_inference.txt
python -m pip check
```

如果运行时检查失败，应先换用/准备匹配容器，不要将训练环境直接覆盖到官方容器。`cost.py` 还会核对驱动及 cuDNN 等环境信息。

准备路径变量。下面三处是**需要替换的占位路径**：`T4_IMAGE_ROOT` 是完整测试原图的根目录；`PICTURES_100` 和 `PICTURES_5` 是主办方提供的恰好 100 张计时图和 5 张预热图，保留其目录结构。它们不是本项目自动生成的文件夹。

```bash
export T4_IMAGE_ROOT=/替换为T4上的完整cxr_first路径
export PICTURES_100=/替换为主办方100张图目录
export PICTURES_5=/替换为主办方5张预热图目录
```

路径准备好后，先在 T4 生成最终提交 CSV：

```bash
python predict.py \
  --model-dir ../deliveries/kd_v1_tiny512/model \
  --image-root "$T4_IMAGE_ROOT" \
  --manifest data/clean_v1_reviewed/test_manifest.csv \
  --output ../deliveries/kd_v1_tiny512/submission.csv
```

预测成功后再计时：

```bash
python cost.py ../deliveries/kd_v1_tiny512/submission.csv "$PICTURES_100" "$PICTURES_5" \
  --model-dir ../deliveries/kd_v1_tiny512/model \
  --manifest data/clean_v1_reviewed/test_manifest.csv \
  --output-dir ../deliveries/kd_v1_tiny512/t4_cost
```

成功条件：`t4_cost/cost_report.json` 中 `official_compliance_pass` 为 `true`，且命令成功退出。脚本按 100 图、batch=1 串行、5 图预热、连续 3 轮计时，包含读图、预处理、模型、聚合和写出 CSV。

如果环境、体积、延迟或 CSV 一致性未满足，脚本会保存报告并以退出码 2 结束。若 100 图不包含相应 Study 的全部视角，一致性也可能标记待确认，应依据主办方实际清单确认口径。不能将这种结果当作已通过；最终仍由主办方复测。

## 8. T4 检查结束后：回训练机打包材料

把 T4 生成的最终 `submission.csv`、`submission.image_predictions.csv`、`submission.csv.report.json` 以及整个 `t4_cost/` 目录传回当前训练机的 `/home/featurize/work/deliveries/kd_v1_tiny512/`。保留 T4 文件原字节，不改写报告或 CSV。传输方式取决于你的 T4 平台和连接地址。

回到原训练机终端，输入：

```bash
cd /home/featurize/work/cloud_project_v2
export KD_PYTHON=/home/featurize/.conda/envs/cxr-kd-v1/bin/python
"$KD_PYTHON" training_evidence_monitor.py refresh \
  --run-dir ../runs/021_convnext_v1_base768_teacher_cv5
"$KD_PYTHON" package_kd_submission.py \
  --model-dir ../deliveries/kd_v1_tiny512/model \
  --output ../deliveries/kd_v1_tiny512/verification_package \
  --submission ../deliveries/kd_v1_tiny512/submission.csv \
  --validation-report ../deliveries/kd_v1_tiny512/fp16_validation/validation_report.json \
  --cost-report ../deliveries/kd_v1_tiny512/t4_cost/cost_report.json \
  --training-evidence ../runs/021_convnext_v1_base768_teacher_cv5/evidence
```

在原训练机打包，是因为脚本需要原工程、固定清单、折计划，以及教师/学生来源记录指向的原始 ImageNet 权重缓存。只把 `model/` 复制到 T4 后直接打包，会找不到这些来源材料；脚本不会自动补下载。

若原缓存迁移了，可以使用 `--pretrain-cache /实际缓存根目录` 指定。支持 Hugging Face `models--timm--.../snapshots/<revision>/model.safetensors` 等布局，并按记录的 SHA256 校验。不要清理原训练机 Hugging Face 缓存，直到核验材料完整生成。

成功条件：`verification_package/package_status.json` 中 `ready_for_organizer_review` 为 `true`。缺少 FP16 验证、最终 CSV 或 T4 证据时会显示 `pending`；打包完成不等于已经向主办方提交。

## 需要暂停时输入：轮末保存与恢复

暂停教师时，在**另一个终端**输入：

```bash
touch /home/featurize/work/runs/021_convnext_v1_base768_teacher_cv5/STOP
```

等待正在运行的整轮完成并保存后，教师会暂停，不再启动下一折。确认原调度器已经退出后，恢复：

```bash
cd /home/featurize/work/cloud_project_v2
rm -f ../runs/021_convnext_v1_base768_teacher_cv5/STOP
bash run_overnight.sh --execute
```

暂停学生时，在另一个终端输入：

```bash
touch /home/featurize/work/runs/022_convnext_v1_tiny512_kd/STOP
```

等待本轮及验证完成并保存、原进程退出后，恢复：

```bash
cd /home/featurize/work/cloud_project_v2
export KD_PYTHON=/home/featurize/.conda/envs/cxr-kd-v1/bin/python
rm -f ../runs/022_convnext_v1_tiny512_kd/STOP
"$KD_PYTHON" -u train_distill.py --execute \
  --fold-plan data/cv5_overnight/folds.json \
  --soft-labels ../runs/021_convnext_v1_base768_teacher_cv5/oof_logits \
  --image-root /home/featurize/data/cxr_first \
  --output ../runs/022_convnext_v1_tiny512_kd \
  --resume ../runs/022_convnext_v1_tiny512_kd/last.pt
```

恢复沿用原目录和配置，不修改固定轮数、损失或缓存。以上按已有完整轮断点恢复；意外中断时，尚未保存的当前轮需要重跑。若学生首次保存前报错、没有任何完整检查点，先检查原因，不要套用已有断点恢复命令。

## 仅将来重建或补装环境时使用

当前依赖已修复，以下不是现在必须再执行的步骤。只在同一环境再次缺少 `socksio` 时补装：

```bash
export KD_PYTHON=/home/featurize/.conda/envs/cxr-kd-v1/bin/python
"$KD_PYTHON" -m pip install socksio==1.0.0
"$KD_PYTHON" -m pip check
```

如果是在新机器重建完整训练环境，先准备 Conda，然后输入：

```bash
cd /home/featurize/work/cloud_project_v2
conda create --prefix /home/featurize/.conda/envs/cxr-kd-v1 --override-channels --channel conda-forge python=3.12 pip -y
export KD_PYTHON=/home/featurize/.conda/envs/cxr-kd-v1/bin/python
"$KD_PYTHON" -m pip install --no-cache-dir -r requirements_kd.txt
"$KD_PYTHON" -m pip check
"$KD_PYTHON" preflight_kd.py --check-files
```

详细实现协议见 [KD V1 操作说明](KD_V1_USAGE.md)。
