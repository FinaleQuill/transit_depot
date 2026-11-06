# cloud_code_v2：ConvNeXt-Tiny完整遍历基线

本次按用户授权完成代码修改和打包，不启动训练、续训、模型前后向或推理。下列命令供用户在云端手动执行。是否提分与实际显存占用尚未在新运行中验证。

## 当前入口和配方

`run_cloud.sh`现调用`train_tiny.py`，使用新`model_tiny.py`和`dataset_tiny.py`。旧Base/类别注意力/Rank代码保留用于历史权重兼容，新入口不调用其训练或损失。`dataset.py`、`dataset_highres.py`及全部原始/清洗CSV保持原样。

| 项目 | 默认值及作用 |
| --- | --- |
| 模型 | timm `convnext_tiny.fb_in1k`，ImageNet-1K预训练；标准GAP、预训练LayerNorm、Dropout、Linear10，无类别注意力。 |
| 输入 | 原图保比例LANCZOS补边384，ImageNet归一化；不读取旧300缓存，不写图片缓存。 |
| 训练样本 | 仅`train_clean.csv`的30328张图、18513个Study；不引入队友的重新划分。 |
| 批次 / 累积 | **32张图片** / 1，非32个完整Study。每轮948批，最后24张正常训练。 |
| 采样 | 普通shuffle，无放回，每轮完整覆盖所有图像，drop_last=False；没有类别感知或加权采样。 |
| 损失 | `BCEWithLogitsLoss(reduction='none')`，按Study逆视图数加权；无Rank、ASL、Focal或Dice。 |
| 类别权重 | 默认`pos_weight`关闭。可选权重仅由训练Study计数计算，默认上限3，绝不看valid/test分布。 |
| 训练长度 | 最多25轮；完整valid Macro-AUC连续5轮未达到有意义改善时早停，min_delta=0.0001；best仍记录实际最优AUC。 |
| 优化 | AdamW，骨干5e-5 / 线性头5e-4；WD0.01；全网从首轮训练，warmup1轮后余弦衰减；梯度裁剪1。 |
| 正则与增强 | Dropout0.1、DropPath0.05；在线亮度/对比与小角度旋转，不加噪声、不裁胸廓、不做Mixup/CutMix。 |
| 4090 | AMP优先BF16，不支持时训练阶段选择FP16；channels_last、cuDNN benchmark、12个持久worker、prefetch2、pin_memory。 |
| EMA | 保留GPU EMA，避免每个优化步复制到CPU；验证、阈值和best权重使用同一EMA。 |
| 激活检查点 | 默认关闭以减少重算；显式OOM备用开关保留。 |
| 验证 | batch64，覆盖全部valid视角，再按Study平均sigmoid概率；记录AUC、AP、F1@0.5及valid调优F1。 |

普通shuffle保证完整样本覆盖，训练端还校验每轮image_id集合与清单相同且无重复。损失权重为`N/(S*n_views)`，N是训练图数，S是训练Study数，n_views是该Study清洗后保留图数。整轮逐图加权BCE均值恰等于“每Study内部视图BCE均值，再对Study均值”，避免多图检查占更大权重。不会再按每个随机batch的权重和重新归一化。

默认每轮948次优化，比旧128 Study配置每轮145次更密。参数更新频率、覆盖与损失目标都改变，不能仅按轮数将新旧训练量视为等价，也不能保证这些变化必然提高验证指标。

模型名称和预训练来源已核对 [timm模型卡](https://huggingface.co/timm/convnext_tiny.fb_in1k)，损失接口见 [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)。包内requirements沿用已固定的Linux Python3.12/CUDA依赖，无新增第三方库。

## 云端解压和启动

上传本地`cloud_code_v2.zip`到`/home/featurize/`。原图继续使用`/home/featurize/data/cxr_first`，无需再次上传或挑拣。加载器兼容其内层还有一层`cxr_first`的情况，始终只按清单读取图像。

解压到全新代码目录，避免影响旧训练正在使用的文件；启动前确认4090上的旧训练已经结束。本次没有停止、连接或改动远程进程。

```bash
mkdir -p /home/featurize/work/cloud_project_v2
unzip -n /home/featurize/cloud_code_v2.zip -d /home/featurize/work/cloud_project_v2
cd /home/featurize/work/cloud_project_v2

/environment/miniconda3/envs/cxr-cloud/bin/python -m pip install -r requirements.txt

env -u OUTPUT_DIR -u LOG_FILE -u SESSION_NAME \
  PYTHON_BIN=/environment/miniconda3/envs/cxr-cloud/bin/python \
  IMAGE_ROOT=/home/featurize/data/cxr_first \
  WORK_ROOT=/home/featurize/work \
  RUN_NAME=015_cloud_tiny_b32_v2 \
  BATCH_SIZE=32 ACCUMULATION_STEPS=1 NUM_WORKERS=12 EPOCHS=25 \
  bash run_cloud.sh
```

使用现有`cxr-cloud` Python3.12环境，避免终端默认Python3.11。脚本以tmux后台运行，SSH断开后继续。新实验目录为`/home/featurize/work/runs/015_cloud_tiny_b32_v2`，日志为`/home/featurize/work/logs/015_cloud_tiny_b32_v2.log`。脚本拒绝覆盖已有run、日志或同名会话；失败时保留日志和退出码，不自动重试或续训。

```bash
tail -f /home/featurize/work/logs/015_cloud_tiny_b32_v2.log
tmux attach -t 015_cloud_tiny_b32_v2
# Ctrl+B 后按 D 脱离会话。
```

仅预览配置可运行`bash run_cloud.sh --dry-run`，不会运行Python、创建日志或启动训练。`tmux ls`中的会话存在不等于训练仍活着，应看图像batch进度、pane状态和`.exitcode`。

## 显存及可选参数

如果实际发生OOM，使用新实验名称降低训练/验证批次；累积2次保持有效32张图：

```bash
env -u OUTPUT_DIR -u LOG_FILE -u SESSION_NAME \
  PYTHON_BIN=/environment/miniconda3/envs/cxr-cloud/bin/python \
  RUN_NAME=016_cloud_tiny_b16_v2 \
  BATCH_SIZE=16 ACCUMULATION_STEPS=2 VALID_BATCH_SIZE=32 \
  bash run_cloud.sh
```

还可显式`GRADIENT_CHECKPOINTING=1`降低激活内存。若CPU/RAM压力大可用`NUM_WORKERS=8`、`PREFETCH_FACTOR=1`。不自动提高分辨率或batch，不把填满显存当作提分证据。

只有后续独立实验需要时才设置`POS_WEIGHT_POWER=0.5 POS_WEIGHT_CAP=3`；默认0保持普通BCE，按训练Study去重后计数。所有配置变化使用新run名，不在test上选择权重、阈值或超参数。

## 权重与推理

新权重采用schema4，保存Tiny模型配置、清单SHA、同源EMA、预处理、验证状态和验证AMP精度。`best.pt`、`thresholds.json`、`best_metrics.json`属于同次保存；`best_f1/`可另保留F1最佳，不能交叉混用阈值。保留`last.pt`用于状态追溯，不提供优化器/RNG精确续训。

`inference.py`已支持schema4并保留schema1/2/3历史模型。schema4推理使用训练验证时保存的BF16/FP16精度；不支持所保存BF16的CUDA设备会报错，CPU显式使用FP32并记录差异。推理恢复时`pretrained=False`，加载EMA，不额外下载骨干；测试端不拟合阈值。

后续经用户授权可手动执行：

```bash
python inference.py \
  --checkpoint /home/featurize/work/runs/015_cloud_tiny_b32_v2/best.pt \
  --thresholds /home/featurize/work/runs/015_cloud_tiny_b32_v2/thresholds.json \
  --test-manifest data/clean_v1_reviewed/test_manifest.csv \
  --image-root /home/featurize/data/cxr_first --work-root /home/featurize/work \
  --output /home/featurize/work/deliveries/015_cloud_tiny_b32_v2 \
  --batch-size 64 --num-workers 8
```

提交格式见[SUBMISSION_FORMAT.md](SUBMISSION_FORMAT.md)：每Study保留全部十类连续概率，不按阈值删行。当前导出三列完整长表，按历史官方答疑可省略Subject_id；四列用户样例也有效。本次不改变该格式约定。

## 代码包与回滚

`cloud_code_v2.zip`为显式白名单代码包，只含代码、文档、依赖和三份清洗CSV，不含原图、缓存、模型权重、环境或训练runs。`pack_cloud.ps1`默认生成这个名称，拒绝覆盖；需要再次打包时指定新的Destination。

同步前版本保存在`artifacts/code_snapshots/tiny_v2_20261018_214520/before`；新增Tiny模块为独立入口，旧Base模块仍原样留存。数据无需恢复。旧4090性能记录针对Base和旧采样，不能充当Tiny新配方的吞吐或显存验收。
