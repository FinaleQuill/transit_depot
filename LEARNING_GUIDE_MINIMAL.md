# 按文件读懂这条训练流水线

这份精简学习包只讲：**原图 → 768×768 → ConvNeXt V2-Base → 纯加权 BCE → 5 折 → Study 概率集成**。
主线需要的训练、恢复、验证、推理代码都保留；历史模型、历史数据集实现不放进来。
中文注释写在独立学习副本中。本次整理没有运行训练、推理或反向传播，也没有操作正在跑的任务。

建议先读本页，再按下面的顺序打开文件。第一遍可以跳过哈希检查、断点协调等工程细节。
文中的 batch、worker 数是**脚本默认值**，不能据此判断服务器此刻训练进程的实际参数。

## 先认清文件之间的关系

```text
run_overnight.sh                     配好默认参数，启动总调度
  └─ run_all_folds.py                安排五个折，逐折轮流训练一轮
       ├─ cv_splits.py               决定这一折哪些记录训练、哪些记录验证
       │    └─ dataset.py            读取 CSV、核对记录、提供等比缩放工具
       ├─ train_overnight.py         单折训练、EMA、验证、保存与恢复
       │    ├─ dataset_overnight.py  读原图 → 增强 → 张量 → 批次
       │    └─ model_overnight.py    创建 ConvNeXt V2-Base 和十输出分类头
       └─ inference_overnight.py     最佳模型集成、官方 valid 阈值、test 提交
            └─ metrics_overnight.py  计算指标、搜索十类阈值
runtime_compat.py                    被模型/训练入口调用的运行时兼容工具
```

**三个词先对齐：**Image 是一张图；Study 是一次检查，可能有多张视角图；Patient 是患者，可能有多次检查。
CSV 通常是一行一图，同一次 Study 的几张图共享十维标签。十个标签位置始终按类别编号 1～10 排列。
“内部验证折”来自 `train_clean.csv`；“官方 valid”是另一个独立集合，两者不能混着理解。

## 1. `dataset.py`：给主线提供清单和原图处理基础工具

一句话作用：**把 CSV 读成可信的图片记录，并提供原图定位、等比缩放和常量。**
这份文件已收窄到当前主线所需工具；真正的 PyTorch Dataset 在下一个文件中。

重点看三个函数：

- `read_manifest(...)`：打开指定 split 的清单 → 检查 ID、路径、标签等字段 → 返回记录字典列表。
  每个字典带着 `image_id`、`subject_id`、`study_key`、`relative_path`、`targets` 等信息；标签不是文件名猜出来的。
- `resolve_native_image_root(...)`：检查传入路径 → 找到原始图片目录。主线从原图读取，不先读取低分辨率缓存再放大。
- `letterbox_rgb(image, size)`：按长边缩到目标尺寸 → 新建黑色正方形画布 → 把缩放图居中贴上去。

例如原图宽 1200、高 900：缩成宽 768、高 576，再上下各补 96 像素，得到 768×768。
这样胸片的长宽比例保持不变；补边不是把短边拉长。`IMAGENET_MEAN/STD` 等常量给后续归一化使用。

## 2. `cv_splits.py`：决定每一折能看到哪些记录

一句话作用：**从干净训练集内部做五折，并把同患者和精确重复内容隔离在同一折。**

重点看三个函数：

- `build_fold_plan(...)`：读 train 和官方 valid 清单 → 按 Study 整理标签 → 把同患者、跨患者精确重复图连成组 → 尽量平衡五折的各类正负 Study 数。
  官方 valid 只参与隔离检查，不用来拟合折内类别分布；test 不进入这里。
- `ensure_fold_plan(...)`：计划不存在时创建；已存在时核对数据、种子、折数。不会悄悄覆盖一份已有计划。
- `load_fold_records(plan, fold, role)`：核对计划 → 重读原始 train 记录 → 按 `study_to_fold` 挑出这一折的训练或验证记录。

```text
假设当前 fold=0：
  role="valid" → Study 映射为 0 的图片记录
  role="train" → Study 映射为 1、2、3、4 的图片记录
```

这里返回的是**记录列表，不是图像张量，也不是已经打开的图片**。
内部验证记录的原始 `split` 仍是 `train`；图片路径不用搬到 `valid/`，只是用途变了。
五折训练出五个独立模型。每个 Study 在一个模型中当验证数据，在其余四个模型中当训练数据。

## 3. `dataset_overnight.py`：把一条记录变成模型能吃的张量

一句话作用：**只读原始图片，做 768 预处理，按批送出张量，同时保留 Study 身份。**

重点看三处：

- `OvernightImageDataset.__getitem__(index)`：拿第 `index` 条记录 → 读图片字节并校验 SHA256 → 按 EXIF 摆正 → 转 RGB → 等比缩放补边 → 增强/归一化 → 返回一张图和标签。
  一张图输出 `image[3,768,768]`、`targets[10]`，还带 `image_id`、`study_key`、`loss_weight`。
- `make_overnight_loader(...)`：创建 Dataset → 配置 CPU worker、随机种子和预取 → 创建 DataLoader。
  它指定训练时打乱顺序、保留尾批，并把 `collate_overnight` 交给 DataLoader 使用。
- `aggregate_study_predictions(...)`：先核对每张图恰好预测一次 → 按 `study_key` 收集 → 对同 Study 的十维概率逐类求平均。

`__getitem__` 中的 `_transforms(training=True, ...)` 按以下顺序处理：

```text
已完成 letterbox 的 RGB 图
→ 50% 概率水平翻转
→ 50% 概率小幅调整亮度/对比度
→ 30% 概率旋转，角度在 -5°～5°
→ Normalize：每通道做 (像素/255 - mean) / std
→ ToTensorV2：高×宽×通道 → 通道×高×宽
```

这些增强独立随机抽样，不是每张图必做三次。标签没有左右侧别，才允许水平翻转。
验证时 `training=False`，只做确定的读取、letterbox、Normalize 和张量转换，不做上述随机增强。
灰度胸片转 RGB 会把灰度复制到三个通道，目的是适配预训练模型的输入接口。

还有一处短函数值得顺手看：`collate_overnight(samples)` 只是堆叠，不改标签。
两张 `[3,768,768]` 变成 `images[2,3,768,768]`，两份标签变成 `targets[2,10]`，ID 保留为列表。
Dataset 初始化时还计算逐图 `loss_weight`：同 Study 视角越多，每张的权重越小，使整个 Study 的总贡献相同。
这个视角权重与后面的类别 `pos_weight` 是两个不同维度的权重。

## 4. `model_overnight.py`：创建从图片到十个分数的网络

一句话作用：**调用 timm 创建 ConvNeXt V2-Base，并让最后一层输出十个独立类别的 logit。**

重点看两个函数：

- `validate_model_config(...)`：检查骨干名称、dropout、drop path 和梯度检查点开关，拒绝拼错的配置。
- `build_overnight_model(pretrained=..., ...)`：创建预训练骨干和十输出分类头 → 开启激活检查点 → 返回网络对象。
  从项目断点恢复或推理时用 `pretrained=False`，之后加载项目自己的参数。

你学过的 ANN 常把输入看作一根向量。这里先保留图片的空间网格，用卷积在不同位置复用同一套检测规则。
可以先把它理解成：“找局部线条/纹理 → 组合成更大的影像特征 → 每类给一个分数”。

```text
[B,3,768,768] → [B,128,192,192] → [B,256,96,96]
              → [B,512,48,48] → [B,1024,24,24]
              → 全局平均池化 → [B,1024] → 分类头 → [B,10]
```

卷积块本体在第三方 `timm` 库中；本地文件负责选择结构和配置。最后的 Linear 层可以接回你熟悉的 ANN 思路。
输出 `[B,10]` 中的十个数先是可正可负的 **logit**；验证时逐个 sigmoid 后才是概率，十个概率不用加起来等于 1。
激活检查点在反向时重算部分中间结果来省显存，与磁盘上保存 `best.pt` 的“断点”不是同一件事。

## 5. `train_overnight.py`：单折训练的核心

一句话作用：**训练一个折，更新在线模型和 EMA，按内部验证 AUC 保存最佳结果，并保留完整恢复状态。**

先抓住三组核心函数：

- `compute_loss_config(...)` / `build_weighted_bce(...)`：只统计当前折训练子集，Study 去重 → 每类算“负 Study 数 / 正 Study 数” → 建立带十维 `pos_weight` 的 BCE。
- `train_epoch(...)`：取一批 → 搬到 GPU → AMP 前向 → FP32 加权 BCE → 反向累积 → 到窗口边界才更新参数。
- `run_training(...)`：核对配置/折记录 → 建模型、优化器、EMA → 恢复完整状态（如指定）→ 逐轮训练和 `evaluate` → 保存 best/last 与状态。

**BCE 在这一份代码里怎么接？**模型直接输出 logits，传给 `BCEWithLogitsLoss` 时不要先 sigmoid。
每张图有十个独立的是/否问题，所以不使用要求类别互斥的 Softmax；一张胸片可以同时有多个阳性标签。
若某类 `pos_weight=20`，真实标签为 1 时，该类 BCE 正项和对应梯度按 20 倍计算；真实标签为 0 时不乘这个 20。
代码没有额外叠加 Rank Loss 或 Focal Loss；`pos_weight` 就是 BCE 内部的正项权重。

```text
criterion(logits.float(), targets)    # [B,10]，暂不自动归约
→ 每张图平均十类                    # [B]
→ 乘该图的 Study 视角权重
→ 求和 / 当前累积窗口的实际图片数
→ backward()，把梯度加到已有 .grad
```

默认微批 2、累积 16：每次只让两张图通过网络，16 次反向后才执行一次优化器更新，完整窗口相当于 32 张图的加权梯度。
这 32 张是图片数，不保证是 32 个不同 Study；尾窗口不足 32 张时按实际张数归一化并保留更新。
`backward()` 计算梯度，不会自己修改参数；`optimizer.step()` 才根据梯度修改卷积核、分类头等参数。

EMA 在 `run_training` 内注册的 `after_step` 中更新：只有真实优化器更新成功，才执行 `ema.update(...)`。
可以理解成另外保存一份“最近多次参数的平滑版本”。目标 decay 默认 0.999，前期 warmup 会调整实际衰减。
在线模型继续参加反向；EMA 不参加反向，供内部验证、最佳模型保存和最终推理使用。

读第二遍时再看这些支撑函数：`build_optimizer` 区分骨干/分类头学习率；`set_learning_rates` 按成功更新次数调学习率。
`evaluate` 用 EMA 做每图 sigmoid，再调用 Study 均值聚合；`calculate_metrics` 算 Study AUC/AP 和诊断 BCE。
每轮以内部 Study Macro-AUC 严格提高作为刷新 best 的条件，这里不搜索最优 F1 阈值。
`save_checkpoint`、`capture_rng/restore_rng`、`reconcile_resume_best` 等负责保存和协调模型、EMA、AdamW、scaler、随机状态与进度。
这些恢复函数保留在包内，因为它们属于真实训练主线；第一遍读 CNN 可以先略过细枝末节。

## 6. `run_all_folds.py`：给五折安排轮班

一句话作用：**同时只训练一个折，每次一轮，轮流接续五折，最后汇总可用的最佳模型。**

重点看三个函数：

- `training_command(...)`：组装单折命令 → 计算 `accumulation=effective_batch/batch` → 优先恢复 `last.pt`，必要时从完整 `best.pt` 恢复。
- `orchestrate(...)`：读取各折状态 → 选完成轮数最少的折 → 调用训练子进程 → 更新状态 → 重复。
  遇到可识别的 OOM 会降低微批并提高累积，尽量保持有效 batch；不会把失败轮的半截梯度接着用。
- `build_ensemble_plan(...)`：收集各折内部 AUC 最佳 EMA 的 `best.pt` → 固定默认等权和 TTA → 写集成计划。

五折都可用时，每折权重为 1/5；若只存在部分折，代码会标记部分集成，不能把它说成已完成五折。
`--skip-inference` 时停在权重和集成计划准备好这一步，官方 valid 阈值搜索和 test 推理不会发生。

## 7. `inference_overnight.py`：把最佳模型变成最终概率

一句话作用：**按固定计划做 Study 概率集成，在官方 valid 找阈值，再输出 test 预测和提交表。**

重点看三个函数：

- `_predict_model(...)`：按原图 768 和水平翻转两种变体前向 → 每图 sigmoid → 同 Study 多视角平均 → TTA 平均。
- `_ensemble_phase(...)`：一次加载一个折的最佳 EMA → 调 `_predict_model` → 按固定折权重平均 Study 概率。
- `run_ensemble_inference(...)`：检查固定计划/数据 → 官方 valid 集成 → 搜十类阈值 → 相同协议预测 test → 按提交模板匹配输出。

平均顺序是：**sigmoid → 原始视角均值 → TTA 均值 → 折加权均值**，不是先平均 logit 再 sigmoid。
内部选 best 用 identity 768 验证；最终集成用计划指定的 768 加水平翻转，不要混淆这两套评价阶段。
阈值用于二值诊断表，正式概率提交保留每个 Study 的十类连续概率，不按阈值删除类别行。

## 8. `metrics_overnight.py`：算分数与找判断线

一句话作用：**接收现成的 Study 标签/概率，计算指标并为每类找 F1 最好的阈值。**

- `search_class_thresholds(...)`：每类遍历概率断点 → 比较 F1 → 选最优阈值；并列时取较高阈值。
- `calculate_metrics(...)`：用连续概率算 AUC/AP → 用各类阈值转 0/1 → 算 F1、精确率、召回率、TP/FP/FN。

这里保留的是当前推理实际需要的指标函数，不包含旧训练器。
阈值只在最终官方 valid 上拟合；不能利用 test 标签找阈值。AUC 比较排序，本身不需要二值阈值。

## 9. `runtime_compat.py` 与 `run_overnight.sh`：工程入口

`runtime_compat.py` 一句话作用：**处理特定 Windows Python 的启动兼容，Linux 上直接返回。**
`prepare_runtime()` 先判断平台/版本，再决定是否调整本进程 WMI 查询；`prepare_windows_runtime()` 是同功能兼容入口。
这里不包含网络层、优化器或训练逻辑，第一次阅读可以跳过。

`run_overnight.sh` 一句话作用：**设置 Python、数据路径、线程数和默认参数，再调用五折调度器。**
阅读时按顺序看三段：设置环境变量 → 检查 batch 是否整除有效 batch → 计算累积并调用 `run_all_folds.py`。
默认 `BATCH_SIZE=2`、`EFFECTIVE_BATCH_SIZE=32`，所以累积 16；CPU worker 默认 8、prefetch 默认 2。
预取是提前准备数据，梯度累积是推迟参数更新，二者不是一回事。这些是配置说明，不代表本次执行过启动脚本。

## 最后：跟着一张图走完整程

```text
① cv_splits.load_fold_records(..., fold=0, role="train")
   选出非0折Study的记录，记录里已经有原图路径与十维标签。
② make_overnight_loader 把记录列表交给 Dataset；DataLoader 抽到其中索引 i。
③ Dataset.__getitem__(i) 读原图 → 校验 → EXIF/RGB → 768等比Pad → 增强/张量。
④ collate_overnight 把它和另一张图堆成 [2,3,768,768]，标签为 [2,10]。
⑤ train_epoch 搬到GPU；model(images) 得到 [2,10] logits。
⑥ 纯BCE内部按类别pos_weight加权，再乘逐图Study视角权重，按累积窗口归一化。
⑦ backward 把这两张图的梯度加进 .grad；释放本微批计算图，参数暂时不变。
⑧ 重复到16个微批 → 梯度裁剪 → optimizer.step → 成功后更新EMA → 清梯度。
⑨ 一轮结束，用该折未参与训练的记录走无随机增强的Dataset与EMA前向。
⑩ 每图sigmoid → 同Study概率均值 → 内部Macro-AUC → 决定是否保存best。
⑪ 五折轮转后，固定各折best → 最终官方valid集成和阈值 → 相同协议预测test。
```

训练时逐图算 BCE，再用视角权重平衡 Study；验证时才把同 Study 的概率合并后算指标。
只要能在源码里定位上面 11 步，你就已经抓住这套系统的主线。之后再逐段阅读恢复和审计代码即可。
