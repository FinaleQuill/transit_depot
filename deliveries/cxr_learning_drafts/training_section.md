## 多标签损失：十道独立的判断题

这个项目的一次检查（Study）可以同时具有多个阳性标签。因此，模型输出的十个数应理解为十道独立判断题各自的原始分数，而不是十个互斥答案。输出尚未经过 sigmoid 的实数叫 **logit**，记作 $z_c$；用 $p_c=\sigma(z_c)=1/(1+e^{-z_c})$ 才得到第 $c$ 类的模型概率。十个概率不必相加等于 1。

Softmax 会把十类放到同一个“总和为 1”的竞争池中，抬高一类通常压低其他类，适用于互斥的单标签分类。这里采用 `BCEWithLogitsLoss`，把十个二分类交叉熵并列计算；内部把 sigmoid 与交叉熵合并成数值更稳定的运算。因此训练前不要手动对 logit 调用 sigmoid 再交给这个损失。验证时才显式 sigmoid 得到供评估和集成使用的概率。

### `pos_weight` 到底乘在什么地方

对一张图的一个标签，令真实标签 $y_c\in\{0,1\}$，其正样本权重为 $q_c$，损失为：

$$
\ell_c(z_c,y_c)=-q_cy_c\log p_c-(1-y_c)\log(1-p_c).
$$

当前 `compute_loss_config()` 对**当前折训练子集**的 Study 去重后计算：

$$
q_c=\frac{N_c^-}{N_c^+}.
$$

同一 Study 的两张图不会被当成两份阳性统计；十类各有自己的权重。内部验证折、官方 valid 和 test 的标签不会参与这个权重的拟合。保存的 `loss_statistics.json` 会记录正负 Study 数、权重、来源和哈希。全训练集的统计只能解释总体分布，真正每折训练用的是该折训练部分重算的值。

例如，一类有 100 个阳性 Study、900 个阴性 Study，$q_c=9$。真实为阳性且模型只给 $p=0.1$，其损失为 $-9\log0.1\approx20.72$；不加权时是 $2.30$。真实为阴性且模型给 $p=0.1$，损失仍是 $-\log0.9\approx0.105$，不会被乘 9。**加权的对象是阳性项，不是这个类别所有样本的整份损失。**

更接近 ANN 反向传播直觉的解释是看对 logit 的导数：

$$
\frac{\partial\ell_c}{\partial z_c}=
\begin{cases}
q_c(p_c-1), & y_c=1,\\
p_c, & y_c=0.
\end{cases}
$$

仍取 $q_c=9$、阳性 $p_c=0.1$：导数是 $-8.1$，未加权时是 $-0.9$。梯度下降会更用力地把这道题的 logit 往上推，纠正漏掉的阳性。它改变训练目标，不能保证召回一定提高，也不意味着输出一定已经校准为真实患病概率。后续为二值预测选择阈值，和直接提交连续概率，是两个不同步骤。

当前 overnight 主线使用**纯正样本加权 BCE**，`auxiliary_losses=[]`，未叠加 Rank、Focal、ASL。旧版入口 `train.py` 仍有 Focal/ASL 等历史配置；不能据此认为当前 overnight 入口启用了它们。

### 同时存在的另一层权重：让 Study 等权

类别稀有程度的 `pos_weight` 和多视图的 Study 权重解决不同问题。设当前折训练集有 $N$ 张图、$S$ 个 Study，某 Study 保留了 $m_s$ 张图。`OvernightImageDataset` 给其中每张图的权重为：

$$
w_i=\frac{N}{S m_s}.
$$

先把图 $i$ 的十类 BCE 求均值 $\bar\ell_i=\frac1{10}\sum_c\ell_{ic}$，再乘 $w_i$。整个 epoch 记录的损失是：

$$
L=\frac1N\sum_i w_i\bar\ell_i
 =\frac1S\sum_s\frac1{m_s}\sum_{i\in s}\bar\ell_i.
$$

这样，一个 Study 有两张图，也不会仅因张数多而获得单图 Study 的两倍影响。`pos_weight` 负责某个类别的阳性稀少，$w_i$ 负责各 Study 的视图数不一致。数据按图片 shuffle、无放回遍历；这里并没有额外使用稀有阳性的过采样。训练是分别对各图求 BCE 再加权，**不是先把一个 Study 的图概率平均再求训练损失**；Study 概率平均主要发生在验证和推理阶段。

## 一次训练循环：从两张图到一次参数更新

按文件阅读顺序：`run_overnight.sh` 调用 `run_all_folds.py`，后者组织五折，逐次调用 `train_overnight.py`。当前脚本的默认值是 `BATCH_SIZE=2`、`EFFECTIVE_BATCH_SIZE=32`，故传给训练器的 `accumulation_steps=16`。这描述**代码默认配置**；环境变量、命令行以及 OOM 自动降档可以改变实际值，应以运行记录为准。不要为了学习而重新运行这些入口。

`train_overnight.py::train_epoch()` 的骨架如下（仅为解释，省略完整校验）：

```python
optimizer.zero_grad(set_to_none=True)       # 清空上一更新窗口的梯度
for batch in loader:
    with autocast(...):
        logits = model(images)             # AMP 前向：图片 -> 十个 logit
    per_image = criterion(logits.float(), targets).mean(1)
    weighted_sum = (per_image * study_weights).sum()
    scaler.scale(weighted_sum / window_images).backward()
    if 当前累积窗口已满或已经到达 epoch 尾部:
        scaler.unscale_(optimizer)         # FP16 时先还原梯度尺度
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)             # 真实成功更新后触发 EMA hook
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

1. **前向传播**：把 `[B,3,768,768]` 图像送进模型，得到 `[B,10]` 的 logit。每道题对照 `[B,10]` 中对应的 0/1 标签产生损失。
2. **反向传播**：`.backward()` 用链式法则计算损失对参数的梯度，写入参数的 `.grad`。此时不会自动修改参数。PyTorch 默认把新梯度加到已有 `.grad` 上，因此可以累积。
3. **梯度累积**：微批 2 表示一次显存里主要处理两张图片，未必是两个不同 Study，更不是同时处理两个完整 Study 的全部视图。连续 16 次前向和反向，直到相当于 32 张图（未必属于 32 个独立 Study）的梯度已经累积，再更新一次参数；不需要同时把 32 张图及其激活放入显存。这里不是算完 16 个批次才进行第一次反向，也不是更新 16 次。
4. **更新**：AdamW 根据累计梯度和自己的历史一、二阶统计更新参数；梯度范数先裁到 1.0 以内，抑制突然过大的更新。之后梯度清空，开始下一个窗口。

代码对每个微批反传的是 `weighted_sum/window_images`。完整窗口分母是 32；假如 epoch 最后仅剩 10 张图，尾窗口分母就是 10，且最后一个微批也会触发更新。这里不能机械地把所有批损失始终除以 16，否则尾窗口会被错误缩小。`drop_last=False` 保证尾批不会被丢掉。分母是这个窗口实际的**图片数**，不是这个窗口内 `loss_weights.sum()`；Study 权重是在整个训练子集上统一归一化的。

“等效 batch 32”表示一次更新累计了相应数量图片的梯度，不应理解为任何情况下都和一次性 batch 32 位级相同。随机增强、dropout、随机深度、数值精度以及其他批间行为都可能造成差别。这里骨干主要用 LayerNorm，不存在依赖大 batch 估计均值方差的 BatchNorm 训练统计这一典型障碍。

### AMP 与梯度检查点分别节省什么

`amp_dtype=auto` 时，有 BF16 支持就用 BF16，否则使用 FP16。4090 支持 BF16；大骨干在 `autocast` 中计算，损失显式用 `logits.float()` 和 FP32 目标计算。AMP 并不意味着模型所有数据都变成 16 位：主参数、优化器状态等仍有对应的精度需求。FP16 时 `GradScaler` 通过缩放避免很小的梯度下溢，检测溢出后可以跳过一次参数更新；BF16 模式下 scaler 关闭，但保留一致的调用接口。

模型配置固定 `gradient_checkpointing=True`：前向传播不长期保存所有中间激活，反向时重算需要的部分，以额外计算换显存。它不是保存 `best.pt`/`last.pt` 的“磁盘检查点”。梯度累积、AMP、激活检查点分别减少同时处理的样本、部分数值占用和保留的中间激活，三者可以一起使用。

### 学习率与 AdamW

骨干已有预训练知识，默认学习率 `2e-5`；新任务分类头需要更快适应，默认 `2e-4`。`build_optimizer()` 把骨干/头、衰减/不衰减参数分组：二维及更高的非 bias 参数使用默认 0.05 weight decay，bias 和一维归一化参数不衰减。全模型可训练，没有先冻结骨干的阶段。

`set_learning_rates()` 按**成功的优化器更新数**安排学习率：默认第一轮从基础学习率的 0.1 倍线性预热，之后余弦下降，最低到基础学习率的 0.05 倍。不要把旧 `train.py` 的分阶段训练流程套到当前入口上。

### EMA：保留一份平滑的模型

可以把训练中的在线参数 $\theta$ 想成每次更新都有些抖动的估计，EMA 参数 $\bar\theta$ 是它的平滑版本：

$$
\bar\theta_t=d_t\bar\theta_{t-1}+(1-d_t)\theta_t.
$$

当前 `ModelEmaV3` 设目标 decay 为 0.999，并启用 `use_warmup=True, warmup_power=0.75`；因此不要把每个时刻的 $d_t$ 都写成从第一步起固定 0.999。EMA 是第二份权重，不参与反向传播（`requires_grad_(False)`），也不取代在线模型供 AdamW 更新。非参数 buffers 按 `exclude_buffers=True` 的策略直接从在线模型同步。

重要实现细节在 `after_step()` 和 `register_step_post_hook()`：**真正执行了 optimizer.step 之后**才增加 `optimizer_updates`、更新 EMA 并增加 `ema_updates`。累积窗口中的每个微批都不更新 EMA；FP16 溢出导致 scaler 跳过参数更新时，也不会更新 EMA。这保证优化器与 EMA 的计数一致。

每个 epoch 训练完，调用 `evaluate(ema.module, ...)`，从 EMA 权重得到内部验证折的指标；用于推理的 `model_state_dict` 保存 EMA，`raw_model_state_dict` 保留在线模型供完整续训。`best.pt` 和 `last.pt` 还保存优化器、scaler、随机状态及进度，因此恢复完整 epoch 状态不只是加载模型参数。

## 验证、五折与最终集成：要分清两种验证集

### 为什么是五折，怎样防止“偷看相似病人”

五折把原始 train 分成五份：某次训练用其中四份，剩下一份评价，轮换得到五个模型。每个训练 Study 都有机会成为一个模型没有见过的内部验证样本（OOF）。这比只挑一次分割更能观察结果是否受某一份数据影响；不同模型的错误也可能互相弥补，概率平均通常比单个模型更稳，但不能保证一定提升。

医学图像不能随意按图片拆分。同一病人可有多个 Study，同一 Study 又可有多张图；若它们跨入训练与验证，模型可能靠病人特征或近乎重复图“认人”，导致验证过于乐观。当前 `cv_splits.py` 不是简单逐图随机，也不是直接套一个普通 `StratifiedKFold`：

- 用病人作为不可拆的组；同一病人的全部 Study 同折。
- 对跨病人记录里的相同文件哈希或相同像素哈希，用并查集合为不可拆的内容组，防止完全重复图跨折。
- 稀有标签较多的组先分配，贪心平衡每类阳性与阴性 Study 数，同时兼顾 Study、病人、图片总量。
- 逐折检查十类都有阳性和阴性，读取分折时再次校验身份、内容和统计。

原始官方 `valid` 是独立留出的验证集，不参加五折训练，也不参与每折 best 的选择。分折时会用身份/哈希核查官方 valid 与 train 隔离，不用其标签来拟合分层。

`run_all_folds.py` 默认每次让一个折训练一个 epoch，然后选择已完成 epoch 最少的可训练折继续，因此五折是轮转推进，不是一次把某折训完再开始下一折。每折单独保存状态和 early stopping。内部验证的 `macro_auc` 严格大于已有最佳才刷新 best；精确相等保留较早的 best。early stopping 另外要求超过参考 AUC 加默认 `min_delta=1e-4` 才算有实质改进，默认 patience=4；保存 best 的比较和早停的比较不是同一条规则。

### 一张图的概率怎样成为一次 Study 的概率

`evaluate()` 对每张图先算 `sigmoid(logit)`，得到十概率，再调用 `aggregate_study_predictions()`。该函数先确认所有预期图恰好出现一次、图 ID 不重复、标签一致，然后按 `subject_id/study_id` 分组，对同一 Study 的原始视图概率求算术平均：

$$
p_{s,c}=\frac1{m_s}\sum_{i\in s}\sigma(z_{i,c}).
$$

例如某类正位图概率为 0.8、侧位图为 0.4，该 Study 的概率就是 0.6。这里是**平均概率**，不是平均 logit 后再 sigmoid，两个操作一般不等价。内部验证使用确定性的 768 预处理、无随机增强、无翻转 TTA。十类逐类算 ROC-AUC、AP 后求均值；诊断 `loss` 是聚合后的普通无加权 BCE，不能拿它和加权的在线训练 loss 直接比较。当前内部折主线不计算阈值调优 F1，checkpoint 中写入的十个 0.5 只是固定占位设置，不用于 AUC 或概率提交。

### 最终推理：Study 平均、TTA 平均、五折平均

冻结的集成计划默认 `scales=[768], hflip=True`。每张原始图产生 768 原图版和水平翻转版两种推理视图；每种变换先按 Study 平均，接着对两种变换平均，然后对五折 EMA best 模型的结果等权平均。用式子表示：

$$
p_{s,c}^{ens}=\sum_{k=1}^{K}\alpha_k
 \frac1T\sum_{t=1}^{T}\frac1{m_s}\sum_{i\in s}
 \sigma\bigl(z_{k,c}(T_t(x_i))\bigr),\quad \sum_k\alpha_k=1.
$$

完整五折默认 $K=5,\alpha_k=1/5$，当前默认 $T=2$。TTA 是推理时多看几个保留语义的版本后平均，模型参数不发生学习。每个分辨率都重新读取原图，避免从旧 512 缩略图放大。代码也允许有 1～4 个可用折的部分集成，会记录 `partial_cv_ensemble`；不要把代码计划的五折误写成运行中已经完成五折。

### 最佳阈值究竟在哪里搜索

当前项目有两个不同概念：内部折验证挑选 `best.pt`，最终官方 valid 供冻结集成的评估与二值阈值拟合。**实际阈值搜索发生在 `inference_overnight.py::run_ensemble_inference()`，不是 `train_overnight.py` 的每个 epoch。**最终推理先用已经冻结的模型、等权和 TTA 对官方 valid 预测，不依该结果再挑模型或调集成权重，然后调用 `train.py::search_class_thresholds()`。

对每类单独选阈值 $t_c$，判断 $\hat y_c=1[p_c\ge t_c]$，使该类 F1 最大：

$$
F1=\frac{2PR}{P+R}=\frac{2TP}{2TP+FP+FN}.
$$

实现用 `precision_recall_curve` 枚举该类验证预测里**所有唯一概率断点**，不是从 0 到 1 每隔 0.01 的粗网格。去掉 PR 曲线末端没有对应阈值的那个点，计算每个候选的 F1；若多个候选的浮点 F1 精确相等且并列最大，就选**更高阈值**。十类独立最大化各自 F1，等价于在这份概率与标签上最大化十类的 Macro-F1。

举一个包含并列最优的四个 Study 例子：真实标签依次为 `[1,0,0,1]`，预测概率为 `[0.9,0.8,0.7,0.6]`。候选是这四个唯一概率：

| 阈值（预测规则为概率 ≥ 阈值） | TP | FP | FN | F1 |
|---:|---:|---:|---:|---:|
| 0.6 | 2 | 2 | 0 | 2/3 |
| 0.7 | 1 | 2 | 1 | 0.4 |
| 0.8 | 1 | 1 | 1 | 0.5 |
| 0.9 | 1 | 0 | 1 | 2/3 |

0.6 和 0.9 的 F1 并列最大，当前代码选择更高的 **0.9**。这个选择只是在同样 F1 时采用确定的规则，并不表示 0.9 对所有类别都更合适。

阈值拟合和报告 F1 用的是同一官方 valid，因此它是调阈值后的拟合结果，不能说成“完全未参与任何选择的外部 F1”。AUC、AP 直接使用连续概率，阈值不会改变它们。只有为了展示或决策生成 0/1 时才使用阈值。

最终 `study_binary.csv` 使用这些阈值；`study_probabilities.csv` 保存完整概率。`submission_final.csv` 严格按提交模板的 `(subject_id,study_id,class_id)` 键与顺序写概率，**每个 Study 的十类都保留，不因低于阈值删行，也不把概率替换成 0/1**。代码检查预测覆盖、行数、身份顺序、概率范围和写出后读回一致性；测试集不用于拟合阈值或模型。

## 给主教程合并时的核对索引（原文件行号）

| 内容 | 已核对的位置 |
|---|---|
| 默认微批/累积/EMA等 | `train_overnight.py:45-65`；`run_overnight.sh:19-34` |
| 每折训练 Study 的 pos_weight，无辅助损失 | `train_overnight.py:114-169`、`:599-600` |
| Study 权重的实际公式 | `dataset_overnight.py:113-115` |
| AdamW 参数组 | `train_overnight.py:235-252` |
| 逐成功更新数的预热/余弦 | `train_overnight.py:255-266` |
| 加权 BCE FP32、尾窗图片数分母、裁剪/step | `train_overnight.py:288-361` |
| 验证先 sigmoid 后按 Study 汇总 | `train_overnight.py:364-382`；`dataset_overnight.py:212-248` |
| AMP 自动选择BF16/FP16 | `train_overnight.py:601-609` |
| EMA 构造、hook | `train_overnight.py:704-726` |
| EMA验证、best严格大于、early stopping | `train_overnight.py:744-764` |
| 固定0.5，EMA/raw权重分别保存 | `train_overnight.py:784-791` |
| 病人/内容隔离、贪心多标签五折 | `cv_splits.py:46-190` |
| 每折每次跑一轮、有效批次自动除法 | `run_all_folds.py:80-96` |
| 当前默认TTA/均匀概率平均 | `run_all_folds.py:133-161` |
| 原图Study平均 -> TTA平均 -> 折均值 | `inference_overnight.py:248-347` |
| 官方valid拟合阈值、概率提交独立 | `inference_overnight.py:405-446` |
| 阈值精确断点和并列取高阈值 | `train.py:241-258` |

核对过程中只读取源码；未 import 项目或 torch，未运行训练、模型、测试、显存探针或任何 GPU 命令。此草稿只写入独立交付目录，不修改正在运行的源码。当前入口默认 2×16 已核实，但此段未读取进程或运行 config，因此不对实时训练实参作断言。
