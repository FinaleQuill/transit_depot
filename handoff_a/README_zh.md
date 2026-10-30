# A训练交接包：冻结教师，直接修改学生继续训练

本包只针对A：ConvNeXt V1 Tiny768、平方根Study正类权重、α=.2/T=2严格OOF蒸馏。A在E14早停，E9最佳；真实FP16+hflip2官方valid Macro-AUC为0.809115940852。类1/4/8仍有不足，不能把它描述为稳定优于033。

## 包里有什么

- `parent_run/best.pt`：E9完整断点，FP32 EMA、raw模型、Adam优化器、Scaler、RNG、早停/学习率调度计数全部保留。
- `parent_run/last.pt`：E14完整末轮断点；原配置、14轮历史、真实控制台日志、源码快照与逐轮哈希账本一并保留，原件不改写。
- `oof/soft_labels.npz`、`metadata.json`：冻结的30328×2×10原始OOF logits和五折教师来源/患者隔离/许可证材料。学生训练不加载教师权重，**不需要重新训练教师或重新生成软标签**。
- `project/`：可编辑的学生训练、数据读取、蒸馏loss、验证/导出代码及原始train/valid/test清单、五折划分。没有B的训练断点，也不自动启动B或教师任务。
- `pretrain/`、`provenance/`：原始合规ImageNet Tiny初始化及完整来源材料。新训练完全离线，禁止自动更换远端权重。
- `environment/`：本次CPython3.12/Linux x86_64的离线附加依赖及精确版本锁；核心torch/CUDA等来自固定比赛镜像。
- `rules/`、`reports/`：两份规则PDF、A验证/耗时/提交审计摘要。**不含原始胸片，也不含test预测CSV**。

## 1. 解压、校验与准备影像

需要Linux x86_64、Docker、支持CUDA12.8的NVIDIA驱动及足够显存/磁盘。请解压到有足够空间的本地盘，建议连同训练输出至少预留12GiB；不要挤占只剩约1GiB的work。可给启动器加`--output-root /你的本地大盘/A_runs`选择宿主训练输出位置，容器内仍统一使用`/experiments`。解压后先在包根运行：

```bash
python tools/verify_files.py
docker pull crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6
python tools/run_in_image.py --cpu -- portable_a.py verify
```

第一条验证包内每个文件SHA；第二条只准备运行环境，不下载数据或模型。第三条用固定镜像实际校验train清单、患者OOF缓存以及best/last完整续训状态，不构建模型、不使用GPU。

原始影像请使用队伍已获授权的**官方数据**，例如`/data/cxr_first/train/...`和`/data/cxr_first/valid/...`。本包不重复分发约52GB影像；脚本逐图核原文件SHA，不能换外部数据或重新编码图片。下面命令中的`/你的路径/cxr_first`需要替换为实际目录。

包内Docker启动器把原工程绝对路径映射到独立容器中，所以无需重写folds、OOF metadata或checkpoint的历史身份字段。所有输入只读挂载，新实验输出在`experiments/`；宿主侧`project/*.py`在开训前可直接编辑。

OOF metadata中的教师checkpoint路径是历史来源证据；运行学生不需要这些教师文件，已经验证其患者分折、来源和缓存哈希。不要重新提取OOF，也不要混入见过对应患者的其他教师输出。

## 2. 推荐：从A最佳权重修改配方，开启新实验

先预览，不会训练：

```bash
python tools/run_in_image.py --data-root /你的路径/cxr_first -- \
  portable_a.py train --output /experiments/A_team_v1 \
  --backbone-lr 2e-5 --head-lr 2e-4 --epochs 20 --patience 5
```

确认配置后加`--execute`：

```bash
python tools/run_in_image.py --data-root /你的路径/cxr_first -- \
  portable_a.py train --output /experiments/A_team_v1 \
  --backbone-lr 2e-5 --head-lr 2e-4 --epochs 20 --patience 5 --execute
```

新入口默认加载**A E9最佳EMA**作为学生起点，复用原OOF，重新初始化优化器、Scaler、EMA更新计数、随机种子和训练日程，epoch从1开始；这是新的学生微调实验，不冒称接续原A的历史优化器轨迹。上面的LR/轮数是演示可编辑参数，不是已经证明更优的配方。所有原A资产保持只读。

可直接修改`project/train_distill.py`、`student_experiments.py`、`dataset_overnight.py`等学生代码，或通过原CLI调整LR、α/T、正类权重、dropout、日程、早停等参数。`--student-init-checkpoint`可显式选择`/bundle/parent_run/last.pt`，默认推荐best.pt。改动后先运行CPU测试；训练启动后不要再改其执行源码。

新实验中断后，用**相同配方、相同output**加`--resume /experiments/A_team_v1/last.pt`恢复；仍保留默认student-init路径作为配置/血缘记录，恢复分支不会重新加载父A权重。不要临时改变已开始run的超参数，想试新配方换新output目录。

训练日志在新run的`evidence/console_*.log`，每轮指标在`history.jsonl`，完整best/last自动保存。建议用tmux启动长训练。

## 3. 如需继承原A完整优化器/RNG，而非新开优化器

保留了两份完整断点，方便高级续训。注意：**直接用原E14 last.pt执行原样resume会因早停已满足而退出**，不是程序卡住。不要把清零早停后改参数冒称原实验无缝延续。

如果要按原配方承接完整状态，可从E9最佳完整断点继续到E10：

```bash
python tools/prepare_exact_resume.py
python tools/run_in_image.py --data-root /你的路径/cxr_first \
  --exact-run experiments/exact_A_best -- \
  portable_a.py exact-resume --execute
```

准备器把best.pt的原字节作为新目录的best/last，继承E9时真实优化器/RNG和未触发的早停状态，并只导入原history的E1–E9。原训练记录仍完整保留在parent_run，不覆写原E14状态。此模式保持原25轮总日程及配方，不允许同时改参数；如果要改参数，使用上一节的新实验入口并明确记录变化。跨GPU/环境不保证逐位重现原轨迹。

## 4. 合规与证据

只用官方train训练；valid保持独立用于早停/选模；test不能进入训练、伪标签、统计拟合、模型选择或增强拟合。不得因榜单反馈调整配方。教师/学生初始化仍限包内已核验的非医学、可再分发、非NC、非gated来源，遵守`rules/`两PDF。

原始执行源码在`parent_run/evidence/source_*/`；`project/train_distill.py`是**为可编辑A微调加入学生初始化开关的交付副本**，差异完整记录在`PORTABILITY.patch`，没有冒充原训练执行源码，也没有改发送方正在训练B的工程文件。新训练会捕获自己的实际代码版本和父A权重SHA/epoch，保存血缘声明。

本ZIP是训练协作包，非T4认证提交包。正式交付还需相同predict/cost路径、真实T4端到端核验和全部材料；A的历史2080Ti约73.94ms/图不能替代T4。
