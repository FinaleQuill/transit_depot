# 合规 V1 教师与离线蒸馏操作说明

> 当前TTA/集成路线见[新操作手册](KD_TTA_ENSEMBLE_USAGE.md)。本页保留此前单Tiny512、identity和原021路径方案的记录，当前执行使用024教师集合及新双视图缓存。


当前已批准20/12/12/12/12混合预算，使用024/teacher_collection.json提取OOF，见[当前操作手册](KD_V1_MIXED_USAGE.md)。以下全20轮调度及021软标签命令保留历史语境。

本次只完成代码、环境和 CPU 验证；训练、模型前向/反向、优化器更新、预训练权重下载均为 0。
历史 020 是 V2 训练产物，不能用于本流程，也没有被删除或覆盖。

上段为14:13代码交付时点记录。用户之后已启动021教师；目前训练继续运行，并已按官方备查通知补齐日志、逐轮时间/哈希及源码留存机制，见[材料修补与验收边界](KD_V1_COMPLIANCE.md)。

## 固定方案与入口

| 阶段 | 入口 | 协议 |
| --- | --- | --- |
| 模型 | `model_overnight.py` | 只允许 V1 Base/Tiny 的 `fb_in22k_ft_in1k_384` 精确来源 |
| 教师 | `run_teachers.py` → `train_teacher.py` | Base / 768 / 患者隔离五折 / 固定20轮最终EMA |
| 软标签 | `extract_soft_labels.py`、`distill_data.py` | 原 train 图像级 FP32 logits / 每图仅一个折外教师 / identity |
| 学生 | `train_distill.py` | 单Tiny / 512 / α=0.2 / T=2 / 前两轮KD预热 |
| 导出 | `export_student.py` | 仅EMA，FP16磁盘权重，无TTA、无集成 |
| 复验 | `validate_student.py`、`predict.py`、`cost.py` | 共享同一512/FP16单图预测接口 |
| 核验包 | `package_kd_submission.py` | 来源原权重、完整许可、XLSX声明、代码、实际部署文件 |

教师和学生训练入口、软标签提取入口默认只预览；添加 `--execute` 才真正执行。
导出、验证和预测入口按名称显式执行自己的操作，不会连带启动训练。
新脚本导入不会构建模型或下载权重。

## 先检查环境与解压结果

独立环境路径：`/home/featurize/.conda/envs/cxr-kd-v1/bin/python`。
训练依赖为 `requirements_kd.txt`，Python3.12、torch2.10.0+cu128、torchvision0.25.0+cu128。
旧环境记录保留在历史资料中；训练使用 `requirements_kd.txt`，官方容器推理依赖使用 `requirements_inference.txt`。

2026-10-25补装：当前主机使用SOCKS代理，Hugging Face下载需要`socksio`。
已在上述独立环境安装`socksio==1.0.0`并加入`requirements_kd.txt`；
Base/Tiny原始预训练权重、SHA256和许可证材料下载验证通过，未启动训练或模型前向。
以后重建环境时使用更新后的依赖清单即可。若仅补装此依赖，使用：

```bash
/home/featurize/.conda/envs/cxr-kd-v1/bin/python -m pip install 'socksio>=1.0.0,<2'
```

各阶段输入时机、tmux、暂停与恢复见[从这里开始](KD_V1_START_HERE.md)。

```bash
cd /home/featurize/work/cloud_project_v2
KD_PYTHON=/home/featurize/.conda/envs/cxr-kd-v1/bin/python
"$KD_PYTHON" preflight_kd.py --check-files
```

该命令只读清单和文件存在性，不解码图像、不加载网络。必须在解压完成后使用。
当前 train 为30328张/18513个Study；valid为1422张/859个Study。Dataset在之后真实读取每张图时核对其已登记SHA256。
单次存在性检查不等于全量图像字节哈希重新审计。

若新机器没有此环境，先手动创建并安装；这些命令不启动训练：

```bash
conda create --prefix /home/featurize/.conda/envs/cxr-kd-v1 --override-channels --channel conda-forge python=3.12 pip -y
/home/featurize/.conda/envs/cxr-kd-v1/bin/python -m pip install --no-cache-dir -r requirements_kd.txt
```

## 第一步：启动教师

先预览五折命令；此命令不创建训练run：

```bash
"$KD_PYTHON" run_teachers.py
```

确认准备好后，以下才是实际训练命令。本次交付期间未执行：

```bash
"$KD_PYTHON" -u run_teachers.py --execute
```

默认输出：`../runs/021_convnext_v1_base768_teacher_cv5/fold_0` 至 `fold_4`。
每折 batch2×累积16，有效批量32，固定768，默认20轮。
首次显式训练会下载相应公开预训练权重和许可证材料，并绑定原始文件SHA256与固定HF commit；实际模型从已绑定的本地文件加载。
不会下载外部训练数据，不运行任何医学自监督或二次预训练。

严格OOF的含义：外折患者不参与梯度、早停、checkpoint选择、类别权重拟合。
本实现不在外折上逐轮计算指标；固定轮数完成后才生成可提取的 `best.pt`，其语义是最终EMA，不是外折选出的最佳轮。
所有恢复状态保存在 `last.pt`；未完成教师会被提取器拒绝。修改固定轮数须使用新run目录，不能中途更改当前run。

通过创建下面的文件请求当前轮完成后停止：

```bash
touch ../runs/021_convnext_v1_base768_teacher_cv5/STOP
```

恢复时先移除自己创建的STOP，再重复同一教师命令；调度器会自动使用各折 `last.pt`。
不要同时运行两个调度器，不要把旧020目录作为新run目录。
如需改变微批量，可保持有效批量32，例如 `--batch-size 1 --effective-batch-size 32`。

## 第二步：全部教师完成后提取OOF

```bash
"$KD_PYTHON" -u extract_soft_labels.py --execute \
  --fold-plan data/cv5_overnight/folds.json \
  --teachers-root ../runs/021_convnext_v1_base768_teacher_cv5 \
  --image-root /home/featurize/data/cxr_first \
  --output ../runs/021_convnext_v1_base768_teacher_cv5/oof_logits
```

去掉 `--execute` 可先预览。输出目录必须是新目录，已有缓存不会被覆盖。
缓存为 `soft_labels.npz` 和 `metadata.json`，按图像ID、原图SHA、患者、fold、教师SHA和预训练材料绑定。
每张train图仅使用其所属外折教师，不能把五个教师全部平均后冒称严格OOF。
缓存不包含官方valid或test。学生加载时重新验收全覆盖和证据，禁止pickle与对象数组。

## 第三步：启动学生

```bash
"$KD_PYTHON" -u train_distill.py --execute \
  --fold-plan data/cv5_overnight/folds.json \
  --soft-labels ../runs/021_convnext_v1_base768_teacher_cv5/oof_logits \
  --image-root /home/featurize/data/cxr_first \
  --output ../runs/022_convnext_v1_tiny512_kd
```

默认batch8×累积4，最多25轮；官方valid按Study均值Macro-AUC早停，patience=5。
valid从不参加梯度训练，test从不用于训练、伪标签或任何拟合。
两项损失都先对10类取均值，再使用同一Study逆视图数权重；软项含Bernoulli阳性/阴性两项与T²，不能换成十类Softmax KL。
硬BCE沿用仅train计算的neg/pos。默认先对齐教师与学生的加权logit尺度：`z_t-log(w_t)+log(w_s)`；这是理论近似，配置如实登记。
仅温和翻转、亮度、旋转增强，无Mixup/CutMix，避免软目标混合不一致。

暂停方式是创建学生输出目录下的 `STOP`。恢复时移除STOP，并在相同命令末尾增加：

```bash
--resume ../runs/022_convnext_v1_tiny512_kd/last.pt
```

恢复核验优化器、RNG、EMA步数、训练目标、缓存及清单哈希，不允许静默改变配置。
若原子提交best之后、提交last之前中断，会使用已完整保存的更新best恢复；首次last尚未创建的边界也支持此恢复方式。
Alpha=0对照需另建学生目录，不可在已有run中途切换目标。

## 第四步：导出并复验真实FP16部署

纯体积估算不需要学生权重，也不构建模型：

```bash
"$KD_PYTHON" export_student.py --estimate-only
```

训练完成后显式导出到新目录：

```bash
"$KD_PYTHON" export_student.py \
  --checkpoint ../runs/022_convnext_v1_tiny512_kd/best.pt \
  --output ../deliveries/kd_v1_tiny512/model
```

导出仅保存一份EMA的FP16权重，静态核对键与形状，不携带raw模型或优化器。
纯参数估算55,655,636字节；实际限制按 `student_fp16.pt` 与 `config.json` 的总字节数核验，必须≤500,000,000。
训练BF16验证AUC不能直接当作最终FP16 AUC。以下命令只评估官方valid，不调整阈值：

```bash
"$KD_PYTHON" validate_student.py \
  --model-dir ../deliveries/kd_v1_tiny512/model \
  --image-root /home/featurize/data/cxr_first \
  --manifest data/clean_v1_reviewed/valid_manifest.csv \
  --output-dir ../deliveries/kd_v1_tiny512/fp16_validation
```

最终预测建议在相同T4部署环境生成，避免不同GPU的FP16舍入差异触发一致性检查：

```bash
python predict.py --model-dir ../deliveries/kd_v1_tiny512/model \
  --image-root /home/featurize/data/cxr_first \
  --manifest data/clean_v1_reviewed/test_manifest.csv \
  --output ../deliveries/kd_v1_tiny512/submission.csv
```

UTF-8长表每Study十类连续概率，当前应有14140行。额外逐图CSV用于重放与计时一致性，不是正式评分文件。
无阈值删行、无TTA、无集成、无测试时更新。

## 第五步：T4端到端计时与核验包

用主办方的100张图、5张预热图及其原目录结构：

```bash
python cost.py ../deliveries/kd_v1_tiny512/submission.csv pictures_100 pictures_5 \
  --model-dir ../deliveries/kd_v1_tiny512/model \
  --manifest data/clean_v1_reviewed/test_manifest.csv \
  --output-dir ../deliveries/kd_v1_tiny512/t4_cost
```

加载与初始化在计时外，5图预热；100图batch1串行连续3轮，逐图三轮平均后再平均。
读图、解码、512预处理、GPU同步、预测、Study汇总、逐图及Study CSV写出均计入。
GPU或环境不符、结果不一致、体积或延迟超限时，不认证合规；官方最终以复测为准。
4090上即使低于100ms，也只是诊断结果，不能替代T4验收。工程目标建议≤80ms。

官方材料存在接口细节未明：评分按Study，计时按图片。若100图没有包含所选Study的全部视角，程序仍真实测完100图，但标记比对待确认并退出2，不读取计时目录外图片，不用预计算Study概率查表。该情况应依据主办方计时清单确认比对口径，不能伪称已通过。

训练、预测和计时完成后，使用核验材料脚本准备新目录。缺失的报告必须如实标记pending，不能据此宣称已完成正式提交：

```bash
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

核验用原始ImageNet权重不参与最终推理，按官方第8条不计入运行模型500MB；核验包总大小可能因此超过500MB。
打包不包含原始比赛图片、旧V2权重或旧运行目录，不自动下载缺失材料。

规则来源：[官网](https://www.ncccu.org.cn/index/Paper/case1.html)、[官方PDF](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf)。
PDF禁止NC/gated权重；本方案保留官方valid用于选型。若未来并训valid，必须另行预先固定全部选择并如实申报，不能再用同一valid调参。
