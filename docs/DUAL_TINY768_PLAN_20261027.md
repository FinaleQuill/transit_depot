# 固定融合验收与双Tiny768下一阶段

> 2026-10-28更新：030训练及实际FP16/固定融合均已完成。双768 25/75=.808384442、50/50=.808370073，患者区间跨0；下一阶段实际组装25/75候选、完整valid及4090计时，见[双768实际FP16复盘](DUAL_TINY768_FP16_REVIEW_20261028.md)。下方训练指令为历史交接，不重复启动；本页预登记三组权重及成员顺序继续有效。

2026-10-27（香港）。029实际FP16验证与唯一预定的512/768等权诊断已完成。用户最新目标为**两个独立Tiny768学生、FP16、固定hflip2、概率加权集成**。下一项只交接030 Tiny768 seed44；先前优先新增Tiny512 seed44的建议已替代。当前没有启动030训练。

## 已完成的结果

同一官方valid，1422图、859 Study、228患者；每个Study十类，实际FP16、原图+水平翻转。CPU独立重算总体/十类/切片AUC与报告一致；两份CSV各8590行，身份覆盖、概率范围、报告/config/权重SHA绑定通过，两份FP16权重全部有限。审查前后18个输入文件未改。

| 候选 | Macro-AUC | 相对512 | 单图391 Study | 多图468 Study |
| --- | ---: | ---: | ---: | ---: |
| 027 Tiny512 seed43 | .805402258 | — | .740976325 | .847151620 |
| 029 Tiny768 seed43 | .803983596 | -.001418662 | .737898841 | .847305203 |
| 512/768各50% | **.807157435** | **+.001755177** | **.741944983** | **.849482205** |

| 类别 | 512 | 768 | 各50% | 融合−512 |
| --- | ---: | ---: | ---: | ---: |
| 1 | .688323 | .692864 | .694611 | +.006287 |
| 2 | .864437 | .853883 | .863648 | -.000789 |
| 3 | .807958 | .800257 | .806835 | -.001123 |
| 4 | .745854 | .745315 | .749907 | +.004053 |
| 5 | .887284 | .880739 | .885414 | -.001870 |
| 6 | .775196 | .776969 | .778750 | +.003554 |
| 7 | .783543 | .789868 | .788623 | +.005080 |
| 8 | .718387 | .720640 | .721082 | +.002695 |
| 9 | .902844 | .901310 | .903454 | +.000610 |
| 10 | .880196 | .877992 | .879250 | -.000946 |

融合六升四降，整体增加约0.1755个百分点；说明单768平均点分稍低仍可能提供互补排序。两类Study切片均有正向点差，但患者/病例构成不同，不能解释为模糊或早期病例改善；当前没有这些诊断标签。

既有1000次患者聚类配对bootstrap（seed20261027）给出：768−512的95%区间[-.007635439,+.004733691]；融合−512为[-.001024356,+.004688074]，均跨0。这里仅量化固定预测的患者采样不确定性，未覆盖训练seed、选epoch及多次历史比较偏差；不能断言稳定提高，也不能断言差距完全是噪声。此次独立审计重算点分及切片，bootstrap仅核对采样元数据，未重新运行。

该融合是保存的Study概率的CPU数学诊断，并非已经验收了实际组合部署的完整valid前向或T4时延。保留它作为已有精度参考，不据此声称两个768也已达到.80716。029 FP16比训练原生BF16低.000193367，与027的精度变化量接近，未见精度导出崩溃。

## 当前怎样平均，随后怎样加权

当前是连续概率**算术平均**，等价于权重0.5/0.5的加权平均，没有阈值硬投票。对图像i、类别c、模型m：

```text
p[m,i,c] = (sigmoid(z[m,原图,c]) + sigmoid(z[m,翻转,c])) / 2
p[i,c]   = w * p[seed43,i,c] + (1-w) * p[seed44,i,c]
p[Study,c] = 该Study全部图像的p[i,c]均值
```

十类各自sigmoid，不作十类softmax，不平均二值标签；TTA先平均概率，不把logits平均与概率平均混为一谈。固定全局权重下，Study均值与成员均值在线性数学上可交换；实际部署仍须核对浮点结果与CSV。

`compose_students.py`已支持`--weights`全局权重与`--weights-json`逐类权重，无需为此改训练代码。不同seed模型误差高度相似或第二成员较弱时，加权集成仍可能降分；多模型方法的文献结果不保证本任务收益。[Deep Ensembles](https://arxiv.org/abs/1612.01474)

在第二个768结果出来前，预先登记三组全局权重，顺序固定为**029 seed43、030 seed44**：25/75、50/50、75/25，文件为[DUAL_TINY768_WEIGHTS_20261027.json](DUAL_TINY768_WEIGHTS_20261027.json)。此映射文件用于`compare_student_candidates.py --combinations-json`；实际组装可用`compose_students.py --weights 0.5 0.5`等，不能将映射原样传给要求数组的`--weights-json`。两个单模型作端点参照；第一项诊断仍为50/50，再看两组粗权重是否有有用差异。不按单模Macro-AUC直接比例赋权，不拟合十套病种权重，也不追加密集扫描。看过验证分数后选权重属于验证选参，必须如实声明，不能将选择后的最高点分视为独立无偏验证。test不参与选择。

## 下一项只训练030

从相同非医学ImageNet初始化**新训练**第二个完整train学生，不能从029的best/last继续训练来冒称独立seed。两名学生不是两个交叉验证折；五折仍指教师严格OOF来源。

- 骨干：`convnext_tiny.fb_in22k_ft_in1k_384`；768输入、seed44，029为seed43。
- 初始化来源：[timm公开权重](https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k_384)，ImageNet-22k→1k、Apache-2.0；已核验本地revision `32c741ba106f9e062174ef829354a16719c3dcac`，原始权重SHA256 `a5cbe10cfa2e90ec787a082253033ac0a61de0264b68210815a988e53a1a5d7c`。
- 与029只改seed/output：alpha.2、T2、两轮KD渐进、batch8×accum4、原25轮余弦上限/patience5/min_delta1e-4、LR5e-5/5e-4、EMA、增强和hflip2全部保持。
- 复用原Base768五教师及严格OOF双视图logits；不重训教师、不重提软标签，不把valid并入训练。

真实CLI已在CUDA隐藏环境预览，`training_started=false`，派生config与029仅seed/output不同；严格OOF加载、train/valid隔离和本地初始化SHA通过。新目录尚未创建，当前空余约3.88GiB；按029实际大小，新best+last+原子临时文件约1.35GB可容纳，无需删除原模型。

训练预计常见早停约**2–2.5小时**，若满25轮约**4小时**；依据029实际13轮总耗时2小时估计，首轮后校准。随后CPU复盘约5–15分钟，实际FP16导出/valid约5–15分钟，保存概率的三组粗权重比较约1–5分钟；这里只给下一训练命令，后续阶段仍先复盘再交接。

先创建全新tmux会话；若提示已存在，先重连查看，不重复启动：

```bash
tmux new-session -s student030
```

在其中执行：

```bash
cd /home/featurize/work/cloud_project_v2
HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
/home/featurize/.conda/envs/cxr-kd-v1/bin/python -u train_distill.py \
  --fold-plan data/cv5_overnight/folds.json \
  --soft-labels ../runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2 \
  --training-mode distillation --alpha 0.2 --temperature 2 \
  --image-size 768 --seed 44 --validation-tta hflip2 \
  --batch-size 8 --accumulation-steps 4 \
  --epochs 25 --patience 5 \
  --output ../runs/030_tiny768_kd_seed44 \
  --execute
```

`HF_HUB_OFFLINE=1`让HF读取已核验本机缓存，避免启动时追随远端main；它不关闭独立的许可证URL下载。若缓存清理或初始化核验失败，先检查来源，不去掉限制直接换权重。训练入口自动保留完整stdout/stderr、原执行源码/环境和逐轮checkpoint账本。

按Ctrl+B、松开后按D离开tmux；重连用`tmux attach-session -t student030`。另一终端监测：

```bash
tail -n 30 -F /home/featurize/work/runs/030_tiny768_kd_seed44/evidence/console_*.log
```

```bash
watch -n 5 cat /home/featurize/work/runs/030_tiny768_kd_seed44/status.json
```

状态轮末更新，轮内看日志；退出tmux界面不等于训练停止。若异常中断，先核验last，再给保持原配置的resume；不要重跑新训或临时改epoch数。完成后检查整体/十类AUC、1/4/8后期退化、训练与验证目标分化和资源，再确定导出阶段。

## 提分与省时分别验证

**已实现的省时：**单图只解码一次；两个768共用缩放/归一化和GPU上的原图、翻转输入。每模型仍独立运行两次，共四次前向；不同权重不能直接共用骨干特征。此复用只在一张图内，不是跨测试图预测缓存或预取隐藏成本。

**后续CLAHE提分候选：**当前学生训练和上述AUC都没有CLAHE。CLAHE改变局部对比度并增加一道计算，不能将其本身当成省时操作。若双768基线完成后仍需提高，将它登记为一项单因素实验，固定参数与应用顺序、训练/验证/部署匹配，保留原学生对照；不把它和换损失、改alpha、改日程同时引入。为两名成员采用同一固定预处理时，可在同图共享CLAHE输出；原图/CLAHE不同成员只能共享其共同处理部分，不能误用同一归一化输入。

**可能的实现加速：**DALI的`fn.clahe`有CPU/GPU后端；官方算子吞吐加速不等于本项目batch1端到端提速。当前没有安装/应用DALI，也没有CLAHE收益或耗时实测。后续若采用，需核验像素语义、FP16完整valid和同一predict/cost实现，并把解码、传输、CLAHE、同步和写结果全部计时。[DALI官方CLAHE](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.clahe.html)

## 部署与规则边界

第6条允许固定TTA及固定多模型投票/平均；最终权重、视图和融合系数须在train/valid阶段确定，不在test上拟合。逐图固定CLAHE按规则允许的独立确定性预处理理解，PDF未点名CLAHE；不跨测试图统计或自适应。[官方规则PDF](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf)

两个Tiny768 FP16现有同架构组合约**128.77MB**，预计低于500,000,000bytes，030实际导出后重新计数。已有双768计时代理在4090约79.24/79.92ms，使用029同run E8/E13，**只代表计算成本**，不代表两个独立学生的精度；无CLAHE/DALI。512+768本机112.11ms，主要多出另一尺寸预处理，不直接拿它否定同尺寸双768。

当前没有T4。4090/MPS不是T4模拟器，无法保证双768在T4≤100ms；最终必须用实际030+029及所有TTA/预处理按官方batch1串行100图/另5图预热/3轮、读图到写出实测。所有原始异常轮保留，后续稳定性和T4余量单独验收。即使固定双768为优化目标，也不能提前声明其速度或成绩合格。

本次只进行CPU完整性/指标/配置检查与文档、权重候选登记，没有新GPU推理、训练、依赖安装或改写原模型/缓存。实际比较报告：`../deliveries/030_kd512_kd768_equal_diagnostic/comparison_report.json`；证据：`artifacts/kd_v1/kd512_kd768_fusion_review_20261027/{integrity.json,next_tiny768_seed44_proposal.json}`。速度详情见[4090计时报告](RESOURCE_STRESS_RESULTS_20261027.md)。
