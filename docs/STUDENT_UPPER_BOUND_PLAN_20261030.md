# 冻结教师后的学生上限实验提案

2026-10-30，Asia/Hong_Kong。状态：**提案，等待用户确认；本轮训练0次**。本阶段授权覆盖必要环境恢复与状态核验，不覆盖启动训练。用户明确暂不以T4的100ms限制筛选学生实验；正式交付仍需另行完成规则验收。

## 本阶段边界与规则

- 教师保持024集合现状，不重训、不更换epoch，不重提OOF。直接复用`oof_logits_tta_v2`的30328×2×10原始logits。教师checkpoint和缓存只读使用。
- 仅用官方train训练，valid保持独立，用于早停、选checkpoint和实验评估；test不参与训练、统计、选参或新预测。原test CSV保留，不重生成、不上传、不删除任何资产。
- 初始化仅复用038内已经核SHA及许可证的ConvNeXt V1 Tiny/Base原始ImageNet权重。Base学生从原始非医学初始化开始，不能从已训练教师checkpoint初始化全train学生并称作严格OOF蒸馏。不使用旧V2 NC权重，不下载新模型或引入外部数据。
- 已逐页读取服务器`rules.pdf`6页、`rules_1.pdf`5页，SHA分别为`ff6aa3adbfd3d000194253055a93dbce8a140031c8432f892046a49de8136fb6`、`836c4f5d3202f2dc65830fd919ca195797c12b7507d750121786a21672dc7a44`。补充第三节允许合规教师的软标签/特征蒸馏，第六节允许训练增强与类别加权。用户要求比允许范围更窄：本阶段不新增外部数据。
- 038目录、同级ZIP及SHA回执均移除写权限；474项manifest文件SHA、ZIP SHA和CRC已核验。033权重、config及原CSV保持原件。当前AVX2/generic＋LUT＋一次D2H推理不回滚、不改动。
- 真实T4的`cost_report.json`继续待办。2080Ti环境恢复或训练成绩不替代T4验收；正式整改阶段不得更换已提交权重的规则仍适用。

## 既有证据与实验动机

当前033实际FP16双Tiny768的历史valid Macro-AUC为0.808384442；单030为0.807512423。两者来自旧4090部署评估，不能当本轮重新实跑。

现有学生硬监督是逐图加权BCE，再以逆Study图数使Study等权；并没有把同Study所有视角聚合后计算硬标签损失。α=.2、T=2，逐图OOF双视图Bernoulli KL在前两轮渐进启用；教师logits先按教师/学生正类权重对齐，再温度化和逐视图求平均。

029/030均13轮正常早停、E8最佳。030的类1在E8→E13下降约0.0483，train loss和valid BCE仍下降；valid类1仅24个阳性Study/16名患者。这支持尝试降低极端正类权重，但不足以证明病因。类1/4/8是重点报告对象，不能仅凭三类点分选择而不看整体。

官方train按Study计算的类1/4/8正类权重约28.764/11.892/3.107，平方根约5.363/3.448/1.763。train共18513个Study/30328图，每个Study最多7图，可做保留全部视角的Study监督。

## 优先实验A：P2，Tiny768平方根正类权重

从原始Tiny ImageNet初始化新训，预先固定seed44。仅将学生硬BCE的`pos_weight`改为`sqrt(negative_studies/positive_studies)`，统计完全来自官方train；其余沿用030：768、α=.2/T=2、两轮KD渐进、EMA=.999、batch8图×accum4、25轮上限/patience5、warmup1、LR5e-5/5e-4、WD=.05、dropout=.2、droppath=.1、grad_clip=1、valid hflip2。

必须同步将KD的学生权重项改为新的平方根权重：`z_teacher - log(w_teacher) + log(w_student_new)`。保留教师各折原权重及OOF logits原字节，不把缓存当概率，也不再次温度化。当前代码尚未提供P2开关，确认后在独立学生实验模块实现，不能直接误用旧命令。

预期作用是缓和稀少阳性的过强梯度，争取减少类1后期排序退化，并观察类4/8。必须重训学生；同卡每轮开销预计约旧Tiny配方的1.0倍（约±10%），不以推理耗时筛选。没有可靠证据预测一定提分。

旧030训练来自4090/BF16；新2080Ti使用FP16＋GradScaler，不能照搬BF16，也不能将新旧差值全部归因P2。当前两实验预算优先探索成绩上限，不额外自动训练同机原配方对照。若以后要估计P2净因果收益，需单独批准同机/同精度的配对重训。

## 优先实验B：Base768学生＋P3完整Study硬监督

从038内原始合规Base ImageNet权重训练一个新学生，仍直接读取当前OOF logits。**预先固定平方根正类权重，不据A的成绩临时搜索B配方。** 这是容量与监督方式的组合探索；即使提分，也不能归因于某个单因素。

硬监督对同一Study所有图的预测概率取均值，再与原始Study标签计算加权BCE；不把Study阳性标签强压到每张视图。实现应使用FP32的稳定log-sigmoid/logsumexp计算均值概率对应的正负对数项，避免FP16概率饱和后直接取log。软监督仍为逐图严格OOF KL，先在Study内按视角求均值，再对Study等权；不能重复乘逆视图权重。

预先固定768、seed44、α=.2/T=2及两轮KD渐进、EMA=.999、25轮上限/patience5、warmup1、LR2e-5/2e-4、WD=.05、dropout=.2、droppath=.15、grad_clip=1、valid hflip2。使用完整Study组批，逻辑有效batch定义为32个Study；先按每微批1个完整Study、累积32次设计，保留1–7个全部视角，启用梯度检查点。最后不足32个Study的累积窗口按实际Study数归一化；物理分块时必须保留同Study硬损失的完整计算图。显存适配只改变物理分块，不能丢视角或悄悄改变有效batch/损失归一化。

预期作用是增加学生容量，并让训练硬目标与Study级评分聚合一致，争取整体及1/4/8改善。必须重训学生；按历史4090 Base/Tiny吞吐推算，同卡每轮暂预算为Tiny的2.5–4倍。该范围并非2080Ti实测；完整Study组批、FP16及保存开销均有不确定性，首轮后更新排期。纯FP16 Base权重预计约176MB，最终实际资产大小另核验，500MB规则并未放宽。

Base虽在模型和来源白名单中，现有`train_distill.py`、`export_student.py`、`deploy_kd.py`仍存在Tiny硬编码。确认后需要独立学生实验训练/验证/导出适配，不能把Base冒充Tiny，不能改写当前033部署或038安全基准。

## 成绩单与停止边界

两个实验各一个预登记配方，不自动连扫alpha、温度、权重、LR、增强或融合比例。+.002～.005 Macro-AUC可作为值得保留的阶段目标，**不是收益预测或承诺**。更复杂的ASL/AUC-margin、CutMix、多尺度训练列为下一批候选，先用本批成绩决定优先级。

Feature Distillation没有现成中间特征缓存；需重新调用每位患者对应的折外冻结教师提特征。虽然规则允许，但会增加教师推理、缓存和对齐成本，先不列入本批；严禁用见过该患者的其余教师混合训练特征。

确认后每轮保留真实loss/硬BCE/KL、十类AUC、AMP跳步、LR、精度、有效batch、耗时和显存。在当前2080Ti上用相同完整valid、相同评估精度/预处理，重新评估冻结030/033及新候选；这些仅是valid前向，不重训教师或重生成test。

最终报告Macro-AUC、十类AUC、1/4/8均值、固定.5的F1、单图/多图Study切片、患者聚类配对置信区间，以及训练资源成本。患者区间不能消除硬件/训练精度/选epoch/valid选参的混杂。若要评估组合，只允许一个预先登记的固定等权组合，不密集扫描、不拟合逐类融合。两个实验结束先交成绩单，等待用户决定下一阶段。

## 环境、原图与断点空间

环境恢复和原图核验的本轮回执集中在`artifacts/kd_v1/student_takeover_20261030/`；该目录不入Git。原图由用户的unzip恢复，本代理未重复解压、未干扰进程；观察进程结束后，train30328/30328、valid1422/1422全部存在且冻结文件SHA匹配，缺失/不匹配/读取错误均0。实际原图根直接是`/home/featurize/data/cxr_first`（含split子目录），不照搬旧双层路径。没有读取test图，test只保留既有清单/CSV身份检查。

本轮实际恢复通过：固定镜像digest `5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6`，Python3.12.3、torch2.10.0+cu128、torchvision0.25.0+cu128、CUDA12.8/cuDNN91002、NumPy1.26.4、Pillow10.4.0、timm1.0.30。当前GPU为2080Ti22GiB、SM75/68SM；非原生BF16，训练显式用FP16。镜像层在本地`/var/lib/docker`，没有写入work。

原生扩展已在该镜像内从当前源码force-build，`native_available=true`、`avx2`、最多8线程；原生7项像素/边界/通用回退测试通过。推理与学生环境实际import/CUDA能力检查通过，另完成一个基础FP16张量运算检查；没有构建模型或模型前向。学生探针首次因脚本位于artifacts而漏加仓库导入路径失败，修正探针后第二次通过，原失败日志保留。

本轮增加环境专用入口`run_student_environment.sh`：固定比赛镜像digest、离线网络、当前真实GPU驱动透传、4GiB共享内存；既有runs（含教师/OOF）、deliveries（含033/038）、项目数据清单和原图均只读挂载。调用无参数只显示用法，不会自动开始训练。新run只能显式指定本地可写输出，不能沿用旧run目录。

推理继续用原`run_in_competition.sh`及其固定8.55MB附加包，不改变当前推理环境。学生入口另加`/tmp/cloud_student_training_addons_cp312`，按030真实`environment_lock.txt`固定14项训练依赖，包括Albumentations2.0.8、OpenCV4.11.0.86、sklearn1.6.1、SciPy1.15.3。训练的sklearn/SciPy覆盖仅发生在独立PYTHONPATH，不覆盖镜像；torch/torchvision/NumPy/Pillow始终来自固定镜像。精确列表及文件SHA保存在本轮回执目录。恢复完成以实际import/CUDA/native回执为准，不能仅凭pip成功宣称可运行。

work约余1.2GiB，本地根盘有充足空间。Tiny完整best/last各约450MB；Base完整断点预计各约1.40GB，保留A/B加原子临时写入、导出与余量应在本地预留至少6–8GiB。不能把新run完整断点写进work或为腾空间删除旧资产。

本地根盘与`/tmp`不保证跨Featurize实例持久。训练前需明确本地run目录及完整断点的保留/迁移办法；小型日志、源码和哈希回执同步work不能替代完整恢复断点。用户尚未确认开训，本轮不创建新训练run或承诺持久断点已解决。

## 证据索引

- 交接：`AGENTS.md`、`AGENT_HANDOFF.md`、`FINAL_OPTIMIZATION_HANDOFF_20261030.md`、`NEW_ENVIRONMENT_HANDOFF_20261029.md`。
- 既有学生目标与来源：`train_distill.py`、`distill_tta_data.py`、`dataset_overnight.py`、`model_overnight.py`、`kd_common.py`。
- 历史成绩：`TINY768_SEED44_REVIEW_20261028.md`、`DUAL_TINY768_FP16_REVIEW_20261028.md`、`OOF_CACHE_REVIEW_20261027.md`。
- 新环境恢复结果另记本轮回执，不覆盖035/036/037/038或历史慢轮。
