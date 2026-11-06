# SSH 4090版本与队友C0训练对照

核查日期：2026-10-18，Asia/Hong_Kong（UTC+8）。本次已同步本地代码，只读取现有源码、清单、JSON和训练记录；新增训练、续训、模型前后向、推理、阈值拟合及平台提交均为0。没有连接远程服务器判断实时运行状态。

## 先区分成绩的来源

按用户最新说明：SSH版本尚未提交平台评测；旧封存版本效果差，没有可用的平台分数；队友版本有成功提交和效果改善的反馈。旧版没有分数不等于AUC为0，SSH尚未提交也不等于线上效果不好。

队友包有明确的**本地验证AUC/F1**，但 [manifest.json](../队友代码/C0算法交接包_20261018/cxr_c0_team/manifest.json) 第14～16行仅记用户反馈改善，`platform_metric_value=null`、`platform_result_independently_verified=false`。因此没有可引用的具体平台AUC或名次，不能据现有材料判断队友与SSH的线上优劣。

## 主工程已完成的同步

来源：`ssh链接服务器/my_code_backup/cloud_project`。

- 15份运行源码、脚本及依赖配置与来源逐字节一致：更新`train_highres.py`、`run_cloud.sh`、`requirements.txt`；新增两个性能工具及两个对应测试文件；其余8份原本相同。
- 引入GPU尾块补齐功能、前台/后台失败日志和退出码保留，以及`socksio==1.0.0`代理依赖。云端脚本当前默认128个Study、128张GPU视角块、12 worker、累积1、补齐倍数16。
- `dataset.py`和三份清洗清单与SSH来源相同，未覆盖或改变。原图、缓存、权重、旧run、队友源码及两边的运行记录均未改写；没有把运行产物搬进本地主工程的runs目录。
- 导入了来源的 [4090性能报告](PERFORMANCE_20261018.md)，合并README和云端说明，保留本地CSV格式文档与既有工作日志。来源报告中的“014尚未启动”是压测交付时点记录；后来提供的014备份已记录12个完整epoch，不能将两者当作同一时点。

同步前快照、文件哈希及回滚依据在 [ssh_sync_20261018_211943](../artifacts/code_snapshots/ssh_sync_20261018_211943/)。`before/`保存已有目标文件原件，`backup_manifest.json`标明原先不存在的新增文件，`sync_manifest.json`记录每个目标与SSH来源的SHA256。

12份Python文件py_compile、Python3.12语法解析，以及run_cloud.sh的Bash语法/dry-run检查通过。没有执行性能压测或包含前后向的单元测试。既有`cloud_code.zip`仍是此前迁移快照，本次没有重新打包。

## 已保存的具体成绩

| 模型 / 评估范围 | Macro-AUC | Macro-F1@0.5 | 调验证阈值后的F1 | Macro-AP |
| --- | ---: | ---: | ---: | ---: |
| SSH 014最佳第6轮 / 官方valid | 0.7977187851 | 0.3184669751 | 0.4861241884 | 0.4256602970 |
| SSH 014备份末轮第12轮 / 官方valid | 0.7871682484 | 0.3823668983 | 0.4613849411 | 0.3999850602 |
| 队友C0固定3轮 / 官方valid | 0.7973891020 | 0.2582292922 | 未提供 | 未提供 |
| 队友C0 / train内部患者audit | 0.8014390079 | 0.2739849035 | 未提供 | 未提供 |
| SSH / 平台测试集 | 尚未提交，无分数 | 无 | 无 | 无 |
| 队友 / 平台测试集 | 未提供具体数值 | 未提供 | 未提供 | 未提供 |

直接证据：

- SSH [best_metrics.json](../ssh链接服务器/my_run_014_backup/runs/014_cloud_4090_max/best_metrics.json) 第193～195行为valid AUC、调阈值F1和AP；第306行为F1@0.5；第331行标明阈值来自同轮valid。第79行的train AUC为null，不要错读成模型没有valid成绩。
- SSH [history.jsonl](../ssh链接服务器/my_run_014_backup/runs/014_cloud_4090_max/history.jsonl) 保存12轮曲线；[status.json](../ssh链接服务器/my_run_014_backup/runs/014_cloud_4090_max/status.json) 记录截至2026-10-18 20:26:34完成12轮、最佳第6轮。它是备份状态，不证明此刻服务器是否还在训练。
- 队友 [valid_report.json](../队友代码/C0算法交接包_20261018/cxr_c0_team/records/valid_report.json) 第11、28、67行为官方valid AUC、F1及固定阈值0.5。
- 队友 [audit_report.json](../队友代码/C0算法交接包_20261018/cxr_c0_team/records/audit_report.json) 第11、28行为内部audit分数，患者bootstrap 500次的AUC 95%区间为[0.79133550,0.81070468]。这不是官方测试成绩。

双方官方valid是相同的859个Study、1422张图、228名患者；原valid与清洗valid的图像身份及标签逐例一致。队友审计记录的原CSV SHA和SSH清单SHA也分别匹配本地文件。可以在这一评估范围下对照结果，但训练集规模、选模及调参过程不同，不能将差值归因于某一个训练模块。

SSH最佳AUC与队友官方valid AUC只差0.0003296831，前者略高；不能据此声称任何一方有明显泛化优势。相同0.5阈值下SSH F1高0.0602376829。SSH的0.4861是在同一valid上拟合逐类阈值得到的调优成绩，不能直接对比队友固定0.5的0.2582，也不能当作独立测试F1。

## 真实训练差异

| 项目 | SSH 014实际配置 | 队友C0实际配置 |
| --- | --- | --- |
| 主干 | timm ConvNeXt V2-Base，FCMAE/ImageNet预训练，约9021万可训练参数 | torchvision ConvNeXt-Tiny，ImageNet1K V1，约2783万参数 |
| 分类头 | 最后两stage特征；十类query、两层Transformer decoder，加全局分类分支 | 标准全局池化及十类线性分类头 |
| 输入 | 只读原图，保比例LANCZOS补边384 | 512 JPEG95缓存，再双线性缩放补边320；可回退原图 |
| 训练数据 | 清洗train：18513 Study / 30328图 | 原始train的development患者：14810 Study / 24266图 |
| 数据留出 | 官方valid选模；train清单保留全部官方train Study | 原始train再留20%患者audit：2211患者 / 3703 Study / 6078图，不参与最终重训 |
| 监督 | 完整Study全部视角先求sigmoid概率均值，再算Study BCE + Pairwise Rank | 每图继承Study标签算BCE，乘该Study图片数倒数修正多图权重 |
| 类别损失 | Rank逐步增加至0.2；不使用ASL或pos_weight | 普通BCE；无Rank、ASL或pos_weight |
| 采样 | 一半类别感知正负配对batch，一半随机Study batch；跨batch有放回 | 图片随机shuffle，drop_last=False，逐轮遍历完整训练子集，无类别重采样 |
| batch / 累积 | 128个完整Study / 1；不是128张图 | 12张图 / 2，有效约24张图每次更新；不是24个完整Study |
| 实际更新 | 最佳E6共870次，其中首轮冻结骨干，完整骨干更新725次；E12共1740次 | 固定3轮共3035次，全程全网训练 |
| LR | 骨干3e-5、头3e-4；首轮冻结骨干、仅训练头部，全网warmup2轮，30轮余弦 | 骨干5e-5、头5e-4；warmup1轮，余弦周期10轮，实际只运行3轮 |
| 优化器 | AdamW，WD0.01；一维/bias不衰减，foreach/fused关闭 | AdamW，WD0.01 |
| 增强 / 正则 | 小亮度对比、±5°旋转、小高斯噪声；dropout0.2、drop_path0.1 | ±5°旋转、亮度对比0.1；dropout0.1，无高斯噪声 |
| EMA | CPU EMA0.999并带warmup，验证及best用EMA | 无EMA，固定第3轮final权重 |
| 选模 | 每轮官方valid；最佳AUC、并列看F1 | 最终训练无valid loader、无早停，固定3轮；轮数来自此前fold0，不是三折CV中位数 |
| GPU设置 | AMP、内外激活检查点、GPU视角分块补齐；确定性开启、benchmark关闭 | AMP、channels_last、benchmark开启、deterministic关闭，无激活检查点 |
| 推理聚合 | 全视角概率均值，identity | 全视角概率均值，identity；无EMA/TTA/集成 |

主要配方来源为SSH [config.json](../ssh链接服务器/my_run_014_backup/runs/014_cloud_4090_max/config.json)、队友 [实际训练配置](../队友代码/C0算法交接包_20261018/cxr_c0_team/records/training/config.json) 与 [summary.json](../队友代码/C0算法交接包_20261018/cxr_c0_team/records/training/summary.json)。以实际配置和history为准，不把源码默认参数当作已运行参数。

队友逐图BCE与SSH聚合后BCE不等价：前者对每张视角独立施加Study标签的损失，后者对整项检查汇总概率施加损失，两者对不同视角的梯度约束不同。两者最终评估都使用Study内概率均值。

## 划分、清洗与公平比较

队友分区完整来自官方train，没有把官方train+valid合并重划。development与audit患者相交为0，与官方valid患者/Study相交也为0。官方valid的标签和覆盖一致，不存在因验证集换成更容易的一批患者就可解释分数的证据。

队友没有使用本工程清洗CSV，其development包含15张本工程排除清单中的图片。该项只说明数据纳入不同；本工程排除原因是同train Study精确重复和技术隔离，不应将纳入差异直接当作队友分数来源。本次未重新做队友全量图像内容重复审计，也不据患者不重叠宣称所有潜在泄漏通道均已排除。

队友0.801439来自train内部audit，SSH训练清单包含这些患者；因此不能把该audit与SSH官方valid分数放在同一排行榜上比较。队友包也明确audit已使用过，后续调整后不得继续把它称为全新盲测。

## 为什么队友的简单方案能取得这样的表现

已知结果是队友用较小模型取得了接近SSH的官方valid AUC，并有用户反馈的线上改善；不是已证明队友在平台上胜过SSH。以下是由源码及记录推导的合理机制，未经过控制变量实验验证：

1. **“3轮”包含更多优化更新。** 队友3轮做3035次全网更新；SSH最佳6轮只有870次总更新、725次全网更新。小图片batch与大Study batch在学习步数和梯度分布上差别很大，不能只按epoch或显存利用率衡量训练充分程度。
2. **每轮完整覆盖，区别于有放回采样。** SSH E6共18560次Study抽样，实际只覆盖11539个不同Study，约62.33%；队友每轮完整遍历其24266张训练图。类感知采样提高部分少数类曝光，但也改变重复率和自然标签分布，未保证在相同步数下取得更高AUC。
3. **较小的模型、较简单的头更容易优化。** Tiny约2783万参数，SSH约9021万并增加注意力头；较少的新增参数及温和增强可能有利于小数据泛化。现有结果不能证明注意力头、Rank或EMA无效，也不能证明某个复杂模块是下降原因。
4. **较早固定结束可能避免继续过拟合。** 队友固定3轮，SSH备份从E6 AUC0.79772降至E12的0.78717；E10无增强训练诊断AUC约0.94497，而valid约0.79412。SSH后续继续拟合并未转化成更好的验证AUC。相应比较应保留best，不能只拿末轮与队友final比较。
5. **完整概率输出也是线上效果的必要条件。** 队友提交保留1414×10行。此前查到的旧阈值过滤CSV缺类，确有格式/覆盖问题；修正输出可能影响提交结果，但没有匹配的平台回执或同模型对照，不能将用户反馈改善直接归因于这一点。

队友也没有解决全部少数类问题：其官方valid第1、3、4类F1@0.5仍为0，第2类约0.05263。没有具体平台高分、0.95 AUC或少数类全面改善的证据。现阶段最值得重视的是两套方案在模型复杂度、完整样本覆盖和实际更新次数上的差异，而不是把更大分辨率、更大batch或更多模块视为提分保证。

独立审计记录：[队友训练与成绩](../artifacts/code_snapshots/ssh_sync_20261018_211943/teammate_training_review.json)、[SSH运行与源码一致性](../artifacts/code_snapshots/ssh_sync_20261018_211943/ssh_training_review.json)、[划分与评估可比性](../artifacts/code_snapshots/ssh_sync_20261018_211943/split_comparability_review.json)。SSH运行provenance中7份源码SHA和train/valid清单SHA均与提供的来源一致。
