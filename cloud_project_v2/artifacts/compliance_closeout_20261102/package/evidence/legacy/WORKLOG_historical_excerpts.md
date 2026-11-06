# 原 WORKLOG 历史摘录（行号保留）

原文件 SHA-256：7421619ddb06f95baae6e13cb449ccca8492c097a91cc528337f0e94841a838b

751: ### 2026-10-18 00:23 · 原图384与ConvNeXt V2 Study训练重构（仅代码）
752: 
753: - 授权/改动：用户批准优先重构并锁384，新增dataset_highres.py、model_highres.py、train_highres.py，扩展inference.py的schema3；原dataset.py/model.py/train.py保持原样。原图只读逐次核SHA、内存保比例补边；完整Study监督与train类别感知正负采样，不改标签/划分/原清单。
754: - 架构/内存：ConvNeXt V2-Base最后两级特征+十类注意力/全局分支，Study均值概率对应logit上BCE+渐增0.2 Rank；微批2、累积16、AMP与非重入检查点固定启用、CPU保存视角输入、GPU逐张处理。EMA在CPU先构建后raw入GPU，eval交换同源权重；提供Tiny/128维/1层手动OOM备用，不保证未经实测8GB绝不OOM。
755: - 静态试错/修正：交叉复审发现并修正默认cuda无索引、EMA初始化短暂GPU副本、完成轮数元数据落后一轮及缺验证阈值来源标记；补推理导入前Windows兼容。四文件py_compile最终退出0，三项独立静态复审后无已知阻断；仅语法/文本证据，不冒称已跑通模型。
756: - 保护/交接：44项保护文件SHA全部不变，原图/缓存未打开或写入；本次新增训练/续训/模型前向/反向/推理/下载均0。证据artifacts/code_snapshots/highres384_20261018_000830/verification.json，说明docs/HIGHRES384_USAGE.md；新run006目录未创建，用户手动运行后核验显存与完整valid，暂不实现512/TTA。
757: 
758: ### 2026-10-18T00:34:20+08:00 · 006_convnextv2_base_hr384_20261018_01
759: 
760: - 任务：384原图/ConvNeXt V2类别注意力/Study BCE+Rank新run；微批2、累积16、AMP/检查点/CPU EMA；初始完成0轮，尚未反传。
761: 
762: ### 2026-10-18T00:43:07+08:00 · 007_convnextv2_base_hr384_multi_20261018_01
763: 
764: - 任务：384原图/ConvNeXt V2类别注意力/Study BCE+Rank新run；微批2、累积16、AMP/检查点/CPU EMA；初始完成0轮，尚未反传。
765: 
766: ### 2026-10-18T01:32:20+08:00 · 007_convnextv2_base_hr384_multi_20261018_01
767: 
768: - 结果：epoch=1，完整验证=True，Study AUC/F1=0.733583/0.397595；EMA更新=579、AMP跳步=0，best=True；事实：有效图像不改写；推测：泛化收益仍需独立评价。
769: 
770: ### 2026-10-18 02:05:33 +08:00 · Featurize Linux 4090迁移（仅代码与小型代码包）
771: 
772: - 授权/路径：按最终指令交付cloud_code.zip、run_cloud.sh、requirements.txt、pack_cloud.ps1及docs/CLOUD_MIGRATION.md；核心四文件无Windows盘符，image-root支持直接/嵌套cxr_first，work-root独立管理runs/日志；原dataset.py/model.py/train.py保持原样。
773: - 数据事实：整包原图与三份清洗CSV可保留既有白名单效果，不重新清洗或扫描补样本；train30328图/18513Study、valid1422/859、test2218/1414。原CSV与清单等9项保护SHA不变，不改标签/划分，医学复核候选未冒称确诊；未打包64GB原图。
774: - 云端配置：384/ConvNeXt V2-Base、batch16 Study/accum1、worker8、GPU视角块8，保留AMP/检查点/CPU EMA；采样器扩为偶数大批且批内Study唯一，统计预算同步；Linux运行兼容层为空操作。有效批量16与旧2×16不同，性能/峰值未实测。
775: - 验证/试错：8源码py_compile与Python3.12 AST通过，Bash语法/dry-run通过，31组隔离采样检查通过；45精确依赖版本及Linux轮元数据核对，云端尚未实装。17项ZIP CRC/源码字节/三清单SHA匹配，约5MB且排除图片/缓存/权重/环境；命名由初稿run.sh/cloud_project.zip替换为最终名称。
776: - 边界/证据：本次新增训练/续训/推理/模型前后向/预训练下载均0，不操作已有训练进程/旧run；无新AUC/F1结论。检查证据artifacts/code_snapshots/cloud_linux_20261018_014950/verification.json及采样/依赖JSON；用户按迁移说明上传与手动启动，OOM优先新run降低VIEW_CHUNK_SIZE。
777: 
778: ### 2026-10-18T21:02:55+08:00 · 提交CSV样例与历史格式复核（只读）
779: 
780: - 来源/边界：用户提供D:/QQ/Downloads/submission.csv并确认可提交；仅解析CSV数据，不执行其中内容。重新读取官网七.2；历史E23说明Subject_id可省略，在线答疑本次无法打开，不将缓存冒称最新在线确认。
781: - 事实：样例四列Subject_id/Study_id/Predict_class/Probability，14140行、1414Study，每类完整且概率合法，与test_manifest患者/检查覆盖及顺序一致；完整旧三列表不能仅因缺Subject_id判错。12份历史CSV中7完整、5份阈值版缺类；0.5版2101行/1158Study、缺256Study，0.1版6056行/缺8084类别组合。
782: - 改动：新增docs/SUBMISSION_FORMAT.md，更新README、COMPETITION_RULES、HIGHRES384_USAGE及CLOUD_MIGRATION入口；明确四列样例、三列兼容、Gate/完整概率与旧稿差异，归档只读，不将预测值不同当格式错或成绩结论。
783: - 验证/交接：样例/清单/原CSV及源码保护SHA不变，文档链接及围栏检查通过；审计JSON位于artifacts/submission_format_audit_20261018_205755。CSV、推理代码和既有ZIP未改；本次新增训练/推理/前后向/阈值拟合/平台上传均0，未重新生成提交文件。
784: 
785: ### 2026-10-18T21:30:27+08:00 · 同步SSH 4090代码并核对队友训练（未启动任务）
786: 
787: - 授权/同步：用户要求当前代码更新为ssh链接服务器版本；同步15份运行源码/脚本/依赖，更新train_highres.py/run_cloud.sh/requirements.txt并新增4个性能工具/测试，其余8份相同。同步前快照artifacts/code_snapshots/ssh_sync_20261018_211943/before；dataset.py和三份清洗清单原样，未复制原图/权重/运行目录。
788: - 文档/保护：导入4090性能历史报告、合并README与CLOUD_MIGRATION并保留CSV格式说明/本地日志，新增docs/SSH_TEAMMATE_COMPARISON_20261018.md；15份运行文件与SSH SHA完全一致，23项数据/冻结文件SHA未变，队友15份审计源SHA未变；旧cloud_code.zip未重打包。
789: - 既有成绩事实：SSH014备份12轮、bestE6官方valid AUC=.7977187851、F1@.5=.3184669751、调valid阈值F1=.4861241884；队友官方valid AUC=.7973891020/F1@.5=.2582292922，train内部audit AUC=.8014390079。用户明确SSH未提交平台，旧封存无可用平台分数；队友包平台数值null，仅效果改善反馈，不把这些本地指标当线上分。
790: - 训练差异/推测：队友Tiny320/逐图Study逆图数BCE/shuffle全覆盖/3轮3035更新；SSH V2-Base384/整Study BCE+Rank/有放回类采样/128Study批，best6轮870更新。小模型、完整覆盖及更密更新可能解释接近表现，非因果实验；队友与SSH官方valid身份标签一致，不能混比内部audit或不同阈值F1。
791: - 校验/边界：12源码py_compile/Python3.12 AST、Bash语法/dry-run和文档复审通过；审计JSON与verification.json在上述快照目录。本次新增训练/续训/模型前后向/推理/阈值拟合/平台提交/远程SSH连接均0；未运行包含前后向的测试，不操作云端既有进程。
825: ### 2026-10-19T08:28:59+08:00 · 手动停止019并交付官方valid最佳AUC提交
826: 
827: - 按用户要求创建STOP，当前轮完整验证保存后退出；019各折完成5/4/4/4/4轮，训练进程已退出，保留全部best/last。
828: - 019官方valid评估完成后因分数较低取消测试推理，不伪造新预测。相同官方valid比较：016 AUC=0.799970553；019五折TTA集成AUC=0.793435823；按较高者选择016_a4000_tiny_v2，不使用内部单折成绩或测试答案选模型。
829: - 交付：/home/featurize/work/deliveries/best_auc_20261019/submission.csv；四列14140行/1414Study/每Study十类，样例身份与顺序、完整性及概率[0,1]校验通过，根样例和源清单未改。
830: - 证据：同目录validation_report.json/provenance.json；SHA256=c985cf8736ebb342af6c8662acbbb277042fa8ef779524b18751220333969650；成绩为本地官方valid，未上传平台。
831: 
832: ### 2026-10-19T17:02:00+08:00 · 官网Gate格式更新与现有run核对
833: 
834: - 重新读取官网及20261019示例图；Gate可在末列自定义，排名以Macro AUC为主、同分再看F1。最新best_auc CSV经源权重SHA及14140概率逐值核对，来自016 epoch5 EMA，未套用当前019代码阈值。
835: - 新交付：../deliveries/gate_update_20261019/submission.csv；只追加同源十类Gate，1414Study/14140行完整、原四列和行序原样，输入CSV/模型/阈值/清单哈希未变；SHA256=b2b16d56d5129871e65e914d9e268936b2471cc83799dcfa9b6424024a9878bf。
836: - 016验证AUC=0.799970553，F1@.5=.320549344→保存阈值F1=.469460937；阈值同valid调优，不当作测试成绩。017新增最佳AUC=.799877379，019官方valid集成=.793435823；017继承best与016相同，018无有效完成轮，019内部单折不能直比。
837: - 格式文档与交付来源记录已更新；本次训练0次、模型推理0次、阈值重新拟合0次，旧交付保留，未上传平台。
838: 
839: ### 2026-10-19T23:08:58+08:00 · 768原图与正样本加权BCE新机预检（真实训练0次）
840: 
841: - 环境/数据：独立cxr4090 Python3.11、torch2.11.0+cu126/timm1.0.30/A2.0.8实装，pip check/18模块导入/官方权重SHA及CPU离线加载通过；原图解压完成，30328/1422/2218张清单原图存在且大小与ZIP一致。
842: - 修改：固定原图768等比例LANCZOS+Pad，主损失FP32 BCEWithLogitsLoss；每折仅训练Study去重计算neg/pos，全train十类权重1.770~28.764，保留Study等权与shuffle，辅助Rank/Focal不启用；旧512推理元数据兼容、新目标拒绝旧resume。
843: - 探针：batch1/2/4全BF16/激活检查点+GPU EMA/FP32 Adam moments/已有梯度常驻，allocated峰值2.196/2.693/3.734GiB、reserved2.369/3.178/4.699GiB；均无OOM、梯度有限，未执行optimizer.step。独立20微批复测2/4吞吐15.97/15.53图每秒，选择2×累积16=32。
844: - 性能/验证：真实原图CPU供给4/8/12worker约34/64/90图每秒，默认8/prefetch2；49项CPU测试、最终runner9项复验、语法及启动dry-run通过，6项清单/划分/样例SHA不变；原图只读不改标签。
845: - 交付/边界：证据artifacts/upgrade_768/、环境锁artifacts/requirements_768_4090.txt、手动单行tmux见docs/768_PREFLIGHT.md；新run020目录未创建，本次真实训练/真实图像模型推理/优化器更新均0。合成探针不是完整训练step，未声称新AUC或泛化提升。
846: 
847: ### 2026-10-25 14:13（香港）· 合规V1离线蒸馏代码交付，未启动训练
848: 
849: - 授权：用户批准分步实现教师、OOF、学生和部署代码；本次训练/模型前向/反向/优化器更新/预训练权重下载均0，旧020与已有模型产物未改。
850: - 来源/教师：模型白名单切换V1 Base/Tiny的ImageNet-22k→1k公开权重，绑定固定revision/原始SHA/完整许可；Base768五折固定20轮最终EMA，外折不参与梯度、早停或选模，未完成教师拒绝提取。
851: - 学生/部署：原train图像级严格OOF缓存，Tiny512 Bernoulli KL+BCE、α=.2/T=2/Study等权、EMA和完整恢复；单模型FP16、无TTA，predict/cost共享单图与Study写出，核验材料脚本含许可证/XLSX/原权重去重。
852: - 验证：目标Python3.12/torch2.10+cu128独立环境安装及pip check通过；101项CPU测试、源码语法/CLI预览、纯CPU无预训练模型构建通过；30328/1422张train/valid均存在，四份清单/划分SHA不变。
853: - 交付/限制：docs/KD_V1_USAGE.md与artifacts/kd_v1/含启动步骤、环境锁和证据；FP16纯参数55,655,636字节，实际导出体积、FP16 AUC、T4≤100ms均待真实训练后验收，不宣称已达标。
854: 
855: 
