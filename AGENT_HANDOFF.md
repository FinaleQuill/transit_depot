# 当前接手：教师冻结，学生上限提案待用户确认

2026-10-30 13:11，Asia/Hong_Kong。用户最新任务优先：不重训/修改教师，直接复用024现有严格OOF；暂不以推理100ms限制筛选学生提分实验。**先恢复环境、核验资产、提出1–2项实验，等用户确认才训练。当前确认尚未收到，本轮训练0次。** 详细配方及边界见[学生上限提案](docs/STUDENT_UPPER_BOUND_PLAN_20261030.md)。

- 当前会话已重新拉取并核验固定digest镜像，恢复802文件推理addon及按030版本锁隔离的14项训练依赖；Python3.12.3/torch2.10.0+cu128/CUDA12.8等实际import通过，2080Ti22GiB FP16基础运算通过。native在镜像内重建，AVX2/最多8线程、7项CPU一致性测试通过。新窗口仍须重新核验，不能据本次状态假定环境持久。
- 038的474项manifest文件、ZIP SHA/CRC通过；038目录、ZIP、SHA回执均已去除写权限。033权重/config/原CSV、清单、OOF哈希匹配；当前10个推理关键源码文件前后SHA相同，未回滚81ms代码。本轮未构建模型、模型前向、重生成test、上传或删除资产。
- 用户自行解压官方ZIP后，train30328/30328、valid1422/1422原图SHA全匹配，test图未读取。**本机原图根直接为`/home/featurize/data/cxr_first`，含train/valid/test子目录，不是旧双层根。** 原图恢复核验新回执替代接手前缺失快照，不改旧记录。
- 新增环境入口`run_student_environment.sh`，不替换正式推理入口：离线、4GiB shm，既有runs/deliveries/数据清单/原图只读，新增断点输出须显式写本地。work约余1.2GiB；A/B完整断点需本地6–8GiB及跨实例完整备份/迁移办法，小日志不等于完整断点持久化。尚未创建训练run。
- 预提案A：Tiny768平方根正类权重（P2，同步KD先验对齐）；B：原始合规Base初始化的Base768学生＋完整Study硬监督P3，固定平方根权重和逐图OOF KL。均需新训学生；不重提教师特征。A约原Tiny同卡开销，B暂估2.5–4倍，2080Ti首轮后校准，不能把新旧GPU/精度差异归因单一配方。
- 本轮完整证据`artifacts/kd_v1/student_takeover_20261030/restoration_summary.json`，含实际探针/原图核验/权限及源哈希。仅真实T4的`cost_report.json`仍待，2080Ti恢复不构成T4认证。

---

# 前阶段交接：81ms等值优化已完成，完整项目待T4复测

2026-10-30，Asia/Hong_Kong。**顶部最新状态与[完整交接](docs/FINAL_OPTIMIZATION_HANDOFF_20261030.md)优先；后文10月29日“默认v1、确认前不实施、尚未达100ms、cost/seed待修、最终包未组装”均为历史，不是当前指令。**

- 已完成默认原生AVX2/generic等值LANCZOS、精确uint8→FP16查表与一次D2H；同环境本地v1 **124.754857→81.509946ms**，NFS **80.781137ms**，完整项目副本独立离线重建 **82.190887ms**。原100图/另5图预热、batch1串行三轮全部保留。
- 105图输入逐bit相同；100图/53Study的三轮逐图/Study概率与冻结v1逐字节相同。033权重/config/FP16/TTA/25-75融合/原test及逐图CSV不变，未训练、未重生成test或上传；本阶段未重跑完整valid，不冒称已跑。
- 当前实现提交65064d5、d539db7；结果提交0dd71a1。新证据037/optimization_summary.json；完整项目038_dual768_optimized_project_20261030及同级ZIP/SHA回执已保存，实际推理文件128,766,272bytes。
- **唯一项目待办为真实T4 cost_report.json。**共享seed43、100逐图/Study语义检查、原sidecar入包及7个真实训练run证据已完成；历史恢复局限保留。官方100图若不闭合Study仍pending，不偷偷补齐。用户最后提交项目，由比赛环境运行，不自动要求租卡/迁服务器，不把2080Ti写成T4认证。
- 新窗口环境可能丢失：按[最新完整交接](docs/FINAL_OPTIMIZATION_HANDOFF_20261030.md)自动恢复固定镜像/依赖/native，旧610驱动挂载和tmp helper不可盲用。源码根无vendor，使用038归档参数；038独立验证在本地副本prepare，原包只读。work仅约1.2GiB，不在work复制大包/镜像，不下载全量数据或再下载包内初始化原件。
- 不重做旧v2/v3/soup或已完成基准；不训练、改预测语义、重生成test、上传或删资产。先必要恢复/只读核验，后续范围以用户新消息为准。

可直接转发：[新agent话术](docs/NEW_AGENT_PROMPT_20261030.md)。用户本次只要求保存交接，本轮无新GPU实验。

---

## 以下为历史交接记录（截至2026-10-29，仅保留原时点语境）

## 2026-10-29新会话：诊断完成，优化实施待确认

按最新用户“先汇报、确认后动代码”完成新环境恢复及固定105图诊断，生产源码未改。v1本地盘三轮120.713995ms；NFS156.409450ms，首轮明显长尾，全部保留。CPU瓶颈JPEG30.956ms/LANCZOS48.954ms；100图GPU路径39.107ms、四前向Event36.650ms。六轮及诊断概率与旧v1逐字节相同。详见[诊断及T4迁移方案](docs/DUAL768_DIAGNOSIS_AND_T4_PLAN_20261029.md)，新证据036，旧035不覆盖。

用户补充必须迁移T4生效。后续按固定镜像、通用CPU指令/能力分派、SM75、最多8核单图内部并行设计；最终须真实T4同机对照、≤100ms及概率/二值/排序验收。当前无T4，不能保证已生效。优先候选是保持Pillow系数与两遍舍入的原生并行缩放（目标≤15ms），再LUT/合并回传；确认前不实施，不训练、不改后处理策略、不重生成test。新窗口仍须核实/恢复临时环境。

## 新窗口优先：环境必须重新恢复

用户明确Featurize新窗口会是新环境、已配置环境会丢失。**不要把下文“推荐镜像可用/环境已就绪”当成新窗口现状。** 先按[新环境完整交接](docs/NEW_ENVIRONMENT_HANDOFF_20261029.md)核对work资产、重新拉取固定镜像、从持久化8.55MB小包恢复附加依赖，并适配新机GPU驱动挂载，再做import/CUDA探测。旧`/tmp`环境、Docker层、原图及zip均不能假定存在。

默认保留v1，不训练、不重生成test、不重复已拒绝实验。恢复后优先新环境下NFS/本地盘I/O对照；105张图已在034包，不需先恢复全量数据。重复读取和核验的子agent使用`high`，避免过度思考。新环境实测、日志及准备回执另存work，不覆盖旧记录。

## 本轮结束状态（优先于下方执行中描述）

已按用户节省token要求停止新实验，默认部署源码为v1等值优化，推荐镜像同环境149.807432→133.443058ms（10.92%），仍未达100ms。完整valid8590概率逐字节不变，AUC .8083852425302762，F1 .4510899186168643；镜像45项CPU测试通过。v2并行更慢已拒绝，v3 CUDA Graph 131.120745ms仅实验保留（额外synthetic初始化口径待核验），参数平均候选AUC .803611602已拒绝。未训练、未重生成test、未上传，GPU任务全部结束。

旧环境已实际验证入口`./run_in_competition.sh`，新窗口须先恢复环境并检查其硬编码驱动挂载；版本/恢复材料见`artifacts/kd_v1/optimization_2080ti_20261029/environment/`。本轮全结果035目录，详见[实际结果与下一步边界](docs/DUAL768_OPTIMIZATION_2080TI_20261029.md)。不要重复已完成基准或误将任何代理卡结果写成T4通过。

## 最新用户授权与状态（2026-10-29，香港）

用户确认2080Ti原样测速已完成，明确**不返回4090，直接在2080Ti降低耗时、研究提分；先不要训练**。新授权替代下方历史“只测速/返回4090后优化”的边界。当前可准备比赛隔离环境、实施推理优化、运行CPU/GPU验证与计时；不自动训练、上传、重生成test或覆盖033和旧证据。

- 原样2080Ti三轮均值138.538797ms/图，CPU预处理独立诊断98.709547ms、四前向Event跨度36.356393ms；正式结果在`../dual768_proxy_timing_results/REPORT.md`。该目录95项结果SHA本次复核通过。
- 同机三轮预测完全一致；相对4090有第3类一对跨Study排序反转，不能声称跨卡语义完全一致或T4认证。
- 当前实施顺序、环境与验证结果统一见[2080Ti优化记录](docs/DUAL768_OPTIMIZATION_2080TI_20261029.md)。先保持033精度做等值降时，提分候选需独立valid核验，不密集扫融合权重。
- 用户正在恢复全量数据；旧conda环境不存在，平台全局Python3.11/torch2.2不符合比赛核心版本，不能拿它执行正式比较。新环境准备和镜像层放本地根盘，work仅余约2.4GiB。
- 下方历史测速命令与快照保留原时点；本节及优化记录优先，不再重复原样测速迁移或要求用户返回4090。

---

# 历史交接：033双Tiny768异卡原样测速

更新时间：2026-10-29，Asia/Hong_Kong（UTC+8）。初版交接前Git基线为`87dcf68`，本次任务前为`d8bf7e8`。本文件供没有前文聊天记录的agent接手；下方历史模型状态继续保留，**顶部本次测速范围优先**。

## 0. 本次任务是测速，禁止顺手优化

用户没有T4且租不到，同一时间只租一台Featurize服务器。当前仅授权准备并在另一张卡上测**原样033双768**；建议先保存/结束4090会话，再租一台RTX2080Ti 22G，完成后带回结果、返回4090，由原代理判断是否及如何优化。3060仅在2080Ti不可用时备用，不要求两台都测，不同时购买。

用户已明确要求做好必要准备，并让新agent自动完成目标机环境检查/隔离环境准备、输入校验、原样GPU测速及结果保存；这些步骤无需重复询问确认。仍不得优化、训练、换模型、上传、购买或归还实例。**新服务器不需要重新下载/解压全量数据集，包内105张图已足够本任务。** 可直接转发给新agent的话见[自动测速任务文本](docs/DUAL768_NEW_AGENT_PROMPT_20261029.md)。

**测速agent先读[仅测速操作交接](docs/DUAL768_PROXY_TIMING_HANDOFF_20261029.md)。不要从下文历史命令启动训练、改Loss/增强/CLAHE/DALI、换模型/TTA/精度/融合权重、优化推理、重做test CSV或上传。** 本轮也不修cost或训练证据工具，只记录其历史待办。

- 唯一对象：033的029/030真实两seed，25%/75%、768、FP16、每模型原图+hflip、四次串行前向。基线4090端到端79.280300ms，固定100张valid图/另5张预热/三轮，不是只测GPU。
- 精确源码、033权重/config、105张图、原valid manifest及4090参考在工作区`deliveries/034_dual768_proxy_timing_20261029/dual768_timing_only.tar.gz`；不是旧031便携包。包实测266,803,659bytes，SHA256为`ae2703c82d5d704b4fe7ed5a6d2702c14b2cca2e14673c4365ee2f0097b5b7a4`，132文件压缩内容逐项核验及CPU元数据预览通过，详见同目录`packaging_verification.json`。此包不等于完整训练资产备份。
- 原样用`benchmark_local_resources.py`，显式8GiB allocator上限、真实卡SM、seed43，三轮无插桩基准先跑；可选分段诊断另进程。不加sleep、不启用MPS、不调功耗/时钟，不把倍率当T4认证。
- 迁移只派生selection的manifest_path，保留原件/图片/manifest字节。报告既要同机三轮一致性，也要与包内4090参考的概率、固定.5二值及排序核对；保留所有慢轮/异常。
- 工作区Git不含权重/数据。原始数据约52GB在`/home/featurize/data/cxr_first`、不属于work；切换实例前确认可恢复。当前旧conda Python路径实际不存在，目标机先核对Python3.12与固定依赖，勿照抄旧路径。原work约余2.9GB，不在这里下载4.607GB镜像或删除训练资产。
- 结果保存到持久work，带回`local_resource_report.json`、三轮Study/逐图CSV、日志/退出码、环境/CPU/GPU/磁盘及一致性报告；测完停止并返回4090再分析。当前**尚未在2080Ti/3060测速**。

2026-10-29只读网页快照推荐RTX2080Ti 22G（8核E5-2686 v4、0.69元/小时）；可用量/价格以[购买时页面](https://featurize.cn/vm/available)为准。agent不得自动购买、归还实例或读取凭证。简单重复核验用medium，复杂分析high，不用ultra。

## 1. 当前结论与用户意图

用户目标是在官方规则和可交付约束内尽力提高成绩，已明确以**两个ConvNeXt V1 Tiny768学生、固定TTA、概率加权集成**为当前主线。

**当前033已完成：真实部署组装、完整valid实际FP16验证、4090三轮端到端计时、完整测试CSV生成和独立完整性审计。教师与学生训练均已结束。**

| 状态 | 交接时事实 |
| --- | --- |
| 当前部署 | 029 seed43 E8 EMA占25%，030 seed44 E8 EMA占75% |
| 输入/精度/TTA | 两个768；FP16；每模型原图+hflip，共四次前向 |
| valid AUC | **0.8083844422198005**，859 Study |
| valid F1@.5 | 0.4510899186168643 |
| 实际推理权重+config | **128,766,272bytes**，小于500,000,000bytes |
| RTX4090端到端 | **79.280300ms/图**，真实双seed、100图/5预热/三轮 |
| 完整test CSV | **已生成并校验**，2218图、1414 Study、14140数据行 |
| 平台 | 代理未上传；未收到平台接收或测试成绩反馈 |
| 正式T4/最终核验包 | **未完成**；当前没有可用T4或已配置的T4连接 |
| CLAHE/DALI | **未训练、未应用、未安装DALI** |

“未上传”是本代理的执行记录；用户若随后手动上传，以用户最新反馈为准。不要把valid AUC当成测试成绩，也不要将4090时间写成T4认证。

此前“代执行”授权对应的部署、验证、4090计时和CSV步骤均已完成。本次最新授权仅为顶部所述异卡原样测速交接；不能据旧授权自动启动新训练、优化、重复test预测或上传。

## 2. 接手必读与协作方式

先读本项目的`AGENTS.md`、`README.md`、`docs/CLEANING_POLICY.md`、`docs/COMPETITION_RULES.md`和`WORKLOG.md`。本项目是`/home/featurize/work/cloud_project_v2`，不要误读工作区根目录的旧WORKLOG或旧工程记录。

- 用户要求每个大阶段完成后核验：总体/十类AUC、患者不确定性、误差切片和资源成本，反馈改进判断后再进入下一阶段。无测试标签时只做交付完整性核验，不做测试效果分析。
- 用户通常需要可直接执行的分阶段命令，也会明确要求agent代跑；以当前授权范围行动，已明确授权的动作不要反复确认。长训练使用tmux，运行中不改其执行源码/配置。
- 重复、技术要求较低的核验任务可委派`gpt-6.1-sol`、`high`，不必用ultra；模型覆盖的子代理使用`fork_turns="none"`并传足够上下文。
- 每项逻辑修改用Git保存。模型、数据、缓存、日志、证据、原始PDF等被Git忽略，不能盲目`git add .`、改全局Git身份或推送远程。
- `WORKLOG.md`只追加关键事实/判断，每块3～6条；它有混合历史换行，新块按CRLF字节追加，保留此前字节前缀。检查可用`git -c core.whitespace=cr-at-eol diff --cached --check`。
- 本机无`rg`；用Python、`find`等替代。不要读取/展示密钥或凭证。

## 3. 最重要的路径

下表均为服务器实际路径。Git克隆只包含代码和记录，**不包含**runs/deliveries/data/artifacts/原图/预训练缓存；异机接手需另行保留这些文件及环境。

| 内容 | 路径 |
| --- | --- |
| 代码/Git仓库 | `/home/featurize/work/cloud_project_v2` |
| 当前CSV | `/home/featurize/work/deliveries/033_dual_tiny768_seed43_25_seed44_75/test_predictions/submission.csv` |
| 当前部署目录 | `/home/featurize/work/deliveries/033_dual_tiny768_seed43_25_seed44_75/fp16_hflip2` |
| 当前valid结果 | 同033根目录的`valid_fp16_hflip2/` |
| 当前4090计时 | 同033根目录的`latency4090/local_resource_report.json` |
| 学生源训练 | `/home/featurize/work/runs/029_tiny768_kd_seed43`、`030_tiny768_kd_seed44` |
| 教师集合 | `/home/featurize/work/runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_collection.json` |
| 严格OOF缓存 | 同024根目录的`oof_logits_tta_v2/{soft_labels.npz,metadata.json}` |
| CLI原图根目录 | `/home/featurize/data/cxr_first` |
| 实际原图/原CSV根目录 | `/home/featurize/data/cxr_first/cxr_first` |
| 原规则PDF | `/home/featurize/work/rules.pdf` |
| 最新补充及官网原件 | 项目内`artifacts/kd_v1/submission_rules_recheck_20261028/` |

**不要拿`/home/featurize/work/submission.csv`上传：那是10月18日旧文件。** 工作区的旧zip/tar、`archive_v1/`、旧V2/Tiny文档同样不是当前模型。

CSV采用UTF-8四列：`Subject_id,Study_id,Predict_class,Probability`。1414个Study各十类，14140数据行，另有表头；无自定义Gate，默认.5，不删低概率行。文件686443bytes，生成约186.41秒。

同目录`submission.image_predictions.csv`有2218×10条逐图概率，只作复现/正式cost核验辅助；`submission.csv.report.json`是生成报告。提交预测文件使用`submission.csv`。不从test预测分布反推类别先验、调权重、阈值、归一化或预处理。

## 4. 当前交付的身份校验

```text
033 config.json:
9363149e143dec2871ac5fb4fbfdee4885f9f62add6a9f2acf4704635311b590
033 student_00_fp16.pt（029 seed43，25%）:
edef1f23e591bb52782a03efdc6ee76e10b4d9f115e90654e40f7dbf35f50269
033 student_01_fp16.pt（030 seed44，75%）:
87ed41a966cf96144cc63acf50613a14c66fe8497b51625f120389a97a563489
033 test_predictions/submission.csv:
ea83de14afb8a8438428dc071b29438b2201a7113cad03aa4181de09284b3661
033 valid_fp16_hflip2/valid_predictions.csv:
7000ce912c0c285fb36ca251897866a314b9c0d33b21c8ed1d92edc0f11a3f94
```

两份推理权重各55,712,511bytes、182个有限FP16张量。完整valid实际预测与保存概率的25/75数学融合最大差5.55e-17，十类排序/并列及AUC一致；计时三轮各53个完整Study与完整valid对应预测误差0。

生成test前后53个受保护文件SHA不变，2218张原图SHA与冻结清单一致，逐图到Study独立聚合误差0。审计未分析测试疾病分布、未读答案或做参数选择。

## 5. 环境与数据

历史训练/033执行Python：`/home/featurize/.conda/envs/cxr-kd-v1/bin/python`，历史版本Python3.12.15、torch2.10.0+cu128、torchvision0.25.0、timm1.0.30、numpy1.26.4、Pillow10.4.0。**2026-10-29当前会话未找到该解释器**，不能把历史环境当成当前已验证可运行；目标机先确认Python3.12和固定推理依赖。推理见`requirements_inference.txt`，训练见`requirements_kd.txt`；不覆盖平台全局CUDA或随意升级。测速不需要训练依赖。

当前仅RTX4090 24GB，16逻辑CPU；交接核对时GPU空闲、无训练/预测/cost/MPS进程，工作盘约剩2.88GiB。新任务先检查实时状态和磁盘，不能据此快照保证以后仍空闲。两学生best/last训练断点各约450MB，含优化器，不是推理体积；保留证据，不为省空间随意删。

| 清单，均在项目`data/clean_v1_reviewed/` | 图 | Study | 患者 |
| --- | ---: | ---: | ---: |
| `train_clean.csv` | 30328 | 18513 | 11055 |
| `valid_manifest.csv` | 1422 | 859 | 228 |
| `test_manifest.csv` | 2218 | 1414 | 136 |

valid从未并训。原图/原CSV只读，不改标签，不无依据补病种或强制No Finding互斥；历史清洗/人工复核见CLEANING_POLICY，不能按模糊度阈值自行删真实胸片。

```text
train_clean.csv: f9a06316cfa8ab10ad4488e60a2dcf934cbc942727edd9ad2c31689823db6aef
valid_manifest.csv: c136b2d9f2646fdfa65e0db91d24b1ab5a021e7b038dcaf243855b8779bb5112
test_manifest.csv: fdfd651ff0663b3dd6f232ecbb0cd32949d78092e162ae2595b4563cdbe224f3
data/cv5_overnight/folds.json: e541696e1418be7848a328a65f12e2a0df1874a0dcc00d506eee581be1eefe67
```

## 6. 教师、预训练与蒸馏逻辑

骨干均为ConvNeXt **V1**：教师`convnext_base.fb_in22k_ft_in1k_384`，学生`convnext_tiny.fb_in22k_ft_in1k_384`；初始化是ImageNet-22k→1k公开权重，模型卡Apache-2.0、上游MIT材料已留存。旧V2 FCMAE的NC来源不得复用。

| 原始初始化 | 固定revision | 原权重SHA256 |
| --- | --- | --- |
| Tiny | `32c741ba106f9e062174ef829354a16719c3dcac` | `a5cbe10cfa2e90ec787a082253033ac0a61de0264b68210815a988e53a1a5d7c` |
| Base | `2b2698d86523262ca8065d0957c7f80d963bb43f` | `bc0cfafc7d7755e8271f51343c4374725bf6409c0434e64028a863c4aca7d9fe` |

原件位于`/home/featurize/.cache/huggingface/hub/models--timm--<backbone>/snapshots/<revision>/model.safetensors`，Tiny约114MB、Base约354MB。每个run的`pretrained_provenance.json`和OOF metadata保留来源及许可全文。新训练的首次HF查询当前仍可能跟随main，后续若需严格对照应使用已核验固定来源；上一轮用`HF_HUB_OFFLINE=1`读取本机缓存，独立许可证URL仍联网。不要默默换成远端新版本。

五教师均768，采用已冻结的20/12/12/12/12轮最终EMA：fold0来自`runs/021_convnext_v1_base768_teacher_cv5/fold_0/best.pt`，fold1–4来自`runs/023_convnext_v1_base768_teacher12_cv5/fold_N/best.pt`，路径均相对工作区根目录。fold0 E13仅历史诊断，不是当前教师；不重训/替换现成教师或用OOF标签挑epoch。

严格患者OOF：每张train图只让未训练其患者的对应外折教师预测，不能把训练见过该患者的其他教师混入。缓存30328×2×10，identity+hflip，保存FP32原始logits（前向BF16），未预先温度化/对齐；不含valid/test。当前两个学生用完整train，**不是学生CV两折或五折**。

```text
OOF soft_labels.npz SHA: 573d0bf0f846f7f802cbb536a67ba640d209fae9d448aa72e3e8a31e2894b6c5
OOF metadata.json SHA: 39b8fd6014b989e7eb8905303410ac963ece1bb985816e29a52253a6b2c9e2bd
```

学生α=.2、T=2，独立Bernoulli KL（含正/负项、乘T²）+加权硬BCE，前两轮渐进启用KD、Study等权。每视图先按`teacher_logit - log(teacher_pos_weight) + log(student_pos_weight)`对齐，再除T、sigmoid后平均软目标；十类不用softmax。详见`train_distill.py`/`distill_tta_data.py`，不要重复对缓存应用温度或把概率当logits。

029/030只改seed/output，其余相同：batch8×accum4、有效batch32，25轮余弦上限、patience5/min_delta1e-4、LR5e-5/5e-4、warmup1、min_ratio.05、WD.05、dropout.2、droppath.1、EMA.999、grad_clip1、训练BF16、valid hflip2。两者均13轮早停、E8最佳。best保存任意真实AUC提升，早停参考只在超过min_delta时重置，二者不要混淆。

## 7. 实验结论：哪些结果可比较

| run | 主要设置/最佳轮 | 原生BF16最佳AUC | 实际FP16+hflip2 AUC |
| --- | --- | ---: | ---: |
| 026 | Tiny512独立BCE，E6 | .803596642 | .803634677 |
| 027 | Tiny512 KD seed43，E7 | .805593834 | .805402258 |
| 028 | Tiny512 KD短12轮日程，E7 | .803890563 | 未额外验证 |
| 029 | Tiny768 KD seed43，E8 | .804176963 | .803983596 |
| 030 | Tiny768 KD seed44，E8 | .807569817 | .807512423 |
| 033 | 029/030概率25/75 | 不适用 | **.808384442** |

已有参考：512+768各50%=.807157435；双768各50%=.808370073；旧43占75%/新44占25%=.806951639。后两行和033只来自预登记三组全局权重，不是密集搜索。25/75与50/50只差.000014369，不宣称25/75显著更好。

033比单44+.000872019，七类升三类降；患者聚类配对1000次、seed20261027的95%区间[-.000714399,+.002738017]跨0。实际部署与数学融合排序/并列一致，沿用该区间而非再跑bootstrap；不修正seed/选epoch/多次比较/valid选参偏差。不能宣称稳定提分，也不能断言全是噪声。

短日程未见改善，不继续围绕余弦长度反复试。后期loss下降而AUC退化，主要稀少类1拖累，并非所有类都下降或数值崩溃；不要据loss振荡任意改alpha、换ASL或恢复已早停run。详细逐类/曲线见各阶段复盘。

## 8. 推理、共享TTA和速度的真实含义

预处理保留EXIF、完整画面、等比例LANCZOS letterbox及ImageNet归一化。已实现灰度先缩放后展开RGB等值提速，同图共用解码/768数组和GPU原图/翻转输入；仍要两模型各两次前向，不能共用不同权重的CNN特征。没有跨图预测缓存，没有CLAHE。

033在4090三轮均值79.375028/79.242428/79.223445ms，总79.280300；每图三轮均值P95=88.537651、max=91.263682，300次原始观测max=91.916501。无复测/删慢样本；模型初始化排除，读图到写CSV包含。原100图来自valid的53完整Study/40患者，另5预热图4Study/4患者。

4090/MPS/16GiB分配器上限均不是T4模拟器。旧`deliveries/031_resource_stress_20261027/portable_t4_diagnostic`包含旧候选及同run E8/E13代理，**不能直接当033最终材料**；迁移需使用033真正两seed和同版源码。当前GPU训练/推理都已结束。

## 9. 官方约束与最新补充

以[官网](https://www.ncccu.org.cn/index/Paper/case1.html)所链[原规则](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf)及[10月27正式补充](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261027/b23d6a1f19602da2a9bd4048108175d0.pdf)为准，优先于FAQ/口头说法。

- 非医学、公开可再分发、非NC/非gated的预训练要求同时约束教师和学生；只用官方数据，不引入外部医学数据。补充已明确仅官方train自监督允许，旧文档的概括禁令已过时；本项目未采用。
- 固定TTA/集成、无拟合的确定性聚合允许；test不得训练、伪标签、估计先验/归一化或用于模型/阈值/权重选择。当前CSV只用于最终预测，之后的实验决定仍只能来自train/valid。
- 全部实际推理权重与必要配置≤500,000,000bytes；不参与最终推理的教师不计，但须如实申报来源和蒸馏关系。
- 正式T4按100图、batch1串行、另5图预热、三轮、读图至写出平均≤100ms。补充要求样本二值/Gate及概率排序一致；|Δp|≤1e-3仅在语义一致时允许，不能只放宽allclose就称合规。
- 超时正式整改阶段不能换权重/改变预测语义。主AUC先保留四位小数再排序，同分比F1，再同分比提交时间。不要据本次test输出优化这些项目。

原规则SHA：`ff6aa3adbfd3d000194253055a93dbce8a140031c8432f892046a49de8136fb6`；补充SHA：`836c4f5d3202f2dc65830fd919ca195797c12b7507d750121786a21672dc7a44`。原件、来源和时间在`artifacts/kd_v1/submission_rules_recheck_20261028/`；详细边界见`docs/COMPETITION_RULES.md`和`docs/OFFICIAL_RULES_CHECKLIST.md`。

## 10. 明确尚未解决的P0待办

**A. 正式cost与新补充的一致性要求。** `cost.py:compare_predictions`当前只做`np.allclose(rtol=0,atol=1e-7)`，未显式核验样本Gate二值/类别排序/并列。裸`predict.py`、`cost.py`及Predictor也未统一显式seed；此次CSV的seed43来自归档`artifacts/kd_v1/dual768_submission_20261028/run_prediction.py`包装器。不能声称普通CLI已实现相同seed入口。后续需按规则完善一致性检查、随机种子及有意义测试，保持现有模型和CSV语义，不覆盖原执行源码证据。

当前cost认证还要求100图包含所选Study全部视图；不闭合则一致性pending/exit2。若官方样本不闭合，须按正式样本口径核对方案，不能读取目录外预测偷偷补齐或把pending改成通过。保留最终`submission.image_predictions.csv`作核验依据。

**B. 最终教师/双学生训练证据。** `/home/featurize/work/runs/024_convnext_v1_base768_mixed20_12_cv5/evidence/evidence_manifest.json`现含五名完整教师及废弃022 student（complete=false，旧config/账本路径已不存在）。`build_training_evidence.py`预览会早返回；实际执行拒绝teacher-evidence含任何student。不能照旧路径预览成功就宣称能合并。需从真实材料派生**新版本teacher-only索引**，保留旧原件，再按029、030顺序合并学生证据。

教师日志亦需核验：fold0 recovered/live各自partial，已有带说明的group声明组合覆盖完整；保留“tmux恢复非原始stdout字节”“E1精确mtime未留”等事实。fold1–4主要关联controller日志，缺独立console_coverage项；结合真实执行回执、每轮账本及源码档案检查覆盖。不得只删旧student记录便宣称所有材料合格，更不能补造历史时间/日志。

**C. 真正T4及最终包。** 尚无T4资源，不要再做“等效T4”推断。得到T4后迁移033、同版代码/依赖/许可，按官方图片及最终CSV核验。最终包需绑定五教师与029/030两学生的初始化、许可、哈希、源码/环境、完整日志和逐轮证据，当前未组装完成。

用户希望继续提分时，先保留033和CSV，再从train/valid设计单因素对照；CLAHE是候选，不是已证实改进或省时操作。DALI有实现加速可能但当前未装，不能声称已有收益。不要把新增训练、损失/alpha/日程/预处理同时改变。

## 11. 代码入口和执行副作用

| 入口 | 用途/注意 |
| --- | --- |
| `model_overnight.py`、`kd_common.py` | V1白名单、初始化来源/许可协议 |
| `train_overnight.py`、`dataset_overnight.py` | 当前训练仍依赖的共享工具；文件名带overnight不代表废弃，不能直接删 |
| `train_teacher.py`、`run_mixed_teachers.py` | 教师/混合轮数，当前已完成，不重复启动 |
| `extract_soft_labels_tta.py`、`distill_tta_data.py` | 严格OOF双视图原始logits及读取 |
| `train_distill.py` | KD/独立BCE；默认配置预览，`--execute`才训练 |
| `run_student_experiments.py` | 旧调度工具，不作为当前阶段自动连跑入口 |
| `compose_students.py` | 默认预览；`--execute`才复制模型并生成config |
| `export_student.py` | 普通调用立即导出；只有`--estimate-only`为估算 |
| `predict.py`、`validate_student.py`、`cost.py` | **无默认preview，普通调用执行GPU推理** |
| `benchmark_local_resources.py` | 默认元数据预览，`--execute`才GPU；不认证T4 |
| `compare_student_candidates.py` | CPU概率比较、患者区间；`--reference`不改变成员/权重顺序 |
| `build_training_evidence.py` | 默认preview，`--execute`写证据；preview不等于深核验通过 |
| `package_kd_submission.py` | **没有preview/--execute，调用即写材料**，缺项返回pending/exit2 |
| `preflight_kd.py` | 只读环境/清单检查，`--check-files`仅stat；传`--output`才写报告 |

`compose --weights-json`要数组M或M×10；预登记`DUAL_TINY768_WEIGHTS_20261027.json`是映射，只可直接用于compare的`--combinations-json`，不能混传。

最近验证：等值推理优化时完整CPU套件269项/365子测试通过；后加`--reference`定向`test_student_workflow.py`17项通过。不是本次重新跑了完整套件。本轮交接仅只读核对与文档，不需要构建模型或重跑预测。

## 12. 第一轮接手的安全检查

```bash
cd /home/featurize/work/cloud_project_v2
git status --short
git log -6 --oneline
df -h /home/featurize/work
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
sha256sum /home/featurize/work/deliveries/033_dual_tiny768_seed43_25_seed44_75/test_predictions/submission.csv
```

后续修改若Git未配置身份，可局部使用`git -c user.name=Codex -c user.email=codex@localhost commit -m '清楚说明本次改动'`，不改全局设置。不要为核对接手状态运行predict/cost/package；先读元数据和当前用户要求。

## 13. 精确证据与阅读索引

最新状态以[CSV交付](docs/SUBMISSION_DUAL768_20261028.md)、[SUBMISSION登记](docs/DUAL_TINY768_SUBMISSION_20261028.json)、[真实部署复盘](docs/DUAL_TINY768_DEPLOYMENT_REVIEW_20261028.md)为准。`ACCEPTANCE`的`final_test_csv_generated=false`是生成CSV前快照，`SELECTION`的待验证字段同理，不能当成当前未完成，也不要回写历史。

| 内容 | 项目内记录/证据 |
| --- | --- |
| 当前CSV、原图/源码保护审计 | `artifacts/kd_v1/dual768_submission_20261028/{execution.json,audit.json,protected_before.json,source/}` |
| 实际组合valid/4090独立核验 | `artifacts/kd_v1/dual768_deployment_review_20261028/{actual_integrity.json,rank_equivalence.json,execution_*.json,source/}` |
| 双768三组概率诊断 | 工作区`deliveries/032_dual_tiny768_fixed_weights_diagnostic/comparison_report.json` |
| 033之前的十类/区间/切片 | `docs/DUAL_TINY768_FP16_REVIEW_20261028.md` |
| 030早停及曲线/来源 | `docs/TINY768_SEED44_REVIEW_20261028.md`及`artifacts/kd_v1/tiny768_seed44_review_20261028/` |
| OOF及教师改进历史 | `docs/OOF_CACHE_REVIEW_20261027.md`、`docs/TEACHER_EPOCH_REVIEW_20261027.md` |
| 老计时、等值优化/MPS边界 | `docs/RESOURCE_STRESS_RESULTS_20261027.md` |
| 历史清理/旧方案 | `docs/history/`；只作文字证据，不复用旧V2流程 |

部分审计JSON的`no_gpu_work=true`仅指对应CPU审计，不是全项目或主代理执行记录；实际033部署和测试生成均做过GPU前向。真实训练归档也不是后来部署等值提速的源码版本，不要求把两者伪造成相同。

重要近期提交：`e405596`等值预处理/TTA输入共享；`e664562`比较参照选项；`f374319`真实双768部署/计时；`00df855`官方补充适用；`87dcf68`当前测试CSV交付记录。接手先保住这一可回退版本，再按用户新任务推进。
