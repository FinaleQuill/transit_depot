**2026-11-01复盘完成：** 两项固定4轮训练和实际组合核验已结束，最高valid 0.817149190/93.49ms，draft 0.816904529/87.10ms；未达0.82，增益区间跨0，原043/044保留、本轮无test预测。已保存实验模型，不重复启动后台训练；先读[最终复盘](docs/BOUNDED_FINETUNE_REVIEW_20261101.md)。

**2026-11-01最新授权与执行：** 用户已恢复提分/训练并允许最多12小时；旧停止说明是历史。当前固定4轮Tiny/Base EMA微调与Study打包，先读[本轮运行交接](docs/BOUNDED_FINETUNE_HANDOFF_20261101.md)，查实际进程再操作，不重复启动。

# 项目交接：043/044实际提交基本持平，valid提分未体现为平台优势

更新时间：2026-10-31，Asia/Hong_Kong（UTC+8）。项目根目录为 `/home/featurize/work/cloud_project_v2`。本文件供没有聊天上下文的下一位agent独立接手；最新用户要求优先于旧交接中的历史阶段。

**最新用户实测反馈（2026-10-31）：**用户报告044（valid 0.816677）的实际提交成绩不如043（valid 0.810570），或可视为基本持平，所述分差约0.01分。两版绝对平台成绩未提供，不换算成AUC差。**valid提分未体现为已证实的平台收益，不能默认044优于单B。** 详见[平台反馈记录](docs/PLATFORM_FEEDBACK_20261031.md)。当前只更新记录，停止优化与后续80ms目标保持不变。

## 1. 用户最新决定与当前任务边界

用户已拿到044的提交CSV，最后明确要求：

> “给我一个0.816677的csv，你先别优化了，84ms现在够了”
>
> “现在够不代表一直够，只是我认为现在不值得继续优化”

因此当前状态是：**保留已经交付的0.816677 / 84.82ms版本，停止继续优化；80ms仍是以后目标。** 不能把暂时接受84ms写成永久取消80ms约束。当前新增请求只是记录用户实际成绩反馈并更新交接文档，不授权恢复实验、训练、重新预测test或上传。

后续用户安排新工作时，再按其新指令推进。简单检查、重复核验的子agent用中等推理强度；核心实验判断由主agent负责。用户不喜欢对已经授权的操作反复确认。

## 2. 项目与当前结果

比赛任务是胸片十类多标签分类。标签与提交单位为Study，一项Study可以有多张图；保留官方原始train/valid/test划分，valid未并入训练。

| 对象 | 完整valid Macro-AUC | 本轮2080Ti端到端均值 | 状态 |
| --- | ---: | ---: | --- |
| 固定B，Base768 / hflip2 | 0.8105695698808949 | 74.549656133ms | 043保留；用户反馈实际提交与044基本相当或略优 |
| 类1用B、其余AB等权，原始解码 | 0.8168300303208815 | 92.866368773ms | 完整实跑，未选作本次CSV |
| **同一融合，JPEG draft768，双hflip2** | **0.8166770734158482** | **84.820141257ms** | **044已交付；最新反馈实际提交未优于043** |
| draft768，A单视图/B双视图 | 无完整结论 | 无完整结论 | 用户要求停止，部分输出保留 |

这里的AUC均为valid结果，**不是平台test成绩**。本agent未自行上传平台；交付后已收到用户对实际提交表现的比较反馈：044略逊于043或基本持平，分差约0.01分，未提供两版绝对平台成绩，代理未独立查询。此前“未收到平台反馈”的状态已更新；正式T4复测仍未完成，2080Ti结果不能冒称T4认证。

044三轮均值为 **86.073288830 / 85.423300830 / 82.963834110ms**。口径是原固定100张valid图、另5张预热图、batch1逐图串行、连续三轮，计入读文件、预处理、全部前向、传输、Study聚合、实际CSV写出与flush，排除模型加载/首次初始化/预热。输入输出位于本地根盘，所有慢轮和300个原始耗时均保留。

## 3. 已交付文件与对照版本

044是上一阶段按valid选择、随后交付的版本；最新实际提交反馈没有确认它优于043。本次没有切换或重新提交指令，两个版本均保留，不把“已交付”解释为“平台最佳”。044路径：

```text
/home/featurize/work/deliveries/044_tiny_base_class1B_draft768_20261031/submission.csv
```

SHA256：

```text
3ff38f2c4c248ce4913c712f30f89ad53d54b98a2e82acb1ac458517833e99d8
```

UTF-8四列为 `Subject_id,Study_id,Predict_class,Probability`，1414个Study各十类，共14140条预测行，另有表头；来源2218张test图、136名患者。没有按阈值过滤行。

044根目录 `/home/featurize/work/deliveries/044_tiny_base_class1B_draft768_20261031/` 已持久保存：

| 相对路径 | 用途 |
| --- | --- |
| `submission.csv` | 上一阶段用户选定并已交付的文件，最新反馈未优于043 |
| `submission.image_predictions.csv` | 22180条逐图概率，仅作核验辅助 |
| `submission.csv.report.json` | CSV/配置/权重身份与生成报告 |
| `independent_audit.json` | 原图、身份、十类覆盖与汇总独立审计 |
| `SELECTION_BEFORE_TEST.json` | 最终预测前固定配置及用户选择记录 |
| `latency_2080ti.json` | 完整valid指标、100×3耗时与一致性记录 |
| `model/` | 本次实际调用的固定推理模型，已持久化 |
| `README.md` | 本版本交付说明 |

原B提交保留在 `/home/featurize/work/deliveries/043_base768_B_submission_20261031/submission.csv`，SHA256为 `f61321af3102e0a08cf4a4aa618bf1d4ae3e0108232b281402a668b2441dacff`。043没有持久部署模型目录；原B导出仍在本机临时目录，可由持久最佳EMA快照恢复。**重放044已经不需要恢复A/B快照，其实际推理权重就在044/model内。**

不要误交工作区根目录的旧 `submission.csv`。043与044是不同的真实交付版本，须按用户指定准确辨认，不自动用任何一版替换另一版。

## 4. 044必须保持的推理配置

以下全部在最终预测前固定，不根据test输出选择或修改：

1. A：`convnext_tiny.fb_in22k_ft_in1k_384`，原039/P2学生，E9最佳EMA。
2. B：`convnext_base.fb_in22k_ft_in1k_384`，原040/P2+完整Study监督学生，E10最佳EMA。
3. 两者均768、FP16、eval；**A和B各使用identity与hflip两个视图（hflip2），共四次前向**。当前044根配置没有 `view_candidate` 字段；不要混入被中止的A单视图方案。
4. 对JPEG且模式为L/RGB的单张图，先固定调用Pillow `draft(mode,(768,768))`，再解码、EXIF转置、原LANCZOS等比例填充至768；其他格式/模式保持原完整解码。归一化仍为原ImageNet设置，采用已验证的AVX2/native与精确uint8→FP16查表路径。
5. 每模型先对两个视图的sigmoid概率取均值；**第1类只用B，其余九类A/B各50%**。固定逐类权重为：

```text
A = [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
B = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
```

6. 在逐图融合之后，按原manifest图像顺序、FP64算术均值汇总成Study概率，十类全部输出。无新阈值拟合，无额外患者跨Study平滑。

JPEG draft会改变像素，这是经完整valid重新选定的新候选，**不能写成与原解码等值的优化**。逐类融合权重也是显式在valid上选择的，不能沿用早期批次“未用valid选逐类权重”的历史声明。

实际推理权重与必要配置共 **241,433,757bytes**，小于500,000,000bytes：

| 文件（相对044/model） | bytes | SHA256 |
| --- | ---: | --- |
| `config.json` | 3579 | `f722b9c3e0e62ac0a675c49163dfac58bc1b52481ded8f2e9e05c1602c2fe92d` |
| `A/config.json` | 5228097 | `6a109c69ca9f15bc56fabf61002221e499e638cc09cdac9d0a7ecfba834fdffd` |
| `A/student_fp16.pt` | 55712511 | `fea2350a479fc65883f687ce038289666345c5b0d5fc7da44576a5f995742b5f` |
| `B/config.json` | 5228541 | `ec8d155fdf1b4a8ca5de7c914ab098efa5d75c449f3a986e950a811d0c632011` |
| `B/student_fp16.pt` | 175261029 | `903ee9b95317536b6120d399d67ddcb1e82b82f7898cc174b2fd8b55d6f8e411` |

## 5. 已完成的核验与证据

原test2218张图片SHA全部匹配冻结manifest；独立审计确认身份、十类唯一覆盖、概率有限且在[0,1]、未知test标签保持未知。逐图按原manifest顺序FP64重算Study均值，与提交14140行逐bit一致，最大误差0。未计算test指标、概率分布或类别先验，未据test改模型或后处理。

完整valid为1422图、859个Study、228名患者。044的100张计时图闭合为53个完整Study；每轮逐图/Study概率均与完整valid逐bit一致，固定二值、排序与并列也一致，三轮CSV哈希相同。

独立CPU复算AUC为0.8166770734158482，相对B点差+0.0061075035；1000次患者配对95%区间为[0.0025178709,0.0105014553]。1/4/8类均值0.7426537148；单图391个Study的AUC为0.7561290380，多图468个Study为0.8520325796。**同一个valid被多次用于epoch、候选和权重选择，区间没有校正选择，不能写成独立泛化确认。** 十类明细保存在完整valid报告。最新用户反馈显示该valid点差没有体现为实际提交优势，不能用此区间反驳该反馈或声称044已证明泛化更好。

主证据目录：

```text
/home/featurize/work/cloud_project_v2/artifacts/kd_v1/student_auc815_20261031/
```

优先读取：

- `actual_pair_draft_target768/valid/validation_report.json`：所选完整valid实跑。
- `actual_pair_draft_target768/actual_report.json`及`timing/`：所选真实三轮和原始CSV。
- `selected_candidate_independent_audit.json`：独立valid、患者区间和计时审计。
- `optimized_submission_complete.json`：044生成及审计完成记录。
- `optimization_stopped_by_user.json`：用户停止优化及保留后续80ms目标。
- `user_platform_feedback_20261031.json`：用户报告044实际提交略逊或近似持平于043，约差0.01分；绝对成绩未知。
- `interrupted_mixed_views/INTERRUPTION.json`：用户中止第八批，exit137，无完整指标。
- `fifth_plan.json`、`seventh_decoder_plan.json`：逐类权重、实际draft768候选登记。

B单模证据在 `artifacts/kd_v1/student_B_submission_20261031/`，包括环境准备、恢复快照、完整valid复现、三轮CSV与B生成记录。固定镜像27项定向CPU检查已通过；真实模型/完整valid/计时及最终CSV审计均已执行，无须仅为接手重复。

## 6. 正确加载入口：不要直接套旧Tiny CLI

| 文件/API | 当前用途 |
| --- | --- |
| `student_fixed_pair.py` / `FixedStudentPair` | 固定类别融合的Tiny/Base双模型加载、来源与体积检查；自身采用原解码 |
| **`student_draft_pair.py` / `DraftStudentPair`** | **044所需加载器，加入固定JPEG draft768；当前双hflip2** |
| `predict.py` / `predict_submission(..., predictor_factory=DraftStudentPair)` | 本次实际共用的提交CSV写出与Study聚合API |
| `cost.py` / `time_round` | 本次调用的共用单图计时循环，调用同一加载器的预测方法 |
| `restore_student_export.py` | 持久最佳EMA快照恢复工具，校验receipt、epoch、双SHA与FP16字节 |
| `submit_student_candidate.py` | 固定单学生B提交入口，不能替代044双模型加载器 |
| `audit_student_submission.py` | 独立CSV/sidecar/原图与Study均值核验 |

**裸 `python predict.py --model-dir 044/model` 和裸 `cost.py` CLI仍默认旧 `deploy_kd.Predictor`，不支持044的 `fixed_classwise_student_pair`。** `predict_submission`提供Python级`predictor_factory`，CLI没有对应开关；`cost.run_cost`当前也没有factory参数。不能宣称现有裸cost CLI已经正式适配044。

本次真实提交使用`DraftStudentPair`注入共用writer，真实计时是同一`DraftStudentPair`加`cost.time_round`。调用`FixedStudentPair`会漏掉JPEG draft，从而改变结果。复现脚本保存在证据目录的`execute_pair.py`与`finalize_selected.py`中，仅作历史回执参考。

`finalize_selected.py`依赖旧`/tmp`且断言044尚不存在，**不能在现在原样重跑**。当前没有重新生成test的必要；如用户以后授权复现，保留固定模型与配置，选择新的输出目录，避免覆盖044。

当前044仍缺完整正式提交项目/官方cost CLI适配及真实T4复测。旧038是旧033的完整项目包，不能冒充044材料。以后用户安排这些工作时再补，现时不要顺手实施。

## 7. 环境现状与新窗口恢复

本轮实际环境：RTX2080Ti约22GiB、SM75/68SM、驱动610.57.04；Python3.12.3、torch2.10.0+cu128、torchvision0.25.0+cu128、CUDA12.8、cuDNN91002、NumPy1.26.4、Pillow10.4.0、timm1.0.30。native实际为AVX2，最多8线程；开发计时allocator限制8GiB，无MPS/时钟/功耗调整。

固定比赛镜像：

```text
crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6
```

**work持久化不等于环境持久化。** 本轮已发生旧/tmp、Docker层和解压图丢失，随后恢复。下列路径目前存在，新窗口不可假定仍在：

| 本机路径 | 内容/局限 |
| --- | --- |
| `/tmp/student_candidate_20261031/data/cxr_first` | 仅已恢复valid1422/test2218图；train没有恢复 |
| `/tmp/dual768_competition_addons` | 离线公开推理附加包 |
| `/tmp/student_candidate_20261031/A_export`、`B_export` | 临时重建导出；044已有持久副本 |
| `/home/featurize/data/cxr_first.zip` | 本机原官方ZIP，约55GB；新实例也须核实 |
| Docker镜像、本机GPU设备与驱动库 | 新窗口可能变化，先查实际环境 |

持久的本机容器回执是 `artifacts/kd_v1/student_B_submission_20261031/run_container.sh`。它硬编码/tmp路径、`--pull=never`、GPU设备、repo绝对路径以及610.57.04驱动库，**不是通用恢复脚本**。新机先查资源与路径，按实际驱动挂载适配，不盲跑、不替换系统驱动。沙箱内GPU/Docker访问受限与宿主GPU丢失是不同情况，本轮沙箱外只读检查证实GPU正常。

推理源码根没有`vendor/`。恢复指引见 `docs/FINAL_OPTIMIZATION_HANDOFF_20261030.md`；离线addon归档和manifest均在：

```text
/home/featurize/work/deliveries/038_dual768_optimized_project_20261030/code/vendor/
  competition_addons_cp312_linux_x86_64.tar.gz
  competition_addons_archive_manifest.json
```

如以后任务需要恢复，在固定镜像内给`prepare_inference.py`显式传上述归档与manifest，构建/确认native可用；不要在只读038原包上force-build或拿034旧源码覆盖当前代码。宿主Python版本不等于比赛环境，CPU不能代替本次FP16 GPU验收。

本轮交接时work仅约**300MiB**空余，/tmp本地盘余量大。不要再在work复制模型/完整项目/镜像/全量数据，也不要删除旧权重或证据腾空间。Git不保存这些被忽略的大资产，换机前需另行保全。

## 8. 训练来源与规则边界

教师与严格OOF保持冻结，本轮新增训练为0。教师集合入口为：

```text
/home/featurize/work/runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_collection.json
/home/featurize/work/runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2/
```

5折教师采用既定20/12/12/12/12轮最终EMA，原合规非医学预训练与许可证来源保留。A/B训练均已结束，旧交接里“B继续训练”的段落只是历史。

持久最佳/末轮EMA与来源、日志在 `artifacts/kd_v1/student_upper_bound_20261030/models/{A,B}/`。这些快照**没有优化器、原始模型、scaler和RNG，不能完整resume**；本轮接手时原/tmp完整断点已丢失，不能把旧receipt中的临时路径当成当前仍存在的完整备份。

044成员保留原完整checkpoint SHA与持久snapshot SHA，恢复后FP16 tensor及历史导出权重文件哈希一致：

| 成员 | 原完整checkpoint SHA256 | 持久最佳snapshot SHA256 |
| --- | --- | --- |
| A E9 | `65cdb72f63a5ec0d36486d4d8dcb34dd2fde5ed85aa52d3a10be2b0dbf633f42` | `aa3c632e7764b57a522920369e16d9cb4768e43e008c0588439b4438aa88fd6f` |
| B E10 | `bf69b6e16917ab03b8f9318287a8ef660e21ebfb71095648527071377ae126eb` | `9cb733eb65b71eeb0fe7633122dcb085b47fc4c13b4f46f139f5b773a5e805a3` |

先读`AGENTS.md`、`docs/CLEANING_POLICY.md`、`docs/COMPETITION_RULES.md`及WORKLOG最新块。规则原件为work下`rules.pdf`、`rules_1.pdf`，正式补充优先于FAQ；逐页文本也保存在 `../deliveries/036_dual768_2080ti_diagnosis_20261029/rules_audit_sources/`。

核心边界：只用官方train训练，保持原valid未并训；test仅最终预测和完整性检查，不做test伪标签、分布/先验估计、选模或拟合。预训练须合规、非医学、非NC、非gated，教师来源也须申报。固定TTA/集成及单图确定性处理允许，但最终仍需真实官方T4、≤100ms、≤500MB及预测一致性/完整材料验收。用户的80ms是更严格的工程目标，当前尚未达到。

## 9. 历史实验与停止点

不要把已完成或已拒绝实验当待办自动重跑：

- 原A/B等权768/hflip2缓存AUC0.814786617，640/768/896多尺度融合0.813881643；logit均值及训练权重校正融合也未超过原等权。
- 第四批固定RMS/max Study聚合四项均下降，完整负结果保留。
- 第五批只比较第1类B75%与B100%，其余九类等权；后者被选作后续实跑。属于显式valid逐类权重选择。
- 原1536 draft目标经valid图头部尺寸检查，对固定100图不会触发缩小解码，在计算候选指标前显式废止；768目标另有登记，原计划/废止回执未覆盖。
- 第八批A单视图/B双视图已按用户要求停止，容器exit137，只有部分valid输出；没有完成的AUC或耗时结果，未经新指令不恢复。

旧033、038、041、043及其原件仍保留。用户没有授权本次上传平台、购买/归还服务器、删旧资产或恢复训练。

## 10. 下一位agent如何接手

当前先只读确认用户的新任务和停止边界，核044的小文件/CSV身份及环境是否仍在，不要为了接手重复GPU、完整大权重哈希、test预测或已完成基准。

可安全执行的轻量检查示例（不启动推理）：

```bash
cd /home/featurize/work/cloud_project_v2
git status --short
git log -5 --oneline
python -m json.tool ../deliveries/044_tiny_base_class1B_draft768_20261031/SELECTION_BEFORE_TEST.json
python -m json.tool ../deliveries/044_tiny_base_class1B_draft768_20261031/model/config.json
sha256sum ../deliveries/044_tiny_base_class1B_draft768_20261031/submission.csv
df -h /home/featurize/work /tmp
bash -n artifacts/kd_v1/student_B_submission_20261031/run_container.sh
```

若用户以后安排完善某版材料，再按其指定版本补齐加载器、predict/cost CLI、可复现包装及真实T4验收；不要因044的valid更高就默认只完善044。若重新优化，保留单B/043作为更简单、已测耗时更低的重要基准，优先检验独立验证与可重复收益，不以反复同valid选择后的点分认定泛化改善，也不据平台反馈扫权重/阈值。两者都不是现在自动执行的任务。

源码保存于`f41bfe1`（恢复/共用提交入口）、`fe7f3f2`（有界CPU筛选）、`6663fe0`（固定融合/draft）、`d80d181`（交付与停止记录）。本交接另提交Git，不推送。

当前另有三个接手前已存在的未跟踪历史实验文件：`fixed_student_candidate.py`、`review_student_upper_bound.py`、`student_probability_fusion.py`。本次交接未改动它们；不要盲删、盲目`git add .`，也不要将其误认为044的正式入口。WORKLOG新块按CRLF字节追加，保留旧前缀；Git身份若缺失，用本次提交的局部配置，不改全局身份。

## 可直接转发给下一位agent

> 项目在`/home/featurize/work/cloud_project_v2`，先读`NEXT_AGENT_HANDOFF.md`与`AGENTS.md`。此前交付044：完整valid AUC0.8166770734，当前2080Ti84.820141ms，CSV和固定模型均已持久化并审计；但最新用户反馈实际提交不如043单B（valid0.810570），或基本持平，约差0.01分，未提供两版绝对成绩。不能把valid提升当成平台优势，043须保留为重要基准，本次没有自动切换提交。双模型都hflip2，JPEG draft768，第1类用B，其余等权；不能混入被中止的A单视图实验。用户明确暂不继续优化，80ms仍保留为后续目标，未经新指令不要恢复实验或重做test。先按我这次的新任务行动；旧038不是044项目包，裸predict/cost默认loader也不能直接载044。临时环境可能丢失，先核实，不要盲跑历史脚本。
