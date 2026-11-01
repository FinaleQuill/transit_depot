# 冻结分类头提分交接（2026-11-01 20:47 香港）

## 最新任务与边界

用户要求：按当前最好成绩生成CSV，随后继续想办法冲AUC0.82；接着提醒Git、因token不多要求保存进度和思路。用户已给全权限，长训练可后台运行，结束后用户会再通知。保持原最多12小时预算，**硬截止2026-11-02 02:37香港**，不能自行延长或用test选模。不推送远程、不删除旧模型/原件。

## 045 CSV已完成

`/home/featurize/work/deliveries/045_X1_B_best817149_submission_20261101/submission.csv`

SHA256 `7379ac54cf4a8c66dd021e0f7c36a5fdaa69f32f6af2efd7ce3a4cf9bc3920da`。当前实际最高valid **0.8171491901128627**，X1 Tiny E4+旧B Base E10，原始完整JPEG解码、各768/FP16/hflip2，第1类仅B，其余等权，原manifest顺序FP64 Study均值。2080Ti端到端93.490616ms，真实T4未验。它不是87ms的draft候选。

1414Study/14140行、2218图原SHA、身份/十类覆盖/有限概率、逐图到Study独立审计均通过，误差0。已保存选择前冻结、生成报告、逐图CSV和独立审计；未上传。只有这一次最终test预测；**后续FC实验不得读取045概率作统计/训练/选模**。043/044原件保留。详情docs/BEST_817149_SUBMISSION_20261101.md。

## 为什么换路线

前轮X1完整Study微调、X2弱KD都让单模下降（.805712/.807006），融合只到.817149，较旧AB+.000319且患者区间跨0；两条E1→E4曲线持续退化，不能继续照同配方续训。此次冻结旧A/Tiny E9和旧B/Base E10表示，只改很小的FC头，把优化目标改为Study级排序；既避免encoder漂移，又不增加正式推理的encoder前向次数，且work仅约60MB，不再持久化新完整大模型。

## 新计划已冻结并提交

`docs/FROZEN_HEAD_PLAN_20261101.json` SHA **a6b906e232c9304983d16644329de8edffbb8a49079e5f0e4cf6a063ce551565**。历史valid结果参与设计已明示，不伪称独立验证。

- 源encoder：原044/model/A与B，精确config/FP16权重SHA写在plan。Tiny feature_dim768、Base1024。eval冻结全部encoder/norm/dropout/drop_path。
- 缓存实际head.fc前hook输入，含原pool/norm/flatten/pre_logits/drop；每原图保留identity与hflip两份FP16特征。train batch16、workers4；valid batch1。生产原解码/FP16归一化，逐图原SHA检查；valid还要求两个旧头对全部1422图概率逐bit复现原valid，否则cache失败。
- F1：完整Study sqrt加权BCE +0.1×原头raw-logit平方锚定。F2：0.5 BCE+0.5逐类等权pos-neg pairwise softplus ranking+0.1锚定。rank score=logit(Study内全部图/双视图sigmoid均值)，tau1。
- 每encoder各F1/F2，共A_F1/A_F2/B_F1/B_F2四个头，**固定12轮最终头**；batch256Study、Adam lr2e-4，1轮warmup后cosine至1e-5，WD0/clip1，seed48/49。只读取train缓存拟合，无valid早停/选epoch。
- 代理评估四single+六条已登记pair；pair第1类只由Base成员输出，其余Tiny/Base等权。不增加新权重组合/逐类扫参。CPU FP32 FC分数仅为代理，按计划最多取两个最高pair做真实GPU完整valid+计时；真实结果再与当前.817149作患者配对区间。
- 本轮后台**不自动生成新test CSV**；完成后用户通知代理再复盘、选最终部署。

## 已启动：不要重复开任务

后台控制器 **PID58479、PPID1**，`python3 -u run_frozen_head_pipeline.py`，附着既有cache，不重复提取。cache launcher **PID57264**，容器`codex-frozen-head-cache-20261101`。ID是本机记录，接手先核实际/proc/docker，不把旧PID当永久身份。

**最新状态必须读新目录**，用户IDE原`bounded_finetune_20261101/pipeline_status.json`是已完成的旧两实验，不代表新任务。

- `artifacts/kd_v1/frozen_head_20261101/STATUS.md`
- `artifacts/kd_v1/frozen_head_20261101/pipeline_status.json`
- `artifacts/kd_v1/frozen_head_20261101/cache.log`、`PIPELINE.log`、各stage.log/command.json。
- 大缓存/合并权重：`/tmp/cloud_student_runs_20261030/frozen_head_20261101/`（不跨实例持久）。
- 小patch/血缘：`artifacts/kd_v1/frozen_head_20261101/fits/{A_F1,A_F2,B_F1,B_F2}/`。

后台流程：等meta.complete及旧头完整valid逐bit证明→四头CPU串行12轮→代理评估与1000患者bootstrap→最高两已登记pair逐一实际FP16验收（先前5valid图overlay-vs-merged逐bit，后1422valid、固定105图/100×3端到端、逐图/Study/二值/排序完整一致）→actual_comparison.json→全局completed/ready_for_review。

每个fit/config必须绑定`training_plan_sha256`、**feature_cache_metadata_sha256**、base/patch SHA、最终E12、全部10类；控制器已补好。meta在valid缓存完成前会变化，**不可提前fit**。控制器记录关键源码SHA，运行中不要修改源文件，否则故意fail closed。

全局failed/budget_exhausted时看pipeline_error.txt及对应log；不自动重启覆盖原输出。cache缺失/不完整不会静默重提取，现存fit也不覆盖。原数据/模型/045 CSV保持只读。

查看命令：

```bash
watch -n 15 cat /home/featurize/work/cloud_project_v2/artifacts/kd_v1/frozen_head_20261101/STATUS.md
```

## 代码、检查与Git

- b60be39：045交付、继续提分授权和记录。
- f5f725f：冻结头计划、精确缓存、纯train头训练、Tiny/Base补丁及临时合并helper。
- 4929ffb：四头/六组合严格代理评估。
- f2de1b0：实际组合验收和脱离聊天的有界串行后台。

新增源码：frozen_head_cache.py / frozen_head_fit.py / frozen_head_predictor.py / frozen_head_evaluate.py / frozen_head_actual.py / run_frozen_head_pipeline.py。20项cache+fit+predictor CPU检查、6项代理评估、9项actual+pipeline检查均通过；真实FC训练尚未开始（接手以状态为准）。缓存前已验证CPU half归一化与真实GPU lookup输入相等，正在处理原train；当时已有约5000张，吞吐稳定。

FrozenHeadPredictor仅覆盖FC权重，原A/B文件不改；export_merged_for_validation强制/tmp输出、只替换head.fc两tensor、其余encoder逐字节保留，兼容原ExperimentalPredictor和StudentEnsemble。正式patch+原encoder部署无额外前向；实际CPU代理和GPU FP16算术不能混称相同，故需要完整实际复评。

使用固定digest官方镜像和已恢复/tmp addons；`STUDENT_RUNTIME_MODE=inference bash run_student_environment.sh ...`挂载本地/tmp工作根，旧deliveries为只读。GPU一次只一个任务。实际T4仍待；均值≤100ms与≤500MB是正式约束，80ms是更严格工程目标。

三份原未跟踪旧文件fixed_student_candidate.py/review_student_upper_bound.py/student_probability_fusion.py不要盲删或git add。Git只收代码/计划/文档，数据模型日志不入库。WORKLOG按原CRLF前缀追加，检查可用git -c core.whitespace=cr-at-eol diff --check。不推送。

## 下一位接手

先读新STATUS与pipeline_status、核进程/GPU，避免重复任务。若completed，先核实际AUC、逐类/切片/患者区间/成本及父权重+小patch血缘，再告知用户是否真正超过.817149/是否到.82；不能把CPU代理分数当实际分数。若无可靠提升，保留045；若新候选有效，按用户最新指令处理新的最终CSV，不根据旧test输出改配方。不要为凑.82临时扫验证权重或改固定E12选择，不自行越过02:37预算。
