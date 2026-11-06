
### 2026-10-18T15:22:47+08:00 · 012_cloud_4090_foreground_debug

- 任务：384原图/ConvNeXt V2类别注意力/Study BCE+Rank新run；微批16、累积1、AMP/检查点/CPU EMA；初始完成0轮，尚未反传。

### 2026-10-18T15:22:54+08:00 · 012_cloud_4090_foreground_debug

- 失败/中断：ImportError: Using SOCKS proxy, but the 'socksio' package is not installed. Make sure to install httpx using `pip install httpx[socks]`.；完成0轮、实际反传0微批；保留此前完整轮权重；未完成轮不视作已验收或可精确续训。

### 2026-10-18T15:25:17+08:00 · 013_cloud_4090_fixed

- 任务：384原图/ConvNeXt V2类别注意力/Study BCE+Rank新run；微批16、累积1、AMP/检查点/CPU EMA；初始完成0轮，尚未反传。

### 2026-10-18T15:50:24+08:00 · 013_cloud_4090_fixed

- 结果：epoch=1，完整验证=True，Study AUC/F1=0.728735/0.400630；EMA更新=1158、AMP跳步=0，best=True；事实：有效图像不改写；推测：泛化收益仍需独立评价。

### 2026-10-18T16:10:07+08:00 · 013_cloud_4090_fixed

- 结果：epoch=2，完整验证=True，Study AUC/F1=0.755249/0.426889；EMA更新=2312、AMP跳步=4，best=True；事实：有效图像不改写；推测：泛化收益仍需独立评价。

### 2026-10-18T16:55:55+08:00 · 014_cloud_4090_max

- 任务：384原图/ConvNeXt V2类别注意力/Study BCE+Rank新run；微批128、累积1、AMP/检查点/CPU EMA；初始完成0轮，尚未反传。

### 2026-10-18T17:19:15+08:00 · 014_cloud_4090_max

- 结果：epoch=1，完整验证=True，Study AUC/F1=0.703631/0.380730；EMA更新=145、AMP跳步=0，best=True；事实：有效图像不改写；推测：泛化收益仍需独立评价。

### 2026-10-18T17:35:30+08:00 · 014_cloud_4090_max

- 结果：epoch=2，完整验证=True，Study AUC/F1=0.727732/0.398279；EMA更新=290、AMP跳步=0，best=True；事实：有效图像不改写；推测：泛化收益仍需独立评价。

### 2026-10-18T17:50:13+08:00 · 014_cloud_4090_max

- 结果：epoch=3，完整验证=True，Study AUC/F1=0.753280/0.419259；EMA更新=435、AMP跳步=0，best=True；事实：有效图像不改写；推测：泛化收益仍需独立评价。

### 2026-10-18T18:04:45+08:00 · 014_cloud_4090_max

- 结果：epoch=4，完整验证=True，Study AUC/F1=0.775642/0.445315；EMA更新=580、AMP跳步=0，best=True；事实：有效图像不改写；推测：泛化收益仍需独立评价。

### 2026-10-18T18:32:52+08:00 · 014_cloud_4090_max

- 结果：epoch=5，完整验证=True，Study AUC/F1=0.790208/0.467324；EMA更新=725、AMP跳步=0，best=True；事实：有效图像不改写；推测：泛化收益仍需独立评价。

### 2026-10-18T18:47:39+08:00 · 014_cloud_4090_max

- 结果：epoch=6，完整验证=True，Study AUC/F1=0.797719/0.486124；EMA更新=870、AMP跳步=0，best=True；事实：有效图像不改写；推测：泛化收益仍需独立评价。

### 2026-10-18T19:01:30+08:00 · 014_cloud_4090_max

- 结果：epoch=7，完整验证=True，Study AUC/F1=0.796407/0.476638；EMA更新=1015、AMP跳步=0，best=False；事实：有效图像不改写；推测：泛化收益仍需独立评价。

### 2026-10-18T19:15:24+08:00 · 014_cloud_4090_max

- 结果：epoch=8，完整验证=True，Study AUC/F1=0.795420/0.477243；EMA更新=1160、AMP跳步=0，best=False；事实：有效图像不改写；推测：泛化收益仍需独立评价。

### 2026-10-18T19:29:46+08:00 · 014_cloud_4090_max

- 结果：epoch=9，完整验证=True，Study AUC/F1=0.795224/0.477177；EMA更新=1305、AMP跳步=0，best=False；事实：有效图像不改写；推测：泛化收益仍需独立评价。

### 2026-10-18T19:57:55+08:00 · 014_cloud_4090_max

- 结果：epoch=10，完整验证=True，Study AUC/F1=0.794116/0.475097；EMA更新=1450、AMP跳步=0，best=False；事实：有效图像不改写；推测：泛化收益仍需独立评价。

### 2026-10-18T20:12:00+08:00 · 014_cloud_4090_max

- 结果：epoch=11，完整验证=True，Study AUC/F1=0.790154/0.467591；EMA更新=1595、AMP跳步=0，best=False；事实：有效图像不改写；推测：泛化收益仍需独立评价。

### 2026-10-18T20:26:34+08:00 · 014_cloud_4090_max

- 结果：epoch=12，完整验证=True，Study AUC/F1=0.787168/0.461385；EMA更新=1740、AMP跳步=0，best=False；事实：有效图像不改写；推测：泛化收益仍需独立评价。

### 2026-10-18T22:46:52+08:00 · 016_a4000_tiny_v2

- 任务：ConvNeXt-Tiny/GAP/Linear原图384新run；逐图shuffle完整遍历，batch=16/accum=2，Study等权BCE，AMP=bfloat16/GPU EMA；初始完成0轮，尚未反传。

### 2026-10-18T23:07:52+08:00 · 016_a4000_tiny_v2

- 结果：epoch=1完整train图数=30328，本轮更新=948，完整valid=True；Study AUC/F1@.5=0.748406/0.160126，best=True；验证调优指标不作为平台测试成绩。

### 2026-10-18T23:27:14+08:00 · 016_a4000_tiny_v2

- 结果：epoch=2完整train图数=30328，本轮更新=948，完整valid=True；Study AUC/F1@.5=0.779049/0.222303，best=True；验证调优指标不作为平台测试成绩。

### 2026-10-18T23:46:32+08:00 · 016_a4000_tiny_v2

- 结果：epoch=3完整train图数=30328，本轮更新=948，完整valid=True；Study AUC/F1@.5=0.793732/0.268608，best=True；验证调优指标不作为平台测试成绩。

### 2026-10-19T00:05:49+08:00 · 016_a4000_tiny_v2

- 结果：epoch=4完整train图数=30328，本轮更新=948，完整valid=True；Study AUC/F1@.5=0.799305/0.298905，best=True；验证调优指标不作为平台测试成绩。

### 2026-10-19T00:25:15+08:00 · 016_a4000_tiny_v2

- 结果：epoch=5完整train图数=30328，本轮更新=948，完整valid=True；Study AUC/F1@.5=0.799971/0.320549，best=True；验证调优指标不作为平台测试成绩。

### 2026-10-19T00:44:20+08:00 · 016_a4000_tiny_v2

- 结果：epoch=6完整train图数=30328，本轮更新=948，完整valid=True；Study AUC/F1@.5=0.797306/0.335820，best=False；验证调优指标不作为平台测试成绩。

### 2026-10-19T01:05:19+08:00 · 017_a4000_tiny_v2_warm

- 任务：ConvNeXt-Tiny/GAP/Linear原图384新run；逐图shuffle完整遍历，batch=16/accum=2，Study等权BCE，AMP=bfloat16/GPU EMA；源检查点完成5轮，本次尚未反传。

### 2026-10-19T01:26:50+08:00 · 017_a4000_tiny_v2_warm

- 结果：epoch=6完整train图数=30328，本轮更新=948，完整valid=True；Study AUC/F1@.5=0.799877/0.329236，best=False；验证调优指标不作为平台测试成绩。

### 2026-10-19T01:46:27+08:00 · 017_a4000_tiny_v2_warm

- 结果：epoch=7完整train图数=30328，本轮更新=948，完整valid=True；Study AUC/F1@.5=0.798576/0.332947，best=False；验证调优指标不作为平台测试成绩。

### 2026-10-19T02:03:11+08:00 · 测试推理

- 任务：按最佳权重推理并导出完整Study概率；checkpoint=/home/featurize/work/runs/016_a4000_tiny_v2/best.pt；输出=/home/featurize/work/deliveries/016_a4000_tiny_v2_inference；work-root=/home/featurize/work；data-root=/home/featurize/data/cxr_first；训练0次，原图/缓存只读，不拟合测试阈值。

### 2026-10-19T02:04:59+08:00 · 测试推理

- 结果：2218图→1414Study，正式表14140行，本地格式校验通过；weights_source=ema，ema={"enabled": true, "decay": 0.999, "use_warmup": true, "warmup_power": 0.75, "updates": 4740, "buffers": "copy_from_raw", "device": "cuda"}；阈值来自epoch=5，CSV SHA=827665e78230072db87c9fe2a26bf7afd718538a9565c16e5a04b6cef2f9cad6；证据=/home/featurize/work/deliveries/016_a4000_tiny_v2_inference/provenance.json。下一步人工核对再提交，尚未上传平台。

### 2026-10-19T02:05:42+08:00 · 016 Tiny 最终测试推理与样例格式提交

- 用户授权：使用016/best.pt（epoch5 EMA）与同源thresholds.json，测试2218图→1414 Study；本次训练0次、测试推理1次，不拟合测试阈值。
- 格式：按原submission.csv的Subject_id,Study_id,Predict_class,Probability四列与逐行身份顺序，只替换概率；UTF-8无BOM、CRLF，保留全部十类。
- 结果：/home/featurize/work/deliveries/016_a4000_tiny_v2_final/submission.csv；14140数据行，缺Study/缺类别/重复/非法概率均0，全部值在[0,1]，原样例和源权重未改。
- 证据：/home/featurize/work/deliveries/016_a4000_tiny_v2_final/validation_report.json；CSV SHA256=c985cf8736ebb342af6c8662acbbb277042fa8ef779524b18751220333969650；原始推理与来源指纹见同目录provenance.json，未上传平台。

### 2026-10-19T08:28:59+08:00 · 手动停止019并交付官方valid最佳AUC提交

- 按用户要求创建STOP，当前轮完整验证保存后退出；019各折完成5/4/4/4/4轮，训练进程已退出，保留全部best/last。
- 019官方valid评估完成后因分数较低取消测试推理，不伪造新预测。相同官方valid比较：016 AUC=0.799970553；019五折TTA集成AUC=0.793435823；按较高者选择016_a4000_tiny_v2，不使用内部单折成绩或测试答案选模型。
- 交付：/home/featurize/work/deliveries/best_auc_20261019/submission.csv；四列14140行/1414Study/每Study十类，样例身份与顺序、完整性及概率[0,1]校验通过，根样例和源清单未改。
- 证据：同目录validation_report.json/provenance.json；SHA256=c985cf8736ebb342af6c8662acbbb277042fa8ef779524b18751220333969650；成绩为本地官方valid，未上传平台。

### 2026-10-19T17:02:00+08:00 · 官网Gate格式更新与现有run核对

- 重新读取官网及20261019示例图；Gate可在末列自定义，排名以Macro AUC为主、同分再看F1。最新best_auc CSV经源权重SHA及14140概率逐值核对，来自016 epoch5 EMA，未套用当前019代码阈值。
- 新交付：deliveries/gate_update_20261019/submission.csv；只追加同源十类Gate，1414Study/14140行完整、原四列和行序原样，输入CSV/模型/阈值/清单哈希未变；SHA256=b2b16d56d5129871e65e914d9e268936b2471cc83799dcfa9b6424024a9878bf。
- 016验证AUC=0.799970553，F1@.5=.320549344→保存阈值F1=.469460937；阈值同valid调优，不当作测试成绩。017新增最佳AUC=.799877379，019官方valid集成=.793435823；017继承best与016相同，018无有效完成轮，019内部单折不能直比。
- 交付目录包含同源阈值/校验/来源记录、官网示例、对比说明和可重放脚本；本次训练0次、模型推理0次、阈值重新拟合0次，旧交付保留，未上传平台。
