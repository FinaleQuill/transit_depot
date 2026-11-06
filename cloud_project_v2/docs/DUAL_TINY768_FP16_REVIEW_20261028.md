# 双Tiny768实际FP16与固定权重复盘

> 阶段更新：下方三条组装/valid/4090计时已由主代理实际完成，不重复运行。真实033 AUC=.808384442、文件128,766,272bytes、4090端到端79.280300ms；三轮CSV与完整valid一致，T4仍待。当前见[真实部署复盘](DUAL_TINY768_DEPLOYMENT_REVIEW_20261028.md)，本页保留此前候选选择和交接。

2026-10-28（香港）。030实际FP16/valid及032三组固定概率融合已完成核验。下一阶段候选固定为**029 seed43占25%、030 seed44占75%，两个768、各hflip2**；先真实组装、验证与本机计时，之后再判断新增训练或CLAHE的必要性。此选择记录于[DUAL_TINY768_SELECTION_20261028.json](DUAL_TINY768_SELECTION_20261028.json)。本次未执行组装、GPU推理或新训练。

## 完整性与同精度结果

030实际FP16源为原E8 EMA，来源SHA与既有训练审计一致；实际FP16张量有限，权重/config/report/CSV绑定通过。两名学生各8590行，覆盖同一1422图、859 Study、228患者；独立CPU重算两单模和三组融合的总体、十类和391单图/468多图Study切片均一致。23个原输入文件前后SHA不变。导出config中pending提示保留导出时语境，另行生成且哈希绑定的complete验证报告证明后续验证完成。

030单模型实际文件合计66,995,857bytes，实际FP16 AUC=.8075124231641647，较原生BF16-.000057394015，未见导出明显退化。

下表全为同一valid、实际FP16输入概率、hflip2。融合行是保存概率的CPU算术诊断，尚非组合模型完整valid前向。

| 候选，权重顺序seed43/seed44 | Macro-AUC | 相对较强单44 | 单图Study | 多图Study |
| --- | ---: | ---: | ---: | ---: |
| 单43 | .803983596 | -.003528827 | .737898841 | .847305203 |
| 单44 | .807512423 | — | .743318836 | .845915123 |
| 50/50 | .808370073 | +.000857650 | .743688489 | .848674656 |
| **25/75** | **.808384442** | **+.000872019** | **.744173220** | .847454066 |
| 75/25 | .806951639 | -.000560784 | .741475045 | .848588777 |

25/75比50/50只高.000014368955，不能据此宣称更优，更不值得继续密集搜索权重。两个方案相对单44均有小幅正向点差；旧43占75%的方案低于单44，说明加权融合本身不保证提分。

## 十类及不确定性

| 类别 | 单44 | 25/75 | 融合−单44 |
| --- | ---: | ---: | ---: |
| 1 | .711627 | .710828 | -.000798 |
| 2 | .844973 | .849017 | +.004044 |
| 3 | .804080 | .804883 | +.000802 |
| 4 | .759073 | .758199 | -.000874 |
| 5 | .880201 | .881241 | +.001040 |
| 6 | .784888 | .785650 | +.000762 |
| 7 | .788109 | .789992 | +.001883 |
| 8 | .717098 | .719228 | +.002130 |
| 9 | .906691 | .906318 | -.000372 |
| 10 | .878385 | .878488 | +.000103 |

25/75和50/50均七升三降。25/75缩小对单44的1/4/9类损失，第6/10类略好；代价是减少2/3/5/7/8类的融合收益。选25/75作为工程候选，是在两者整体点分接近时偏重较强单44、保留弱成员补充作用的取舍，不是证明它统计上超过50/50。

既有1000次患者聚类配对bootstrap（seed20261027，以单44为参照）：

- 25/75−单44：95%区间[-.000714399,+.002738017]。
- 50/50−单44：95%区间[-.002008016,+.004537826]。
- 75/25−单44：95%区间[-.004983042,+.004825153]。

区间均跨0，不能确认稳定收益，也不能认定所有差异是纯噪声。25/75更接近单44，区间较窄不构成独立优胜证据；没有25/75与50/50直接差值区间，不能用两个对单44的区间代替。所有逐类区间亦跨0。此次独立审计重算点分及切片，核验bootstrap算法/元数据及区间结构，未重新跑1000次。

25/75相对单44在单图/多图切片分别+.000854384/+.001538943；两切片病例构成及患者有差异，没有切片区间，不能解释为早期病变、模糊或病灶大小改善。

三组权重来自训练030前的预登记，仅从这三组中选择，没有新权重或十套类别拟合。但当前选择仍使用valid，应如实申报；患者区间不覆盖seed/选epoch、历史多次比较和选参乐观偏差。报告的weights_fitted=false表示工具没有连续拟合，不表示选择独立于valid。原032报告的deployment_selected=false保留当时事实，后续候选选择另记新文件，不覆盖历史报告。

## 下一阶段：真实组装、验证与4090计时

CPU组装预览通过：成员顺序43→44、权重1:3、两个768各hflip2，预计实际权重+config **128,766,272bytes**，小于500,000,000bytes；实际组装后重新统计。未来033目录未创建。固定100计时图和另5预热图的字节、原valid身份、完整Study覆盖及患者隔离再次核验通过；复用同一批图，不按新结果重新挑选。

以下三条仍由用户依次运行，每条成功再运行下一条。合计暂估5–15分钟，无需重训或重跑两个单模型valid。

第一条CPU组装两个真实独立seed，通常约10–60秒：

```bash
cd /home/featurize/work/cloud_project_v2
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/featurize/.conda/envs/cxr-kd-v1/bin/python compose_students.py \
  --model-dirs \
    ../deliveries/029_tiny768_kd_seed43/fp16_hflip2 \
    ../deliveries/030_tiny768_kd_seed44/fp16_hflip2 \
  --weights 1 3 --tta hflip2 \
  --output ../deliveries/033_dual_tiny768_seed43_25_seed44_75/fp16_hflip2 \
  --execute
```

第二条实际双模型、共享逐图TTA输入、完整valid，暂估3–8分钟：

```bash
/home/featurize/.conda/envs/cxr-kd-v1/bin/python validate_student.py \
  --model-dir ../deliveries/033_dual_tiny768_seed43_25_seed44_75/fp16_hflip2 \
  --image-root /home/featurize/data/cxr_first \
  --manifest data/clean_v1_reviewed/valid_manifest.csv \
  --output-dir ../deliveries/033_dual_tiny768_seed43_25_seed44_75/valid_fp16_hflip2
```

第三条对同一真实双模型做4090端到端计时，暂估1–3分钟；运行前保持无其它GPU计算任务，不运行旧MPS压测调度器：

```bash
/home/featurize/.conda/envs/cxr-kd-v1/bin/python benchmark_local_resources.py \
  --model-dir ../deliveries/033_dual_tiny768_seed43_25_seed44_75/fp16_hflip2 \
  --pictures-100 ../deliveries/031_resource_stress_20261027/inputs/pictures_100 \
  --pictures-5 ../deliveries/031_resource_stress_20261027/inputs/pictures_5 \
  --selection-json ../deliveries/031_resource_stress_20261027/inputs/selection.json \
  --output-dir ../deliveries/033_dual_tiny768_seed43_25_seed44_75/latency4090 \
  --profile dual768_seed43_25_seed44_75_native4090 \
  --memory-limit-gib 16 --execute
```

第三条不设置MPS比例；16GiB仅为4090分配器上限，不模拟T4。脚本与cost共用Predictor/time_round，100图、另5图预热、三轮、FP16、batch1串行，含读图至CSV写出，所有成员/视图均计时，初始化排除。报告比较三轮输出一致性，不冒称已与最终test submission.csv完成官方核对。两条GPU命令不要带`CUDA_VISIBLE_DEVICES=''`。

完成后核对实际组合逐Study概率与预登记25/75数学结果、十类和AUC；组合validation_report的source_training_validation沿用首成员来源，不把它误当双模型训练分数或对单44的差值。目标是确认真实部署复现约.808384的诊断结果并了解实际成本，不根据微小浮点差异重新搜权重。

## 资源、规则与后续优化

当前省时实现共用单图解码/768缩放/归一化和原图/翻转输入，仍然两个模型各两次前向；权重25/75相对50/50不会减少前向次数。旧4090约80ms结果来自同run E8/E13计时代理，此次需要真实029+030重新测，不能沿用旧结果认定通过。

目前无T4；此次4090实测即使小于100ms也只是开发诊断，最终仍须实际T4按官方完整口径、同predict/cost与最终CSV复测。只有实际使用的两名学生权重与必要配置计体积，教师和未调用训练断点不计；最终提交材料尚未完成。

只用官方train/未并训valid和已核验的非医学ImageNet权重，固定TTA/集成在规则6边界内，test未参与选模/权重拟合。CLAHE/DALI未加入。先完成真实双768部署闭环，再决定是否单独登记CLAHE提分实验；不把对比度增强当加速操作，不同时改损失/alpha/日程。

证据：`artifacts/kd_v1/dual768_fp16_review_20261028/{integrity.json,composition_preview.json,latency_input_preview.json}`；原概率比较：`../deliveries/032_dual_tiny768_fixed_weights_diagnostic/comparison_report.json`。本次只做CPU核查/组装预览与文档登记，没有新权重导出、组装、模型前向、GPU任务或训练。
