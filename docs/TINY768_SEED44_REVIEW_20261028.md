# 第二个Tiny768完成复盘与FP16融合交接

2026-10-28（香港）。030正常完成13轮，E8最佳；下一阶段只做030实际FP16导出/valid，以及预定双768三组概率权重诊断。继续双768目标，先验收现有成果，再决定是否需要CLAHE等新增实验。本次未启动训练或GPU预测。

## 正常早停与来源核验

030单次命令从10月27日23:18:36至28日01:17:58正常exit0，耗时1小时59分21秒。13轮完整覆盖30328张train，E9–13连续五轮未超过早停参考AUC+.0001，state=early_stopped；25是上限，不需要续训到25。

E8 best AUC=.8075698171794465，E13 last=.8046978962388822。best/last真实SHA、raw/EMA有限性、完整恢复状态、21条账本链及41份原执行源码归档通过CPU核验；49283批反向、12324次优化器/EMA更新，AMP跳步0。原run53文件前后SHA不变，不用后来修改的比较工具冒充原执行源码。

配置与029仅seed43→44及output不同，原ImageNet初始化SHA、train/valid清单与严格OOF缓存绑定通过；11055个train患者在五教师外折中互斥，train/valid身份/内容隔离通过。缓存仍为30328×2×10双视图原始logits，原教师/缓存保留；此次没有重新全量哈希原图或五教师大权重。

## 成绩与后期变化

下表均为训练时原生BF16、hflip2、同一859 Study的valid，不混入实际FP16分数。

| 学生 | 最佳轮 | Macro-AUC |
| --- | ---: | ---: |
| 029 Tiny768 seed43 | 8 | .804176963 |
| 030 Tiny768 seed44 | 8 | **.807569817** |

提高.003392854，约0.3393个百分点，六类升四类降；相对027 Tiny512同BF16最佳.805593834提高.001975983、五升五降。两个seed使用完整train，是两个学生而非两个CV折。当前不能从一个较高点分证明稳定优势，也不能将030原生BF16与此前512+768实际FP16的.807157435微小差值直接判胜。

| 类别 | 029最佳 | 030最佳 | 030−029 | 030 E13−E8 |
| --- | ---: | ---: | ---: | ---: |
| 1 | .693064 | .711602 | +.018538 | -.048328 |
| 2 | .854179 | .845203 | -.008976 | +.008943 |
| 3 | .800417 | .804000 | +.003583 | +.003503 |
| 4 | .745464 | .759454 | +.013990 | -.001887 |
| 5 | .880774 | .880265 | -.000508 | +.000532 |
| 6 | .777482 | .784901 | +.007418 | +.006089 |
| 7 | .789658 | .788343 | -.001315 | +.000105 |
| 8 | .721365 | .717094 | -.004271 | +.004365 |
| 9 | .901358 | .906588 | +.005230 | +.000498 |
| 10 | .878009 | .878249 | +.000239 | -.002538 |

1/4/6/9是030主要增益，029在2/5/7/8仍有更高点分，保留其互补可能；边际AUC不能证明逐病例错误互补，更不能保证融合胜过030。030目前没有保存的逐Study部署概率，因此患者配对区间、真实融合分数及单图/多图误差切片均待下阶段。

E8→13总体AUC下降.002871921，train loss .670808→.517713、valid BCE .470006→.431231继续下降；十类七升三降，主要1类下降.048328拖累，其valid只有24阳性Study/16患者，不能说所有病种都过拟合。现有结果支持后期训练目标与排序表现分化，尚不能确定病理原因。KL从E9最低.183896升至E13 .207154不足以推出应修改alpha。保留E8，按原协议早停合理。

训练循环平均8分19.65秒/轮，与029的8分18.57秒接近；记录allocated峰值2.159GiB不是整卡峰值。训练checkpoint含优化器等，每份约450MB，不作为部署大小。

## 下一阶段三条命令，合计暂估5–15分钟

依次执行，前一条成功后再执行下一条；若报错就停在该步骤，不删除或覆盖旧输出。030导出/验证及032比较目录在交接时均不存在。无需重跑029导出/valid。

第一条只在CPU导出E8，通常约10–60秒，不训练、不做前向：

```bash
cd /home/featurize/work/cloud_project_v2
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/featurize/.conda/envs/cxr-kd-v1/bin/python export_student.py \
  --checkpoint ../runs/030_tiny768_kd_seed44/best.pt \
  --tta hflip2 \
  --output ../deliveries/030_tiny768_kd_seed44/fp16_hflip2
```

第二条使用实际FP16与predict/cost相同的Predictor，验证完整valid，暂估3–8分钟；本条不要设置`CUDA_VISIBLE_DEVICES=''`：

```bash
/home/featurize/.conda/envs/cxr-kd-v1/bin/python validate_student.py \
  --model-dir ../deliveries/030_tiny768_kd_seed44/fp16_hflip2 \
  --image-root /home/featurize/data/cxr_first \
  --manifest data/clean_v1_reviewed/valid_manifest.csv \
  --output-dir ../deliveries/030_tiny768_kd_seed44/valid_fp16_hflip2
```

第三条只用CPU比较两份实际FP16概率，暂估1–5分钟；三组权重仍按预先登记的seed43、seed44顺序，不添加其它候选：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/featurize/.conda/envs/cxr-kd-v1/bin/python compare_student_candidates.py \
  --validation-dirs \
    ../deliveries/029_tiny768_kd_seed43/valid_fp16_hflip2 \
    ../deliveries/030_tiny768_kd_seed44/valid_fp16_hflip2 \
  --names kd768_seed43_fp16_hflip2 kd768_seed44_fp16_hflip2 \
  --reference kd768_seed44_fp16_hflip2 \
  --manifest data/clean_v1_reviewed/valid_manifest.csv \
  --combinations-json docs/DUAL_TINY768_WEIGHTS_20261027.json \
  --bootstrap 1000 --seed 20261027 \
  --output ../deliveries/032_dual_tiny768_fixed_weights_diagnostic
```

新增`--reference`只指定点分差及患者配对区间的参照，不改变成员/来源/权重的顺序、候选概率或任何权重。这里事先用030作参照，重点检查融合能否超过新学生；如果实际FP16使单模排名变化，后续审查必须同时面对真实更强的单模型。省略选项仍保持首成员参照的历史行为。17项定向CPU测试通过，包括第二成员参照、权重顺序、独立点分/患者bootstrap重算及非法参照无输出。

比较报告包含50/50、25/75、75/25全局概率平均及两个单模型，十类AUC、患者聚类配对区间和单图/多图Study切片；不拟合阈值或十套类别权重。候选权重虽事先登记，按valid选择仍是选参，患者区间不修正历史选模/多重比较或训练随机性；`weights_fitted=false`只表示脚本没有连续权重拟合，不表示后续选择独立于valid。

**三条完成后回来复盘。**再决定实际组装哪一组权重、同一组合完整FP16验证及计时；CPU概率平均不替代真实组合部署验收，不自动开始新训练或CLAHE实验。

## 规则与交付状态

当前只使用官方train、未并训valid及已核验的非医学公开预训练；第6条允许固定TTA与集成，权重/预处理须在train/valid固定，test不参与选择。rules.pdf哈希与既有官方版本一致，未改规则或数据协议。

预计单Tiny约67MB、双Tiny约129MB，030实际导出后重新统计。已有4090双768 E8/E13同run计时代理约80ms不代表029+030正式成绩，也不构成T4认证；最终实际组合须按官方batch1串行、100图/另5图预热/3轮、读图至写出平均≤100ms验收。CLAHE/DALI未加入，本阶段不改预处理。

CPU审计证据：`artifacts/kd_v1/tiny768_seed44_review_20261028/{integrity.json,curve_analysis.json}`。实际FP16概率与区间尚未生成，本次只修改比较工具和文档，未导出模型、执行GPU预测或新训练；原执行源码/权重/缓存保持原位。
