# 当前方案：教师折外TTA蒸馏与多分辨率学生集成

> 2026-10-27当前阶段：027 KD已12轮早停，E7最佳.805593834。按[KD512验收与改进判断](KD512_REVIEW_20261027.md)先导出E7 FP16/valid，再配对比较及固定BCE/KD互补诊断；028短日程、768和额外seed不自动启动。下方旧compare调度器仅保留为后续入口，T4计时仍待完成。


用户已批准用固定TTA和多个学生争取成绩。现有五个Base768教师保持原权重与20/12/12/12/12轮协议；学生采用同一ImageNet来源的V1 Tiny，先做512/768同seed对照，再按验证结果补模型。
本次交付只修改和检查代码，没有运行下面的GPU命令。

## 教师状态与运行环境

五教师已于2026-10-26 19:26（香港）全部完成。最终权重和患者划分不变，集合入口为`../runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_collection.json`。

```bash
cd /home/featurize/work/cloud_project_v2
KD_PYTHON=/home/featurize/.conda/envs/cxr-kd-v1/bin/python
bash run_mixed_teachers.sh --status
```

状态应为`phase: completed`和`all_teachers_complete: true`。历史`error`字符串可能仍保留前次失败原因，实际完成以阶段、最终权重及完成记录为准。
新入口默认只预览；显式`--execute`才开始提取、评估或实验训练。原始图像、标签、清单与教师共享源码保持冻结。

## 1. 评估教师的固定TTA和五模型平均

```bash
"$KD_PYTHON" validate_teacher_tta.py \
  --fold-plan data/cv5_overnight/folds.json \
  --teacher-collection ../runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_collection.json \
  --image-root /home/featurize/data/cxr_first \
  --output-dir ../runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_tta_valid \
  --execute
```

只读取折表绑定的官方valid。比较每个固定最终EMA和五教师等权平均的identity、原图＋水平翻转，报告Macro-AUC与十类AUC并保存Study概率。不依据外折重新挑教师epoch，不把valid并入训练。

## 2. 提取严格OOF逐视图logits

```bash
"$KD_PYTHON" extract_soft_labels_tta.py \
  --fold-plan data/cv5_overnight/folds.json \
  --teacher-collection ../runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_collection.json \
  --image-root /home/featurize/data/cxr_first \
  --tta identity_hflip \
  --output ../runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2 \
  --execute
```

默认原图＋水平翻转，保存30328张train的`N×2×10` FP32原始logits；仅对应的外折教师预测该图，不能让训练过该患者的其他教师参与平均。新schema为`cxr_strict_oof_view_logits_v2`，输出`soft_labels.npz`和`metadata.json`，新目录不能已存在。

学生对每个视图先执行类别权重logit对齐，再除T=2，sigmoid后平均，构造完整Bernoulli KL乘T²；总损失为`0.8×硬标签BCE + 0.2×软目标KL`，含原有KD渐进启用与Study等权。缓存不提前施加温度或伪称概率为原始logits。

## 3. 先训练512/768两个学生

先预览：

```bash
"$KD_PYTHON" run_student_experiments.py --stage compare
```

准备开始这两个学生时，显式执行：

```bash
"$KD_PYTHON" run_student_experiments.py --stage compare --execute
```

该阶段只依次训练`../runs/025_tiny_multires_kd/tiny512_seed43`与`tiny768_seed43`。两者同seed43、同预训练来源、同OOF缓存、相同有效batch32与损失；512微批8×累积4，768微批4×累积8。最多25轮、patience5，实际按各自双视图TTA的官方valid Study Macro-AUC早停/选EMA检查点。

每次仍只训练一个模型。它们使用完整train，是多seed/多分辨率成员，不是学生交叉验证折。现有教师缓存不能直接作为新学生CV的独立OOF证明。
异常或暂停时不启下一学生，重放相同命令从原目录完整断点恢复；首轮best先于last落盘的中断也可恢复。`STOP`文件表示暂停意图，恢复前先确认并移除相应实验根目录和学生目录的STOP。不要同时运行两个实验调度器。

额外成员分别显式选择，不随compare自动启动：

```bash
"$KD_PYTHON" run_student_experiments.py --stage extra512 --execute
"$KD_PYTHON" run_student_experiments.py --stage extra768 --execute
```

分别为seed47的512、768。`--stage four512`提供512的43/47/53/59四个seed；已完成成员只核验恢复状态，不重新训练。

## 4. FP16导出、TTA对照与组合

```bash
"$KD_PYTHON" export_student.py \
  --checkpoint ../runs/025_tiny_multires_kd/tiny512_seed43/best.pt \
  --output ../deliveries/025_tiny_multires_kd/tiny512_seed43

"$KD_PYTHON" export_student.py \
  --checkpoint ../runs/025_tiny_multires_kd/tiny768_seed43/best.pt \
  --output ../deliveries/025_tiny_multires_kd/tiny768_seed43
```

新schema8检查点默认按训练声明导出hflip2；可用`--tta identity`另建输出做无TTA对照。原schema7默认保持原identity512语义，不能把它改写成新训练记录。

组合两个已导出的模型：

```bash
"$KD_PYTHON" compose_students.py \
  --model-dirs ../deliveries/025_tiny_multires_kd/tiny512_seed43 ../deliveries/025_tiny_multires_kd/tiny768_seed43 \
  --tta hflip2 \
  --output ../deliveries/025_tiny_multires_kd/ensemble_512_768 \
  --execute
```

默认等权平均十类连续概率。`--weights 1 1`指定固定成员权重；`--weights-json PATH`支持固定M×10逐类权重。权重应仅从train/valid阶段确定，不在test上拟合。先比较少量预定组合，避免在小valid上搜索大量逐类参数。

支持1–4个独立学生、各自512/768和identity/hflip2。重复权重或同一源checkpoint不能充当不同成员。每张原图只解码一次，按分辨率复用预处理；每成员与每TTA视图均按batch1串行计算，再进行Study概率平均。

| 候选 | FP16纯参数估算 | 双视图前向数 |
| --- | ---: | ---: |
| 512＋768 | 111.31MB | 4 |
| 2×512＋768 | 166.97MB | 6 |
| 2×768 | 111.31MB | 4 |
| 4×512 | 222.62MB | 8 |

估算不含必要config；导出/组合会实际统计所有加载权重与必要配置≤500,000,000 bytes。以上不是T4耗时测量，也不自动表示最终选择。

## 5. 按十类AUC比较真实部署

对每个候选，包括单模型、不同TTA和最终组合，分别运行：

```bash
"$KD_PYTHON" validate_student.py \
  --model-dir ../deliveries/025_tiny_multires_kd/ensemble_512_768 \
  --image-root /home/featurize/data/cxr_first \
  --output-dir ../deliveries/025_tiny_multires_kd/valid_ensemble_512_768
```

`--model-dir`和`--output-dir`换成相应单学生即可。验证采用实际导出的FP16、TTA和所有成员，输出逐Study十类概率及其SHA256、每类AUC、Macro-AUC和固定0.5 F1，不拟合阈值。

512和768单学生的验证结果都生成后，可以仅比较保存概率：

```bash
"$KD_PYTHON" compare_student_candidates.py \
  --validation-dirs ../deliveries/025_tiny_multires_kd/valid_tiny512_seed43 ../deliveries/025_tiny_multires_kd/valid_tiny768_seed43 \
  --names tiny512 tiny768 --bootstrap 500 \
  --output ../deliveries/025_tiny_multires_kd/comparison_512_768
```

报告十类差异、等权概率组合、单图/多图Study切片，以及可选的患者聚类配对bootstrap区间。该步骤不加载模型、不拟合权重、不自动挑最终部署；没有病灶大小或早期分期标签时，不把切片结果冒称病理分期结论。

## 6. 最终预测、T4计时和材料

确定候选后使用现有`predict.py`生成完整submission.csv；`cost.py`加载同一部署，同样执行所有TTA和成员：100图、batch1串行、另5图预热、3轮，计入解码、预处理、传输、概率汇总及结果写出。非T4结果只作诊断，最终平均≤100ms须实测，预测必须与最终CSV一致。

打包前合并实际使用的每个学生训练证据，学生顺序与部署members顺序相同：

```bash
"$KD_PYTHON" build_training_evidence.py \
  --model-dir ../deliveries/025_tiny_multires_kd/ensemble_512_768 \
  --teacher-evidence ../runs/024_convnext_v1_base768_mixed20_12_cv5/evidence \
  --student-runs ../runs/025_tiny_multires_kd/tiny512_seed43 ../runs/025_tiny_multires_kd/tiny768_seed43 \
  --output ../deliveries/025_tiny_multires_kd/combined_evidence --execute
```

再将`package_kd_submission.py --training-evidence`指向该combined_evidence。打包遍历所有成员收集初始化来源、完整许可、实际执行源码、日志、逐轮时间/hash；原始预训练文件按SHA去重。只剩resume日志时继续标partial，不补造完整记录。
缺少最终CSV、同一组合的FP16验证、合格T4计时或完整训练证据时，核验包保持pending。
