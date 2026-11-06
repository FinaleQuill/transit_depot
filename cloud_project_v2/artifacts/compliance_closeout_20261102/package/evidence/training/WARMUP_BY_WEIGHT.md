# 逐权重训练配置（据原始配置与日志）

LR warmup 与 KD warmup 是不同参数，不能称统一配方。下表为历史真实训练记录，不启动训练。

| 权重/产物 | 是否用软标签 | alpha曲线 | KD warmup轮数 | LR warmup轮数 | seed | 起点 | 配置证据 | 日志证据 | 用途 |
|---|---|---|---|---|---|---|---|---|---|
| teacher_fold_0 | 否 | 不适用（监督BCE） | 不适用 | 1 | 43 | ImageNet Base预训练 | evidence/teachers/fold_0/config.json | evidence/teachers/fold_0/history.jsonl | 仅离线产train软标签；不推理 |
| teacher_fold_1 | 否 | 不适用（监督BCE） | 不适用 | 1 | 43 | ImageNet Base预训练 | evidence/teachers/fold_1/config.json | evidence/teachers/fold_1/history.jsonl | 仅离线产train软标签；不推理 |
| teacher_fold_2 | 否 | 不适用（监督BCE） | 不适用 | 1 | 43 | ImageNet Base预训练 | evidence/teachers/fold_2/config.json | evidence/teachers/fold_2/history.jsonl | 仅离线产train软标签；不推理 |
| teacher_fold_3 | 否 | 不适用（监督BCE） | 不适用 | 1 | 43 | ImageNet Base预训练 | evidence/teachers/fold_3/config.json | evidence/teachers/fold_3/history.jsonl | 仅离线产train软标签；不推理 |
| teacher_fold_4 | 否 | 不适用（监督BCE） | 不适用 | 1 | 43 | ImageNet Base预训练 | evidence/teachers/fold_4/config.json | evidence/teachers/fold_4/history.jsonl | 仅离线产train软标签；不推理 |
| A（041 Tiny） | 是（train折外logits） | alpha=0.2*min(1,((epoch-1)+(batch_index+1)/batches)/2)，E1末0.10，E2末0.20；首条日志四舍五入为0.0000 | 2 | 1 | 44 | ImageNet Tiny预训练 | evidence/training/student_A/config.json | evidence/training/student_A/console_20261030T052954Z_dc99f91e.log:1 | 历史候选/祖先；非最终 |
| B（043/045 Base学生） | 是（train折外logits） | alpha=0.2*min(1,((epoch-1)+(batch_index+1)/batches)/2)，E1末0.10，E2末0.20；首条日志四舍五入为0.0000 | 2 | 1 | 44 | ImageNet Base预训练 | evidence/training/student_B/config.json | evidence/training/student_B/console_20261030T103505Z_a5480d3c.log:1 | 历史候选/祖先；非最终 |
| X1（045 Tiny） | 是（train折外logits） | 从首个训练pack开始固定alpha=0.20；首轮日志alpha=0.2000 | 0 | 1 | 46 | Tiny A E9 FP32 EMA | evidence/training/X1/config.json | evidence/training/X1/console_20261101T065749Z_f2a2138c.log:1 | 历史候选/祖先；非最终 |
| 裁剪对照A_tiny（弃用） | 是（train折外logits） | 从首个训练pack开始固定alpha=0.20；首轮日志alpha=0.2000 | 0 | 1 | 48 | Tiny A E9 FP32 EMA；E1后保留状态迁移 | evidence/training/crop_A_tiny/config.json | evidence/training/crop_A_tiny/original_E1_console.log:1 | 弃用实验；不推理 |
| 裁剪对照A_base（弃用） | 是（train折外logits） | 从首个训练pack开始固定alpha=0.20；首轮日志alpha=0.2000 | 0 | 1 | 48 | Base B E10 FP32 EMA | evidence/training/crop_A_base/config.json | evidence/training/crop_A_base/console_20261101T221700Z_64e06eeb.log:1 | 弃用实验；不推理 |
| 裁剪B_tiny（弃用） | 是（train折外logits） | 从首个训练pack开始固定alpha=0.20；首轮日志alpha=0.2000 | 0 | 1 | 48 | Tiny A E9 FP32 EMA | evidence/training/crop_B_tiny/config.json | evidence/training/crop_B_tiny/console_20261102T024037Z_61c80be4.log:1 | 弃用实验；不推理 |
| 裁剪B_base（弃用） | 是（train折外logits） | 从首个训练pack开始固定alpha=0.20；首轮日志alpha=0.2000 | 0 | 1 | 48 | Base B E10 FP32 EMA | evidence/training/crop_B_base/config.json | evidence/training/crop_B_base/console_20261102T055123Z_db7ec019.log:1 | 弃用实验；不推理 |

初始A的hard loss为逐图BCE再按Study图数加权，Base B及后续X1/裁剪A/B的hard loss为完整Study概率均值BCE。所有有KD者均为Bernoulli KL(T=2)。
初始A alpha公式源码：`evidence/training/student_A/source/train_distill.py:121`；X1/crop的原始config及首条日志显示KD warmup=0、alpha=0.20。
教师全程不读取held-out图片做训练或选epoch；固定终点best.pt证据在`evidence/teachers/SELECTION_EVIDENCE.md`。
