# A4000 Tiny：从最佳权重热重启

本次只修改代码并做CPU权重/逻辑检查，没有启动训练、验证或模型前后向。
源检查点为 `runs/016_a4000_tiny_v2/best.pt`：epoch 5，完整valid AUC 0.7999705530021188。
第6轮的AUC/BCE回落是过拟合迹象，单轮波动不足以证明持续过拟合。

`train_tiny.py --resume` 的约定：

- 在线模型严格加载 `raw_model_state_dict`，EMA严格加载 `model_state_dict`；不再下载ImageNet预训练。
- epoch从源epoch+1开始；`--epochs 10`表示全局最后一轮10，因此本次最多跑6–10轮编号，共5轮。
- 旧检查点没有AdamW/GradScaler/RNG状态。这里重建优化器和GradScaler，是权重热重启，不是精确状态恢复。
- 默认峰值LR为源检查点基础LR乘0.1，即骨干5e-6、头5e-5；不再次warmup；局部成功更新从0开始做剩余轮数的余弦衰减，最低为峰值的0.01。
- 累计优化器/EMA步数从4740继续，避免EMA重新warmup覆盖旧EMA。早停从源AUC基线重新计数，patience最多3，min_delta=1e-4。
- 保留原BCE、Study等权、增强、Dropout、DropPath和WD；本轮不同时引入pos_weight或混合标签增强。
- 类别顺序、模型配置、预处理、清单SHA和EMA协议必须匹配，加载前后核对检查点SHA。
- 新run先保存继承的best及其阈值绑定；若新轮次未超过源AUC，保留源best。原实验目录只读。

手动启动命令（本次没有执行）：

```bash
mkdir -p /home/featurize/work/logs
tmux new-session -d -s 017_a4000_tiny_v2_warm \
  -c /home/featurize/work/cloud_project_v2 \
  'exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1 \
    /environment/miniconda3/envs/cxr-cloud/bin/python -u train_tiny.py \
    --resume /home/featurize/work/runs/016_a4000_tiny_v2/best.pt \
    --output /home/featurize/work/runs/017_a4000_tiny_v2_warm \
    --image-root /home/featurize/data/cxr_first \
    --work-root /home/featurize/work \
    --train-manifest /home/featurize/work/cloud_project_v2/data/clean_v1_reviewed/train_clean.csv \
    --valid-manifest /home/featurize/work/cloud_project_v2/data/clean_v1_reviewed/valid_manifest.csv \
    --epochs 10 --resume-lr-factor 0.1 \
    --batch-size 16 --accumulation-steps 2 --valid-batch-size 32 \
    --num-workers 4 --prefetch-factor 1 --patience 3 \
    >> /home/featurize/work/logs/017_a4000_tiny_v2_warm.log 2>&1' \
  \; set-option -w -t 017_a4000_tiny_v2_warm:0 remain-on-exit on
```

日志：`tail -f /home/featurize/work/logs/017_a4000_tiny_v2_warm.log`。
第一行策略日志应显示 `epoch 5 -> 6`、`baseline AUC=0.79997055`、`LR=5e-06/5e-05`。
新run/log/会话名需要保持唯一；已有output会被训练入口拒绝覆盖。
`run_cloud.sh`没有接入resume参数，因此本次直接在tmux里调用Python入口。

原入口备份：`/home/featurize/work/code_backups/warm_restart_20261019_005643/train_tiny.py`。

检查只在CPU进行：真实检查点两套各182个张量strict加载；学习率首末值、早停逻辑、阈值绑定和错误输入拒绝由 `tests/test_tiny_warm_restart.py` 验证。
