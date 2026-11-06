# 028短日程中断核查与续训

> 阶段更新：028已于17:55正常E12结束，当前无需续训。下方中断和恢复命令保留为历史记录；当前见[完成复盘及029交接](KD512_SHORT12_REVIEW_20261027.md)。

2026-10-27 17:10（香港）核查：028没有训练完，也没有正常早停；当前训练进程不存在、4090空闲，需由用户执行下方续训命令。

## 已完成与未完成

- 原运行15:21启动，E1–7完整训练/验证并保存；七轮AUC依次为.728622/.765069/.790449/.798969/.802221/.803111/.803891。
- E7最佳AUC=.8038905628482913，early stopping bad_epochs=0，AMP跳步0；best/last均是E7完整状态。
- 最后控制台在16:15:20停于`epoch=8 batch=2000/3791`，没有E8验证/保存、异常栈或正常退出记录；执行凭据exit_code仍为null，原child_pid不存在。退出原因没有记录，不推断是断连、OOM或主动终止。
- `status.json`遗留running且完成数为7，不能仅以这个字段判断进程活性。原状态、凭据、日志、归档及权重全部保留，没有手工改写成completed/failed。

## CPU恢复核验

生产恢复校验与best/last协调通过，实际权重SHA绑定status/history一致；14条逐轮账本链、33份归档Python源码与当前版本一致。已核对train/valid隔离、清单/折表及OOF缓存哈希，raw模型、EMA、优化器、scaler、RNG和计数均在检查点中。精确CLI参数预览通过，未执行模型构建/前向/训练。审查前后原run全部文件哈希一致。

| 检查点 | SHA256 |
| --- | --- |
| best.pt | `4e08a64bb236431c40b0e8e05ad91feee628c5b63385c3f7eceb2b4ccb380b70` |
| last.pt | `2f89f16aa80d3025faa7d42f7e94fd2ca6a34d004ea8c02319167f5faa90d1b3` |

本轮尚未完成，不做最终患者配对区间、误差切片或实际FP16排名结论。当前E7低于027原BF16最佳.805593834；短日程仍在提高，继续预定12轮后再判断是否值得替换。旧027/BCE及已验证FP16不变。

## 当前唯一续训命令

建议在tmux终端执行；保持原全部参数，新增`--resume`。从已保存E7恢复优化器/EMA等状态后重新跑E8，未保存的半轮工作会重算；`epochs`仍为12，不能改成剩余5轮，否则改变学习率日程/恢复配置。预计还需约35–45分钟，取决于实际吞吐与启动开销。

```bash
cd /home/featurize/work/cloud_project_v2
NO_ALBUMENTATIONS_UPDATE=1 /home/featurize/.conda/envs/cxr-kd-v1/bin/python -u train_distill.py \
  --fold-plan data/cv5_overnight/folds.json \
  --soft-labels ../runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2 \
  --training-mode distillation --alpha 0.2 --temperature 2 \
  --image-size 512 --seed 43 --validation-tta hflip2 \
  --batch-size 8 --accumulation-steps 4 \
  --epochs 12 --patience 5 \
  --output ../runs/028_tiny512_kd_short12_seed43 \
  --resume ../runs/028_tiny512_kd_short12_seed43/last.pt \
  --execute
```

另开终端监测（原中断日志保留，续训新增自己的日志）：

```bash
tail -n 20 -F /home/featurize/work/runs/028_tiny512_kd_short12_seed43/evidence/console_*.log
```

完成后先复盘整体/十类及资源，再决定实际FP16验证或下一训练；不自动追加768/seed/损失修改。恢复沿用原train/独立valid与严格OOF、固定TTA，不引入test选择；最终T4≤100ms、实际文件≤500MB和提交材料仍待独立验收。

证据：`artifacts/kd_v1/short12_interruption_20261027/{audit.json,audit_script.py}`。此次只有CPU读取/哈希/元数据检查和文档更新，未启动或恢复训练，未修改运行源码、原始数据及已有模型。
