# 唯一25/75权重平均候选：仅诊断、未执行

这是无训练候选准备材料，不是033等值加速，也不是可提交模型。固定现成seed43/44的033 FP16张量，CPU上先转FP32，逐张量按0.25/0.75乘、加，再转FP16。整数张量要求完全相同。保留768、FP16和原图+hflip，以一个成员两次前向代替两个成员四次前向；参数平均不等于概率平均，不能预先承诺AUC收益。

registration.json在候选GPU执行前登记唯一公式和停止规则。validate_weight_average.py默认只打印登记；显式--execute才读取源权重和做完整valid。它不生成生产config，不伪造真实训练断点、教师或单模型训练来源，不接触test，也不替换033。输出目录必须新建且位于源模型、图片、基线结果目录之外。

脚本使用同环境033完整valid预测/report/receipt作为基线，并核验Python3.12、依赖、GPU及环境探针的CUDA/cuDNN。必须先完成原033及等值加速的验收；候选只由root独立决定是否执行。

下列路径对应本轮准备。若解释器或基线目录变化，应明确核对而非改用历史4090指标。

```bash
PYTHONHASHSEED=43 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
/tmp/dual768_timing_env/bin/python -B \
/home/featurize/work/cloud_project_v2/artifacts/kd_v1/optimization_2080ti_20261029/model_soup/validate_weight_average.py \
  --execute \
  --model-dir /home/featurize/work/deliveries/033_dual_tiny768_seed43_25_seed44_75/fp16_hflip2 \
  --image-root /home/featurize/data/cxr_first \
  --manifest /home/featurize/work/cloud_project_v2/data/clean_v1_reviewed/valid_manifest.csv \
  --baseline-predictions /home/featurize/work/deliveries/035_dual768_2080ti_optimization_20261029/valid_original/valid_predictions.csv \
  --baseline-report /home/featurize/work/deliveries/035_dual768_2080ti_optimization_20261029/valid_original/validation_report.json \
  --baseline-receipt /home/featurize/work/deliveries/035_dual768_2080ti_optimization_20261029/valid_original.receipt.json \
  --environment-probe /home/featurize/work/cloud_project_v2/artifacts/kd_v1/optimization_2080ti_20261029/environment/fallback_environment_probe_full.json \
  --output-dir /home/featurize/work/deliveries/035_dual768_2080ti_optimization_20261029/model_soup_25_75_diagnostic \
  > /home/featurize/work/deliveries/035_dual768_2080ti_optimization_20261029/model_soup_25_75_diagnostic.console.log 2>&1
```

输出：derived_weight_average_fp16.pt、derivation.json、逐图及Study valid CSV、diagnostic_report.json、execution_receipt.json。报告包含源/派生SHA、总体和十类AUC、固定0.5 F1及实际2844次前向计数。完整valid执行中加入forward计数hook，其墙钟不是正式100图测速；T4及官方通过字段保持false。

若候选AUC不严格高于同环境033，则停止，不追加比例搜索。即使点分提高，也须另做患者级不确定性、切片、资源及来源审查，不能声称稳定提分或直接提交。失败和无提升结果全部保留。
