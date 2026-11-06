# LUT归一化及单次D2H候选

此目录是独立原型，未修改项目源码，未训练、未运行GPU。根代理统一安排GPU核验及正式计时。

`lut_prototype.py`提供：

- `normalization_table(mean,std)`：按原NumPy float32 copy→subtract→divide生成3×256值。没有重新整理数学公式。
- `load_image_pixels(path,sizes,letterbox)`：原解码、EXIF、非8bit拒绝、颜色转换及letterbox，返回uint8。第三参数可接入另行验证的原生letterbox，不会提前读图或跨图缓存。
- `ExactU8Input`：表按原FP32→CUDA FP16转换生成。灰度uint8上传、int32索引、Torch embedding查表，HWC输出直接permute为channels_last，无新增浮点运算；RGB保留原CPU FP32归一化路径。默认按模型尺寸初始化uint8 pinned host和device缓冲，正常串行`predict_image`内async H2D后同stream查表，最终D2H完成后才允许下一张复用。
- `member_results_to_cpu`：20个原FP32概率加1个0/1有限值标志打包，一次84bytes D2H。仍核验原logits，不以sigmoid输出代替finite检查，原float64融合函数不变。
- `CandidatePredictor(base,deploy_module,use_lut=True,merge_d2h=True,pixel_loader=None)`：复用已加载原模型，四前向和各项算术不变。两个开关可做单因素检查；不修改原模块全局函数。

像素表不从valid/test拟合，完整256值是固定8bit定义域；全部样本仍各自独立处理。GPU操作为固定PyTorch2.10基础算子，无新CUDA/TensorRT依赖、无本机编译架构标记。SM75/T4运行及实际时延仍须真实环境验证，不能预先保证加速。

安全复用约定：`ExactU8Input`的预分配staging是为batch1串行Predictor设计，必须在下一图片覆盖之前完成本图D2H。不要用它做跨图异步流水线，或脱离完整Predictor连续覆盖缓冲。CandidatePredictor若在D2H前遇到shape/模型等异常，会仅在错误路径同步后重抛原异常，避免调用者捕获并重试时覆盖尚未完成H2D的host缓冲。正常路径没有该显式同步；非有限logit异常在打包D2H完成后抛出。

## 已执行CPU检查

固定比赛镜像实际torch2.10.0+cu128/NumPy1.26.4，`cpu_checks.json`全部通过：穷举3×256 FP32字节一致、CPU int32 embedding和FP16/channels_last布局、20概率原样打包、Inf/-Inf/NaN原logit标志拒绝、24模式/EXIF组合和512/768、非8bit策略拒绝。它没有替代GPU输入逐bit验证。

CPU脚本原型修改后应另取新输出文件，不覆盖既有回执：

```bash
python /home/featurize/work/deliveries/036_dual768_2080ti_diagnosis_20261029/run_container.py \
  /home/featurize/work/deliveries/037_dual768_native_optimization_20261029/lut_candidate/check_cpu.py \
  --source-root /home/featurize/work/cloud_project_v2/artifacts/kd_v1/optimization_2080ti_20261029/source_optimized_v1 \
  --output /home/featurize/work/deliveries/037_dual768_native_optimization_20261029/lut_candidate/cpu_checks_next.json
```

## 待根代理执行的GPU核验

`verify_gpu.py`默认只打印metadata，不碰GPU；显式`--execute`才执行。它验证全256值、随机形状及105张原图的**最终FP16模型输入逐bit一致**，然后固定033、5预热、100计时图，对LUT单项/D2H单项/两项组合分别与原v1概率逐字节比较。该脚本是诊断，包含校验同步与重复读图，不是正式时延结果。

```bash
python /home/featurize/work/deliveries/036_dual768_2080ti_diagnosis_20261029/run_container.py \
  /home/featurize/work/deliveries/037_dual768_native_optimization_20261029/lut_candidate/verify_gpu.py \
  --source-root /home/featurize/work/cloud_project_v2/artifacts/kd_v1/optimization_2080ti_20261029/source_optimized_v1 \
  --model-dir /home/featurize/work/deliveries/033_dual_tiny768_seed43_25_seed44_75/fp16_hflip2 \
  --inputs-root /tmp/dual768_2080ti_diagnosis/inputs \
  --output-dir /home/featurize/work/deliveries/037_dual768_native_optimization_20261029/lut_candidate/gpu_verification_01 \
  --execute
```

主代理集成后仍需同一正式`cost.time_round`计时、三轮100图、保留全部输出及慢轮；逐图一致不能代替最终Study/二值/排序检查。原归一化仅约2.25ms、H2D设备活动约0.32ms，此候选不应被当作90ms目标的主要收益来源。
