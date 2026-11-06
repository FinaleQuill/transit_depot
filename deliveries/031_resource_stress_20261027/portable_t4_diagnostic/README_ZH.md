# 可迁移 T4 诊断材料

这是四个候选的本地计时材料合集，不是最终提交包。当前没有真实 T4 验收结果。
本包不包含训练集、教师权重或训练断点；config 中的教师/预训练来源文字是核验材料，
旧机器绝对路径仅作来源证据，推理不读取旧机器路径或下载原始预训练权重。

四候选：kd512（027 E7）、kd768（029 E8）、equal512_768（两者固定等权），
two768_timing_only（同一次 029 训练的 E8/E13，仅诊断两个 768 模型的计时开销）。
最后一项不是两个独立 seed/fold，不作精度或集成收益声明。各候选保持原始 FP16
权重、配置及导出/组合报告字节不变。单独候选的实际调用权重和必要配置均小于
500,000,000 bytes；合集总大小不能当成最终提交模型大小。

输入来自原始 valid：100 张、53 个完整 Study、40 位患者；另 5 张预热、4 个完整
Study、4 位患者。两组患者不相交。所有 105 张都是独立文件拷贝，未重编码。
valid_manifest.csv 是原始清单逐字节副本；inputs/selection.original.json 保留原选样记录。
inputs/selection.json 只将 manifest_path 改为 valid_manifest.csv，并增加
original_selection_sha256 / original_manifest_path 两项来源证据；身份、原图 SHA、
Study 完整性及其它原字段保留。output_relative_path 仍相对于 inputs/。

请先切换到本包根目录。以下默认仅 CPU 元数据核验，隐藏 CUDA 亦可运行：

```sh
CUDA_VISIBLE_DEVICES='' python run_native.py
```

依赖见 requirements_inference.txt。在指定 T4 环境使用已有兼容 CUDA/PyTorch；
不要用训练环境 wheel 替换官方环境。这里不自动安装依赖。明确执行时，每个命令
使用全新报告目录；启动器清除 CUDA_MPS_*，不设置 MPS 或线程百分比，使用
14 GiB 的 PyTorch allocator 上限（它不模拟带宽/架构，也不限制整个进程显存）：

```sh
python run_native.py --candidate kd512 --output-base reports/t4_kd512 --execute
python run_native.py --candidate kd768 --output-base reports/t4_kd768 --execute
python run_native.py --candidate equal512_768 --output-base reports/t4_equal --execute
python run_native.py --candidate two768_timing_only --output-base reports/t4_two768 --execute
```

启动器先核验 inventory.json 所列文件 SHA/大小，再校验部署配置和完整 Study 输入。
也可直接使用原包装器（必须在本包根目录运行；增加 --execute 才会启用 CUDA）：

```sh
python benchmark_local_resources.py --model-dir models/kd512 --pictures-100 inputs/pictures_100 --pictures-5 inputs/pictures_5 --selection-json inputs/selection.json --output-dir reports/direct_kd512 --profile actual_t4_diagnostic --memory-limit-gib 14
```

协议为 FP16、batch=1 串行、固定 hflip2 TTA、5 图预热、100 图连续 3 轮，
读图到预处理/前向/汇总/实际 CSV 写出计时。没有 DALI、CLAHE、跨图预取或预测缓存；
仅在同一张图内共享解码/相同尺寸预处理及 TTA 输入。权重/模型初始化不计时。
原生设备名称、SM 数、环境和 mean_ms 等由执行报告保存，必须核对实际硬件。
profile 名称本身不能证明使用 T4。包装器始终保留 T4_latency_certified=false、
official_compliance_pass=false；真实 T4 上的这些 valid 诊断也不等于官方选定
100 张 test 的验收。4090/MPS/allocator 上限结果都不能换算或宣称 T4 已通过。

最终仍须在官方指定 T4/输入/环境，独立运行 predict.py 与 cost.py，核对完整
submission.csv、全部实际部署成员体积、三轮时延和输出一致性。完整 test 清单和
图片不在本包中，须明确提供 --manifest / --image-root，不能使用旧机器默认路径。
预训练许可证/模型卡另存于 LICENSES/，并保留在原配置；未给本项目源码新增许可证。
inventory.json 的 files 记录本包固定文件及来源；它不列自身（避免自引用哈希），
执行后新生成的 reports/ 也不属于固定原始材料。

diagnostics/ 保留本机既有报告：results_v1 为旧预处理的 MPS 压测，native_controls
为旧原生结果，optimized_native_controls / optimized_native_repeat 为新预处理的
原生结果；其原始绝对路径是来源证据。provenance/deploy_before_fast_inputs.py
保留加速前源码，运行入口始终是本包根目录的最新版 deploy_kd.py。
这些都是本机 4090 诊断，不是真实 T4 统计。新版单 768 首轮完整均值 109.235 ms，
三轮均值 80.024 / 79.693 / 167.990 ms，异常慢轮原样保留；复测均值 77.335 ms。
双 768 E8/E13 计时代理两次均值 79.244 / 79.917 ms，不声明两个独立训练的精度。
CLAHE 与 DALI 均未启用；具体完整精度数字、每轮均值和环境以原始 JSON 为准。
