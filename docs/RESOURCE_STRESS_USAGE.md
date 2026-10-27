# 隔离资源压测与未来T4复测

本流程在4090上用独立MPS客户端限制计算资源，不修改GPU时钟、功耗、计算模式或既有推理实现。它不是T4仿真器，所有报告明确保留T4未认证。真实T4不可用时，只用于开发阶段了解资源收紧后的端到端耗时。

NVIDIA的MPS支持按客户端设置可用线程比例，部分比例会按硬件粒度取整；本机已验证100/30/20/10%分别得到128/38/24/12个CUDA可见SM，并核对过实际MPS客户端连接。16GiB限制同时声明在MPS客户端与PyTorch分配器；它不把显存带宽、缓存、Tensor Core或CUDA内核变成T4。[MPS官方说明](https://docs.nvidia.com/deploy/mps/latest/appendix-environment-variables.html)

T4官方规格为Turing架构、16GB GDDR6、320+GB/s带宽、FP32峰值8.1TFLOPS及混合FP16/FP32峰值65TFLOPS。因此相同SM数或一个FP16矩阵乘吞吐都不能证明两卡等效。[T4官方规格](https://www.nvidia.com/en-us/data-center/tesla-t4/)

## 输入与候选

`prepare_latency_inputs.py`仅按valid身份、每个Study图数与固定seed选择完整Study：100张计时图、另5张预热图，患者不重叠。只复制原图字节并核对SHA，保留valid身份和原清单，不读取标签/质量/预测作选择，不伪造test清单。默认预览，`--execute`才复制。

```bash
python prepare_latency_inputs.py \
  --manifest data/clean_v1_reviewed/valid_manifest.csv \
  --image-root /home/featurize/data/cxr_first \
  --output NEW_INPUT_DIRECTORY --seed 20261027 --execute
```

`benchmark_local_resources.py`复用原`Predictor`和`cost.time_round`，FP16、batch1串行、5图预热、三轮100图，计入读文件/解码/预处理/传输/所有成员与TTA/后处理/实际CSV写出，模型初始化不计时。按逐图三轮均值再100图平均，并报告中位数/P95/最大值、实际文件体积及三轮预测一致性。受限档位若CUDA可见SM数没有下降，会在构建模型之前失败。

`run_resource_stress.py`统一创建独立临时MPS控制器，启动全新子进程应用各档位，默认只预览；每档另记录4096阶FP16 GEMM吞吐和大缓冲拷贝带宽作为环境诊断，不拟合成T4时间。无后台占用负载、无插入sleep。正常结束及异常都尝试关闭本次私有控制器，保留日志和成功/失败状态，不操作其它控制器。

## 本机操作示例

训练环境Python为`/home/featurize/.conda/envs/cxr-kd-v1/bin/python`。下面路径属于031本地开发压测，未验证准确率的组合不等于已选定的部署方案。新输出目录不得存在；运行前确保无其它GPU计算任务，结束后检查控制器停止记录和GPU进程。

```bash
/home/featurize/.conda/envs/cxr-kd-v1/bin/python run_resource_stress.py \
  --model-dirs \
    ../deliveries/027_tiny512_kd_seed43/fp16_hflip2 \
    ../deliveries/029_tiny768_kd_seed43/fp16_hflip2 \
    ../deliveries/031_resource_stress_20261027/models/kd512_kd768_equal_hflip2 \
  --names kd512 kd768 equal512_768 \
  --inputs ../deliveries/031_resource_stress_20261027/inputs \
  --output NEW_RESULT_DIRECTORY \
  --percentages 100 30 20 10 --execute
```

每个候选均为固定hflip2：单模型每图两次前向，512+768各一个成员则四次；融合是预先固定的等权连续概率平均，不做阈值投票或参数拟合。计时结果不说明组合精度提高，完整valid实际FP16验证仍是另一阶段。

## 迁移与边界

输入selection.json保留原valid清单的路径和SHA。便携材料需要复制同一清单，并显式调整副本的manifest_path，同时记录原selection SHA及原路径；图像/清单字节和来源不改写。运行源码、依赖、所有候选权重/config和选择记录一起保留，禁止仅拷一份模型后依赖在线下载。

将来在真实T4上可用相同105图和同一Predictor做开发复测；T4通常不应使用4090的MPS档位，分配器上限也须小于实际可用容量。最终正式验收仍执行官方`predict.py`/`cost.py`，绑定最终CSV、实际部署权重/config、官方完整test清单与评测图片；必须按规则核验T4环境、平均≤100ms、全部实际推理文件≤500,000,000bytes及CSV一致性。本地valid压测不能替代官方复现。

此次新增工具仅支持推理开发诊断，不训练、不改变数据/标签/教师/蒸馏来源。使用CPU模拟测试验证输入闭合、覆盖、隔离、配额失效拒绝、计时汇总、非T4认证和控制器异常清理；真实压测来源及结果另存版本化目录。
