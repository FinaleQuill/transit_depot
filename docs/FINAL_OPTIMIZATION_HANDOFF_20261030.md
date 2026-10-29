# 81ms等值优化与完整项目交接

更新于2026-10-30，Asia/Hong_Kong。本文与AGENT_HANDOFF顶部优先于所有10月29日历史状态。用户已授权实施并完成优化；旧“确认前不实施、默认v1、尚未达100ms”不是当前指令。

## 当前成果及验收范围

目标2080Ti端到端≤90ms已完成，当前默认是**原生AVX2/generic并行LANCZOS + 精确uint8→FP16 LUT + 单次概率/有限值标志D2H**，不是旧v1。

| 同一固定比赛镜像，105张固定valid图 | 三轮均值ms/图 | 总均值ms/图 |
|---|---|---:|
| 本轮新测v1，本地盘 | 123.015035 / 123.690501 / 127.559035 | 124.754857 |
| pthread初版，本地盘 | 94.899195 / 89.492180 / 95.352051 | 93.247809 |
| AVX2-only，本地盘 | 89.696651 / 82.674156 / 81.530947 | 84.633918 |
| **默认组合版，本地盘** | **79.697875 / 82.149228 / 82.682735** | **81.509946** |
| **默认组合版，NFS读写** | **79.847088 / 80.855821 / 81.640502** | **80.781137** |
| **完整提交项目副本，独立离线重建** | **85.290579 / 75.751842 / 85.530240** | **82.190887** |

本地同环境相对新测v1降低34.66%。旧149.81→133.44ms属于历史会话；不要混用不同会话、磁盘或Profiler口径计算收益。全部候选、轮次及慢值保留，无复跑择优。主计时均为100图、另5预热、batch1串行三轮，从读文件到实际CSV写出，初始化排除；真实68SM、8GiB本地allocator协议，无MPS/功耗/时钟调整。

105张图的uint8缩放、float32及最终FP16输入字节一致；100图、53Study的所有候选/三轮/独立项目输出与冻结2080Ti v1逐字节相同，概率、固定二值、排序、并列不变。**本阶段未重跑完整1422图valid**，不能把035旧v1完整valid成绩写成新组合版完整valid实跑。跨卡浮点及排序差异旧记录仍保留，不能宣称跨卡逐位一致。

033两模型、config、原submission.csv、原逐图CSV均未改；768、FP16、两成员25%/75%、各identity+hflip四次串行前向及Study融合不变。本轮训练0次、test重生成0次、上传0次。

## 实现及源码版本

- `65064d5`：原生等值缩放、LUT、一次D2H及测试。`native_lanczos/resample.c`保持Pillow10.4的系数生成、22位INT32、H→uint8→V及舍入；pthread持久池最多8个活跃线程。AVX2运行时检测，generic fallback；无`-march=native`、OpenMP或CUDA编译依赖。
- `d539db7`：共享seed43、逐图及Study概率/二值/Gate/排序/并列核验、离线准备及项目打包。`0dd71a1`记录验收结果；后续交接提交仅文档。
- `inference_inputs.py`穷举3×256原NumPy float32运算值，再按原CUDA转换成FP16；灰度uint8单图上传并查表成channels_last，RGB保留原CPU算术。最终21个FP32包含20个成员概率及有限值标志，一次D2H；原FP64融合不变。仅异常路径有额外fence，保护pinned缓冲重试。
- `prepare_inference.py`不加载模型：恢复固定公开addon、检查core不被覆盖、用新子进程实际import核心/timm/native、构建CPU算子。`--force-build`父进程不会预先映射待覆盖的.so。
- 定向CPU集成39项、原生7项、LUT6项、打包24项及一致性17项等已通过（部分交叉覆盖，不把次数相加冒称唯一测试数）。原型GPU错误路径、105输入及100概率、最终独立项目三轮均有真实回执。未重新跑整个训练测试套件。

原生wheel1.1.0为36,908bytes，SHA256 `2da965c87d512e1170aeeaa96f811c7a73ec419e22d6ca1b7c0257c1e475e621`；已验证inplace so SHA `565419dc32e7f45e7d0c1c155c2d8b4b1da74d298299a43199563f1711ea391f`。独立重建因构建路径可有不同二进制SHA，须记录新回执和像素一致性，不能冒称原so字节不变。ABI固定CPython3.12/Pillow10.4/x86_64；GLIBC≥2.34，比赛镜像2.39满足。不支持启动原生线程池后再fork，当前单进程串行CLI不触及。

## 权威材料位置

均相对`/home/featurize/work`：

| 位置 | 内容/用途 |
|---|---|
| `cloud_project_v2/` | 当前Git源码；不要拿034旧源码覆盖 |
| `deliveries/033_dual_tiny768_seed43_25_seed44_75/` | 冻结模型/config、完整valid及最终test原CSV |
| `deliveries/034_dual768_proxy_timing_20261029/bundle/` | 105张原图/selection/manifest及旧源码；只取输入与原参考 |
| `deliveries/035_dual768_2080ti_optimization_20261029/` | 历史v1/v2/v3/soup及旧完整valid证据，只读保留 |
| `deliveries/036_dual768_2080ti_diagnosis_20261029/` | 环境恢复、PDF审计、CPU/GPU分段及NFS长尾诊断 |
| `deliveries/037_dual768_native_optimization_20261029/` | 本轮全部源码快照、native/wheel、测试、GPU/三轮/独立项目证据；总入口`optimization_summary.json` |
| `deliveries/038_dual768_optimized_project_20261030/` | 已组装完整提交项目，保持原件只读 |
| `deliveries/038_dual768_optimized_project_20261030.zip` | 567,557,306bytes，475项，CRC已通过 |
| `deliveries/038_dual768_optimized_project_20261030.zip.sha256.json` | ZIP回执，注意在目录旁边，不在038目录内部 |
| `cloud_project_v2/artifacts/kd_v1/final_dual768_evidence_20261029/` | 新teacher_only/combined/collected证据；旧024索引未改 |

ZIP SHA256：`dd3dc01d57567942d4c46d23345706e9530fcf13f718fea0644ac335fdd29200`。038的474项manifest文件最后逐项SHA通过。实际推理模型/config共**128,766,272bytes**，500MB上限统计此推理资产；ZIP另含初始化原件/日志/源码/依赖，不能用ZIP总大小误判。

038内包括`model/`、原`submission.csv`及`submission.image_predictions.csv`、`code/`、native源码/许可/so、`code/vendor/`固定8.55MB addon归档及manifest、`pretrain_model/`两份公开非医学原始初始化权重、许可证/modelcards、七个真实训练run证据及`audit/optimization_2080ti/`。两份初始化原件已在本轮按冻结revision下载并核SHA，只作核验，从未加载进推理。新环境不必再次下载，038内已有。

训练证据：5教师20/12/12/12/12轮及029/030各13轮/E8，350份小文件collector `complete=true,pending=[]`；fold0恢复时间语义及controller receipt未finalize等三项历史局限仍保留，不能宣称历史日志字节或时间完全无缺。

## 新Featurize环境恢复

`work`持久不等于环境持久。`/tmp`、Docker层、旧addon路径、宿主驱动及原图均不假定存在。当前最后探测2080Ti22GB、E5-2686v4/8vCPU、work NFS约剩**1.2GiB**，根盘本地ext4约余550GiB；新机重新核实。不要在work重复解压/复制038、下载镜像或全量数据，不删除训练资产腾空间。完整数据不需要，034原105图足够必要回放。

固定镜像：

```text
crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6
```

镜像本地层约8.5GB，下载还需临时空间，放有余量的本地盘。固定版本Python3.12.3、torch2.10.0+cu128、torchvision0.25.0+cu128、CUDA12.8/cuDNN91002、NumPy1.26.4、Pillow10.4.0、timm1.0.30；不要在宿主Python3.11/torch2.2执行正式比较。

先核查GPU/挂载/空闲/空间及Docker。优先新机正常NVIDIA runtime；旧`run_in_competition.sh`和036`run_container.py`硬编码610.57.04驱动、设备、tmp addon及本地绑定，**不能盲用**。按新机实际路径适配并保存差异，无需重复征求恢复权限，不替换全局驱动。参考旧恢复细节见`NEW_ENVIRONMENT_HANDOFF_20261029.md`，该页当前状态以本文为准。

两条准备路线均应在已核验的固定镜像内执行，不是host命令：

1. **提交项目独立验证：**将038复制/解压到有空间的本地目录，绑定进容器；从副本根运行`python code/prepare_inference.py --force-build --output /new/preparation.json`（先创建输出父目录）。它从`code/vendor`离线恢复依赖。不要在只读原038上force-build改写.so，使原manifest失效。推理/计时命令见副本README，使用副本model与code，取消外部PYTHONPATH，避免偷用源码仓库模块。
2. **当前源码工程：**源码根没有`vendor/`，使用038的归档参数，或先按旧小包恢复`/tmp/dual768_competition_addons`供已适配helper使用。源码根准备命令为：

```bash
python prepare_inference.py \
  --addons-archive /home/featurize/work/deliveries/038_dual768_optimized_project_20261030/code/vendor/competition_addons_cp312_linux_x86_64.tar.gz \
  --addons-manifest /home/featurize/work/deliveries/038_dual768_optimized_project_20261030/code/vendor/competition_addons_archive_manifest.json \
  --force-build --output /new/preparation.json
```

`--check-only`同样须在恢复好的固定镜像执行，会实际import依赖，但不构建或做模型前向。新native so要先构建/确认，Git不含编译二进制，不能因Pillow fallback能运行就称已启用加速。准备回执应显示`native_available=true`，实际CPU支持时`native_implementation=avx2`；generic仅保证语义，不保证同样速度。

## 剩余工作与执行边界

**038/package_status.json当前唯一待办是`T4 cost_report.json`，ready=false。**用户明确最终交付是提交项目，由比赛环境运行，不是要求其租T4或手工迁卡。当前无真实T4连接；不自动购买/归还实例、不上传项目，不把2080Ti结果或SM75兼容性写成T4认证。

官方T4/100+5图及最终CSV可用时，按038 README运行共享predict/cost，核对单张概率、二值/Gate、排序、并列与身份，均值≤100ms。cost已修复共享seed43与逐张sidecar检查；**旧交接P0中“仅allclose、未统一seed、训练证据混有022、最终包未组装”已解决，不再重做。**100图若不闭合所选Study，既有Study一致性仍会pending；不拿目录外预测偷偷补齐，也不伪造通过，须按官方提供材料核对这一口径。

不训练、不改033/标签/模型/预处理语义/FP16/TTA/融合/后处理策略，不重生成test或用test调参；严格遵守work下`rules.pdf`（用户称rule.pdf）及`rules_1.pdf`。两PDF SHA及逐页文本在036规则审计。逐图batch1串行、完整读至写计时，不跨图预取/流水线，不通过删同步漏计GPU工作。规则未点名的技术不能冒称官方逐项认证。

不重做已拒绝旧v2 Python条带并行、v3 Graph合成初始化、model soup或密集融合比例实验。当前81ms目标已完成，默认先只读核验/必要环境恢复，后续工作范围以用户新消息为准；无需自动重跑已完成GPU基准。简单核验子agent用high，复杂任务可并行，但正式计时不得与安装/下载/解压/其他GPU或重CPU任务重叠。新日志与回执另存work，不覆盖037/038或历史慢轮。
