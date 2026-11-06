# 仅测速交接：033双Tiny768在2080Ti上的原样端到端基准

更新时间：2026-10-29，Asia/Hong_Kong。**本任务仅测速，不优化、不训练、不换模型、不重新生成test CSV。** 用户同一时间只租一台服务器：先结束4090会话，再租一台代理卡；带回结果后返回4090，由原代理分析是否以及如何优化。

当前没有新代理卡测速结果。本次在4090只准备文档、只读核验与最小迁移包，没有运行模型或修改推理源码。所有后续命令都供目标服务器接手agent执行。

用户已明确授权新agent自动完成目标机的必要环境准备、包校验、原样GPU测速和结果留存，无需对这些步骤重复询问。**不需要重新下载或解压全量数据集**，包内105张图片覆盖本次全部输入。可转发的完整任务文本见项目docs/DUAL768_NEW_AGENT_PROMPT_20261029.md。

## 1. 租哪一台

2026-10-29查看[Featurize可用实例页](https://featurize.cn/vm/available)：优先选择页面上的 **RTX2080Ti 22G**，当时列价0.69元/小时、8核E5-2686 v4；只有该机型无货或无法运行既定环境时，才改用RTX3060。不要同时租两台，也无需先后把所有卡测完。库存、价格及具体宿主机以购买时页面为准。

2080Ti与T4同属Turing，优先作为算子性能代理；22GB显存不等于T4，也不能假定该卡的时钟、功耗或带宽与标准11GB卡相同。3060属另一代架构，只作为备选。页面上的3090、3080、A4000、V100没有更适合本次任务的架构对应关系，不必为了测速升级租金。[NVIDIA Turing资料](https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/)

原服务器与代理服务器CPU、磁盘不同，特别是E5与当前4090宿主机可能有明显单核差异。端到端变慢不应全部归因于GPU。2080Ti/3060的报告始终是代理测速，不能标记T4通过。

## 2. 唯一测速对象与基线

| 项目 | 冻结内容 |
| --- | --- |
| 部署 | 033：029 seed43 E8 EMA占25%，030 seed44 E8 EMA占75% |
| 模型 | 两个ConvNeXt V1 Tiny，输入768，FP16，channels_last |
| TTA | 每成员identity+hflip，四次串行前向 |
| 聚合 | 保持现有每视图sigmoid/TTA均值、成员概率加权、Study均值 |
| 推理文件 | 两权重及config合计128,766,272 bytes |
| 已有valid AUC | 0.8083844422198005；此行不是代理卡重新评估结果 |
| 4090端到端 | 79.280300ms/图；三轮79.375028/79.242428/79.223445 |
| 固定测量图片 | 原valid中100张，53完整Study/40患者 |
| 另5张预热图 | 4完整Study/4患者；与测量患者互斥 |

当前实现本来就是batch=1、eager PyTorch、无TensorRT、无跨图预取。不要加sleep、MPS、降功耗、锁时钟、关FP16、增减TTA或更换预处理来“模拟T4”。旧031便携包包含同run E8/E13计时代理，不是本次033真实双seed；禁止拿旧启动器直接跑。

三个关键SHA256：

~~~text
config.json
9363149e143dec2871ac5fb4fbfdee4885f9f62add6a9f2acf4704635311b590
student_00_fp16.pt
edef1f23e591bb52782a03efdc6ee76e10b4d9f115e90654e40f7dbf35f50269
student_01_fp16.pt
87ed41a966cf96144cc63acf50613a14c66fe8497b51625f120389a97a563489
~~~

## 3. 最小测速包和换机前保存

包在工作区：

~~~text
/home/featurize/work/deliveries/034_dual768_proxy_timing_20261029/
  dual768_timing_only.tar.gz
  packaging_verification.json
  metadata_preview.json
  bundle/
    README_TIMING_ONLY.md
    BUNDLE_MANIFEST.json
    SHA256SUMS
    benchmark_local_resources.py / cost.py / predict.py / deploy_kd.py
    model_overnight.py / runtime_compat.py / kd_common.py
    requirements_inference.txt
    model/033/{config.json,student_00_fp16.pt,student_01_fp16.pt}
    inputs/{selection.original.json,valid_manifest.csv,pictures_100/,pictures_5/}
    reference_4090/{local_resource_report.json,round_*.csv,valid_predictions.csv,
                   validation_report.json,environment_lock_training_original.txt}
~~~

最小模型、105张原图、selection和manifest约291MB，另加少量源码与参考报告；压缩包以packaging_verification.json中的实测大小/SHA为准。包不含test图或test预测，不含教师、OOF或训练断点；它不能代替完整项目备份。

本次八项推理相关源文件与033实际执行源码逐项核对一致；打包不修改这些源码。SHA256SUMS用于检查迁移字节，BUNDLE_MANIFEST记录来源，不能用Git克隆代替权重/图像迁移。

[Featurize最佳实践](https://docs.featurize.cn/docs/manual/bestpractise)要求需保留的代码和模型放在work。当前work是NFS持久工作盘，runs、deliveries及Git在其中；但原数据在 **/home/featurize/data/cxr_first**，约52GB，属于实例根盘且不在work内。停止/归还4090前：

- 确认work确属当前账号持久云盘，并保住整个项目、runs、deliveries及旧证据；不要为腾空间删除任何模型、日志或checkpoint。
- 原数据须有可恢复的官方下载入口、已有私有数据集或授权备份。不要误认为小测速包或work已包含全量原图，也不要把52GB强行复制进只余约2.9GB的work。
- 保存本次tar包及其SHA；云盘无法确认持久化时，先将小测速包下载到本人电脑核对。不要上传公开数据集、公开仓库或公开网盘。
- 用户自行完成实例归还/购买；agent不得自动下单、归还实例、删除数据或读取凭证。同一时间只保持一台付费实例。

换机后原数据目录可能不存在，但本次只需要包内105张图。测试结束返回4090时，训练数据和环境的恢复仍需单独核对。

## 4. 目标机环境与存储检查

优先对齐已归档环境：Python3.12、torch2.10.0+cu128、torchvision0.25.0+cu128、timm1.0.30、numpy1.26.4、Pillow10.4.0；cuDNN与驱动实际版本也要记录。requirements_inference.txt中的范围是部署依赖声明，测速时不要借此随意升级NumPy/Pillow。原机器旧conda路径在当前会话不存在，不能照抄为保证可用的解释器。

官方镜像是软件环境，不提供T4。此前只读确认的镜像为：

~~~text
crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:1.0
sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6
~~~

压缩层约4.607GB，解压还需更多空间。旧work剩余空间不足，当前不拉取。目标机若支持Docker且使用该镜像，应固定digest并将镜像层放在空间足够的本地磁盘；仍须核对实际包版本及GPU透传。若不支持Docker，在隔离Python3.12环境按上述版本准备，不覆盖平台全局驱动/CUDA/训练环境。缺环境是环境问题，不能改成CPU/FP32替代测速。

推理不需要OpenCV、sklearn、Albumentations、教师或HF预训练下载；模型按pretrained=False加载现有权重。保留许可证来源声明，不新增权重下载。

由接手agent选择实际解释器，并在下列命令前设置：

~~~bash
TIMING_PYTHON="/实际可用的Python3.12路径/bin/python"
TIMING_BUNDLE_ROOT="/home/featurize/work/deliveries/034_dual768_proxy_timing_20261029/bundle"
TIMING_SAVE_ROOT="/home/featurize/work/dual768_proxy_timing_results"
TIMING_MEMORY_GIB=8
mkdir -p "$TIMING_SAVE_ROOT"
cd "$TIMING_BUNDLE_ROOT"
sha256sum -c SHA256SUMS
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
lscpu
df -h . "$TIMING_SAVE_ROOT"
findmnt -T .
findmnt -T "$TIMING_SAVE_ROOT"
~~~

首次使用压缩包时，先在空目录解压；不要覆盖已有bundle或旧结果。若同账号云盘原包已可见，直接核对即可，无需重复复制模型。输入与输出默认放work以便与4090旧基线的存储位置一致，并记录实际NFS/本地盘；它与官方SSD并不完全相同。若怀疑NFS主导，再另做标记清楚的本地SSD诊断，不能用后者覆盖前者或只回传较快结果。

若只迁移了tar文件，可在工作目录解压到新的空目录（下列目标不能已存在）：

~~~bash
mkdir /home/featurize/work/dual768_timing_transfer
tar -xzf /实际路径/dual768_timing_only.tar.gz -C /home/featurize/work/dual768_timing_transfer
TIMING_BUNDLE_ROOT="/home/featurize/work/dual768_timing_transfer/bundle"
cd "$TIMING_BUNDLE_ROOT"
sha256sum -c SHA256SUMS
~~~

这只解压约292MB测速材料，不是全量数据集。解压后继续按实际路径设置前述变量；不要执行README中举例的占位解释器路径。

确认没有其他训练、预测、压测进程；不启用MPS，不与上传/下载/解压/安装同时测速，不杀别人的进程。不改显卡功耗、时钟或驱动。记录22GB卡实际名称、显存、SM、温度、功耗和实际时钟；页面备选是3060 12GB。若实际改租8GB卡，CUDA可见显存可能小于8GiB，先把TIMING_MEMORY_GIB设为6，并在预览/执行中统一使用该值和记录差异；不要让8GiB上限超过物理显存。

## 5. 只派生路径，不改原始清单

在bundle根目录执行；此步骤仅生成新selection文件，原件及manifest不改。只有manifest_path需要重定位，image_root/source_path仍作为旧来源记录保留。

~~~bash
"$TIMING_PYTHON" -B - <<'PY'
from pathlib import Path
import hashlib
import json

root = Path.cwd()
source = root / "inputs/selection.original.json"
target = root / "inputs/selection.portable.json"
value = json.loads(source.read_text(encoding="utf-8"))
manifest = root / "inputs/valid_manifest.csv"
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == value["manifest_sha256"]
value["manifest_path"] = str(manifest.resolve(strict=True))
with target.open("x", encoding="utf-8") as stream:
    json.dump(value, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print("selection.portable.json ready; original preserved")
PY
~~~

路径派生只执行一次；文件已存在时先核对内容，不覆盖。不要改图片、split、ID、相对路径、SHA、Study计数或选择seed，也不重新挑100张图。

先做CPU元数据预览，不加execute：

~~~bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$TIMING_PYTHON" -B benchmark_local_resources.py \
  --model-dir model/033 \
  --pictures-100 inputs/pictures_100 --pictures-5 inputs/pictures_5 \
  --selection-json inputs/selection.portable.json \
  --output-dir "$TIMING_SAVE_ROOT/not_created_by_preview" \
  --profile dual768_033_proxy_preview --memory-limit-gib "$TIMING_MEMORY_GIB" \
  > "$TIMING_SAVE_ROOT/metadata_preview.json"
~~~

预览须通过输入SHA、Study闭包、患者互斥和128,766,272bytes库存检查。预览不构建模型、不运行CUDA，也不证明目标机环境已能推理。

## 6. 唯一主基准：同一033、无插桩、三轮

在同一个终端保留前述变量。先确认可见的是租用的一张卡；若平台指定CUDA_VISIBLE_DEVICES，不盲目覆盖设备映射。存在CUDA_MPS_变量或MPS服务时先查清，不启动MPS。

以下包装器只在导入/加载模型前设seed43并查询真实SM数，不改模型或推理函数。8GiB是PyTorch分配器上限，远高于033历史约0.2GiB allocated峰值；避免默认16GiB在12GB3060/11GB2080Ti上直接报错，不表示模拟T4。旧4090用16GiB，该配置差异需如实保留。

~~~bash
TIMING_RUN_ID="dual768_033_native_$(TZ=Asia/Hong_Kong date +%Y%m%d_%H%M%S)"
TIMING_OUTPUT_DIR="$TIMING_SAVE_ROOT/$TIMING_RUN_ID"
TIMING_START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TIMING_START_EPOCH="$(date +%s)"
if PYTHONHASHSEED=43 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 "$TIMING_PYTHON" -B - \
  --model-dir model/033 \
  --pictures-100 inputs/pictures_100 --pictures-5 inputs/pictures_5 \
  --selection-json inputs/selection.portable.json \
  --output-dir "$TIMING_OUTPUT_DIR" --profile "$TIMING_RUN_ID" \
  --memory-limit-gib "$TIMING_MEMORY_GIB" --execute \
  > "$TIMING_SAVE_ROOT/$TIMING_RUN_ID.console.log" 2>&1 <<'PY'
import os
import random
import sys

if any(key.startswith("CUDA_MPS_") for key in os.environ):
    raise RuntimeError("Unexpected MPS environment; stop and inspect rather than hiding it")
import numpy as np
import torch

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__.split("+")[0] == "2.10.0", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, "Require one visible rented GPU"
random.seed(43)
np.random.seed(43)
torch.manual_seed(43)
torch.cuda.manual_seed_all(43)
actual_sm = torch.cuda.get_device_properties(0).multi_processor_count
from benchmark_local_resources import main
raise SystemExit(main(sys.argv[1:] + ["--baseline-sm-count", str(actual_sm)]))
PY
then
  TIMING_EXIT_CODE=0
else
  TIMING_EXIT_CODE=$?
fi
TIMING_END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TIMING_END_EPOCH="$(date +%s)"
printf '%s\n' "$TIMING_EXIT_CODE" > "$TIMING_SAVE_ROOT/$TIMING_RUN_ID.exit_code.txt"
printf '%s\n' "started_utc=$TIMING_START_UTC" "ended_utc=$TIMING_END_UTC" \
  "wall_seconds_approx=$((TIMING_END_EPOCH - TIMING_START_EPOCH))" \
  "exit_code=$TIMING_EXIT_CODE" \
  > "$TIMING_SAVE_ROOT/$TIMING_RUN_ID.execution_receipt.txt"
~~~

输出目录须从未存在，父目录可存在。非0先保存并回传日志，不反复重跑取最好结果，不改源码/权重/容差让它“通过”。安装/环境不符先停止修环境，环境准备时间不计为模型推理时间。

计时口径：另5图预热一次；100图逐张串行三轮；从读图片到逐图/Study CSV实际flush，含预处理、所有四次前向、回传、后处理和GPU同步。模型加载、首次初始化、5图预热排除；独立receipt记录整个脚本起止、秒级墙钟和退出码。SHA256SUMS、metadata preview及benchmark自身输入校验会提前读过105张图，正式第一轮也可能已有页缓存，因此本次不能称为冷盘读取或模型冷启动测试。保留三轮自然缓存状态，不只报较快轮，也不清全机页缓存或跨图预解码。

报告自动保留三轮100点、每图均值、总均值、P95/max、设备环境、显存及三轮CSV。T4_latency_certified和official_compliance_pass为false是正确结果；latency_pass_on_current_hardware不等于T4通过。

## 7. 测完必须补的核对与可选诊断

主基准结束后做CPU核对，不能把核对耗时混进主基准：

- 再检查SHA256SUMS，确认源文件、模型和105张图未变。
- 核对退出码、100图×3、5预热图、两成员、768/hflip2、25/75和真实SM，无删图/删慢轮。
- benchmark自带comparison只核同机三轮相互一致；另将新三轮逐图和Study CSV与reference_4090对应CSV按身份/类别对齐，记录覆盖、最大/平均绝对概率差、超过1e-3数量、固定0.5二值差异、样本内类别排序/并列变化，并额外记录每类跨Study排序变化。禁止修改原概率或放宽原脚本阈值来消除差异。
- 异卡FP16可能有微差，先如实记录。仅代理诊断；不宣称完成官方100张test样本一致性核验，不重新计算或拟合test指标。
- 保存Python/torch/torchvision/timm/NumPy/Pillow版本、CUDA/cuDNN/驱动、CPU/可用核数、GPU名称/显存/SM/功耗/时钟、输入输出磁盘类型；完整日志中不要包含凭证、token或带密钥的pip源URL。

若接手agent能稳定完成，可在**主基准之后、独立进程/输出目录**做一轮100图分段诊断：运行时包裹当前load_image_arrays记录CPU预处理，给两模型根模块挂CUDA Event hooks记录四次前向，保留端到端时间。只能在外部诊断包装器中插桩，保存包装器源码，不改受版本控制的生产推理文件。CUDA Event跨度可能包含主机提交空隙，不能写成纯kernel忙时；需纯内核占比时另存短Profiler trace。所有诊断时间都不能替代上节无插桩基准。

不要执行旧breakdown脚本：它固定旧模型、SM128、16GiB及旧目录，可能测错对象。没有分段能力时先把主基准原样带回，不为补诊断启动任何优化。

本次输入是valid，使用benchmark_local_resources.py。**不要拿最终test CSV搭配valid图片运行cost.py，不改valid清单split来冒充正式核验。** cost.py现有语义/seed缺口留待返回4090后独立处理，本次不修改。

## 8. 带回什么、何时停止

将以下内容保存到work中的本次结果目录，并确认归还测试实例后仍可读取；必要时下载给本人，再结束租用并返回4090：

1. local_resource_report.json、全部三轮Study/逐图CSV、console.log、退出码及execution_receipt.txt。
2. 原始与portable selection、包SHA、执行命令/seed、元数据预览、环境和硬件/磁盘记录。
3. 与4090参考的CPU一致性报告及核对脚本。
4. 若执行分段诊断，带回报告、逐图时间、trace和诊断包装器；没有执行就明确未执行。

测速后先停止，不切换单模、不重训、不重生成test、不上传平台。返回4090后由原代理结合CPU/存储差异、GPU算子成本、长尾及概率差异评估T4风险；不能把2080Ti端到端简单乘2、将3060当T4，或称任何倍率为保证上界。

## 9. 仅记录、暂不执行的优化思路

以下不是测速agent的任务，只供返回4090后的决策：

- 如CPU预处理占主导，先研究保留LANCZOS/几何及float32算术的灰度归一化查表，逐字节验证后再测。
- 如主机同步占明显比例，研究延后汇总有限值检查、合并D2H，但保持原概率算术顺序。
- 若JPEG解码仍主导，再以新候选研究draft/IDCT缩放；会改变像素，须重新完整valid核验，不能冒充已提交033的等值整改。
- 如四前向确为瓶颈，返回4090后才讨论固定权重平均、单成员/TTA取舍；027单512和030单768备用仍保存，但它们不是本次测速对象。
- 提分训练候选保留Tiny512 seed44或单独Loss对照；不在测试卡训练，不同时改变seed、分辨率、Loss及增强。

所有后续独立修改先验证再Git提交，模型/日志/图片不入Git。简单重复检查用medium，复杂诊断用high，不使用ultra做重复核验。
