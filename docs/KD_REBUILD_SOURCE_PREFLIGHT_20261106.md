# 大教师公开初始化与环境预检

2026-11-06，香港。用户已明确授权按新方案启动训练。本项执行公开权重下载、来源核验、CPU模型构造与checkpoint加载；没有模型前向、GPU操作、训练或submission生成。

## 公开权重

资产全部位于 `/home/featurize/data/kd_rebuild_20261106/assets/`，不占接近满载的work持久盘。匿名直接访问固定revision的公开、非gated Hugging Face模型，无token，无登录，无gated端点绕行。完整README、config、许可证、API/tree快照与逐文件SHA保存在各模型目录。

| 资产 | 固定模型与revision | 参数/字节 | 实际SHA256 |
|---|---|---|---|
| `dinov3_l/model.safetensors` | `timm/vit_large_patch16_dinov3.lvd1689m`，`30c1109559f65dea34316b0d4842d35c5771fe11` | 303,079,424参数；1,212,347,640 bytes | `45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941` |
| `convnext_large/model.safetensors` | `timm/convnext_large.fb_in22k_ft_in1k_384`，`1c3c222eca50028b09e888e12746ac10f2403a36` | 197,767,336参数；791,103,952 bytes | `d2d9196af67de8731710f67cfe0f6edcb4ce47986c29af3544d24f327e167b43` |

两者实际完整SHA与固定revision公开LFS对象SHA逐一相等。`convnext_large`的384版revision与此前审计224版的`dc2e53...`不同，不能交换来源声明。

DINOv3模型卡标注LVD-1689M通用网页图像及7B教师蒸馏；许可证为非标准DINOv3 License，允许使用/复制/再分发/衍生，分发须附协议；不冒称Apache或已由主办方核准具体权重。CNN权重模型卡为Apache-2.0，ImageNet-22k预训练、ImageNet-1k微调；附Apache全文及上游ConvNeXt代码MIT全文，区分权重声明与代码许可。离线教师不计最终500MB，但须申报其来源及蒸馏关系；若参与最终推理则计入。

总完成清单：`assets/teacher_source_manifest.json`。单模型 `source.json` 提供统一字段 `repo/revision/sha256/license/gated/pretraining_dataset/state`；均为 `state=complete`、`gated=false`。

## 官方容器与轻量依赖

固定官方镜像沿用 `reconstruction.runtime.IMAGE`：

`crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:125d8bcb4ce5a2a9acec6d9838b37e2ae1e2014c09b848700f9530f2f7856a0b`

镜像已有Python3.12.3、torch2.10.0+cu128、torchvision0.25.0+cu128、numpy1.26.4、Pillow10.4.0、scikit-learn1.5.2、pandas2.2.3。没有重装或覆盖任何上述核心库。

轻量依赖安装在 `/home/featurize/data/kd_rebuild_20261106/environment/site`，正式挂载 `/opt/kd_addons`：timm1.0.30、safetensors0.6.2、huggingface-hub0.36.0、pytest8.4.2、tqdm4.67.1、packaging25.0、pluggy1.6.0、iniconfig2.3.0、pygments2.19.2。下载wheel按PyPI公开SHA核验，在Python3.12容器内使用 `--no-index --no-deps --target` 离线安装。不存在torch/torchvision/numpy/PIL/nvidia/triton目录覆盖。hf-xet不参与此离线训练环境，关闭Xet并不影响本地safetensors加载。

环境逐文件清单 `environment/manifest.json`：1,971文件，33,371,932 bytes，清单SHA256 `f212ba9c9bf2b0d23e6e62efca75df8281983805e254d01e2049e588ee1fc42e`。训练应设置 `PYTHONDONTWRITEBYTECODE=1`、`PYTHONNOUSERSITE=1`、`HF_HUB_OFFLINE=1`、`HF_HUB_DISABLE_XET=1`，避免现场网络和库影子加载。

## CPU兼容检查

使用同一官方固定digest镜像、network=none、无GPU设备，真实创建CPU模型并加载本地完整checkpoint，没有执行forward。证据为 `environment/cpu_preflight.json`；可复现脚本 `teacher_distill_20261106/source_preflight.py`。

- DINOv3-L由timm的`Eva`类实现，`num_features=1024`；`img_size=512,dynamic_img_size=True,num_classes=0`可构造；完整checkpoint骨干缺失/额外key均0。
- CNN-L为`ConvNeXt`，`num_features=1536`；仅移除原ImageNet分类器`head.fc.weight/head.fc.bias`，剩余骨干缺失/额外key均0。去掉旧分类器后196,230,336参数。
- 两骨干均提供`set_grad_checkpointing`。真实GPU显存、前后向吞吐、优化器和训练链路验证由主代理执行；本CPU检查不冒充已训练或GPU可行性通过。

可断点公开下载脚本为 `teacher_distill_20261106/provenance.py`；失败不覆盖完整权重，续传严格核对Content-Range，完成后核对固定LFS SHA。公开权重来源：[DINOv3-L](https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m/tree/30c1109559f65dea34316b0d4842d35c5771fe11)、[CNN-L384](https://huggingface.co/timm/convnext_large.fb_in22k_ft_in1k_384/tree/1c3c222eca50028b09e888e12746ac10f2403a36)。
