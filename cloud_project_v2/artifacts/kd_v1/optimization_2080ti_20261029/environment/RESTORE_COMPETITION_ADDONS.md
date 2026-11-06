# 2080Ti 比赛环境附加包恢复

**新窗口前提：**本文件只恢复附加包。用户明确新窗口环境会丢失，Docker镜像也须先检查/重新拉取，不能直接运行旧helper。完整顺序及新机驱动挂载适配见项目`docs/NEW_ENVIRONMENT_HANDOFF_20261029.md`；下方“镜像目前留在本地”仅是旧环境快照。

本包仅含13个公开pip附加包，802个文件、原始文件28342126 bytes；不含账号凭证、模型、数据集、完整Python环境或Docker镜像。凭证特征扫描仅命中httpx上游URL编码文档示例。789项RECORD文件哈希通过，全部纳入文件均属于这13包，未纳入空`.lock`及已单独保存的`resolution.json`。

压缩包8554684 bytes，SHA256：

```text
eb7ecd612df5568f9f87b9f3bc3ec71cf775ed3d00c4b072467f308e2f32fe10
```

逐文件SHA、大小和13包版本见`competition_addons_archive_manifest.json`；精确安装版本见`competition_addons_requirements.txt`。包内无torch、torchvision、NumPy、Pillow、SciPy、sklearn、nvidia或triton模块，核心包继续从原官方镜像加载。

## 首选：恢复现成小包

仅在目标目录不存在时执行，避免覆盖正在运行的环境：

```bash
set -eu
cd /home/featurize/work/cloud_project_v2/artifacts/kd_v1/optimization_2080ti_20261029/environment
sha256sum -c competition_addons_archive.sha256
test ! -e /tmp/dual768_competition_addons
tar --no-same-owner --no-same-permissions -xzf competition_addons_cp312_linux_x86_64.tar.gz -C /tmp
cd /home/featurize/work/cloud_project_v2
./run_in_competition.sh --version
./run_in_competition.sh artifacts/kd_v1/optimization_2080ti_20261029/environment/probe_container.py
```

解压只创建`/tmp/dual768_competition_addons`，不向官方镜像或全局Python安装。CLI入口脚本不是本项目推理入口；统一使用`run_in_competition.sh`运行Python模块。

## 无现成小包时：按固定版本重新安装附加包

先确认Docker Root Dir位于有足够空间的本地根盘，不将镜像放work。原镜像目前留在本地`/var/lib/docker`；新服务器需要时再执行固定digest拉取：

```bash
docker info --format '{{.DockerRootDir}}'
docker pull crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6
```

以下安装仅向一个全新target写入13个固定附加包，`--no-deps`禁止重新解析并安装torch等核心依赖：

```bash
set -eu
test ! -e /tmp/dual768_competition_addons
mkdir /tmp/dual768_competition_addons
docker run --rm --pull=never --runtime=runc --network host \
  --user "$(id -u):$(id -g)" \
  --volume /tmp/dual768_competition_addons:/opt/dual768_addons \
  --volume /home/featurize/work/cloud_project_v2/artifacts/kd_v1/optimization_2080ti_20261029/environment:/bundle:ro \
  --entrypoint python \
  crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6 \
  -m pip install --no-cache-dir --no-deps --target /opt/dual768_addons \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r /bundle/competition_addons_requirements.txt
```

随后运行上面的版本与import探测。当前helper明确绑定本机610.57.04驱动库和GPU设备；换机后需重新核对路径，不能把本机2080Ti结果写为T4认证。原镜像核心版本：Python3.12.3、torch2.10.0+cu128、torchvision0.25.0+cu128、cuDNN91002、NumPy1.26.4、Pillow10.4.0、SciPy1.17.1、sklearn1.5.2。
