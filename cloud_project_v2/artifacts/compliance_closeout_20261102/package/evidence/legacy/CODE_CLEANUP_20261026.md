# 2026-10-26 弃用代码清理

用户明确授权删除已经抛弃的代码、保留历史文字记录。本次未启动训练、模型前向或推理，也未停止、重启或改参当前混合教师任务。

清理前核对了现用入口的静态依赖、源码指纹与打包清单；删除45个文件：根目录15个旧入口/配置、4个旧测试，以及26个旧备份代码或编译缓存文件，合计896199字节。逐项路径、大小与SHA256见[机器清单](CODE_CLEANUP_20261026.json)。

## 已删除的根目录旧入口

```text
train.py                 model.py                 inference.py
train_highres.py         dataset_highres.py       model_highres.py
train_tiny.py            dataset_tiny.py           model_tiny.py
run_all_folds.py         inference_overnight.py   probe_gpu_memory.py
run_cloud.sh             pack_cloud.ps1          requirements.txt
```

同时删除专门对应这些旧入口的4个测试，以及 `artifacts/code_snapshots/`、`artifacts/upgrade_768/` 内已弃用代码副本。其文本记录仍保留；历史报告中引用的代码快照可能已经按本清单移除，不改写旧报告伪造其当时状态。

## 保留的必要内容

- 原图、三份清洗清单、fold计划，以及项目外的训练run、教师/学生权重和预测结果。
- 所有历史文字说明、日志、JSON证据、官方规则原件及哈希记录。
- `train_overnight.py`、`dataset.py` 等仍被现行教师和学生导入的公共模块。
- 已被真实训练账本引用的源码版本，例如 `artifacts/kd_v1/compliance_repair_20261025/source_before_patch/` 和 `source_after_patch/`。它们仍用于官方备查，属于必要材料，保持原样。

打包脚本已移除对旧 `requirements.txt` 的依赖，继续使用当前训练/推理依赖文件。核验包按官方目录约定生成的 `code/train.py`、`code/model.py` 是当前实现的别名，不是已删旧代码。

当前入口与文件职责见[项目README](../../README.md)。[清理前README原文](README_BEFORE_CLEANUP_20261026.md)保留原历史说明。

## 清理后验证

145项CPU测试及42项子测试通过；数据清单与fold计划SHA不变，原教师进程继续运行。证据见[测试报告](../../artifacts/kd_v1/cleanup_20261026_tests.xml)与[保护核查](../../artifacts/kd_v1/cleanup_20261026_verification.json)。
