# CXR 纯代码学习包

从 [LEARNING_GUIDE.md](LEARNING_GUIDE.md) 开始：为“懂 ANN、正在学习 CNN”的读者编写，逐步解释数据、ConvNeXt V2-Base、十类加权 BCE、梯度累积、EMA、五折、Study 概率和阈值。

这是当前项目的独立源码快照。主线核心部分已增加中文注释；Python 可执行 AST 与源项目一致。制作时原项目只读，没有修改其代码、训练设置或 tmux 任务，也没有导入项目、构模、运行测试或使用 GPU。

包内不含 data/、runs/、.pt 权重、图片、CSV 数据、缓存、日志或虚拟环境。代码保留数据与模型文件的原路径引用；实际运行需要另行准备这些资源。当前阅读无需运行训练、推理、显存探针或启动脚本。

主要文件：

- `LEARNING_GUIDE.md`：完整教程，含图解、数学直觉、示例与自测题。
- `CODE_MAP.md`：函数阅读地图及本副本行号。
- `*_overnight.py`、`run_all_folds.py`、`cv_splits.py`：当前 768 主线。
- `dataset.py`：清单解析与 letterbox 等通用接口。
- `train.py`：历史训练代码，同时提供当前集成复用的阈值/指标函数。
- 其他 Python / Shell / PowerShell：保留的历史方案和工程入口，中文导航标明阅读位置。
- `tests/`：测试源码；本次未执行。
- `requirements.txt`、`requirements.lock.txt`：依赖规格与此前实装版本锁。
- `docs/`：项目原有说明；其中历史配置和路径不一定是当前主线。`PROJECT_README_ORIGINAL.md` 是源 README 的原样副本。
- `PACKAGE_MANIFEST.json`：来源、哈希、语义一致性及排除内容记录。

默认主线配置为 768 原图、ConvNeXt V2-Base、微批 2 / 累积 16 / 有效 32 图、每折 train Study 去重正权重 BCE。默认值可被环境变量或 CLI 覆盖；本包不声称读取了正在运行进程的实际参数。
