# 胸片多标签数据清洗工程

2026-10-25：当前主入口改为**许可证合规的 ConvNeXt V1 Base768 五折教师 → 严格OOF图像级软标签 → 单个V1 Tiny512蒸馏 → FP16无TTA部署**。操作见 [KD V1分步说明](docs/KD_V1_USAGE.md)。所有新训练入口默认预览，需显式`--execute`；本次没有训练、模型前向/反向或预训练权重下载。旧V2来源为NC许可证，新流程明确拒绝，以下V2及历史性能段仅为历史记录。

2026-10-19 23:08（香港）：当前 `run_overnight.sh` 已升级固定原图 768×768 等比例 Pad、每折 Study 去重 neg/pos 加权 BCE。新机环境/权重缓存及 batch1/2/4 BF16+全检查点显存探针均通过；实测 batch2 吞吐略高，默认 2×累积16=32、workers8/prefetch2。只做修改与测试，真实训练/优化器更新均为0；结果、十类权重及单行手动 tmux 命令见 [768 预检报告](docs/768_PREFLIGHT.md)。下述512段落为此前配置记录。

2026-10-19 通宵五折：新增 ConvNeXt V2-Base、512/多尺度、患者隔离 5 折、完整断点恢复、TTA/概率集成与样例格式提交。手动入口 `run_overnight.sh`，新机 RTX 4090 已通过 BF16/激活检查点的 512 合成前后向显存探针，默认微批8/累积4、workers=4；本次未启动真实训练。详见 [通宵方案与单行 tmux 命令](docs/OVERNIGHT_CV.md)。下方 Tiny/SSH 说明为保留的既有实验入口。

2026-10-18 Tiny v2：当前`run_cloud.sh`入口已改为`train_tiny.py`，使用ConvNeXt-Tiny、GAP+Linear、Study等权逐图BCE；batch32图片、完整shuffle、最多25轮及valid早停。去掉新训练路径中的类别注意力、Rank和类别重采样，保留384原图、AMP和GPU EMA。交付`cloud_code_v2.zip`；解压启动见 [Tiny v2说明](docs/TINY_V2_USAGE.md)。本次仅修改/静态检查/打包，训练与模型前后向0次；下方SSH Base同步记录为此前版本历史。

2026-10-18 代码同步：主工程15份运行代码/配置已逐字节对齐 `ssh链接服务器/my_code_backup/cloud_project`，包括128 Study、128视角块、12 worker、尾块补16的4090启动配置。原清洗清单与dataset.py保持原样。历史清洗段的“训练0次”仅为当时时点；本次同步与对比新增训练/推理/前后向均为0。

本地014备份截至2026-10-18 20:26:34记录完成12轮、最佳第6轮；SSH尚未提交平台评测。队友包有本地验证成绩和平台改善反馈，具体平台分数字段为空。来源及训练差异见 [SSH与队友训练对照](docs/SSH_TEAMMATE_COMPARISON_20261018.md)，性能历史记录见 [4090性能报告](docs/PERFORMANCE_20261018.md)。这次未连接或启动远程训练，备份不代表服务器当前实时状态。

清洗阶段原始记录：先完成全量、可追溯的数据清洗，再讨论训练；该阶段训练次数为0。后续既有训练与本次同步边界见上方说明。

- 原始数据：cxr_first/cxr_first，保留原处，只读。
- 旧工程：archive/legacy_20261016_2345，含旧源码、Git历史、文档、模型、实验及交付；不作为新工程实现。
- 共用资源：.venv、.venv-competition及tools内Python/uv运行时保留；原图片缓存与规则截图仍在原处，新清洗直接检查原图。
- 新代码：cxr_clean/；清洗输出：data/；规则和研究依据：docs/。
- 清洗原则：确定故障在新清单排除；质量、近重复和医学标签疑点只标记复核；原图和原标签不改写。

本次已完成33984张原图审计和300RGB缓存；训练/模型推理0次。最终训练清单30328张：排除14张精确重复、技术复核隔离2张无完整胸片内容的图；18513 Study、11055患者及十类Study阳性计数均不变。valid/test保持1422/2218图及859/1414 Study。

当前交付：

- `data/clean_v1_reviewed/train_clean.csv`：后续训练推荐清单（不是比赛预测提交）。
- [提交CSV格式与历史文件对照](docs/SUBMISSION_FORMAT.md)：用户确认可用的四列样例、可省略Subject_id的历史答疑，以及不能提交的阈值删行旧稿。
- `data/clean_v1_reviewed/DELIVERY_REPORT.md`：中文交付说明与限制。
- `data/clean_v1_reviewed/exclusions.csv` / `manual_decisions.json`：16项排除及2项视觉技术复核证据。
- `data/clean_v1_reviewed/review_queue.csv`：2769个候选，含已隔离2项；其余2767项保持原样待专业复核，不冒称全是错图。
- `data/clean_v1/`：通过验收的自动清洗父版本及33970张缓存；复核版引用其中33968张，两个目录须一起保留。

保留十类Study多标签和全部正常视角，附带视角、方向、尺寸、质量与重复特征。没有自动添加疾病答案，没有测试伪标签，没有为了清洗重采样。统一尺寸已实际生成；归一化与温和在线增强在加载接口提供，随机增强未写入清洗缓存。

38项合成/策略测试通过；同数据同划分17份清洗决策重放一致。执行命令与回滚见docs/USAGE.md，过程和替换理由见WORKLOG.md，实验对比见EXPERIMENTS.md。已验收输出禁止覆写，修改策略另建版本。
