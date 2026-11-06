# KD V1本次验证结果

本次实现完成；训练、网络前向、反向、优化器更新、预训练权重下载均未执行。

- Python3.12.15、torch2.10.0+cu128、torchvision0.25.0+cu128、timm1.0.30；pip check通过，4090可见，BF16支持已查询。
- 101项CPU测试通过，另9项subtests通过。覆盖严格OOF与缓存链、KL公式、来源许可材料、断点恢复、部署计时与核验材料，包含相关旧数据/恢复/输出逻辑回归。
- 14份源码语法检查、10个CLI帮助入口、4个无执行预览通过。教师/学生只在CPU构建pretrained=False核对参数量，没有网络前向。
- train30328图、valid1422图均存在且非空；五折患者交叉0。三份清洗清单与folds.json哈希保持不变。
- Tiny十类参数27,827,818，FP16纯参数55,655,636字节。该数字不含实际序列化与配置开销。

证据：[测试XML](tests.xml)、[目标环境](environment.json)、[准备检查](preflight_ready.json)、[参数核对](architecture_check.json)、[输入保护](protected_inputs.json)、[依赖锁](requirements_training_lock.txt)。

未生成训练好的教师、软标签或学生；未生成实际FP16部署包、预测CSV或T4计时成绩。最终体积、AUC与延迟需在后续训练和同一部署协议下测量，官方复测为准。

开始步骤见[分步操作说明](../../docs/KD_V1_USAGE.md)。
