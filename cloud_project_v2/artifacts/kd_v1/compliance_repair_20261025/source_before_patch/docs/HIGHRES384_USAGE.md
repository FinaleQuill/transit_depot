# 高分辨率384 · 首轮运行说明

2026-10-18，Asia/Hong_Kong。当前只完成代码重构及静态检查，新增训练、推理、模型前向/反向、权重下载均为0；没有384指标或显存实测结果。

## 启动

在PowerShell中手动执行（本次未执行、未预建run目录）：

```powershell
& "E:\PycharmCodes\计挑人工智能赛道\.venv\Scripts\python.exe" -X utf8 -u "E:\PycharmCodes\计挑人工智能赛道\train_highres.py" `
  --output "E:\PycharmCodes\计挑人工智能赛道\runs\006_convnextv2_base_hr384_20261018_01" `
  --backbone convnextv2_base.fcmae_ft_in1k --image-size 384 `
  --batch-size 2 --accumulation-steps 16 --epochs 30 --num-workers 0 --seed 43 --device cuda
```

首次手动运行时，timm若无本地预训练权重，可能下载ImageNet-1K权重。本次未导入执行项目模块、构建模型或下载。

## 固定的内存策略

- 输入固定384，拒绝512；AMP和非重入激活检查点始终启用，没有关闭开关。
- 微批固定两个完整Study。每个Study保留清单内全部视角，GPU每次只处理一个视角；两Study合计最多14张图不会整体搬到GPU。
- 外层检查点保存CPU输入；空CUDA anchor登记随机数设备，使Dropout/DropPath重算重放。骨干另逐块检查点；每微批反传后释放计算图，累积只保存参数梯度。
- 默认累积16微批，即完整更新窗口32次Study抽样；尾窗口按实际微批数归一化。排序只在当前微批有正负样本的类别计算，不冒称跨32个Study构造排序对。
- EMA在CPU创建和更新。验证临时把EMA权重拷入唯一GPU模型，结束恢复raw，保留Adam状态；AdamW关闭foreach/fused以减少临时显存。
- 初始化和每次完整运行的显存峰值仍受CUDA环境、占用与算子影响，代码不能保证任意8GB环境绝不OOM。日志记录GPU实际allocated/reserved峰值。

若仍OOM，用全新run目录，将模型参数改为：

```text
--backbone convnextv2_tiny.fcmae_ft_in1k --attention-dim 128 --attention-layers 1
```

建议新目录名`runs/006_convnextv2_tiny_hr384_20261018_01`。不要用Base权重续训Tiny。减小梯度累积次数不会降低单视角前向峰值；本入口固定两个Study以保留有效正负排序对，不提供微批1的虚假等价替换。`--num-workers 0`降低CPU原图解码/预取压力，主要影响主存与吞吐，不能代替GPU显存优化。

## 数据、监督与采样

`dataset.py`、原始CSV、原图、300缓存与清洗清单保持原样。`dataset_highres.py`复用原`read_manifest()`校验，严格从清单relative_path读取原图，每次核对SHA，内存中EXIF转正/RGB/保比例LANCZOS/居中黑边到384，再归一化；不会扫描目录重新纳入排除项，不会回退300缓存。训练保留亮度/对比度、轻旋转与灰度共享噪声，关闭逐图Mixup/CutMix和Cutout。

训练每轮9257微批、18514次Study抽样，跨批允许重复；这不是18513个Study的唯一全覆盖。50%批次均匀选两个不同Study，50%批次先均匀选一个类别再选一正一负。权重仅由train Study标签形成，history记录各类曝光、重复与有效样本量。完整训练诊断与valid采用独立顺序无增强loader。

模型取ConvNeXt-V2最后两级特征，经过十类查询注意力头，与保留预训练归一化的全局分支相加输出十维logits。训练用稳定公式计算`logit(mean(sigmoid(view_logits)))`，随后执行Study BCE + 0.2倍成对logistic排序；排序前3轮关闭，第4～6轮渐增。没有额外pos_weight，没有对原标签补写或强制互斥。

首轮头部冻结预热1轮，全网warmup2轮，骨干/头峰值LR为3e-5/3e-4，随后余弦降至1%下界；总30轮，无自动512切换。valid每轮完整；无增强全训练诊断默认第1、每5轮及最后一轮执行，可用`--train-eval-every 0`关闭训练诊断以节省时间，不影响完整valid。

## 结果与推理协议

提交CSV字段、可选Subject_id/Gate及历史阈值过滤错误，统一见 [SUBMISSION_FORMAT.md](SUBMISSION_FORMAT.md)。当前推理默认三列完整长表；用户提供的可用四列样例另含Subject_id，两者不能仅因这一列的差别判定有效性。

日志输出Study AUC、逐类AP/F1、调阈值F1、固定0.5 F1、EMA更新/AMP跳步和显存峰值。验证缺图则该轮指标与阈值无效，不替换best；worker故障可沿用原串行复核机制，但不丢掉某些视角后继续冒称完整验证。

- `best.pt`及同目录`thresholds.json`：按完整valid Macro-AUC选取，并列比较调优F1。
- `best_f1/best.pt`及其同目录侧文件：另保留调优Macro-F1最佳，阈值单独绑定。
- `last.pt`：最近完整优化轮的快照，包含EMA及raw参数；未保存优化器/RNG精确恢复状态，不支持续训。缺验证时标记inference_ready=False。
- schema3记录模型配置、384原图预处理、EMA来源、完整验证状态和阈值预处理SHA；扩展后的`inference.py`按此协议只读原图、单张CUDA AMP预测，保持原Study概率均值。历史schema1/2 B3仍兼容。

本阶段推理固定identity，不启用512、多尺度TTA或融合。阈值只来自完整valid，不读取测试标签，不用test做选择。所有F1调优成绩均属于同一valid的样本内调优，不能当独立测试成绩。

运行时可创建本run下STOP文件或按一次Ctrl+C请求轮末保存退出；第二次Ctrl+C立即退出，当前不完整轮不保证恢复。本次未创建STOP或操作任何历史训练进程。

## 验证与回滚

语法检查命令：`.venv/Scripts/python.exe -m py_compile dataset_highres.py model_highres.py train_highres.py inference.py`。该命令不导入执行项目模块，不验证运行时张量形状、显存或提分。

改前快照与保护哈希在`artifacts/code_snapshots/highres384_20261018_000830`；恢复其中inference.py并移除本次三个新模块即可回到原B3入口，数据与旧权重无需改动。静态复审已修正默认CUDA索引、CPU EMA初始化及完成轮数元数据时序。
