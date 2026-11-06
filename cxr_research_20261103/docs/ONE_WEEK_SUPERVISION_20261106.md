# 一周学生冲刺：启动与监督

2026-11-06 19:09（香港）核验。用户本轮已明确授权方案后立即训练学生、禁止重训教师。核心源码提交`ac8fe0c`，正式入口`python -m one_week.pipeline --plan one_week/plan.json --execute`。

**当前新学生已发生真实优化器更新。** 控制器PID **20353**、PPID1、独立SID20353、stdin空设备，容器`cxr-one-week-raw_anchor_seed44`。启动时间2026-11-06 19:05:53（香港）；核验快照为epoch1、已读2,200/30,344图、68次真实optimizer更新、68次EMA更新、AMP跳步0。完整epoch仍为0；这些是启动快照，不是首轮成绩。

## 冻结流水线

1. raw完整30,344图/18,513 Study，Tiny768/P2 seed44；固定9轮、LR horizon25，仅末轮完整valid质量验收。
2. 只有A正常完成、完整valid859 Study/1,422图且Macro≥.800时，才运行train+valid seed44，31,766图/19,372 Study，固定同日程；valid并入后不再评估选模。
3. 上一stage正常完成后，运行同配方train+valid seed43。预登记主候选单44，等权双seed为备选，不按并训valid选择。

总预算最多27个学生epoch；首轮尚未完成，按历史吞吐暂预算10–14小时，后续按完整epoch实际时间更新。**本次实际启动学生1个，未来2个仅已登记；新教师0个，没有平台新成绩。**

## 当前证据与路径

运行根：`/home/featurize/work/reconstruction_runs_20261103/ONE_WEEK_20261106/`。

| 文件 | 用途 |
|---|---|
| pipeline_status.json | 当前stage、命令/PID、完成列表、质量否决及失败终态 |
| 各stage/progress.json | 实时batch/图数/optimizer/EMA/AMP，不能把一轮中途当完整 |
| 各stage/history.jsonl | 仅完整epoch的loss、覆盖、耗时、LR、断点SHA |
| 各stage/configuration.json、data_audit.json、source/ | 冻结源码、原图/CSV、旧OOF/教师血缘与数据用途 |
| 各stage/latest_ema_fp16.pt | 每完整轮原子替换；仅training_complete且epoch9才是最终候选 |
| 各stage/recovery.json | 完整last路径、SHA与匹配FP16的轮次 |
| LAUNCH_VERIFIED.json | 本次真实optimizer更新及旧资产保护快照 |
| runtime_preflight/CPU_IMPORTS_VERIFIED.json | 官方镜像、1971项依赖SHA、只读挂载和实际导入 |
| audit/history_inventory_20261106/ | 3648份历史文本、58份不同history内容的库存与完整解析 |

完整可恢复checkpoint：`/home/featurize/data/one_week_20261106/state/<stage>/last.pt`，含raw/EMA/optimizer/scaler/RNG。完整状态只在本机盘，不冒充跨实例备份；工作区持久保留FP16及审计，不能由FP16恢复优化器。后台控制器独立于本对话，主机/容器停止仍会中断。

## 已完成的验证与保留项

13项官方Docker CPU测试通过；已有KD行的新旧损失/梯度一致，缺缓存图完整BCE，混split增强/路径保持、全局Study等权、原图SHA拒绝、尾窗梯度及9/25 LR日程通过。真实数据预审完整30344/18513，30328图有旧严格OOF，16图涉及11 Study/10患者只有BCE，未删图。

已完整复核两PDF/答疑。冠军CSV SHA`44dde95b…`、最佳FP32 EMA SHA`aa3c632e…`与原归档相符；所有旧STOP保留，没有恢复旧训练。用户本来未跟踪的四个raw导出源码未修改、未混入本轮提交。没有新test前向、伪标签、平台上传或对测试输出拟合。

## 停止与失败

运行根写`STOP`取消后续stage，训练器在完整轮边界保存后退出；紧急停止可向当前容器发SIGTERM，未完成epoch丢弃并保留最近完整last。不自动移除STOP、不自动重试、不自动延长。遇到数据/数值/质量否决立即停止后续流水线。

完整决策见[训练方案](ONE_WEEK_ULTIMATE_PLAN_20261106.md)。最终模型加载体积、正式T4计时、predict/cost完整Study一致性与材料交付尚待训练完成后验收；当前不宣称最终合规包已经完成。
