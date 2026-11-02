# B_base 完成后的早停、蒸馏与部署复盘

本文件登记用户要求的后续分析，不修改正在运行的四项训练及其冻结参数。
新增后处理控制器等待 `earlystop25` 全部训练、固定融合评估、历史参考评估正常完成，
并取得原流程锁、确认没有其他 GPU 计算进程后，才启动本轮计时。

要回答的三个问题：A_base 为什么最佳在 E1、现有 OOF 教师是否仍有蒸馏价值、
100ms 约束下 Tiny 直训/Tiny 蒸馏/单 Base/双模型分别有何证据。

已经核实 A_base 在 E6 早停，E1 为最佳轮；初始化为旧 Base 学生 E10 EMA。
教师是早已完成的五折 Base 最终 EMA（20/12/12/12/12），当前微调不生成或覆盖软标签。
中期证据保存在 `earlystop25/review_inputs/`，不能把其中 B 尚未完成的快照当作终局。

后续固定分析为 A/B 各单 Tiny、单 Base、固定 Tiny＋Base 共六项，使用已经保存的
实际 FP16 hflip2 逐图概率。只重放固定概率融合，不拟合权重。报告总体及逐类 AUC、
AP-only、AP 单图/多图、非 AP 和患者配对 Bootstrap1000次。对照包括各架构 B−A、
同组 Base−Tiny、融合−Tiny、融合−Base；沿用原融合 B−A 区间。

计时复用原034已登记的100张 valid 图和另5张预热，验证 Study 完整覆盖与原图SHA，
复制到本地盘后做3轮 batch1串行读图至CSV写出。每一轮的逐图与Study概率需与保存
的完整valid输出精确一致；原始逐图时延、环境、实际权重/配置大小和SHA均保存。
不引用旧draft时延替代新full-decode/Tight Crop计时；2080Ti只能做本机筛选，T4验收另列。

历史Tiny512纯BCE与KD是已有辅助证据，本轮没有新的同768条件的纯BCE消融。
报告必须说明这个缺口，不以“从KD权重继续训练但关闭KL”冒充纯直训。
AP用于早停与选最佳轮，最终区间不校正选模偏差；没有新训练、test读取或自动部署。

- 控制器：`run_tight_crop_postreview.py`
- 数值复盘和计时：`review_tight_crop_ab_results.py`
- 中文分析报告：`render_tight_crop_strategy_review.py`
- 状态：`artifacts/tight_crop_ab_20261102/earlystop25/postreview_controller/STATUS.md`
- 最终报告：`artifacts/tight_crop_ab_20261102/earlystop25/STRATEGY_REVIEW.md`
- 数值和计时证据：`artifacts/tight_crop_ab_20261102/earlystop25/postreview/`

报告依据终局数据生成结论，不能预先承诺 B 提升或某路线一定获胜。
