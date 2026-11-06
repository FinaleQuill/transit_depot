# 单图LANCZOS并行原型：尚未部署

本目录只含CPU研究原型及验证材料，未修改生产源、未运行GPU、未训练。不是跨图预取，也没有缓存预测或缩小JPEG解码分辨率。

Pillow10.4的[原始实现](https://github.com/python-pillow/Pillow/blob/10.4.0/src/libImaging/Resample.c#L604-L653)先做横向再做纵向卷积，两个阶段之间保存8位中间图。两种原型均重用同一Pillow的LANCZOS卷积和量化：

- parallel_lanczos_prototype.py：同一图片按行分块，只并行横向卷积，拼接后串行纵向卷积。
- parallel_lanczos_both_axes.py：每图新建≤4线程pool，横向按行分块并全部完成/拼接；纵向按列分块并全部完成/拼接。返回前关闭线程池，再做原CHW float32归一化和后续GPU。

只支持EXIF完成后的L/RGB。P/1/LA/RGBA按生产路径先转RGB，避免alpha模式在拆分两次Python resize时重复预乘/去预乘引入误差；非8位图继续拒绝。所有band保持完整的卷积方向输入，横向band不裁列、纵向band不裁行，不改变核边界。小图、单worker或无需缩放可回退同一Pillow原2D实现。

两种原型各自验证通过：168个随机及极端直接resize案例、480个模式/EXIF/几何/512768案例、105张原valid图×512/768共210组float32逐字节一致，以及3个高位深拒绝案例。原图SHA和mtime保持不变；没有放宽任何像素容差。证据为prototype_report.json和both_axes_report.json及对应console日志。

最终净诊断在两套验证任务均结束后执行，只比较当前v1和两轴4线程。固定同8图、每方法3次、交替先后顺序；每图取中位数再求8图均值：

| CPU load768 | ms/图 |
|---|---:|
| 当前v1（已含直接CHW归一化） | 97.757 |
| 两轴4线程＋同一归一化 | 88.070 |
| 相对v1净省 | 9.687 |

这是CPU诊断，不是正式GPU端到端测速，不能拿旧033的归一化收益再计一次。前5张6–8MP各省12.2–16.7ms；3.1–3.5MP的三张收益较小，其中一张慢0.85ms。因此如root决定进一步验收，建议保守地仅对≥4MP大图使用4线程，其余保留v1；不继续扩大线程数/阈值搜索。当前原型默认1MP仅为实验参数，不代表已接受的生产阈值。

可见CPU affinity为0–7。当前cgroup叶及中间层没有cpu.max，user.slice为max100000；cpu.stat未见节流记录，没有发现可见quota限制。这里只能说明可读cgroup状态，不能排除宿主机调度或CPU共享。Pillow每个band的resize还会重复系数预计算及图像分配，线程启动、裁剪/拼接及串行步骤会抵消部分并行收益。

净诊断脚本及记录为diagnose_net_v1.py、net_vs_v1_report.json、net_vs_v1.console.log。没有正式预测/AUC证据前维持v1生产状态；若部署候选，仍须root先做独立三轮100图GPU测速及完整valid概率/排序核验。代理卡结果不能称为T4认证。
