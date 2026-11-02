# Living package — REVIEW-2 前置维护

本目录是可持续更新的审查主包，最终成员仍未冻结，不能据此声明正式合规通过。
A1–A5 的 62 格及其样式、许可证和来源材料继续冻结；本轮整个 XLSX 字节亦保持不变。

原交付 ZIP 和 DELIVERY_RECEIPT.json 保留原位原字节；独立只读快照及其 SHA-256 元数据位于 `../snapshots/closeout_c8bd868/`。文件 0444、目录 0555，只读权限不等同不可被管理员修改的 WORM 存储。

主包 code/ 可以接入新的合规记录实现；每次明确同步的来源文件与 SHA 记在 LIVING_PACKAGE.json，历史执行源码仍在 evidence/ 和原 ZIP 中，不能将今天的源码冒称历史运行版本。

未来经单独授权的新 run 在其 compliance/ 下同步记录启动、逐 epoch、checkpoint 落盘和结束事件；续训追加新会话，合成演示明确标注 synthetic，不充当真实训练或官方验收。运行证据与本包每次组装时的索引分开保存，组装后必须刷新并核验 PACKAGE_SHA256SUMS.json。

七项最终冻结登记见 REVIEW2_FREEZE_CHECKLIST_20261103.md；登记清单不等于通过，最终成员和官方 T4/指定样本/Study 验收仍待。本轮不自动训练、推理、优化、租机或上传。
