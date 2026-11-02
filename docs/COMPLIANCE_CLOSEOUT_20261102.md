# 2026-11-02 合规材料收口与待验收状态

本轮只整理合规材料、两轮自查及用户明确要求的100样本cost一致性复核；没有训练、优化、租机、改权重、选模或重新生成提交CSV。

包内根目录：[审查材料](../artifacts/compliance_closeout_20261102/package/README.md)。最终ZIP及SHA见 [交付回执](../artifacts/compliance_closeout_20261102/DELIVERY_RECEIPT.json)。正式上传与平台操作未执行。

## 五项验收入口

1. [compliance_matrix.md](../artifacts/compliance_closeout_20261102/package/compliance_matrix.md)：规则1–11、A1–A6、补充I–XI及附录总则共29行；19已满足、7待补、3不适用（附理由）。
2. [compliance_selfcheck.md](../artifacts/compliance_closeout_20261102/package/compliance_selfcheck.md)：两轮记录、实际打开引用、首ZIP陌生人审查及补正。
3. [三节声明](../artifacts/compliance_closeout_20261102/package/declarations.md)：valid用途、固定Study均值/FP16/TTA、V1范围的否定声明；保留历史V2隔离事实。
4. [XLSX](../artifacts/compliance_closeout_20261102/package/模型核验材料与数据声明清单.xlsx)：C区FP16，教师五份SHA，候选A/B/X1/045与弃用产物SHA；A1–A5冻结62单元格未变。
5. [包结构](../artifacts/compliance_closeout_20261102/package/PACKAGE_STRUCTURE.md) 与 [cost一致性报告](../artifacts/compliance_closeout_20261102/package/cost_consistency_report.md)。

## 已补齐的关键证据

教师best.pt由原源码与逐轮记录证实是固定E20/12/12/12/12最终EMA，不按held-out分数选epoch。教师LRwarmup1/无KD，初始学生KDwarmup2，X1及裁剪支线KDwarmup0，分别列seed/起点/日志。初始Tiny与Base hard-loss实现差异也已如实申报。

041原config从A EMA保留metadata与原导出函数恢复，完整SHA348901…fbeb39、11,284,048bytes与原回执一致，权重SHAfea235…42b5f保持原字节；历史加载合计66,996,559bytes。根submission.csv是历史最高分76.92对应041，SHA44dde95b…f66e未改。

在官方3.0镜像digest125d8b…b0b、当前2080Ti上执行一次规定的三轮核验：100图/54完整Study，另5图预热。三轮逐图/Studymax差0、二值/排序差异0；独立标准库复算亦FP64字节一致。只验证固定样本一致性，不算AUC或选参；其66.296177ms/图是旧逐图口径参考，不能当T4或新StudyGate。退出码2明确因official_compliance_pass=false，数值比较本身matched。

## 必须保留的缺口

当前不满足“零待补”完成定义，已停止等待验收决定：

- 最终成员尚未冻结；不能替提交人把041/043/045填成最终成员，最终去重体积与申报身份未闭合。
- 主办方指定完整100样本、T4及逐Study严格验收缺证；本地功能样本按身份字典序确定性选择，明确非官方指定集。正式书面与群公告计时口径原样记录，不将旧计时伪装新规通过。
- 队伍名尚未给定，ZIP使用待确认占位；附录A命名/模板/目录为示例，不新增空白模板认证闸门。

预训练二进制目录是附录提交示例；离线教师不参与推理，补充III允许不附教师二进制，但本包已申报五份SHA与许可证。没有将示例误当必须上传所有预训练文件。

完整二进制、原始数据元信息、历史日志与证据包留在工作区，不纳入Git；Git保存本次审阅文档、声明、工作簿及包哈希索引。无需也不允许因此自动恢复原训练/优化排期。
