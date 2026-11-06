# 工作区约定

- 当前停止全部训练；未经用户再次明确授权，不启动或恢复训练。
- 保留源码；旧方案、复盘及思路材料仅在隐藏Git归档中，除非用户要求，不主动恢复或读取。
- 规则原件：rules.pdf、rules_1.pdf；答疑：RULES_QA_CLARIFICATION_20261104.md。
- 原始数据：/home/featurize/data/cxr_first/；原图与CSV只读。获准训练时完整保留官方原始30,344张train图、18,513个Study，缺图报错。
- 已训练教师、模型必要配置/来源/许可证、既有STOP及submission_best.csv保留。
- 历史完整内容保存于本地 .project_history.git；源码与文档归档远程为 https://github.com/FinaleQuill/transit_depot.git。数据、影像与权重留在本机。
