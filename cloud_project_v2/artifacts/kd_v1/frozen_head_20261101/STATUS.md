# 冻结骨干、重训分类头：后台状态

状态：completed；阶段：ready_for_review
更新：2026-11-01T21:21:44.284238+08:00
原预算截止：2026-11-02 02:37（香港）

045最高分CSV已经生成并审计；本后台不读取它作训练/选择，不自动生成新的test CSV。
缓存：[CACHE_COMPLETE] 1462.9778683209988

A_F1: completed，12/12轮
A_F2: completed，12/12轮
B_F1: completed，12/12轮
B_F2: completed，12/12轮

CPU缓存分数只是代理；最终以GPU FP16完整valid和端到端计时为准。

实际GPU完整valid：
- A_F2_B_F2: AUC=0.817614546，102.960ms
- A_B_F2: AUC=0.817614160，95.410ms

说明：四个固定12轮小头、预登记组合代理评估和最高两项实际FP16/完整valid/计时均完成；等待复盘。045CSV保持既定版本，没有自动新test预测。

全局state=completed时通知代理复盘；failed/budget_exhausted时也可直接通知。
不要重复启动本pipeline或缓存。大文件在/tmp；小patch与日志已保存在此目录。
