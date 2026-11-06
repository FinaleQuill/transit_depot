# 本轮运行入口

用户已授权：先发 ROI 图册，发完继续训练，不等回复；只使用官方 train，valid 仅用于早停/比较与固定计时；不访问 test，不生成赛事 CSV，不提交、不修改申报。

## 状态、监测和报告

固定状态文件：`/home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/STATUS.json`

```sh
watch -n 20 'cat /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/STATUS.json'
```

此文件每约20秒刷新当前臂、训练批次、已完成指标、compliance回执、预计结束UTC、失败原因与当前日志。ETA来自当前训练速度加后续保守估算，不是承诺。

晨间报告：`MORNING_REPORT.md`，分辨率、ROI、主比较、best次级比较完成时更新。未完成项明确标为未完成，不使用历史结果填补。
控制器日志：`queue/controller_v2.log`。各步骤日志：`queue/logs/`。每步证据回执：`queue/receipts/`。每个训练run的原始生命周期：对应run_dir/compliance/events.jsonl。

## 当前计划

已完成640三轮只核验复用，不重训；按顺序完成640同协议评估、768、896及三臂逐类/患者配对CI；之后原041起点同协议评估、纯ROI、MIL、固定50:50的768+896及768+ROI；最后补充各best EMA并给一条人工取舍建议。
主因果比较固定E3，次级候选用各自best EMA。全部共同768推理+hflip2；identity只作TTA对照。
敏感性地图只支持“训练分辨率在共同768推理下的类别差异”，不能证明每类最终该用哪个推理尺寸。按图确定下一步通路，不能将探索CI当独立验证。
时间不作本轮准入；每项留三图Study P50/P95/max，2080Ti结果不是官方T4认证。

## ROI

四视角12张图册和自审：`../roi_magnify_explore_20261103_v2_2/gallery/index.md`、`SELF_REVIEW.md`。已向用户发送链接。
经典包络+2% margin，等比长边填768、短边补边、只放大、必要时原letterbox NO CUT；无Random Erasing。48张9 apply、39 NO CUT、0缩小，倍率1.0136–1.2152。未见新增截断；源片遮挡/触边和启发式限制已登记，不作全量解剖保证。85%占比版本作废。

## 需要用户验收的清单（不阻断已授权训练）

| 项目 | 待判断内容 |
|---|---|
| 解剖图 | 是否有新增截肺；看到问题明确叫停 |
| 最终成员 | 根据完整指标/CI/耗时，是否接受候选建议；未冻结成员 |
| 规则边界 | 出现官方书面口径冲突时转官方确认 |
| 线上提交 | 用户执行与判断；本轮无接口、不生成CSV、不提交 |

## 失败与停止

任何命令失败、预注册缺失、SHA变化、compliance或指标不完整，控制器停止推进并写本目录和queue目录的`FAILED`以及失败原因，不静默跳过。不自动恢复失败。

```sh
touch /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/STOP
```

STOP由控制器轮询触发停止，不以强制删除权重/证据清理。恢复必须先查失败日志和进程，保留失败尝试；不得重复开跑或删除证据。正常命令异常可落盘，机器掉电/SIGKILL无法保证写出终态，若心跳停止须先查进程。

自动化是本地主机nohup串行控制器，不依赖对话在线，也不承诺训练完自动唤醒LLM；指标、规则门槛、报告和候选建议由已冻结程序自行写出。

## 来源差异的审计

640两个session使用旧trainer SHA，新代码只多legacy resume validation_size默认值；精确剔除该三行与两个归档文件逐字节相同，且两session已有validation_size=768。新臂使用EMA warm-start不执行该默认分支。真实SHA分别绑定，不回写旧证据；证明见`queue/attempts/reviewed_legacy_resume_source_equivalence.json`。
首个shell-background启动在controller启动前消失，确认没有新任务/状态后改用nohup+独立session启动；保留`queue/attempts/shell_launch_failed_before_start.json`。
