# 历史实验与当前申报链路边界

本目录保留历史原件/摘录，不改写过去。所有源副本的 SHA-256 见 `sources.json`。

## 已有文件能证明的事实

- `run019_v2_run_config.json:3`、`run020_v2_run_config.json:3` 的模型为 `convnextv2_base.fcmae_ft_in1k`。`run020_v2_runner_status.json` 记录五折已执行训练、完成轮数 6/5/5/5/5。因此“本项目从未使用 ConvNeXt V2”是错误声明。
- `run019_v2_status.json:2` 状态为 `cancelled_after_validation`；其 `test_inference_complete` 和 `test_submission_generated` 均为 false。`selected_v1_016_provenance.json` 记载因为 019 valid 分较低，选择 016，`platform_upload_performed=false`。
- `run016_v1_config.json:7` 的 `backbone_name` 是 `convnext_tiny.fb_in1k`。016 目录名中的 `v2` 是旧工程版本标识，不代表 ConvNeXt V2；禁止只按目录字符串判断架构。
- `KD_V1_USAGE.md:9` 明确历史 020 是 V2 训练产物，“不能用于本流程”，并且没有被删除或覆盖。`model_overnight_current.py:19` 至配置校验明确当前仅允许 ConvNeXt V1 Base/Tiny；`kd_common_current.py:13` 至 `validate_pretrained_provenance` 保留实际来源白名单。
- `WORKLOG_historical_excerpts.md` 保留原 WORKLOG 行号和源 SHA：原 L825–831 记录 019 test 推理被取消、交付016且未上传；原 L847–855 记录迁移到 V1 非医学公开权重和隔离旧 020。`CODE_CLEANUP_20261026.md` 记录删除旧入口、保留历史文字/模型证据。

## 当前可以写的否定声明

“本次申报的 V1 教师—学生链路、041 历史最高分 CSV 所对应推理模型未使用历史 ConvNeXt V2 权重或其软标签/特征；历史 V2 实验是单独弃用分支，其实际存在如上披露。”

前半句必须与本包本次 V1 checkpoint/source/soft-label 血缘证据共同核对，本目录本身只提供历史边界及静态白名单证据。

## 当前仍不能从本地材料断言的事实

现有019记录证明当时该候选没有生成 test CSV，016记录证明该次本地交付未执行平台上传。这些不能证明所有时间、所有平台账号从未提交过任何 V2 产物。当前没有完整平台历史提交/撤销台账，不能写“从未提交V2”，也不能反过来把缺失台账说成“已发生违规提交”。该点需要队伍历史台账或平台记录方可收口。

`KD_V1_COMPLIANCE.md`、`README_current_snapshot.md` 等副本包含各自时间点的旧方案和待办，只作历史证据，不能把其中旧部署候选或旧计时口径当作本包已选定的最终成员/验收结论。
