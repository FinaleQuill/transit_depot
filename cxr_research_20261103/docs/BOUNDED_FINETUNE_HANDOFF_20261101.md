**2026-11-01复盘完成：** 两项固定4轮训练和实际组合核验已结束，最高valid 0.817149190/93.49ms，draft 0.816904529/87.10ms；未达0.82，增益区间跨0，原043/044保留、本轮无test预测。已保存实验模型，不重复启动后台训练；先读[最终复盘](BOUNDED_FINETUNE_REVIEW_20261101.md)。

**2026-11-01 15:08最新后台交接：** 用户要求长训练后台运行、提供查看文件，完成后由用户再通知代理。现已启动脱离聊天的串行控制器（4ef0f28），PID9897、PPID1，自动接管正在运行的X1_packed，完成最终FP16/valid核验后，预算足够时自动启动X2_packed及完整valid评估。**不要手动启动第二份X1、X2或pipeline。** 先读`artifacts/kd_v1/bounded_finetune_20261101/STATUS.md`和`pipeline_status.json`，全局state=completed/stage=ready_for_review后再复盘、选模、计时。不会自动预测test、覆盖044或上传。若failed/budget_exhausted，查看pipeline_error.txt/各日志后接手。以下“下一步”的手动命令仅供故障接手参考，不能在后台任务活着时直接执行。

# 2026-11-01 有界提分执行交接

用户已重新授权提分和训练，目标AUC≥0.82，最多12小时，原停止边界已被替代。用户给全权限，当前无需逐步确认。硬截止香港2026-11-02 02:37；保留最终验证余量。

此前已知最好完整valid为旧AB完整解码0.8168300303、044 draft为0.8166770734；用户平台反馈044与单B043基本持平或略逊。新版7814f75的B2报告只到0.800967459、R2到0.813657853，B2模型已丢，不采用。当前没有本轮完成的新AUC。

固定主计划见BOUNDED_STUDY_FINETUNE_20261101.json（SHA c73aef018b7841bf3d47cd0e88b79f79d4c32aef1555ec8cdadd119ece76ee93）；为实际吞吐另登记Study打包补充BOUNDED_STUDY_PACKING_20261101.json（SHA d6ff67eb13776b66e1737bd953f9929919d82039108151edc158625e81cb5471）。在任何新valid前登记，不回写主计划。

X1：旧A E9最佳FP32 EMA→Tiny完整Study4轮，seed46，LR1e-5/1e-4、α=.2、drop_path=.1。X2：旧B E10→Base完整Study4轮，seed47，LR5e-6/5e-5、α=.05、drop_path=.15。两者768/FP16、sqrt posweight、Study batch32、LR warmup1、KD warmup0、最后EMA；不挑中间最佳轮，不追加valid权重搜索。

## 当前运行

X1原逐Study尝试已中止，0完整epoch/0valid，exit137；证据X1.execution.json及X1_unpacked_interruption.json保留。实际打包尝试于香港14:57启动，host launcher用`python3 -u run_bounded_finetune.py --stage X1 --execute`，容器`codex-bounded-study-x1-packed-20261101`。**先查进程，禁止重复启动。**

- 持久证据：`artifacts/kd_v1/bounded_finetune_20261101/`，看`X1_packed.training.log`、`X1_packed.execution.json`。
- 完整本地状态：`/tmp/cloud_student_runs_20261030/bounded_finetune_20261101/X1_packed/`。
- 控制器持续复制小文件/source/log/history到work；训练完成自动严格核E4最终EMA，导出到证据`models/X1/`，写`X1.final_ema.json`。训练仅放/tmp，work只余300MB，两个新FP16约241MB含config；不得再复制大包或删旧资产。
- 原逐Study目录`.../X1/`保留，不把它的state=running旧status当仍活着；实际以容器和execution/interruption为准。

## 下一步

1. 监测X1打包吞吐、finite loss、显存与终态。首轮实测成本后更新总预算；不根据首轮valid延长训练。
2. X1四轮完成且导出成功：`STUDENT_RUNTIME_MODE=inference bash run_student_environment.sh -u evaluate_bounded_finetune.py --stage X1`。该入口核最终来源、valid全图SHA、实际FP16/hflip2，并以1000患者bootstrap复算固定规则；保留选择偏差说明。
3. 复盘X1总体/逐类/切片与成本并反馈，再在剩余预算足够时启动`python3 -u run_bounded_finetune.py --stage X2 --execute`；X2不与其他GPU任务并行。X2同理实际评估。
4. 固定6条候选见计划；不要扫权重或混用test。选择前还需实际组合完整valid、100+5/连续3轮端到端计时（旧034 inputs）、体积和逐图/Study/排序一致性。优选80ms，本机硬100ms；2080Ti不冒称真实T4认证。
5. 优于旧版才做新交付；选定及记录SHA后仅最终test预测一次并独立审计，不上传。若没有改进如实报告，保留043/044。最终新模型权重用hardlink同work已有资产以省空间，勿原地改任何被链接权重。

## 环境与代码

比赛digest镜像已存在，GPU2080Ti 22GiB、驱动610.57.04。原train30328图/valid1422图齐全；样本SHA与患者/Study/图像内容隔离通过，训练loader逐图验证原SHA。只用官方train；教师和严格OOF冻结，test尚未读取。inference addon已验证从038恢复，训练addon在/tmp/cloud_student_training_addons_cp312精确恢复。`run_student_environment.sh`默认training；`STUDENT_RUNTIME_MODE=inference`只用官方镜像核心和timm addon，挂载同一训练本地盘用于导出/验证。

新增严格EMA warm-start位于train_distill.py，血缘`student_warm_start`传播完整checkpoint/快照/export；可继续真正resume新run，不能说旧EMA是完整resume。StudyConfig支持显式backbone_name与pack_images=8，Study打包独立梯度测试通过。训练59项、打包23项、CLI/一致性33项、固定规则4项和ensemble3项检查通过（有交集，不累计声称全不同用例）；初次测试命令模块路径错误日志保留，final/corrected通过。

`predict.py`和`cost.py` CLI现在共用`submission_runtime.load_submission_predictor`，支持固定single/ensemble/044 draft，旧Python factory API仍保留。新builder `make_ensemble.py`需--rule-name并验证plan成员/权重/TTA/decoder/finalepoch/init血缘；选定后要实际运行验收。裸cost CLI的真实T4验收仍待。

当前提交：b577baa主计划，0838e4c EMA初始化，caa77df builder绑定，e167dd5 shared CLI，92e80dc final实际评估入口，7b3c641 Study打包。不要推送。三份原未跟踪文件fixed_student_candidate.py/review_student_upper_bound.py/student_probability_fusion.py是旧实验，不盲删或add。原WORKLOG前缀按CRLF字节追加，检查用git -c core.whitespace=cr-at-eol diff --check。
