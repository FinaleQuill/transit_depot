# 正式入口一致性与项目打包复核

2026-10-29，Asia/Hong_Kong。仅CPU代码/已保存输出复核；模型前向0次、GPU工作0次、训练0次、test预测重生成0次。没有修改033模型/config/原CSV。本文不是T4时延或整包合规认证。

## 已实现

- `predict.py`和`cost.py`在模型加载前调用同一个`configure_inference()`：固定Python/NumPy/Torch seed43、关闭TF32及cuDNN benchmark，沿用已有模型eval路径；没有改用新的deterministic算法策略，没有改概率计算。
- `cost.compare_predictions`保留旧接口与`atol=1e-7`（比PDF允许的1e-3更严），增加有限/合法概率、身份覆盖、默认0.5及config Gate的`>=`二值判定、样本内类别排序/并列、同类跨比较样本排序/并列检查；复杂度O(n log n)，无全量n²矩阵。
- 默认要求两套身份均覆盖；完整submission作为计时子集参考时，必须显式给出计时身份并允许参考超集。空集、缺失身份、复用subject-less引用、非法向量都不通过。
- `cost.py`另核每张计时图。读取已有最终`submission.image_predictions.csv`，用官方manifest的image_id映射计时目录的subject/study/filename，比较本轮100张逐图sidecar；再独立比较Study聚合。没有读取目录外图像或重新生成预测。缺最终逐图sidecar时仍可计时，但结果pending、不能认证。
- 新参数`--image-submission`可指定最终逐图sidecar位置；默认与submission.csv同目录，名称为submission.image_predictions.csv。

PDF没有进一步定义“概率排序”的范围及tie处理；本实现同时检查样本内类别顺序和跨样本各类顺序/精确并列，属于保守工程检查，不宣称这些细节是官方另行明示。排序不一致统计使用每样本排序位置/并列组签名，未把pairwise inversion数冒充样本数。

## 验证

固定比赛镜像内CPU unittest：新17项一致性/入口测试、原deployment12项、local-resource11项，共40项通过。入口测试的模型/T4/时间均为显式mock，只验证代码路径，不代表真实模型或T4运行。覆盖微于1e-7的排序/二值/Gate翻转、并列变化、缺失/额外身份、非法概率、默认与自定义sidecar以及共享seed调用顺序；缺sidecar必须pending也有测试。

另只读回放036已有NFS/本地v1输出：100张逐图及53Study均通过加强后的全部检查，最大概率误差0。具体报告为`test_receipt.json`、各测试log和`saved_output_replay.json`，源码SHA在`source_sha256.json`。生产代码改变仅涉及元数据、确定性入口及计时外核验，未改计时区间和模型数学。

## 主代理必须纳入最终项目的打包改动

授权后已扩展`package_kd_submission.py`的显式白名单与support文件收集，以下原审计要求已实现并通过24项package测试：

- `inference_reproducibility.py`、`fast_lanczos.py`，以及最终采用的其他runtime Python模块。
- `native_lanczos/__init__.py`、`resample.c`、`setup.py`、README和上游LICENSE/PROVENANCE；若保留上游源码/header作出处，亦一起入包。
- 通用x86_64/CPython3.12编译产物或可离线构建的wheel/源码构建入口，记录SHA和编译参数，构建不得依赖本机绝对路径、`-march=native`、驱动610.57.04或68SM。需要在固定镜像的新目录内实际构建并import验证，不能靠仓库PYTHONPATH碰巧找到缺失文件。
- 包README写出自动准备/构建步骤及实际依赖；正式推理仍从原始文件逐张读到写出。原生fallback只保证语义，不能因为fallback可运行就宣称有加速或达标；报告应明确实际backend/线程/能力。

打包现会校验最终逐图sidecar的完整原始test身份集合，将原`submission.image_predictions.csv`逐字节复制到包根并登记SHA；支持`--image-submission`覆盖来源路径。没有重新生成或改写原CSV。

建议`verify_reports`拒绝缺共享seed及新语义证据的旧cost报告：检查seed43、3轮各100张逐图覆盖/零不一致、Study覆盖/零不一致、sidecar SHA与包中引用一致、同config/实际推理文件绑定，再接受其通过标志；不要仅信一个旧`official_compliance_pass=true`布尔值。这些检查已实现，并用旧布尔伪通过、seed变化、99张样本、语义失败、sidecar SHA变化及超100ms的合成报告测试覆盖。

根cost.py本身是插入code路径的转发器，使用相同`code/cost.py`与`code/predict.py`逻辑，结构无需改成独立推理实现。实际权重/config仍按500,000,000bytes核算；native实现属于代码/环境依赖，不是额外模型权重，编译推理引擎如实际采用则按规则计入。

## 保留的材料边界

已保留原024索引，真实派生teacher-only并合并029/030，执行collector后7个run、350份小文件complete=true、pending=[]。fold0恢复和controller原receipt未finalize等历史局限仍随材料保留。当前无真实T4复测；两份原始ImageNet预训练权重在已知路径缺失，未下载，仍是完整核验包实际缺项。固定样本一致与81.51ms代理卡耗时不能替代主办方指定T4及官方样本复测。


## 教师/学生证据追加只读核验

`training_evidence_readonly.json`记录本次实际读取：5名教师ledger链与last epoch覆盖20/12/12/12/12完整；只缺废弃022 student的config及audit引用。029/030均13轮早停、E8最佳，fresh-start receipt完整exit0且原生日志声明/实际bytes分别30489/30351相同。未读取大checkpoint二进制，尚未重做最终权重与账本SHA绑定。

控制器第一个日志仅等待后KeyboardInterrupt，不含fold1–4训练；第二个日志覆盖fold1/2/3后child RuntimeError，原receipt未finalize；第三个覆盖fold3/4且exit0。后两日志合计48个epoch的预期首/每100步/末批标记及完成summary全部齐全。可在新teacher-only索引按实际日志scope建立逐fold group coverage，绑定本次audit的原日志SHA，保留失败/恢复、未finalize receipt及fold0 tmux恢复局限。不要把原索引第一个日志的complete声明误当四fold覆盖，也不修改原始receipt的结束时间或exit。

建议新索引路径为`artifacts/kd_v1/final_dual768_evidence_20261029/teacher_only/evidence_manifest.json`，只移除不参与033的旧student run及对应timeline声明，原件与所有历史来源保留；再按029、030顺序合并至同目录`combined/`。实际teacher-only派生、029/030顺序合并及collector深核验已完成，见`training_evidence_assembly.json`。它表示训练证据收集器通过，不表示最终整个提交包或T4已认证。


最终测试：package相关24项、加强一致性17项本次通过；原deployment12项和local-resource11项此前也通过。首次package sidecar测试发现局部变量覆盖pretrain清单，修复后重跑全24项通过，初始失败保留在`package_tests.failed_initial.log`。所有测试使用合成小文件或已保存输出，无模型/GPU运行。
