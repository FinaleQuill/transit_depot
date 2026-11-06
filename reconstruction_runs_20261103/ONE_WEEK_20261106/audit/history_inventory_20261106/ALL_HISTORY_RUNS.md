# 全部不同history内容的运行摘要

此表依据完整读取的history及同目录status机械提取，精确副本合并；非同内容的续训/迁移快照仍可能重叠。AUC只标识记录口径，不与其他数据/精度直接比较；“—”为未记录，不能当成零。

| history代表路径 | 完整轮/最后轮 | 记录中最高Macro-AUC/轮 | 状态；跳步 | 备注 |
|---|---:|---:|---|---|
| `archive_v1/runs/013_cloud_4090_fixed/history.jsonl` | 2/2 | 0.755249/E2 | epoch_complete; 4 |  |
| `archive_v1/runs/014_cloud_4090_max/history.jsonl` | 12/12 | 0.797719/E6 | epoch_complete; 0 |  |
| `cloud_project_v2/artifacts/compliance_closeout_20261102/package/evidence/training/crop_A_base/history.jsonl` | 6/6 | 0.737627/E1 | early_stopped; 1 | 2份精确副本 |
| `cloud_project_v2/artifacts/compliance_closeout_20261102/package/evidence/training/crop_A_tiny/history.jsonl` | 7/7 | 0.739349/E2 | early_stopped; 2 | 2份精确副本 |
| `cloud_project_v2/artifacts/compliance_closeout_20261102/package/evidence/training/crop_B_base/history.jsonl` | 3/3 | 0.736490/E1 | paused; 1 | 2份精确副本 |
| `cloud_project_v2/artifacts/compliance_closeout_20261102/package/evidence/training/crop_B_tiny/history.jsonl` | 7/7 | 0.739769/E2 | early_stopped; 2 | 2份精确副本 |
| `cloud_project_v2/artifacts/compliance_closeout_20261102/package/evidence/training/student_A/history.jsonl` | 14/14 | 0.809129/E9 | early_stopped; 6 | 3份精确副本 |
| `cloud_project_v2/artifacts/compliance_closeout_20261102/package/evidence/training/student_B/history.jsonl` | 15/15 | 0.810569/E10 | early_stopped; 3 | 2份精确副本 |
| `cloud_project_v2/artifacts/kd_v1/bounded_finetune_20261101/training/X1_packed/history.jsonl` | 4/4 | 0.808357/E1 | completed; 1 | 2份精确副本 |
| `cloud_project_v2/artifacts/kd_v1/bounded_finetune_20261101/training/X2_packed/history.jsonl` | 4/4 | 0.809130/E1 | completed; 1 |  |
| `cloud_project_v2/artifacts/kd_v1/frozen_head_20261101/fits/A_F1/history.jsonl` | 12/12 | — | —; — |  |
| `cloud_project_v2/artifacts/kd_v1/frozen_head_20261101/fits/A_F2/history.jsonl` | 12/12 | — | —; — |  |
| `cloud_project_v2/artifacts/kd_v1/frozen_head_20261101/fits/B_F1/history.jsonl` | 12/12 | — | —; — |  |
| `cloud_project_v2/artifacts/kd_v1/frozen_head_20261101/fits/B_F2/history.jsonl` | 12/12 | — | —; — |  |
| `cloud_project_v2/artifacts/tight_crop_ab_20261102/runs/A_tiny/history.jsonl` | 1/1 | 0.808717/E1 | paused; 1 | 2份精确副本 |
| `reconstruction_runs_20261103/BREAKTHROUGH_20261105/DINOv2_B518_fold0123/history.jsonl` | 1/1 | 0.708464/E1 | epoch_completed; 0 |  |
| `reconstruction_runs_20261103/BREAKTHROUGH_20261105/RAW_REBUILD/DINOv2_B518_fold0123/history.jsonl` | 21/21 | 0.804905/E20 | stopped_by_user; 0 |  |
| `reconstruction_runs_20261103/BREAKTHROUGH_20261105/RAW_REBUILD/smoke_B518/history.jsonl` | 1/1 | — | smoke_completed; 0 | 机械试跑 |
| `reconstruction_runs_20261103/BREAKTHROUGH_20261105/smoke_B518/history.jsonl` | 1/1 | — | smoke_completed; 0 | 机械试跑 |
| `reconstruction_runs_20261103/BREAKTHROUGH_20261105/smoke_B518_pack16/history.jsonl` | 2/2 | — | smoke_completed; 0 | 机械试跑 |
| `reconstruction_runs_20261103/E0_base768_seed20261103/history.jsonl` | 14/14 | 0.797840/E8 | early_stopped; 3 |  |
| `reconstruction_runs_20261103/E1_base768_seed20261103/history.jsonl` | 12/12 | 0.799239/E7 | interrupted; — |  |
| `reconstruction_runs_20261103/E2_20261104/controller/performance_transition/global_control_history.jsonl` | 1/1 | 0.796436/E1 | —; 0 | 初始epoch0最佳，不把训练记录最高误作所选最佳；续训/迁移快照 |
| `reconstruction_runs_20261103/E2_20261104/global_control_seed20261104/history.jsonl` | 3/3 | 0.796436/E1 | paused; 1 | 初始epoch0最佳，不把训练记录最高误作所选最佳；3份精确副本 |
| `reconstruction_runs_20261103/E2_20261104/global_local_seed20261104/history.jsonl` | 2/2 | 0.796341/E1 | interrupted; — | 3份精确副本 |
| `reconstruction_runs_20261103/ENSEMBLE_SPRINT_20261105/context512/history.jsonl` | 4/4 | — | completed; — |  |
| `reconstruction_runs_20261103/ENSEMBLE_SPRINT_20261105/opacity512/history.jsonl` | 6/6 | — | completed; — |  |
| `reconstruction_runs_20261103/KD_REBUILD_20261106/dinov3_l_inner0/history.jsonl` | 2/2 | — | interrupted; — |  |
| `reconstruction_runs_20261103/KD_REBUILD_20261106/mechanics/convnext_l/history.jsonl` | 2/2 | — | mechanics_smoke_completed; 0 | 机械试跑 |
| `reconstruction_runs_20261103/KD_REBUILD_20261106/mechanics/dinov3_l/history.jsonl` | 2/2 | — | mechanics_smoke_completed; 0 | 机械试跑 |
| `reconstruction_runs_20261103/SPRINT_20261104/history.jsonl` | 4/4 | — | completed; — |  |
| `reconstruction_runs_20261103/smoke_base768/history.jsonl` | 1/1 | — | smoke_completed; 0 | 机械试跑 |
| `reconstruction_runs_20261103/smoke_packed_base768/history.jsonl` | 1/1 | — | smoke_completed; 0 | 机械试跑 |
| `runs/016_a4000_tiny_v2/history.jsonl` | 6/6 | 0.799971/E5 | epoch_complete; 0 |  |
| `runs/017_a4000_tiny_v2_warm/history.jsonl` | 2/7 | 0.799877/E6 | epoch_complete; 0 | 续训/迁移快照 |
| `runs/019_convnextv2_base512_cv5_4090/fold_0/history.jsonl` | 5/5 | 0.802403/E5 | paused; 0 |  |
| `runs/019_convnextv2_base512_cv5_4090/fold_1/history.jsonl` | 4/4 | 0.790213/E4 | paused; 0 |  |
| `runs/019_convnextv2_base512_cv5_4090/fold_2/history.jsonl` | 4/4 | 0.788720/E4 | paused; 0 |  |
| `runs/019_convnextv2_base512_cv5_4090/fold_3/history.jsonl` | 4/4 | 0.798739/E4 | paused; 0 |  |
| `runs/019_convnextv2_base512_cv5_4090/fold_4/history.jsonl` | 4/4 | 0.794513/E4 | paused; 0 |  |
| `runs/020_convnextv2_base768_posbce_cv5_4090/fold_0/history.jsonl` | 6/6 | 0.806735/E6 | paused; 0 |  |
| `runs/020_convnextv2_base768_posbce_cv5_4090/fold_1/history.jsonl` | 5/5 | 0.801396/E5 | paused; 0 |  |
| `runs/020_convnextv2_base768_posbce_cv5_4090/fold_2/history.jsonl` | 5/5 | 0.794795/E5 | paused; 0 |  |
| `runs/020_convnextv2_base768_posbce_cv5_4090/fold_3/history.jsonl` | 5/5 | 0.803268/E5 | paused; 0 |  |
| `runs/020_convnextv2_base768_posbce_cv5_4090/fold_4/history.jsonl` | 5/5 | 0.799602/E5 | paused; 0 |  |
| `runs/021_convnext_v1_base768_teacher_cv5/fold_0/history.jsonl` | 20/20 | — | completed; 0 | 2份精确副本 |
| `runs/023_convnext_v1_base768_teacher12_cv5/fold_1/history.jsonl` | 12/12 | — | completed; 0 | 2份精确副本 |
| `runs/023_convnext_v1_base768_teacher12_cv5/fold_2/history.jsonl` | 12/12 | — | completed; 0 | 2份精确副本 |
| `runs/023_convnext_v1_base768_teacher12_cv5/fold_3/history.jsonl` | 12/12 | — | completed; 0 | 2份精确副本 |
| `runs/023_convnext_v1_base768_teacher12_cv5/fold_4/history.jsonl` | 12/12 | — | completed; 0 | 2份精确副本 |
| `runs/026_tiny512_bce_seed43/history.jsonl` | 11/11 | 0.803597/E6 | early_stopped; 0 |  |
| `runs/027_tiny512_kd_seed43/history.jsonl` | 12/12 | 0.805594/E7 | early_stopped; 0 |  |
| `runs/028_tiny512_kd_short12_seed43/history.jsonl` | 12/12 | 0.803891/E7 | early_stopped; 0 |  |
| `runs/029_tiny768_kd_seed43/history.jsonl` | 13/13 | 0.804177/E8 | early_stopped; 0 |  |
| `runs/030_tiny768_kd_seed44/history.jsonl` | 13/13 | 0.807570/E8 | early_stopped; 0 |  |
| `runs/031_resolution_train640_eval768_20261103/history.jsonl` | 3/3 | 0.803222/E1 | completed; 2 |  |
| `runs/032_resolution_train768_eval768_20261103/history.jsonl` | 3/3 | 0.807565/E1 | completed; 2 |  |
| `runs/033_resolution_train896_eval768_20261103/history.jsonl` | 3/3 | 0.807890/E1 | completed; 2 |  |
