# Tight-crop A/B training curve snapshot

Snapshot updated: 2026-11-02 08:09:27 UTC. Read-only extraction from the four `history.jsonl`, `status.json`, and `config.json` files under `/tmp/cloud_student_runs_20261030/tight_crop_ab_20261102/earlystop25/`. No model execution or image reads. B_base is `paused` after the user-requested STOP, with E1-E3 fully validated. This is a deliberate interruption, not natural patience-based early stopping or training failure. Every row below is a completed, validated epoch.

`Overall` means all 859 valid Studies' macro-AUC. `AP-only` means the 442 AP-only valid Studies' macro-AUC, the early-stop monitor. These are AUC values, not average precision. BCE and KL are the training history's `hard_bce` and `soft_kl_T2`, not validation losses. LR is the recorded end-of-epoch `backbone_decay`/`head_decay` rate; the no-decay groups match their corresponding decay groups in these records.

| Run | Epoch | Overall AUC | AP-only AUC | Train BCE | Train KL T2 | Backbone LR | Head LR | New best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| A_tiny | 1 | .808700868 | .738964615 | .416886 | .124483 | 9.968912e-6 | 9.968912e-5 | yes |
| A_tiny | 2 | .808482322 | .739348586 | .409869 | .133381 | 9.959637e-6 | 9.959637e-5 | yes |
| A_tiny | 3 | .807752355 | .738614416 | .401176 | .136624 | 9.838680e-6 | 9.838680e-5 | no |
| A_tiny | 4 | .805512267 | .737183711 | .394051 | .140029 | 9.639198e-6 | 9.639198e-5 | no |
| A_tiny | 5 | .804567110 | .736078895 | .386497 | .143286 | 9.365141e-6 | 9.365141e-5 | no |
| A_tiny | 6 | .803441121 | .735559333 | .380256 | .146324 | 9.020253e-6 | 9.020253e-5 | no |
| A_tiny | 7 | .801519897 | .733449053 | .373653 | .150914 | 8.610845e-6 | 8.610845e-5 | no |
| A_base | 1 | .809010809 | .737626633 | .407840 | .117695 | 4.992228e-6 | 4.992228e-5 | yes |
| A_base | 2 | .808271469 | .736543067 | .404954 | .122902 | 4.979819e-6 | 4.979819e-5 | no |
| A_base | 3 | .807664108 | .735220467 | .397737 | .124374 | 4.919340e-6 | 4.919340e-5 | no |
| A_base | 4 | .806908200 | .734861101 | .390944 | .128210 | 4.819599e-6 | 4.819599e-5 | no |
| A_base | 5 | .806638491 | .735072981 | .386578 | .130518 | 4.682302e-6 | 4.682302e-5 | no |
| A_base | 6 | .806458595 | .734866739 | .379493 | .133059 | 4.509800e-6 | 4.509800e-5 | no |
| B_tiny | 1 | .808854430 | .739087500 | .421230 | .125680 | 9.968912e-6 | 9.968912e-5 | yes |
| B_tiny | 2 | .808773078 | .739769114 | .413956 | .134062 | 9.959637e-6 | 9.959637e-5 | yes |
| B_tiny | 3 | .808092739 | .738903796 | .405908 | .136810 | 9.838680e-6 | 9.838680e-5 | no |
| B_tiny | 4 | .806142730 | .737416707 | .398954 | .139722 | 9.639198e-6 | 9.639198e-5 | no |
| B_tiny | 5 | .805059298 | .736530291 | .391167 | .142580 | 9.365141e-6 | 9.365141e-5 | no |
| B_tiny | 6 | .804606534 | .736147803 | .385603 | .145258 | 9.020253e-6 | 9.020253e-5 | no |
| B_tiny | 7 | .803098379 | .734878264 | .378485 | .149700 | 8.610845e-6 | 8.610845e-5 | no |
| B_base | 1 | .808987478 | .736490349 | .411839 | .118690 | 4.992228e-6 | 4.992228e-5 | yes |
| B_base | 2 | .808794301 | .736283911 | .409183 | .123323 | 4.979819e-6 | 4.979819e-5 | no |
| B_base | 3 | .808169222 | .735103166 | .402043 | .124776 | 4.919340e-6 | 4.919340e-5 | no |

A_base reached its best AP-only monitor at E1 and early-stopped after E6. Across E1-E6, train BCE fell from .407840 to .379493, while train KL T2 rose from .117695 to .133059; overall AUC fell from .809011 to .806459 and AP-only AUC from .737627 to .734867. Its final E6 is not the selected `best.pt`; the selected epoch is E1. This is evidence of degradation during this continuation on the monitored valid split, with no causal attribution to the OOF teacher.

B_base's E1-E3 overall AUC fell from .808987 to .808169 and AP-only AUC from .736490 to .735103 while train BCE fell from .411839 to .402043 and KL rose from .118690 to .124776. Its `status.json` records `state=paused`, `completed_epochs=validated_epochs=3`, `best_epoch=1`, and `selected_checkpoint=best.pt`; the `STOP` marker remains present. E3 was completed before this user stop took effect.

## Model and OOF lineage

- The current Base runs are **students**, warm-started from `artifacts/kd_v1/student_upper_bound_20261030/models/B/best_ema_fp32.pt`, whose run config identifies source epoch 10. A_base and B_base use the same recorded warm-start SHA-256 `9cb733eb65b71eeb0fe7633122dcb085b47fc4c13b4f46f139f5b773a5e805a3`. Tiny starts from `models/A/best_ema_fp32.pt`, source epoch 9. See each run's `config.json` fields `student_warm_start`, `init_student_ema`, and `training_mode=distillation`.
- Their soft labels point to `/home/featurize/work/runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2/`. Its `metadata.json` declares `source_split=train`, `output_type=per_view_raw_logits`, `tta=identity_hflip`, `complete=true`, and five teacher records. The `teacher_collection.json` lists fold epochs 20/12/12/12/12, fixed final EMA. Those five CV teachers are distinct from the current Base student.
- The run config records `alpha=.2`, `temperature=2`, Study-equal weighting, and a hard Study BCE plus image-level soft KL T2 objective. The decreasing BCE alongside rising KL is compatible with a tradeoff in the recorded objective; it does not establish that the teacher was wrong or that its quality caused valid decline.

## Source receipt

The compact numeric snapshot is in `TRAINING_REVIEW.json`. Its SHA-256 source hashes bind the four history and config files used here. Teacher collection SHA-256: `0ec962fab2cc0db3b87619ea6d9c05bf6205ad8441b4e479d0732d4bd4f03b3e`; OOF metadata SHA-256: `39b8fd6014b989e7eb8905303410ac963ece1bb985816e29a52253a6b2c9e2bd`. The original training files and frozen pipeline were not edited.
