# Exploration comparison

All metrics are official-valid exploration results; no test access or platform submission.

| Candidate | Overall AUC | AP-only AUC | Three-image P50/P95 ms |
| --- | ---: | ---: | --- |
| res640_identity | 0.802002 | 0.731810 | 603.40/635.71 |
| res640_hflip2 | 0.803244 | 0.734243 | 593.56/652.78 |
| res768_identity | 0.806706 | 0.737397 | 570.45/638.52 |
| res768_hflip2 | 0.807556 | 0.738682 | 596.77/670.18 |
| res896_identity | 0.806932 | 0.736726 | 565.59/622.13 |
| res896_hflip2 | 0.807891 | 0.737578 | 635.88/711.19 |
| res768_res896_equal_identity | 0.807240 | 0.737250 | unmeasured |
| res768_res896_equal_hflip2 | 0.808187 | 0.738433 | unmeasured |

Intervals resample patients with all their Studies. They are conditional on selected checkpoints and do not remove repeated-validation selection bias.

Resolution evidence concerns training size at common 768 inference; disease-specific inference-size recommendations remain untested.

- res640_identity_minus_res768_hflip2: delta -0.005554; patient 95% CI [-0.009134112026579225, -0.0017633934070613625].
- res640_hflip2_minus_res768_hflip2: delta -0.004312; patient 95% CI [-0.007988135783158351, -0.0008727396469892777].
- res768_identity_minus_res768_hflip2: delta -0.000850; patient 95% CI [-0.0025450569665189994, 0.001002222463582671].
- res896_identity_minus_res768_hflip2: delta -0.000624; patient 95% CI [-0.004511673678138911, 0.0038899485207697774].
- res896_hflip2_minus_res768_hflip2: delta +0.000334; patient 95% CI [-0.0029158639801002474, 0.004208199978455412].
- res768_res896_equal_identity_minus_res768_hflip2: delta -0.000316; patient 95% CI [-0.002772337286592319, 0.00252871280617953].
- res768_res896_equal_hflip2_minus_res768_hflip2: delta +0.000630; patient 95% CI [-0.0010731541998793281, 0.002410789138291951].
- res640_hflip2_minus_identity: delta +0.001242; patient 95% CI [-0.00033432779311998726, 0.0028179331420155325].
- res768_hflip2_minus_identity: delta +0.000850; patient 95% CI [-0.0010022224635826721, 0.0025450569665189976].
- res896_hflip2_minus_identity: delta +0.000959; patient 95% CI [-0.0007970276124149411, 0.0025269445545907806].
- res768_res896_equal_identity_minus_res640_identity: delta +0.005238; patient 95% CI [0.0001230554863914879, 0.010609086306693391].
- res768_res896_equal_identity_minus_res768_identity: delta +0.000534; patient 95% CI [-0.00104330236630919, 0.00227146139595328].
- res768_res896_equal_identity_minus_res896_identity: delta +0.000308; patient 95% CI [-0.0017017501827982701, 0.0020901813818467083].
- res768_res896_equal_hflip2_minus_res640_hflip2: delta +0.004942; patient 95% CI [0.00012050678980471687, 0.01010990856351432].
- res768_res896_equal_hflip2_minus_res896_hflip2: delta +0.000296; patient 95% CI [-0.0018471340114100704, 0.002063922792344377].
