# Exploration comparison

All metrics are official-valid exploration results; no test access or platform submission.

| Candidate | Overall AUC | AP-only AUC | Three-image P50/P95 ms |
| --- | ---: | ---: | --- |
| res640_identity | 0.801676 | 0.732567 | 570.20/9520.94 |
| res640_hflip2 | 0.802279 | 0.733931 | 576.55/632.07 |
| res768_identity | 0.806490 | 0.738203 | 556.73/663.92 |
| res768_hflip2 | 0.807222 | 0.738913 | 614.10/643.93 |
| res896_identity | 0.806485 | 0.736665 | 600.60/646.62 |
| res896_hflip2 | 0.807227 | 0.737040 | 591.67/677.08 |
| res768_res896_equal_identity | 0.807010 | 0.738063 | 591.19/611.21 |
| res768_res896_equal_hflip2 | 0.807634 | 0.738431 | 651.18/674.05 |

Intervals resample patients with all their Studies. They are conditional on selected checkpoints and do not remove repeated-validation selection bias.

Resolution evidence concerns training size at common 768 inference; disease-specific inference-size recommendations remain untested.

- res640_identity_minus_res768_hflip2: delta -0.005546; patient 95% CI [-0.009374471090200762, -0.0012871670107992598].
- res640_hflip2_minus_res768_hflip2: delta -0.004943; patient 95% CI [-0.008746080638816187, -0.0009827343285651862].
- res768_identity_minus_res768_hflip2: delta -0.000733; patient 95% CI [-0.0022172565577284363, 0.0009903515554948476].
- res896_identity_minus_res768_hflip2: delta -0.000737; patient 95% CI [-0.004581717287352102, 0.003914845868907007].
- res896_hflip2_minus_res768_hflip2: delta +0.000005; patient 95% CI [-0.003346861025129241, 0.00373483877847825].
- res768_res896_equal_identity_minus_res768_hflip2: delta -0.000212; patient 95% CI [-0.0024092424230339774, 0.0027136441130453763].
- res768_res896_equal_hflip2_minus_res768_hflip2: delta +0.000412; patient 95% CI [-0.0011098799977412115, 0.0021222216222445906].
- res640_hflip2_minus_identity: delta +0.000603; patient 95% CI [-0.0011783376785144324, 0.0023293903151187867].
- res768_hflip2_minus_identity: delta +0.000733; patient 95% CI [-0.0009903515554948478, 0.0022172565577284354].
- res896_hflip2_minus_identity: delta +0.000742; patient 95% CI [-0.0010643029444296764, 0.0022324664368015964].
- res768_res896_equal_identity_minus_res640_identity: delta +0.005334; patient 95% CI [0.00010793695065961233, 0.01062988393477989].
- res768_res896_equal_identity_minus_res768_identity: delta +0.000520; patient 95% CI [-0.0010212182008200621, 0.0023472810510779592].
- res768_res896_equal_identity_minus_res896_identity: delta +0.000525; patient 95% CI [-0.0015886467098214369, 0.0025640651216201175].
- res768_res896_equal_hflip2_minus_res640_hflip2: delta +0.005355; patient 95% CI [0.00023475612356458622, 0.010477872255349888].
- res768_res896_equal_hflip2_minus_res896_hflip2: delta +0.000407; patient 95% CI [-0.001795684869605857, 0.0024237865817456146].
