# E0 versus E1 post-stop paired comparison

This is a CPU comparison of the complete selected best epochs. E0 completed normally; E1 epoch 7 is complete and was its best, but the E1 training invocation was forcibly interrupted during epoch 13 and is therefore not a terminal successful queue run.

| Slice | E0 Macro-AUC | E1 Macro-AUC | E1-E0 | Paired patient 95% CI |
| --- | ---: | ---: | ---: | --- |
| all | 0.7978396535270589 | 0.7992390384597867 | 0.001399384932727843 | [-0.0010006361110716316, 0.003703471865285304] |
| ap_projection_only | 0.7292861071887053 | 0.7315480976574801 | 0.0022619904687747328 | [-0.0012740305710989357, 0.005696670224806549] |
| single_image_study | 0.7356524378406736 | 0.737130737772906 | 0.0014782999322324741 | [-0.00216862085287462, 0.005120176134539269] |
| multiple_image_study | 0.8247313731330308 | 0.8246637434725125 | -6.76296605183202e-05 | [-0.004226159170641008, 0.003960008488186722] |

Overall E0 best epoch 8: 0.7978396535270589
Overall E1 best epoch 7: 0.7992390384597867
E1-E0 paired delta: 0.001399384932727843
Paired patient 95% CI: [-0.0010006361110716316, 0.003703471865285304]

No H, official-validation, test, image, weight, CSV, submission or deployment action was performed.
This report is exploratory internal-development evidence and does not certify the interrupted E1 queue as completed.
