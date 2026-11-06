# Tight-crop A/B AUC progress

Updated Hong Kong: 2026-11-02T16:08:22.549163+08:00
Controller: failed / train_B_base

The overall column is all-Study valid Macro-AUC. AP-only controls native early stopping.

| Job | Run state | Latest epoch | Overall AUC | AP-only AUC | Best AP epoch | Best AP AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A_tiny | early_stopped | 7 | 0.801519897 | 0.733449053 | 2 | 0.739348586 |
| A_base | early_stopped | 6 | 0.806458595 | 0.734866739 | 1 | 0.737626633 |
| B_tiny | early_stopped | 7 | 0.803098379 | 0.734878264 | 2 | 0.739769114 |
| B_base | paused | 3 | 0.808169222 | 0.735103166 | 1 | 0.736490349 |

Per-class valid AUC for classes 1-10 is in `auc_progress.csv`. No test data or checkpoint weights are read by this watcher.
