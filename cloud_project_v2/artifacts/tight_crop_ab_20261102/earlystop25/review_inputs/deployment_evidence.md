# Historical deployment evidence for post-B_base review

All AUC values are official **valid** Study Macro-AUC, not platform scores. Latency
is measured end to end only where a GPU and protocol are listed. This inventory
does not rerun models or timing.

| Candidate | Valid AUC | Local end-to-end ms/image | FP16 assets | Distinguishing input |
| --- | ---: | ---: | ---: | --- |
| [026 direct BCE Tiny512](../../../../docs/BCE_FP16_REVIEW_20261027.md) | .803635 | unmeasured | 55.78MB | full decode; hflip2 |
| [027 KD Tiny512](../../../../docs/KD512_FP16_REVIEW_20261027.md) | .805402 | unmeasured | 67.00MB | full decode; hflip2 |
| [033 dual KD Tiny768](../../../../docs/DUAL_TINY768_DEPLOYMENT_REVIEW_20261028.md) | .808384 | 79.280 on **4090** | 128.77MB | full decode; four forwards |
| [041 KD Tiny768 A](../../../../docs/A_SUBMISSION_AND_SPEED_20261030.md) | .809116 | 73.936 on **2080Ti** | 67.00MB | full decode; two forwards |
| [043 KD Base768 B](../../../../docs/B_SUBMISSION_AND_OPTIMIZATION_20261031.md) | .810570 | 74.550 on **2080Ti** | not separately established here | full decode; two forwards |
| [044 Tiny/Base fusion](../../../../docs/SELECTED_816677_SUBMISSION_20261031.md) | .816677 | 84.820 on **2080Ti** | 241.43MB | JPEG **draft768**; four forwards |
| Historical AB full decode | .816830 | 92.866 on **2080Ti** | not separately established here | full decode; four forwards |
| [045 X1+old B](../../../../docs/BEST_817149_SUBMISSION_20261101.md) | .817149 | 93.491 on **2080Ti** | about 241.45MB | full decode; four forwards |

The 026 and 027 single Tiny512 candidates are a clean training contrast at the
same resolution and seed, but they lack a comparable actual end-to-end timing.
041 and 043 are the single-model 2080Ti references; their training recipes and
architectures differ. The 033 timing is 4090 development evidence. 044 changes
JPEG pixels via draft; its 84.82ms must not be combined with the new full-decode
crop cost. The old full-decode AB is the closer historical pixel-path reference,
but the new A/B training, best-epoch selection and B crop still differ.

The user reported 044's actual platform submission did not improve on 043, or
was slightly worse, with no absolute platform scores available. All candidates
still need genuine organizer T4 cost verification for the 100ms rule. The new
tight-crop A/B pair has no complete deployment AUC or end-to-end timing yet.
Its final comparison must include both AP-only and all-Study AUC, per-class
results, actual FP16 assets, and batch-1 serial read-to-output timing on the
chosen model and decoder. The full source paths and exact values are in
`deployment_evidence.json`.
