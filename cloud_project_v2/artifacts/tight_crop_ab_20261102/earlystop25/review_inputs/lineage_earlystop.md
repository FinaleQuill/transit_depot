# A/B lineage and early-stop evidence (interim)

B_tiny was still running and B_base had not started at the 2026-11-02 04:34 UTC snapshot. This is an A-branch evidence extract, not the final four-model comparison. Exact paths and hashes are in `lineage_earlystop.json`.

| Run | Terminal epoch | Selected epoch | Best AP-only Study AUC | Best full-valid AUC |
| --- | ---: | ---: | ---: | ---: |
| A_tiny | 7 | 2 | 0.739348586 | 0.808482322 |
| A_base | 6 | 1 | 0.737626633 | 0.809010809 |
| B_tiny | running, E4 snapshot | 2 then | 0.739769114 then | pending |
| B_base | pending | pending | pending | pending |

A_base selected E1 because its AP-only AUC fell to 0.736543067 at E2 and remained below the E1 best plus 0.0001 through E6. Five consecutive non-improving epochs triggered patience5. Its full-valid AUC also declined from 0.809010809 to 0.806458595. Training hard BCE fell from 0.407840061 to 0.379493260 while OOF KL rose from 0.117694864 to 0.133058551. These curves show training progress accompanied by validation ranking loss, but do not identify the cause.

A_base is a **student** initialized from the prior 040 Base E10 FP32 best EMA (SHA `9cb733eb...805a3`), not one of the teachers. The frozen soft targets come from five Base768 patient-out-of-fold teachers, fixed final EMA epochs 20/12/12/12/12, on 30,328 train images with identity/hflip raw logits. The documented pooled OOF train Study AUC is 0.789509541; its fold-mixed train scope cannot be directly ranked against the current official-valid student AUC. Official valid and test were not used to form the teacher cache.

The five-teacher ensemble's historical **official-valid** hflip2 AUC was 0.802467016. A_base E1's overall valid AUC is 0.809010809, but the ensemble and student have different training and selection histories. This score gap does not isolate the effect or quality of distillation.

AP-only selection uses 442 Studies, with 15 class-1 positives. The same official valid has supported repeated model decisions, so these comparisons remain exploratory even with patient-cluster intervals.
