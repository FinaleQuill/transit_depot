# A/B terminal evidence inventory (2026-11-02)

## Missing result first

`earlystop25/scores/B_base/` had no `image_probs.npz`, `score.json`, or FP16 export at the 2026-11-02 07:47 UTC run snapshot. The `B_base` run then had two completed, validated epochs and an AP-selected `best.pt` (E1); its status was still `running`. A finished B arm, the four-member A/B fused comparison, patient-paired confidence intervals, and new Study-level latency were absent at that snapshot. A training-validation best AP score is not a deployment FP16 score. This inventory does not score `B_base`, read test, or use GPU.

## Existing actual FP16 valid outputs

All three scored members use actual FP16, batch-1, hflip2, full valid, with 1,422 image probabilities aggregated into 859 Studies. Source: `artifacts/tight_crop_ab_20261102/earlystop25/scores/{name}/score.json` and `image_probs.npz`.

| Member | Run epochs | Selected EMA | All Study AUC | AP-only AUC, 442 Studies | AP single, 381 | AP multiple, 61 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_tiny | 7, early stopped | E2 | .808457918 | .739334358 | .744979823 | .685344954 |
| A_base | 6, early stopped | E1 | .809009403 | .737582484 | .746062004 | .640978744 |
| B_tiny | 7, early stopped | E2 | .808749508 | .739679838 | .746112907 | .683225619 |
| B_base | 2 validated at read time | E1 training selection only | missing | missing | missing | missing |

AP-only means every image in the Study has AP view. The 61 multi-image AP Studies have as few as one positive Study for class 1; per-class estimates and any interval will be unstable. Full per-class AP AUC and positive/negative counts already reside in each `score.json` under `scores.ap_only`, `scores.ap_only_single_image`, and `scores.ap_only_multiple_image`. These single-member scores are not the fixed Tiny/Base fusion result.

The three actual export weight files are 55,712,135 bytes (each Tiny) and 175,260,329 bytes (A Base). A's two weights are 230,972,464 bytes; adding its two short export configs (1,020 and 1,021 bytes) gives 230,974,505 bytes. This is an export-file inventory, not a measured final deployment package. B's final two-member size is unknown until B Base export exists.

## Historical comparison, same FP16 protocol

The clean seed-43, Tiny512 direct BCE E6 versus strict OOF KD E7 contrast uses actual FP16/batch-1/hflip2/full-decode official valid predictions. BCE AUC is .803634677176; KD AUC is .805402257887. KD minus BCE is **+.001767580711**, with 1,000 patient-paired percentile bootstrap 95% CI **[-.001751749262, +.005884143402]** across 228 patients and 859 Studies. The CI crosses zero and does not correct valid checkpoint selection. BCE assets are 55,778,119 bytes; KD assets 66,995,853 bytes. No comparable actual end-to-end timing is established for either. Source: `artifacts/kd_v1/kd512_fp16_review_20261027/comparison.json` and `docs/KD512_FP16_REVIEW_20261027.md`.

## Historical candidate context

All AUC values below are actual FP16 official-valid Study AUC. All listed latency values are **historical per-image proxies**, from a 100-image/5-warmup/three-round local read-to-output protocol. They are **not measurements under the new Study timing rule** and must not be multiplied by an average image count to claim Study cost. Hardware and decoder also differ.

| Candidate | Valid AUC | Counted FP16 assets | Historical local ms/image | Input / hardware |
| --- | ---: | ---: | ---: | --- |
| 043 old B, Base768 | .810569569881 | 180,489,570 bytes | 74.549656 | full decode, 2080 Ti |
| 044 old A/B fusion | .816677073416 | 241,433,757 bytes | 84.820141 | JPEG draft768, 2080 Ti |
| 045 X1 + old B | .817149190113 | 241,448,464 bytes | 93.490616 | full decode, 2080 Ti |

044's user-reported platform result was roughly tied with or worse than 043; absolute platform scores were not provided. The 045 AUC is a valid result, not a platform score. Source: `docs/B_SUBMISSION_AND_OPTIMIZATION_20261031.md`, `docs/SELECTED_816677_SUBMISSION_20261031.md`, `docs/BEST_817149_SUBMISSION_20261101.md`, the 043 deployment inventory in `artifacts/kd_v1/student_B_submission_20261031/execution_complete.json`, and the 045 size audit in `artifacts/kd_v1/bounded_finetune_20261101/actual_combination_review/full_decode/actual_audit.json`. Actual official T4 and new Study-level timing are unmeasured for these candidates.

The counted size breaks down as follows: 043 = 175,261,029-byte Base weight + 5,228,541-byte member config; 044 = the same Base weight + 55,712,511-byte Tiny weight + 10,460,217 bytes across the two member configs and pair config; 045 = the same two weight sizes + 10,471,751 bytes across member/pair configs + a 3,173-byte `make_report.json` included by its stored inventory. The old member configs each carry a large `teacher_provenance` field. The newer A export uses much shorter configs and slightly different serialized weight sizes, so only their actual file inventory is comparable, not a claim that its eventual complete package is 10 MB smaller.

Existing full-valid Study CSVs also permit CPU-only AP re-scoring without model inference. Recomputed all-Study AUC exactly matches each original report. The AP-only slice has 442 Studies, with 381 single-image and 61 multiple-image AP Studies:

| Historical candidate | AP-only AUC | AP single AUC | AP multiple AUC | All single AUC | All multiple AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 043 | .738561693 | .747834393 | .637592971 | .747111684 | .847950867 |
| 044 | .747384463 | .756480566 | .647253086 | .756129038 | .852032580 |
| 045 | .749014506 | .756820108 | .657578612 | .756468288 | .852944642 |

AP-only per-class AUC vectors (class 1 through 10) and source CSV paths are recorded in `AB_EVIDENCE.json`. Inputs were the preserved **valid** Study CSVs at `artifacts/kd_v1/student_B_submission_20261031/full_valid_reproduced/valid_predictions.csv`, `artifacts/kd_v1/student_auc815_20261031/actual_pair_draft_target768/valid/valid_predictions.csv`, and `artifacts/kd_v1/bounded_finetune_20261101/actual_combination_review/full_decode/valid/valid_predictions.csv`. Re-scoring used `student_probability_fusion.py::class_aucs` on official-valid labels and view slices from `tight_crop_ab_evaluate.py::validation_records`; it preserves rank ties by midrank. These historical candidates have different training, decoder and selection paths, so their AP differences are descriptive and not the paired A/B treatment effect.

## CPU-only terminal evaluation entry

`tight_crop_ab_earlystop_evaluate.py::evaluate_selected_pair` is the existing final CPU route. It requires a selection JSON with exactly `selected.A_tiny`, `selected.A_base`, `selected.B_tiny`, and `selected.B_base`, each pointing to an immutable score directory. Its CLI is `python tight_crop_ab_earlystop_evaluate.py --action final --selection PATH --output NEW_DIRECTORY`. It verifies plan/valid identities, score and export hashes, ordered 1,422-image coverage, and then uses the registered class rule (class 1 Base only; classes 2-10 equal Tiny/Base) and FP64 Study mean. It writes `metrics.json` and `REPORT.md`, including all-Study, AP-only, AP-only single/multiple, class AUC/counts, and paired patient intervals. It cannot run successfully until a legitimate B Base FP16 score directory exists; an E1 training score or native/BF16 validation output must not fill that gap.

The reusable CPU functions are `tight_crop_ab_evaluate.py::score_arm`, `score_slice`, `paired_bootstrap`, and `paired_report`. `paired_bootstrap` draws patients with replacement (all their Studies travel together), uses the same draws for A and B, 1,000 samples and seed 20261102, and reports percentile 95% intervals for A, B, and B-A on all/AP/AP-single/AP-multiple slices. AP-only selected each best checkpoint, so these exploratory intervals do not remove selection bias. For a safety-stopped B Base, report its stop reason and the selected epoch explicitly rather than calling the four-member protocol completed. The current `evaluate_selected_pair` enforces the AP-selected `best.pt` source identity but does not itself establish Study timing.

The latest cost rule uses a Study as the timing unit. A future cost check must time complete Studies including every image, inference and Study aggregation through output, with the specified hardware/protocol and actual frozen FP16 candidate. No such Study-level A/B measurement is present in this inventory; the table above remains historical context only.
