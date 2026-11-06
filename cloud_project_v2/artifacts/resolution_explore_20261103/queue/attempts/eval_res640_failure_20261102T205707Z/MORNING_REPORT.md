# Morning exploration report

Queue state: **FAILED**; current arm: **eval_res640 / stopped_without_advancing**.
Estimated finish UTC: not available.
Failure: RuntimeError: Command exit 1: eval_res640; see /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/queue/logs/eval_res640.log.
Test access: no. Platform submission: no. Declaration changes: no.

Class names follow the official task label sheet (SHA-256 034a098696d60c6a3d6ee041d75b6d22ec85c5c1dc272fc0ad44846c12364cb5).

Metrics below are emitted only when their evidence files exist; missing results are marked incomplete.

## Job evidence

| Job | Receipt | Preregistration SHA-256 | Run ledger SHA-256 | Receipt SHA-256 |
| --- | --- | --- | --- | --- |
| res640 | /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/queue/receipts/res640.json | a7b7f9f7a99b99f9f061867049c72a244d674b426669fd646efa0066d4646420 | c97dbd213ea49c22dd9e2cadb6fe6aed7c7ff0cecb12d388365f8925968120fe | e1928c351235ed5420b72926d71cf11ee571d7d996041195f53d778cdfd61936 |
## ROI atlas

- AP: [train_AP_contact.jpg](/home/featurize/work/cloud_project_v2/artifacts/roi_magnify_explore_20261103_v2_2/gallery/train_AP_contact.jpg)
- PA: [train_PA_contact.jpg](/home/featurize/work/cloud_project_v2/artifacts/roi_magnify_explore_20261103_v2_2/gallery/train_PA_contact.jpg)
- Lateral: [train_Lateral_contact.jpg](/home/featurize/work/cloud_project_v2/artifacts/roi_magnify_explore_20261103_v2_2/gallery/train_Lateral_contact.jpg)
- Missing: [train_Missing_contact.jpg](/home/featurize/work/cloud_project_v2/artifacts/roi_magnify_explore_20261103_v2_2/gallery/train_Missing_contact.jpg)

Gallery review: /home/featurize/work/cloud_project_v2/artifacts/roi_magnify_explore_20261103_v2_2/gallery/index.md

## Candidate metrics

| Candidate | Selection | Macro-AUC | AP-only Macro-AUC | Registered sampled-train minus full-valid gap | Three-image P50/P95 ms |
| --- | --- | ---: | ---: | --- | ---: |
| res640 | not complete | not complete | not complete | not complete | not measured |
| res768 | not complete | not complete | not complete | not complete | not measured |
| res896 | not complete | not complete | not complete | not complete | not measured |
| champion041 | not complete | not complete | not complete | not complete | not measured |
| roi | not complete | not complete | not complete | not complete | not measured |
| mil | not complete | not complete | not complete | not complete | not measured |
| res640_best | not complete | not complete | not complete | not complete | not measured |
| res768_best | not complete | not complete | not complete | not complete | not measured |
| res896_best | not complete | not complete | not complete | not complete | not measured |
| roi_best | not complete | not complete | not complete | not complete | not measured |
| mil_best | not complete | not complete | not complete | not complete | not measured |

## Resolution sensitivity

- not complete

## Fusion and spatial pooling

- not complete

## Best-checkpoint comparison (supplementary)

- Not complete.

## Best-checkpoint review suggestion

- Neither registered 50:50 pair (res768+res896, res768+roi) has sufficient best-checkpoint evidence for a fusion assessment.
- Suggested decision: retain the original champion041 baseline. No new candidate has established the required local gain with paired support and the AP-only point safeguard; do not claim a stronger model from this round.
- The >=0.004 and AP-only >=-0.003 gates use local point estimates. A positive paired valid CI is separately required for this report to suggest a new candidate; a CI crossing zero does not establish reliable improvement.
- This is local valid screening only; valid-selected results are supplementary and selection-conditional. Online performance is unknown, final members require human choice, and this report does not freeze members.

## Human review

- Anatomy: confirm no lung tissue was cut in the ROI atlas.
- Final members: owner review remains required; this report does not freeze members.
- Rules: only unresolved interpretation boundaries need review.
- Platform: no submission interface was used; platform outcome remains unavailable.

The resolution and fusion sensitivity estimates are conditional on this fixed valid set and checkpoint selection; they are not online-score claims.
