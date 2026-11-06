# Morning exploration report

Queue state: **FAILED**; current arm: **compare_resolution / stopped_without_advancing**.
Estimated finish UTC: not available.
Failure: RuntimeError: Command exit 1: compare_resolution; see /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/queue/logs/compare_resolution.log.
Test access: no. Platform submission: no. Declaration changes: no.

Class names follow the official task label sheet (SHA-256 034a098696d60c6a3d6ee041d75b6d22ec85c5c1dc272fc0ad44846c12364cb5).

Metrics below are emitted only when their evidence files exist; missing results are marked incomplete.

## Job evidence

| Job | Receipt | Preregistration SHA-256 | Run ledger SHA-256 | Receipt SHA-256 |
| --- | --- | --- | --- | --- |
| res640 | /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/queue/receipts/res640.json | a7b7f9f7a99b99f9f061867049c72a244d674b426669fd646efa0066d4646420 | c97dbd213ea49c22dd9e2cadb6fe6aed7c7ff0cecb12d388365f8925968120fe | f19144f9dd64aa223b7cbfac7c1f94529cc694a7351625c7b0c50e4c7391fe19 |
| eval_res640_retry1 | /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/queue/receipts/eval_res640_retry1.json | 43af0619f754d96a8f5f89a22b1a595c6f1a426625c687b368bb1f60a0fafbe7 | c97dbd213ea49c22dd9e2cadb6fe6aed7c7ff0cecb12d388365f8925968120fe | 9da6ba2c60166b4070ab55a6be57eaaa0882b4fe5c67e35cf578b57b64a526f9 |
| res768 | /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/queue/receipts/res768.json | d755c48b4277b5a0ee88bcfb8a88cffc5a2cc3db293319df7299eb09da435124 | 24c31318be1b993c83fd4de93a68fcfbfa165956eba340987055612db206a58d | 8428b56bccd4abcbdd905cde7d3e538d716889a658f35fb3559b406d4860cc04 |
| eval_res768 | /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/queue/receipts/eval_res768.json | 04dcbe91a1f40aa73539ebe559d74659eb3ff7d0d1bbbcf326f779a4d02fa623 | 24c31318be1b993c83fd4de93a68fcfbfa165956eba340987055612db206a58d | 098f42b6b2194e22384d91d9fd3af4157e3910bd0d936ead432c004a43495de9 |
| res896 | /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/queue/receipts/res896.json | dbb9cca88a5362f32eb97abb2c9d71ca0ea5566516f7ad3896b9fbdc5ed5c2f2 | fc068e4c8fb321f96b669f7d52867eae15adbc87120eb8f57a861a52ca6d3176 | fd1342110e192f30ded35ac83a5f9a2e50e80dcfef85f0cd0d34fe387b12001d |
| eval_res896 | /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/queue/receipts/eval_res896.json | 3646fcb75e03bb1773144393cae4fc3da06ddb5cd52bf5805a39fe4057b3ed79 | fc068e4c8fb321f96b669f7d52867eae15adbc87120eb8f57a861a52ca6d3176 | d433eeb8198519c30d097bbcf7592598eb3900c678802048810c68c52aa8818c |
## ROI atlas

- AP: [train_AP_contact.jpg](/home/featurize/work/cloud_project_v2/artifacts/roi_magnify_explore_20261103_v2_2/gallery/train_AP_contact.jpg)
- PA: [train_PA_contact.jpg](/home/featurize/work/cloud_project_v2/artifacts/roi_magnify_explore_20261103_v2_2/gallery/train_PA_contact.jpg)
- Lateral: [train_Lateral_contact.jpg](/home/featurize/work/cloud_project_v2/artifacts/roi_magnify_explore_20261103_v2_2/gallery/train_Lateral_contact.jpg)
- Missing: [train_Missing_contact.jpg](/home/featurize/work/cloud_project_v2/artifacts/roi_magnify_explore_20261103_v2_2/gallery/train_Missing_contact.jpg)

Gallery review: /home/featurize/work/cloud_project_v2/artifacts/roi_magnify_explore_20261103_v2_2/gallery/index.md

## Candidate metrics

| Candidate | Selection | Macro-AUC | AP-only Macro-AUC | Registered sampled-train minus full-valid gap | Three-image P50/P95 ms |
| --- | --- | ---: | ---: | --- | ---: |
| res640 | fixed_final_E3 | 0.802279 | 0.733931 | 1000 train Studies: 0.108795 | 576.55/632.07 |
| res768 | fixed_final_E3 | 0.807222 | 0.738913 | 1000 train Studies: 0.109375 | 614.10/643.93 |
| res896 | fixed_final_E3 | 0.807227 | 0.737040 | 1000 train Studies: 0.102166 | 591.67/677.08 |
| champion041 | not complete | not complete | not complete | not complete | not measured |
| roi | not complete | not complete | not complete | not complete | not measured |
| mil | not complete | not complete | not complete | not complete | not measured |
| res640_best | not complete | not complete | not complete | not complete | not measured |
| res768_best | not complete | not complete | not complete | not complete | not measured |
| res896_best | not complete | not complete | not complete | not complete | not measured |
| roi_best | not complete | not complete | not complete | not complete | not measured |
| mil_best | not complete | not complete | not complete | not complete | not measured |

Per-class valid AUC (res640): Enlarged Cardiomediastinum=0.662625, Pneumothorax=0.860262, Consolidation=0.790550, Pneumonia=0.736540, Edema=0.876543, Cardiomegaly=0.792377, Atelectasis=0.799452, Lung Opacity=0.707882, Pleural Effusion=0.912277, No Finding=0.884285
Per-class AP-only AUC (res640): Enlarged Cardiomediastinum=0.552537, Pneumothorax=0.834521, Consolidation=0.749009, Pneumonia=0.677010, Edema=0.803648, Cardiomegaly=0.702787, Atelectasis=0.744173, Lung Opacity=0.642371, Pleural Effusion=0.846199, No Finding=0.787058
Compliance: checkpoint SHA 071634889fe44132053ec20a1e8f11ad1ebb158013006cd6f29bb0b924eabdd8; lifecycle ledger SHA c97dbd213ea49c22dd9e2cadb6fe6aed7c7ff0cecb12d388365f8925968120fe.

Per-class valid AUC (res768): Enlarged Cardiomediastinum=0.675150, Pneumothorax=0.853456, Consolidation=0.810204, Pneumonia=0.746375, Edema=0.876777, Cardiomegaly=0.795450, Atelectasis=0.800534, Lung Opacity=0.716091, Pleural Effusion=0.912283, No Finding=0.885905
Per-class AP-only AUC (res768): Enlarged Cardiomediastinum=0.566745, Pneumothorax=0.828074, Consolidation=0.768445, Pneumonia=0.683217, Edema=0.803648, Cardiomegaly=0.706162, Atelectasis=0.742993, Lung Opacity=0.656315, Pleural Effusion=0.847961, No Finding=0.785574
Compliance: checkpoint SHA 07980cb5669876dddd97f5e3338c64504399f243dedc6b662d313845df02d97e; lifecycle ledger SHA 24c31318be1b993c83fd4de93a68fcfbfa165956eba340987055612db206a58d.

Per-class valid AUC (res896): Enlarged Cardiomediastinum=0.684731, Pneumothorax=0.847504, Consolidation=0.813707, Pneumonia=0.748383, Edema=0.876274, Cardiomegaly=0.787017, Atelectasis=0.798192, Lung Opacity=0.721148, Pleural Effusion=0.911365, No Finding=0.883949
Per-class AP-only AUC (res896): Enlarged Cardiomediastinum=0.583607, Pneumothorax=0.818686, Consolidation=0.763415, Pneumonia=0.679947, Edema=0.801029, Cardiomegaly=0.695363, Atelectasis=0.739125, Lung Opacity=0.664225, Pleural Effusion=0.848548, No Finding=0.776452
Compliance: checkpoint SHA dbde1f6a2e2cb1d9672735a388d6b893db1d7c8454be7ed1991517c0ae58ecbc; lifecycle ledger SHA fc068e4c8fb321f96b669f7d52867eae15adbc87120eb8f57a861a52ca6d3176.

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
