# Existing-member audit before the new fixed fusion

Recorded 2026-11-05T13:01:20.665053+08:00 (Hong Kong). CPU only; no new fusion evaluated.

All six saved prediction sets reproduce the previous report within 1e-12 using independent positive-negative pair counting. Input hashes remained unchanged.

| Existing member | Study Macro | Patient-equal Macro | Study class 8 | Patient-equal class 8 | Single-image Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| preserved_sprint | 0.817963 | 0.871672 | 0.724980 | 0.765057 | 0.758661 |
| previous_three_member | 0.817044 | 0.871285 | 0.726509 | 0.770073 | 0.755281 |
| sprint_base768 | 0.809591 | 0.864041 | 0.721292 | 0.756946 | 0.747521 |
| context512 | 0.802066 | 0.834312 | 0.710874 | 0.761621 | 0.729012 |
| opacity512 | 0.803235 | 0.852236 | 0.718054 | 0.772457 | 0.734460 |
| online_reference_041 | 0.809116 | 0.840424 | 0.717076 | 0.759640 | 0.745910 |

Opacity512 has lower ordinary Macro than Sprint, but higher patient-equal class 8 (.772457 versus .765057). Within single-image Studies, its patient-equal class 8 is .722222 versus .705341 and class 4 is .794118 versus .744289. This suggests complementary information but does not prove a blended gain.

Opacity class-8 probability correlation with Sprint is .923233 (Context .937081; 041 .959446). It wins 5.2735% of all class-8 positive-negative pairs that Sprint does not win, while losing 5.9661% that Sprint wins. These are individual-member diagnostics, not candidate-fusion performance.

The former three-member ensemble removed the original Tiny768; its losses in single-image classes 2/4/6 support preserving that member. Context512 has especially weak single-image class 4 (.653764).

The ten patients with the most Studies contribute 295/859 Studies; the largest has 89. Patient-equal metrics give each patient equal total weight within each slice, retaining all labels. They are sensitivity analyses on the same repeatedly used validation set, not an unseen-patient confirmation.

The recommendation fixed before new-result access is to preserve Sprint and add existing Opacity512 at 20% for classes 2–10, keeping class 1 exactly unchanged. No alternative weight or class-specific mixture was evaluated here.

Recommended conservative replacement gates, frozen before candidate evaluation: strictly higher ordinary Macro; strictly higher class-8 AUC; no lower single-image Macro; no lower patient-equal Macro (numerical tolerance 1e-12 only). Report every class and both image-count slices. Any optional class-2/4/6 veto must also be frozen before result access; arbitrary -.0005/-.002 tolerances have no independently established practical meaning.

See existing_members_audit.json for full classwise/slice metrics, prediction-source paths and SHA256 values. audit_existing_members.py reproduces the audit without model inference or test data.
