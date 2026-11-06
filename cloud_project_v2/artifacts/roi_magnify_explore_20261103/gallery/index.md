# Train-only ROI magnify atlas — SUPERSEDED

This v2 edge-contact-veto atlas is retained as historical evidence. It is superseded for direct comparison by the v2.1 paired replay at `../../roi_magnify_explore_20261103_v2_1/gallery/index.md`, which reuses the exact same 48 train image IDs and records the original metrics SHA256.

12 deliberately varied images per available view; no labels selected, no valid/test read.
Classical body-envelope heuristic, not a medical segmentation. Source-edge clipping is rejected, not repaired.
Crop body envelope with about 2% margin per side; long side fills 768 and short side is padded isotropically.
Only proposals with effective magnification 1.01-1.25 that do not worsen square padding are applied. Other cases exactly fall back to full-frame letterbox.
Selected-case counts do not estimate train prevalence; no erasing, training, or model inference.

## Review

In this deliberately varied 48-image gallery, the crop was applied to 5 images and the remaining 43 used exact full-frame fallback after body-envelope contact with a source edge. Applied magnification ranged from 1.0248x to 1.2152x; no proposal shrank an image. Visual review covered all four view contact sheets and all five applied crops. No newly cropped lung apex, costophrenic angle, or visible device/tube was found cut off. This is a visual spot check, not a completeness guarantee or train-prevalence estimate.

## AP

![AP](train_AP_contact.jpg)

- [train:020367: lowest_effective_magnification](AP/01_train_020367/comparison.jpg) — body_envelope_magnify_only
- [train:014217: widest_frame](AP/02_train_014217/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:002628: tallest_frame](AP/03_train_002628/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:026589: smallest_native_image](AP/04_train_026589/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:002814: largest_native_image](AP/05_train_002814/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:019225: source_edge_contact_veto](AP/06_train_019225/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:027401: no_cut_fallback](AP/07_train_027401/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:023875: deterministic_hash_sample](AP/08_train_023875/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:015552: deterministic_hash_sample](AP/09_train_015552/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:030268: deterministic_hash_sample](AP/10_train_030268/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:013335: deterministic_hash_sample](AP/11_train_013335/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:017438: deterministic_hash_sample](AP/12_train_017438/comparison.jpg) — body_envelope_clipped_at_source_edge

## PA

![PA](train_PA_contact.jpg)

- [train:007796: widest_frame](PA/01_train_007796/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:020264: tallest_frame](PA/02_train_020264/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:008539: smallest_native_image](PA/03_train_008539/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:001782: largest_native_image](PA/04_train_001782/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:018249: source_edge_contact_veto](PA/05_train_018249/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:004186: no_cut_fallback](PA/06_train_004186/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:008352: deterministic_hash_sample](PA/07_train_008352/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:015493: deterministic_hash_sample](PA/08_train_015493/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:004574: deterministic_hash_sample](PA/09_train_004574/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:007430: deterministic_hash_sample](PA/10_train_007430/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:027077: deterministic_hash_sample](PA/11_train_027077/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:011695: deterministic_hash_sample](PA/12_train_011695/comparison.jpg) — body_envelope_clipped_at_source_edge

## Lateral

![Lateral](train_Lateral_contact.jpg)

- [train:011769: lowest_effective_magnification](Lateral/01_train_011769/comparison.jpg) — body_envelope_magnify_only
- [train:007246: widest_frame](Lateral/02_train_007246/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:029302: tallest_frame](Lateral/03_train_029302/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:027784: smallest_native_image](Lateral/04_train_027784/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:000693: largest_native_image](Lateral/05_train_000693/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:027974: source_edge_contact_veto](Lateral/06_train_027974/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:013220: no_cut_fallback](Lateral/07_train_013220/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:014559: deterministic_hash_sample](Lateral/08_train_014559/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:025314: deterministic_hash_sample](Lateral/09_train_025314/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:022013: deterministic_hash_sample](Lateral/10_train_022013/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:014921: deterministic_hash_sample](Lateral/11_train_014921/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:013215: deterministic_hash_sample](Lateral/12_train_013215/comparison.jpg) — body_envelope_clipped_at_source_edge

## Missing

![Missing](train_Missing_contact.jpg)

- [train:017355: lowest_effective_magnification](Missing/01_train_017355/comparison.jpg) — body_envelope_magnify_only
- [train:020354: highest_effective_magnification](Missing/02_train_020354/comparison.jpg) — body_envelope_magnify_only
- [train:000207: widest_frame](Missing/03_train_000207/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:000006: tallest_frame](Missing/04_train_000006/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:000025: smallest_native_image](Missing/05_train_000025/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:017762: largest_native_image](Missing/06_train_017762/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:024052: largest_removed_source_area](Missing/07_train_024052/comparison.jpg) — body_envelope_magnify_only
- [train:011770: source_edge_contact_veto](Missing/08_train_011770/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:011772: no_cut_fallback](Missing/09_train_011772/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:024973: deterministic_hash_sample](Missing/10_train_024973/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:002581: deterministic_hash_sample](Missing/11_train_002581/comparison.jpg) — body_envelope_clipped_at_source_edge
- [train:024435: deterministic_hash_sample](Missing/12_train_024435/comparison.jpg) — body_envelope_clipped_at_source_edge

