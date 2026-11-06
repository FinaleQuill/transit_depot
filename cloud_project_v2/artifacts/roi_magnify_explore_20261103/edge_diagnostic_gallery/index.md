# Train-only ROI magnify atlas

12 deliberately varied images per available view; no labels selected, no valid/test read.
Classical body-envelope heuristic, not a medical segmentation. Edge contact is diagnostic; the crop is clipped to source bounds and preserves the touched side.
Crop body envelope with about 2% margin per side; long side fills 768 and short side is padded isotropically.
Only proposals with effective magnification 1.01-1.25 that do not worsen square padding are applied. Other cases exactly fall back to full-frame letterbox.
Selected-case counts do not estimate train prevalence; no erasing, training, or model inference.

## AP

![AP](train_AP_contact.jpg)

- [train:023155: lowest_effective_magnification](AP/01_train_023155/comparison.jpg) — body_envelope_magnify_only
- [train:020367: highest_effective_magnification](AP/02_train_020367/comparison.jpg) — body_envelope_magnify_only
- [train:014217: widest_frame](AP/03_train_014217/comparison.jpg) — crop_aspect_worsens_padding
- [train:002628: tallest_frame](AP/04_train_002628/comparison.jpg) — crop_aspect_worsens_padding
- [train:026589: smallest_native_image](AP/05_train_026589/comparison.jpg) — minimum_magnification_limit
- [train:002814: largest_native_image](AP/06_train_002814/comparison.jpg) — minimum_magnification_limit
- [train:030327: largest_removed_source_area](AP/07_train_030327/comparison.jpg) — body_envelope_magnify_only
- [train:015552: source_edge_contact_diagnostic](AP/08_train_015552/comparison.jpg) — body_envelope_magnify_only
- [train:019225: no_cut_fallback](AP/09_train_019225/comparison.jpg) — minimum_magnification_limit
- [train:027401: deterministic_hash_sample](AP/10_train_027401/comparison.jpg) — minimum_magnification_limit
- [train:023875: deterministic_hash_sample](AP/11_train_023875/comparison.jpg) — minimum_magnification_limit
- [train:030268: deterministic_hash_sample](AP/12_train_030268/comparison.jpg) — minimum_magnification_limit

## PA

![PA](train_PA_contact.jpg)

- [train:030323: lowest_effective_magnification](PA/01_train_030323/comparison.jpg) — body_envelope_magnify_only
- [train:011339: highest_effective_magnification](PA/02_train_011339/comparison.jpg) — body_envelope_magnify_only
- [train:007796: widest_frame](PA/03_train_007796/comparison.jpg) — minimum_magnification_limit
- [train:020264: tallest_frame](PA/04_train_020264/comparison.jpg) — crop_aspect_worsens_padding
- [train:008539: smallest_native_image](PA/05_train_008539/comparison.jpg) — crop_aspect_worsens_padding
- [train:001782: largest_native_image](PA/06_train_001782/comparison.jpg) — minimum_magnification_limit
- [train:027077: largest_removed_source_area](PA/07_train_027077/comparison.jpg) — body_envelope_magnify_only
- [train:002507: source_edge_contact_diagnostic](PA/08_train_002507/comparison.jpg) — body_envelope_magnify_only
- [train:018249: no_cut_fallback](PA/09_train_018249/comparison.jpg) — minimum_magnification_limit
- [train:004186: deterministic_hash_sample](PA/10_train_004186/comparison.jpg) — minimum_magnification_limit
- [train:008352: deterministic_hash_sample](PA/11_train_008352/comparison.jpg) — crop_aspect_worsens_padding
- [train:015493: deterministic_hash_sample](PA/12_train_015493/comparison.jpg) — minimum_magnification_limit

## Lateral

![Lateral](train_Lateral_contact.jpg)

- [train:023895: lowest_effective_magnification](Lateral/01_train_023895/comparison.jpg) — body_envelope_magnify_only
- [train:030081: highest_effective_magnification](Lateral/02_train_030081/comparison.jpg) — body_envelope_magnify_only
- [train:007246: widest_frame](Lateral/03_train_007246/comparison.jpg) — minimum_magnification_limit
- [train:029302: tallest_frame](Lateral/04_train_029302/comparison.jpg) — crop_aspect_too_extreme
- [train:027784: smallest_native_image](Lateral/05_train_027784/comparison.jpg) — minimum_magnification_limit
- [train:000693: largest_native_image](Lateral/06_train_000693/comparison.jpg) — minimum_magnification_limit
- [train:001949: largest_removed_source_area](Lateral/07_train_001949/comparison.jpg) — body_envelope_magnify_only
- [train:014559: source_edge_contact_diagnostic](Lateral/08_train_014559/comparison.jpg) — body_envelope_magnify_only
- [train:027974: no_cut_fallback](Lateral/09_train_027974/comparison.jpg) — minimum_magnification_limit
- [train:013220: deterministic_hash_sample](Lateral/10_train_013220/comparison.jpg) — minimum_magnification_limit
- [train:025314: deterministic_hash_sample](Lateral/11_train_025314/comparison.jpg) — crop_aspect_worsens_padding
- [train:022013: deterministic_hash_sample](Lateral/12_train_022013/comparison.jpg) — minimum_magnification_limit

## Missing

![Missing](train_Missing_contact.jpg)

- [train:030125: lowest_effective_magnification](Missing/01_train_030125/comparison.jpg) — body_envelope_magnify_only
- [train:027339: highest_effective_magnification](Missing/02_train_027339/comparison.jpg) — body_envelope_magnify_only
- [train:000207: widest_frame](Missing/03_train_000207/comparison.jpg) — minimum_magnification_limit
- [train:000006: tallest_frame](Missing/04_train_000006/comparison.jpg) — body_envelope_magnify_only
- [train:000025: smallest_native_image](Missing/05_train_000025/comparison.jpg) — crop_aspect_worsens_padding
- [train:017762: largest_native_image](Missing/06_train_017762/comparison.jpg) — minimum_magnification_limit
- [train:024052: largest_removed_source_area](Missing/07_train_024052/comparison.jpg) — body_envelope_magnify_only
- [train:026261: source_edge_contact_diagnostic](Missing/08_train_026261/comparison.jpg) — body_envelope_magnify_only
- [train:011770: no_cut_fallback](Missing/09_train_011770/comparison.jpg) — crop_aspect_worsens_padding
- [train:011772: deterministic_hash_sample](Missing/10_train_011772/comparison.jpg) — minimum_magnification_limit
- [train:024973: deterministic_hash_sample](Missing/11_train_024973/comparison.jpg) — minimum_magnification_limit
- [train:002581: deterministic_hash_sample](Missing/12_train_002581/comparison.jpg) — minimum_magnification_limit

