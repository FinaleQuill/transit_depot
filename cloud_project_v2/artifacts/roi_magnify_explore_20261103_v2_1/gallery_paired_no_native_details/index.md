# Train-only ROI magnify atlas v2.1 — DRAFT SUPERSEDED

This paired sample was generated before adding native-resolution edge details. It is retained for provenance; use the final sibling `gallery/index.md` for review.

12 deliberately varied images per available view; no labels selected, no valid/test read.
Classical body-envelope heuristic, not a medical segmentation. Edge contact is diagnostic; the crop is clipped to source bounds and preserves the touched side.
Crop body envelope with about 2% margin per side; long side fills 768 and short side is padded isotropically.
Only proposals with effective magnification 1.01-1.25 that do not worsen square padding are applied. Other cases exactly fall back to full-frame letterbox.
Selected-case counts do not estimate train prevalence; no erasing, training, or model inference.

## Review

The replayed selected set contains 9 applied crops and 39 exact full-frame fallbacks (9/48 applied, 18.75%). Applied magnification is 1.0136x-1.2152x. The deterministic 320-image train candidate pool contains 40 applied proposals (12.5%); neither rate estimates full-train prevalence. The selected image IDs and per-view order match the prior v2 atlas exactly. Its metrics SHA256 is recorded in `metrics.json`.

Four applied crops have source-edge contact: AP `train:015552` (left, bottom), PA `train:027077` (left, top), Lateral `train:014559` (left, right, bottom), and Missing `train:000006` (left). In each case, the accepted crop box is clipped to the source rectangle, so the touched source side is preserved. Visual review of all four contact sheets and all four edge-contact comparisons found no new apical, costophrenic, or visible line/device truncation. In AP and Missing, original opacity/overexposure already obscures part of the lower base, so that contour is not assessable; the PA upper-right annotation is already near the source edge. This review cannot establish anatomical completeness or repair source truncation.

The geometric tradeoff is to treat detected edge contact as diagnostic, not an automatic veto: a blanket veto rejected four proposals that preserve the touched source boundary and pass every other registered guard. The crop still requires a reliable detector candidate, about 2% margin, at least 1.01x and at most 1.25x magnification, no worse square padding than the source, and a crop short/long ratio of at least 0.55. Geometry failures return the exact full-frame letterbox. No ROI efficacy claim follows from this atlas.

Historical B combined ROI cropping with random erasing and therefore cannot isolate the crop effect. Its overall Macro-AUC change was +0.000052; AP-only change was -0.001267 with paired 95% CI [-0.004323, +0.001287]. That result is not reliable evidence of ROI gain or harm and motivates an isolated paired ROI experiment before retaining the method.

## AP

![AP](train_AP_contact.jpg)

- [train:020367: lowest_effective_magnification](AP/01_train_020367/comparison.jpg) — body_envelope_magnify_only
- [train:014217: widest_frame](AP/02_train_014217/comparison.jpg) — crop_aspect_worsens_padding
- [train:002628: tallest_frame](AP/03_train_002628/comparison.jpg) — crop_aspect_worsens_padding
- [train:026589: smallest_native_image](AP/04_train_026589/comparison.jpg) — minimum_magnification_limit
- [train:002814: largest_native_image](AP/05_train_002814/comparison.jpg) — minimum_magnification_limit
- [train:019225: source_edge_contact_veto](AP/06_train_019225/comparison.jpg) — minimum_magnification_limit
- [train:027401: no_cut_fallback](AP/07_train_027401/comparison.jpg) — minimum_magnification_limit
- [train:023875: deterministic_hash_sample](AP/08_train_023875/comparison.jpg) — minimum_magnification_limit
- [train:015552: deterministic_hash_sample](AP/09_train_015552/comparison.jpg) — body_envelope_magnify_only
- [train:030268: deterministic_hash_sample](AP/10_train_030268/comparison.jpg) — minimum_magnification_limit
- [train:013335: deterministic_hash_sample](AP/11_train_013335/comparison.jpg) — minimum_magnification_limit
- [train:017438: deterministic_hash_sample](AP/12_train_017438/comparison.jpg) — minimum_magnification_limit

## PA

![PA](train_PA_contact.jpg)

- [train:007796: widest_frame](PA/01_train_007796/comparison.jpg) — minimum_magnification_limit
- [train:020264: tallest_frame](PA/02_train_020264/comparison.jpg) — crop_aspect_worsens_padding
- [train:008539: smallest_native_image](PA/03_train_008539/comparison.jpg) — crop_aspect_worsens_padding
- [train:001782: largest_native_image](PA/04_train_001782/comparison.jpg) — minimum_magnification_limit
- [train:018249: source_edge_contact_veto](PA/05_train_018249/comparison.jpg) — minimum_magnification_limit
- [train:004186: no_cut_fallback](PA/06_train_004186/comparison.jpg) — minimum_magnification_limit
- [train:008352: deterministic_hash_sample](PA/07_train_008352/comparison.jpg) — crop_aspect_worsens_padding
- [train:015493: deterministic_hash_sample](PA/08_train_015493/comparison.jpg) — minimum_magnification_limit
- [train:004574: deterministic_hash_sample](PA/09_train_004574/comparison.jpg) — minimum_magnification_limit
- [train:007430: deterministic_hash_sample](PA/10_train_007430/comparison.jpg) — minimum_magnification_limit
- [train:027077: deterministic_hash_sample](PA/11_train_027077/comparison.jpg) — body_envelope_magnify_only
- [train:011695: deterministic_hash_sample](PA/12_train_011695/comparison.jpg) — minimum_magnification_limit

## Lateral

![Lateral](train_Lateral_contact.jpg)

- [train:011769: lowest_effective_magnification](Lateral/01_train_011769/comparison.jpg) — body_envelope_magnify_only
- [train:007246: widest_frame](Lateral/02_train_007246/comparison.jpg) — minimum_magnification_limit
- [train:029302: tallest_frame](Lateral/03_train_029302/comparison.jpg) — crop_aspect_too_extreme
- [train:027784: smallest_native_image](Lateral/04_train_027784/comparison.jpg) — minimum_magnification_limit
- [train:000693: largest_native_image](Lateral/05_train_000693/comparison.jpg) — minimum_magnification_limit
- [train:027974: source_edge_contact_veto](Lateral/06_train_027974/comparison.jpg) — minimum_magnification_limit
- [train:013220: no_cut_fallback](Lateral/07_train_013220/comparison.jpg) — minimum_magnification_limit
- [train:014559: deterministic_hash_sample](Lateral/08_train_014559/comparison.jpg) — body_envelope_magnify_only
- [train:025314: deterministic_hash_sample](Lateral/09_train_025314/comparison.jpg) — crop_aspect_worsens_padding
- [train:022013: deterministic_hash_sample](Lateral/10_train_022013/comparison.jpg) — minimum_magnification_limit
- [train:014921: deterministic_hash_sample](Lateral/11_train_014921/comparison.jpg) — crop_aspect_worsens_padding
- [train:013215: deterministic_hash_sample](Lateral/12_train_013215/comparison.jpg) — crop_aspect_worsens_padding

## Missing

![Missing](train_Missing_contact.jpg)

- [train:017355: lowest_effective_magnification](Missing/01_train_017355/comparison.jpg) — body_envelope_magnify_only
- [train:020354: highest_effective_magnification](Missing/02_train_020354/comparison.jpg) — body_envelope_magnify_only
- [train:000207: widest_frame](Missing/03_train_000207/comparison.jpg) — minimum_magnification_limit
- [train:000006: tallest_frame](Missing/04_train_000006/comparison.jpg) — body_envelope_magnify_only
- [train:000025: smallest_native_image](Missing/05_train_000025/comparison.jpg) — crop_aspect_worsens_padding
- [train:017762: largest_native_image](Missing/06_train_017762/comparison.jpg) — minimum_magnification_limit
- [train:024052: largest_removed_source_area](Missing/07_train_024052/comparison.jpg) — body_envelope_magnify_only
- [train:011770: source_edge_contact_veto](Missing/08_train_011770/comparison.jpg) — crop_aspect_worsens_padding
- [train:011772: no_cut_fallback](Missing/09_train_011772/comparison.jpg) — minimum_magnification_limit
- [train:024973: deterministic_hash_sample](Missing/10_train_024973/comparison.jpg) — minimum_magnification_limit
- [train:002581: deterministic_hash_sample](Missing/11_train_002581/comparison.jpg) — minimum_magnification_limit
- [train:024435: deterministic_hash_sample](Missing/12_train_024435/comparison.jpg) — minimum_magnification_limit

