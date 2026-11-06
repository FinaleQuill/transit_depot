# Train-only ROI magnify atlas v2.1

12 deliberately varied images per available view; no labels selected, no valid/test read.
Classical body-envelope heuristic, not a medical segmentation. Edge contact is diagnostic; the crop is clipped to source bounds and preserves the touched side.
Crop body envelope with about 2% margin per side; long side fills 768 and short side is padded isotropically.
Only proposals with effective magnification 1.01-1.25 that do not worsen square padding are applied. Other cases exactly fall back to full-frame letterbox.
Selected-case counts do not estimate train prevalence; no erasing, training, or model inference.

## Review

This paired replay contains 9 applied crops and 39 exact full-frame fallbacks (9/48, 18.75%). Applied magnification ranges from 1.0136x to 1.2152x. The deterministic train candidate pool contains 40 applied proposals among 320 inspected images (12.5%); neither rate estimates full-train prevalence. The 48 selected image IDs and per-view order exactly match the superseded v2 atlas. `metrics.json` binds this replay to the prior metrics SHA256.

Four applied crops have source-edge contact: AP `train:015552` (left, bottom), PA `train:027077` (left, top), Lateral `train:014559` (left, right, bottom), and Missing `train:000006` (left). The crop box remains inside source bounds and retains each touched side. The four contact sheets and all four native-source comparisons were reviewed; the AP and Lateral bottom-edge detail strips are linked below. No new apical, costophrenic, or visible line/device truncation was found. In AP/09 and Missing/04 the original already obscures part of the base, so completeness there cannot be assessed. The PA upper-right annotation is already close to the source edge. The gallery is a visual review, not proof that the source anatomy is complete.

The selected-set geometry supports treating edge contact as diagnostic rather than an automatic veto: the four paired examples preserve their touched source boundaries and pass the other registered checks. The crop remains bounded to the source and must contain the detector hull, use about 2% margin, magnify 1.01-1.25x, preserve or improve square-padding efficiency, and retain a crop short/long ratio of at least 0.55. The classical detector continues its broad-seed, protected-secondary-support, and removed-area checks. One limitation remains: its broad-lost check is performed at its own 5% box; the downstream crop adds 2% around the hull. Hull containment is checked, but every small support structure in the 2%-to-5% band is not proven retained. Visible devices and lines are not targeted or guaranteed to be preserved. Treat v2.1 as an isolated experiment candidate, not a validated default preprocessing policy.

Historical B combined ROI crop with random erasing, so it cannot identify the crop's independent effect. Overall Macro-AUC changed by +0.000052; AP-only changed by -0.001267 with paired 95% CI [-0.004323, +0.001287]. This is inconclusive evidence of ROI benefit or harm; an isolated paired ROI training comparison is needed before adoption.

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
  Native source and accepted crop: [train:015552](AP/09_train_015552/native_source_crop.jpg)
  Bottom-edge detail: [train:015552](AP/09_train_015552/bottom_edge_detail.jpg)
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
  Native source and accepted crop: [train:027077](PA/11_train_027077/native_source_crop.jpg)
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
  Native source and accepted crop: [train:014559](Lateral/08_train_014559/native_source_crop.jpg)
  Bottom-edge detail: [train:014559](Lateral/08_train_014559/bottom_edge_detail.jpg)
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
  Native source and accepted crop: [train:000006](Missing/04_train_000006/native_source_crop.jpg)
- [train:000025: smallest_native_image](Missing/05_train_000025/comparison.jpg) — crop_aspect_worsens_padding
- [train:017762: largest_native_image](Missing/06_train_017762/comparison.jpg) — minimum_magnification_limit
- [train:024052: largest_removed_source_area](Missing/07_train_024052/comparison.jpg) — body_envelope_magnify_only
- [train:011770: source_edge_contact_veto](Missing/08_train_011770/comparison.jpg) — crop_aspect_worsens_padding
- [train:011772: no_cut_fallback](Missing/09_train_011772/comparison.jpg) — minimum_magnification_limit
- [train:024973: deterministic_hash_sample](Missing/10_train_024973/comparison.jpg) — minimum_magnification_limit
- [train:002581: deterministic_hash_sample](Missing/11_train_002581/comparison.jpg) — minimum_magnification_limit
- [train:024435: deterministic_hash_sample](Missing/12_train_024435/comparison.jpg) — minimum_magnification_limit

