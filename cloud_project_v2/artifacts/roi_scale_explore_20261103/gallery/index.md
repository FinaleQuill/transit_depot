# Train-only ROI scale atlas

12 deliberately varied images per available view; no labels selected, no valid/test read.
Classical body-envelope heuristic, not a medical segmentation. Source-edge clipping is not repaired.
Compare full-frame baseline with a fixed 85% body long-edge occupancy. Unsafe estimates use exact baseline fallback.
Selected-case counts do not estimate train prevalence; no erasing, training, or model inference.

## AP

![AP](train_AP_contact.jpg)

- [train:003409: lowest_detected_body_occupancy](AP/01_train_003409/comparison.jpg) — fixed_body_envelope_occupancy
- [train:000600: highest_detected_body_occupancy](AP/02_train_000600/comparison.jpg) — fixed_body_envelope_occupancy
- [train:014217: widest_frame](AP/03_train_014217/comparison.jpg) — fixed_body_envelope_occupancy
- [train:002628: tallest_frame](AP/04_train_002628/comparison.jpg) — fixed_body_envelope_occupancy
- [train:026589: smallest_native_image](AP/05_train_026589/comparison.jpg) — fixed_body_envelope_occupancy
- [train:000817: largest_native_image](AP/06_train_000817/comparison.jpg) — fixed_body_envelope_occupancy
- [train:007520: largest_removed_background_area](AP/07_train_007520/comparison.jpg) — fixed_body_envelope_occupancy
- [train:018121: source_edge_contact](AP/08_train_018121/comparison.jpg) — fixed_body_envelope_occupancy
- [train:022541: deterministic_hash_sample](AP/09_train_022541/comparison.jpg) — fixed_body_envelope_occupancy
- [train:011938: deterministic_hash_sample](AP/10_train_011938/comparison.jpg) — fixed_body_envelope_occupancy
- [train:005128: deterministic_hash_sample](AP/11_train_005128/comparison.jpg) — fixed_body_envelope_occupancy
- [train:020889: deterministic_hash_sample](AP/12_train_020889/comparison.jpg) — fixed_body_envelope_occupancy

## PA

![PA](train_PA_contact.jpg)

- [train:002523: lowest_detected_body_occupancy](PA/01_train_002523/comparison.jpg) — fixed_body_envelope_occupancy
- [train:000618: highest_detected_body_occupancy](PA/02_train_000618/comparison.jpg) — fixed_body_envelope_occupancy
- [train:007796: widest_frame](PA/03_train_007796/comparison.jpg) — fixed_body_envelope_occupancy
- [train:020264: tallest_frame](PA/04_train_020264/comparison.jpg) — fixed_body_envelope_occupancy
- [train:008539: smallest_native_image](PA/05_train_008539/comparison.jpg) — fixed_body_envelope_occupancy
- [train:001422: largest_native_image](PA/06_train_001422/comparison.jpg) — fixed_body_envelope_occupancy
- [train:000241: largest_removed_background_area](PA/07_train_000241/comparison.jpg) — fixed_body_envelope_occupancy
- [train:024856: source_edge_contact](PA/08_train_024856/comparison.jpg) — fixed_body_envelope_occupancy
- [train:017306: deterministic_hash_sample](PA/09_train_017306/comparison.jpg) — fixed_body_envelope_occupancy
- [train:009263: deterministic_hash_sample](PA/10_train_009263/comparison.jpg) — fixed_body_envelope_occupancy
- [train:008537: deterministic_hash_sample](PA/11_train_008537/comparison.jpg) — fixed_body_envelope_occupancy
- [train:012706: deterministic_hash_sample](PA/12_train_012706/comparison.jpg) — fixed_body_envelope_occupancy

## Lateral

![Lateral](train_Lateral_contact.jpg)

- [train:004668: lowest_detected_body_occupancy](Lateral/01_train_004668/comparison.jpg) — fixed_body_envelope_occupancy
- [train:000121: highest_detected_body_occupancy](Lateral/02_train_000121/comparison.jpg) — fixed_body_envelope_occupancy
- [train:007246: widest_frame](Lateral/03_train_007246/comparison.jpg) — fixed_body_envelope_occupancy
- [train:029302: tallest_frame](Lateral/04_train_029302/comparison.jpg) — fixed_body_envelope_occupancy
- [train:027784: smallest_native_image](Lateral/05_train_027784/comparison.jpg) — fixed_body_envelope_occupancy
- [train:001288: largest_native_image](Lateral/06_train_001288/comparison.jpg) — fixed_body_envelope_occupancy
- [train:014341: largest_removed_background_area](Lateral/07_train_014341/comparison.jpg) — fixed_body_envelope_occupancy
- [train:018634: source_edge_contact](Lateral/08_train_018634/comparison.jpg) — fixed_body_envelope_occupancy
- [train:004519: detector_fallback](Lateral/09_train_004519/comparison.jpg) — detector_broad_support_outside_candidate
- [train:023219: deterministic_hash_sample](Lateral/10_train_023219/comparison.jpg) — fixed_body_envelope_occupancy
- [train:027650: deterministic_hash_sample](Lateral/11_train_027650/comparison.jpg) — fixed_body_envelope_occupancy
- [train:004814: deterministic_hash_sample](Lateral/12_train_004814/comparison.jpg) — fixed_body_envelope_occupancy

## Missing

![Missing](train_Missing_contact.jpg)

- [train:026977: lowest_detected_body_occupancy](Missing/01_train_026977/comparison.jpg) — fixed_body_envelope_occupancy
- [train:000207: highest_detected_body_occupancy](Missing/02_train_000207/comparison.jpg) — fixed_body_envelope_occupancy
- [train:000304: widest_frame](Missing/03_train_000304/comparison.jpg) — fixed_body_envelope_occupancy
- [train:000006: tallest_frame](Missing/04_train_000006/comparison.jpg) — fixed_body_envelope_occupancy
- [train:000025: smallest_native_image](Missing/05_train_000025/comparison.jpg) — fixed_body_envelope_occupancy
- [train:005525: largest_native_image](Missing/06_train_005525/comparison.jpg) — fixed_body_envelope_occupancy
- [train:018101: largest_removed_background_area](Missing/07_train_018101/comparison.jpg) — fixed_body_envelope_occupancy
- [train:015053: source_edge_contact](Missing/08_train_015053/comparison.jpg) — fixed_body_envelope_occupancy
- [train:002813: detector_fallback](Missing/09_train_002813/comparison.jpg) — detector_broad_support_outside_candidate
- [train:015045: deterministic_hash_sample](Missing/10_train_015045/comparison.jpg) — fixed_body_envelope_occupancy
- [train:004971: deterministic_hash_sample](Missing/11_train_004971/comparison.jpg) — fixed_body_envelope_occupancy
- [train:019500: deterministic_hash_sample](Missing/12_train_019500/comparison.jpg) — fixed_body_envelope_occupancy

