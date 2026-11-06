# Tight crop two-image preview

Proposal only, not anatomy segmentation. Letterbox results have no random erasing.

Manifest SHA256: `f9a06316cfa8ab10ad4488e60a2dcf934cbc942727edd9ad2c31689823db6aef`  
Detector SHA256: `85ec065e2b4a9bb23a0defcafc3681cd474a579563affd3501ddcb4d2426cd8d`  
Runner SHA256: `907f9f16c7ab36449f8c34c3cf2e4b0ee8e25eefb443f8f94d903e1218e4073d`

## train:006758

Source: `/home/featurize/data/cxr_first/train/subject_01151/study_04482/pic_007369.jpg`; SHA256 before/after: `f6edda4534b6fc0266cde6529c864469939f005e4bc1c80684560a62816e6508` / `f6edda4534b6fc0266cde6529c864469939f005e4bc1c80684560a62816e6508`

Box: `(0, 0, 3056, 1443)`; body: `(0, 0, 3056, 1374)`; size: `[3056, 1443]`

Removed px: `{'left': 0, 'top': 0, 'right': 0, 'bottom': 1101}`; area: `0.432783`

768 scale full/crop: `0.251309` / `0.251309`; linear factor: `1.0000`

[Four-panel review](train_006758/review_four_panel.png) | [Raw crop](train_006758/crop.png) | [768 letterbox](train_006758/letterbox768.png) | [Mask](train_006758/mask.png) | [Seed](train_006758/seed_mask.png)

[Raw bottom strip, 1:1](train_006758/bottom_raw_1to1.png)  

## train:005642

Source: `/home/featurize/data/cxr_first/train/subject_03252/study_03738/pic_006139.jpg`; SHA256 before/after: `9134f0aa54629b39765c680ae49737021d7a4031bdf08ddd4fa79231812e5b8c` / `9134f0aa54629b39765c680ae49737021d7a4031bdf08ddd4fa79231812e5b8c`

Box: `(0, 0, 2539, 2990)`; body: `(0, 0, 2539, 2848)`; size: `[2539, 2990]`

Removed px: `{'left': 0, 'top': 0, 'right': 0, 'bottom': 60}`; area: `0.019672`

768 scale full/crop: `0.251803` / `0.256856`; linear factor: `1.0201`

[Four-panel review](train_005642/review_four_panel.png) | [Raw crop](train_005642/crop.png) | [768 letterbox](train_005642/letterbox768.png) | [Mask](train_005642/mask.png) | [Seed](train_005642/seed_mask.png)

[Raw bottom strip, 1:1](train_005642/bottom_raw_1to1.png)  

## Extra training augmentation examples

These fixed-seed p=0.5 examples do not replace the no-erasing 768 previews. A non-triggered draw stays unchanged; a triggered draw uses one uniformly located black patch of at most 2% content area.

- train:006758 seed 0: actual trigger NO; [comparison](train_006758/training_erasing_comparison.png) / [example](train_006758/training_erasing_example.png)
- train:005642 seed 2: actual trigger YES; [comparison](train_005642/training_erasing_comparison.png) / [example](train_005642/training_erasing_example.png)
