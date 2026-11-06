# Train-only text-candidate review

These boxes are unconfirmed contrast/shape candidates, not verified watermarks. No training pixels were removed.

48 images / 48 distinct patients; 22 images with candidates.
The sample is stratified and does not estimate prevalence. No H/dev/test images, OCR, model weights or GPU were used.

- [contact_01.jpg](contact_01.jpg)
- [contact_02.jpg](contact_02.jpg)
- [contact_03.jpg](contact_03.jpg)
- [contact_04.jpg](contact_04.jpg)

Possible future experiment: after review, add low-probability synthetic border text overlays or narrowly limited border occlusion on TRAIN only, while preserving the original image branch. Use a fixed train-selected mask policy; never infer labels from text. Broad white-pixel removal and automatic masking of these boxes are not justified.
