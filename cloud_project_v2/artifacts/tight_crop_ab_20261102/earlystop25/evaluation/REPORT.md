# Tight-crop A/B official-valid evaluation

Images: 1422; Studies: 859; patients: 228.
AP-only early-stop selected best.pt EMA (up to E25), full decode, FP16, 768, hflip2. Class 1 uses Base; other classes use equal Tiny/Base probabilities.
AP-only requires every image in the Study to be AP; non-AP includes Studies with any other view.

| Study slice | Studies | Eligible classes | A macro-AUC | B macro-AUC | B - A |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 859 | 10 | 0.817187 | 0.817239 | 0.000052 |
| ap_only | 442 | 10 | 0.750163 | 0.748897 | -0.001267 |
| ap_only_single_image | 381 | 10 | 0.757759 | 0.757066 | -0.000692 |
| ap_only_multiple_image | 61 | 10 | 0.662779 | 0.659941 | -0.002838 |
| non_ap_only | 417 | 10 | 0.869318 | 0.870830 | 0.001511 |
| single_image | 391 | 10 | 0.757608 | 0.757043 | -0.000564 |
| multiple_image | 468 | 10 | 0.852234 | 0.853810 | 0.001577 |

## Per-class AUC: all

| Class | Positive / negative Studies | A | B | B - A |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 24 / 835 | 0.742315 | 0.747505 | 0.005190 |
| 2 | 37 / 822 | 0.857007 | 0.858190 | 0.001184 |
| 3 | 46 / 813 | 0.815873 | 0.814108 | -0.001765 |
| 4 | 68 / 791 | 0.754890 | 0.753086 | -0.001803 |
| 5 | 115 / 744 | 0.882574 | 0.882024 | -0.000549 |
| 6 | 177 / 682 | 0.794929 | 0.793860 | -0.001069 |
| 7 | 193 / 666 | 0.801039 | 0.801646 | 0.000607 |
| 8 | 214 / 645 | 0.726784 | 0.726357 | -0.000427 |
| 9 | 260 / 599 | 0.911699 | 0.911455 | -0.000244 |
| 10 | 306 / 553 | 0.884758 | 0.884155 | -0.000603 |

## Per-class AUC: ap_only

| Class | Positive / negative Studies | A | B | B - A |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15 / 427 | 0.644496 | 0.641062 | -0.003435 |
| 2 | 21 / 421 | 0.839498 | 0.842439 | 0.002941 |
| 3 | 32 / 410 | 0.772180 | 0.769970 | -0.002210 |
| 4 | 37 / 405 | 0.690757 | 0.683684 | -0.007074 |
| 5 | 93 / 349 | 0.811597 | 0.810888 | -0.000709 |
| 6 | 127 / 315 | 0.704787 | 0.703737 | -0.001050 |
| 7 | 126 / 316 | 0.750603 | 0.752486 | 0.001884 |
| 8 | 142 / 300 | 0.664671 | 0.663075 | -0.001596 |
| 9 | 187 / 255 | 0.844857 | 0.844354 | -0.000503 |
| 10 | 90 / 352 | 0.778188 | 0.777273 | -0.000915 |

## Per-class AUC: ap_only_single_image

| Class | Positive / negative Studies | A | B | B - A |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14 / 367 | 0.662320 | 0.661152 | -0.001168 |
| 2 | 18 / 363 | 0.870217 | 0.873890 | 0.003673 |
| 3 | 30 / 351 | 0.782811 | 0.781102 | -0.001709 |
| 4 | 33 / 348 | 0.695141 | 0.689655 | -0.005486 |
| 5 | 75 / 306 | 0.812985 | 0.812549 | -0.000436 |
| 6 | 110 / 271 | 0.699061 | 0.699094 | 0.000034 |
| 7 | 106 / 275 | 0.749160 | 0.750086 | 0.000926 |
| 8 | 133 / 248 | 0.669870 | 0.667566 | -0.002304 |
| 9 | 158 / 223 | 0.846427 | 0.845973 | -0.000454 |
| 10 | 74 / 307 | 0.789594 | 0.789594 | 0.000000 |

## Per-class AUC: ap_only_multiple_image

| Class | Positive / negative Studies | A | B | B - A |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 / 60 | 0.333333 | 0.333333 | 0.000000 |
| 2 | 3 / 58 | 0.643678 | 0.643678 | 0.000000 |
| 3 | 2 / 59 | 0.720339 | 0.720339 | 0.000000 |
| 4 | 4 / 57 | 0.644737 | 0.609649 | -0.035088 |
| 5 | 18 / 43 | 0.757106 | 0.757106 | 0.000000 |
| 6 | 17 / 44 | 0.760695 | 0.758021 | -0.002674 |
| 7 | 20 / 41 | 0.756098 | 0.767073 | 0.010976 |
| 8 | 9 / 52 | 0.489316 | 0.493590 | 0.004274 |
| 9 | 29 / 32 | 0.830819 | 0.831897 | 0.001078 |
| 10 | 16 / 45 | 0.691667 | 0.684722 | -0.006944 |

## Patient-cluster bootstrap

1,000 paired patient resamples; seed 20261102. Each interval uses classes eligible in the original slice.

| Slice | Eligible classes | A 95% CI | B 95% CI | B - A 95% CI | Valid paired resamples |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 10 | [0.785640, 0.845561] | [0.785399, 0.845489] | [-0.001933, 0.002530] | 1000 |
| ap_only | 10 | [0.711725, 0.789661] | [0.710846, 0.788259] | [-0.004323, 0.001287] | 1000 |
| ap_only_single_image | 10 | [0.717557, 0.794636] | [0.717889, 0.792951] | [-0.004380, 0.002265] | 1000 |
| ap_only_multiple_image | 10 | [0.572870, 0.734169] | [0.567598, 0.734310] | [-0.007896, 0.005506] | 575 |

Historical 044 is reported separately and is not a paired treatment arm.
Validation comparisons are exploratory; no test selection or T4 certification.

Selected epochs: A_tiny=E2, A_base=E1, B_tiny=E2, B_base=E1
AP-only official valid selected each member's epoch and then evaluated the pair. The patient intervals are conditional on this selection and do not correct selection bias or prove online generalization.
