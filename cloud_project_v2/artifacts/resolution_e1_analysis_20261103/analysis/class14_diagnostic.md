# Class 1 / class 4 diagnostic

All analyses use official valid predictions. E1 means the valid-selected best checkpoint, confirmed epoch 1 for all three training resolutions; E3 is fixed-final context. Inference is common 768 with hflip2.

Class 1 = Enlarged Cardiomediastinum (array index 0); class 4 = Pneumonia (array index 3).

## Support and patient clustering

| Class | Positive Studies / patients | AP-only positive Studies / patients | Mixed-label multi-Study patients |
| --- | ---: | ---: | ---: |
| 1 Enlarged Cardiomediastinum | 24 / 16 | 15 / 11 | 15 / 109 |
| 4 Pneumonia | 68 / 37 | 37 / 20 | 32 / 109 |

Patient labels belong to Studies: a patient can contribute both positive and negative Studies. Mixed labels within a patient are not proof of erroneous labels.

## E1 absolute AUC and patient bootstrap intervals

2,000 draws, seed 20261103; resample all 228 patients with replacement, retain each patient's complete set of Studies, and use tie-aware weighted AUC. AP-only uses the same patient draws followed by the AP-only slice. Intervals are 2.5/97.5 percentile intervals; unusable single-label draws are excluded and counted. These are absolute per-model AUC intervals, not paired difference intervals.

| Class | res640 AUC [95% CI]; width | res768 AUC [95% CI]; width | res896 AUC [95% CI]; width |
| --- | --- | --- | --- |
| 1 Enlarged Cardiomediastinum | 0.660828 [0.485367, 0.783114]; 0.297747 | 0.675299 [0.502577, 0.795283]; 0.292707 | 0.687500 [0.528605, 0.810674]; 0.282069 |
| 2 Pneumothorax | 0.860229 [0.767839, 0.926905]; 0.159065 | 0.852239 [0.758320, 0.920598]; 0.162279 | 0.846025 [0.751752, 0.913191]; 0.161439 |
| 3 Consolidation | 0.795069 [0.749348, 0.847358]; 0.098010 | 0.811327 [0.764440, 0.858783]; 0.094342 | 0.812396 [0.762540, 0.859802]; 0.097262 |
| 4 Pneumonia | 0.738064 [0.657540, 0.800976]; 0.143436 | 0.747174 [0.673746, 0.805505]; 0.131759 | 0.752584 [0.682205, 0.808354]; 0.126148 |
| 5 Edema | 0.878448 [0.838610, 0.922313]; 0.083703 | 0.878214 [0.836694, 0.921967]; 0.085272 | 0.877676 [0.835738, 0.921571]; 0.085833 |
| 6 Cardiomegaly | 0.792278 [0.747998, 0.838585]; 0.090587 | 0.795956 [0.749191, 0.843432]; 0.094241 | 0.788989 [0.737831, 0.838914]; 0.101084 |
| 7 Atelectasis | 0.800573 [0.759864, 0.841163]; 0.081299 | 0.801156 [0.761876, 0.842158]; 0.080281 | 0.798643 [0.758342, 0.839861]; 0.081519 |
| 8 Lung Opacity | 0.709389 [0.661601, 0.757641]; 0.096040 | 0.715533 [0.668916, 0.762770]; 0.093854 | 0.719496 [0.671891, 0.766905]; 0.095014 |
| 9 Pleural Effusion | 0.912771 [0.863419, 0.949818]; 0.086398 | 0.912848 [0.864311, 0.949363]; 0.085052 | 0.911860 [0.864687, 0.947438]; 0.082751 |
| 10 No Finding | 0.884794 [0.849458, 0.917476]; 0.068018 | 0.885816 [0.850425, 0.918190]; 0.067765 | 0.883736 [0.847739, 0.915988]; 0.068249 |

| AP-only class | res640 AUC [95% CI]; width | res768 AUC [95% CI]; width | res896 AUC [95% CI]; width |
| --- | --- | --- | --- |
| 1 | 0.544887 [0.345804, 0.724295]; 0.378490 | 0.565652 [0.372557, 0.744913]; 0.372356 | 0.585012 [0.391023, 0.761156]; 0.370133 |
| 4 | 0.677077 [0.559982, 0.767977]; 0.207995 | 0.683150 [0.578449, 0.765972]; 0.187524 | 0.684351 [0.581668, 0.767624]; 0.185955 |

## E1 errors at fixed 0.5, for review only

A probability >=0.5 counts as positive. No threshold was fitted or changed. These error counts describe this operating point and do not replace AUC. Shared means the intersection of errors across all three E1 models.

| Class | Model / set | Error | Studies / patients | AP-only / errors | Label-matched pool AP-only | Largest patient share |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | res640 | FP | 16 / 9 | 8 / 16 | 427 / 835 | 4 / 16 |
| 1 | res640 | FN | 23 / 16 | 15 / 23 | 15 / 24 | 4 / 23 |
| 1 | res768 | FP | 18 / 8 | 9 / 18 | 427 / 835 | 6 / 18 |
| 1 | res768 | FN | 22 / 16 | 14 / 22 | 15 / 24 | 3 / 22 |
| 1 | res896 | FP | 15 / 8 | 8 / 15 | 427 / 835 | 5 / 15 |
| 1 | res896 | FN | 22 / 16 | 14 / 22 | 15 / 24 | 3 / 22 |
| 1 | shared E1 all three | FP | 13 / 7 | 7 / 13 | 427 / 835 | 4 / 13 |
| 1 | shared E1 all three | FN | 22 / 16 | 14 / 22 | 15 / 24 | 3 / 22 |
| 4 | res640 | FP | 23 / 12 | 16 / 23 | 405 / 791 | 10 / 23 |
| 4 | res640 | FN | 53 / 32 | 29 / 53 | 37 / 68 | 6 / 53 |
| 4 | res768 | FP | 43 / 16 | 32 / 43 | 405 / 791 | 17 / 43 |
| 4 | res768 | FN | 48 / 30 | 26 / 48 | 37 / 68 | 4 / 48 |
| 4 | res896 | FP | 47 / 15 | 36 / 47 | 405 / 791 | 19 / 47 |
| 4 | res896 | FN | 47 / 30 | 26 / 47 | 37 / 68 | 4 / 47 |
| 4 | shared E1 all three | FP | 18 / 7 | 14 / 18 | 405 / 791 | 10 / 18 |
| 4 | shared E1 all three | FN | 47 / 30 | 26 / 47 | 37 / 68 | 4 / 47 |

The JSON provides every error Study, all referenced image IDs and valid relative paths, view positions, all three E1/E3 Study and image probabilities, exclusive error signatures, and full patient counts. Image labels are inherited from the Study: a high per-image probability does not establish an image-level annotation error.

## Co-labels of shared E1 errors with explicit controls

The pool has the same target-class label as the error set (negative for FP, positive for FN). The complement control excludes the shared error set; it can include errors from a subset of models. Per-model JSON controls are TN for FP and TP for FN. Counts are descriptive, without independence, multiplicity, or confounder adjustment.

Class 1 shared FP:

| Co-label | Shared errors positive / total | Label-matched pool positive / total | Complement control positive / total |
| --- | ---: | ---: | ---: |
| 2 Pneumothorax | 1 / 13 | 36 / 835 | 35 / 822 |
| 3 Consolidation | 3 / 13 | 43 / 835 | 40 / 822 |
| 4 Pneumonia | 2 / 13 | 63 / 835 | 61 / 822 |
| 5 Edema | 0 / 13 | 112 / 835 | 112 / 822 |
| 6 Cardiomegaly | 2 / 13 | 173 / 835 | 171 / 822 |
| 7 Atelectasis | 6 / 13 | 178 / 835 | 172 / 822 |
| 8 Lung Opacity | 5 / 13 | 204 / 835 | 199 / 822 |
| 9 Pleural Effusion | 4 / 13 | 252 / 835 | 248 / 822 |
| 10 No Finding | 2 / 13 | 306 / 835 | 304 / 822 |

Top error patients: subject_11846: 4 errors / 6 total valid Studies; subject_11855: 3 errors / 16 total valid Studies; subject_11868: 2 errors / 18 total valid Studies; subject_11815: 1 errors / 8 total valid Studies; subject_11827: 1 errors / 89 total valid Studies.
Single patient contributes more than half of errors: False.

Class 1 shared FN:

| Co-label | Shared errors positive / total | Label-matched pool positive / total | Complement control positive / total |
| --- | ---: | ---: | ---: |
| 2 Pneumothorax | 1 / 22 | 1 / 24 | 0 / 2 |
| 3 Consolidation | 3 / 22 | 3 / 24 | 0 / 2 |
| 4 Pneumonia | 5 / 22 | 5 / 24 | 0 / 2 |
| 5 Edema | 3 / 22 | 3 / 24 | 0 / 2 |
| 6 Cardiomegaly | 4 / 22 | 4 / 24 | 0 / 2 |
| 7 Atelectasis | 14 / 22 | 15 / 24 | 1 / 2 |
| 8 Lung Opacity | 10 / 22 | 10 / 24 | 0 / 2 |
| 9 Pleural Effusion | 8 / 22 | 8 / 24 | 0 / 2 |
| 10 No Finding | 0 / 22 | 0 / 24 | 0 / 2 |

Top error patients: subject_11868: 3 errors / 18 total valid Studies; subject_11827: 2 errors / 89 total valid Studies; subject_11829: 2 errors / 36 total valid Studies; subject_11836: 2 errors / 39 total valid Studies; subject_11981: 2 errors / 3 total valid Studies.
Single patient contributes more than half of errors: False.

Class 4 shared FP:

| Co-label | Shared errors positive / total | Label-matched pool positive / total | Complement control positive / total |
| --- | ---: | ---: | ---: |
| 1 Enlarged Cardiomediastinum | 0 / 18 | 19 / 791 | 19 / 773 |
| 2 Pneumothorax | 0 / 18 | 37 / 791 | 37 / 773 |
| 3 Consolidation | 3 / 18 | 36 / 791 | 33 / 773 |
| 5 Edema | 4 / 18 | 104 / 791 | 100 / 773 |
| 6 Cardiomegaly | 7 / 18 | 165 / 791 | 158 / 773 |
| 7 Atelectasis | 3 / 18 | 182 / 791 | 179 / 773 |
| 8 Lung Opacity | 12 / 18 | 178 / 791 | 166 / 773 |
| 9 Pleural Effusion | 7 / 18 | 237 / 791 | 230 / 773 |
| 10 No Finding | 1 / 18 | 306 / 791 | 305 / 773 |

Top error patients: subject_11827: 10 errors / 89 total valid Studies; subject_11829: 2 errors / 36 total valid Studies; subject_11889: 2 errors / 2 total valid Studies; subject_11832: 1 errors / 13 total valid Studies; subject_11837: 1 errors / 26 total valid Studies.
Single patient contributes more than half of errors: True.

Class 4 shared FN:

| Co-label | Shared errors positive / total | Label-matched pool positive / total | Complement control positive / total |
| --- | ---: | ---: | ---: |
| 1 Enlarged Cardiomediastinum | 3 / 47 | 5 / 68 | 2 / 21 |
| 2 Pneumothorax | 0 / 47 | 0 / 68 | 0 / 21 |
| 3 Consolidation | 3 / 47 | 10 / 68 | 7 / 21 |
| 5 Edema | 6 / 47 | 11 / 68 | 5 / 21 |
| 6 Cardiomegaly | 6 / 47 | 12 / 68 | 6 / 21 |
| 7 Atelectasis | 9 / 47 | 11 / 68 | 2 / 21 |
| 8 Lung Opacity | 29 / 47 | 36 / 68 | 7 / 21 |
| 9 Pleural Effusion | 15 / 47 | 23 / 68 | 8 / 21 |
| 10 No Finding | 0 / 47 | 0 / 68 | 0 / 21 |

Top error patients: subject_11827: 4 errors / 89 total valid Studies; subject_11837: 4 errors / 26 total valid Studies; subject_11817: 3 errors / 26 total valid Studies; subject_11904: 3 errors / 10 total valid Studies; subject_11825: 2 errors / 13 total valid Studies.
Single patient contributes more than half of errors: False.

## Shared E1 case references

Three highest-confidence shared FP and three lowest-probability shared FN Studies are listed per class, ordered by the mean of the three E1 probabilities. These are metadata references to all images of each Study; this analysis does not open the images. The full JSON contains every case.

| Class / error | Study | E1 probabilities 640 / 768 / 896 | AP-only | Valid image references |
| --- | --- | --- | --- | --- |
| 1 / FP | subject_11987/study_20501 | 0.874768 / 0.943766 / 0.946399 | True | valid/subject_11987/study_20501/pic_033887.jpg |
| 1 / FP | subject_11855/study_20254 | 0.893383 / 0.930870 / 0.895865 | True | valid/subject_11855/study_20254/pic_033480.jpg |
| 1 / FP | subject_11827/study_20439 | 0.675586 / 0.746371 / 0.727120 | True | valid/subject_11827/study_20439/pic_033781.jpg |
| 1 / FN | subject_11875/study_20088 | 0.006202 / 0.001454 / 0.000851 | True | valid/subject_11875/study_20088/pic_033199.jpg |
| 1 / FN | subject_11954/study_20828 | 0.015810 / 0.002918 / 0.001289 | True | valid/subject_11954/study_20828/pic_034424.jpg |
| 1 / FN | subject_11827/study_20781 | 0.016243 / 0.005621 / 0.004673 | True | valid/subject_11827/study_20781/pic_034342.jpg |
| 4 / FP | subject_11829/study_20307 | 0.827788 / 0.911120 / 0.931666 | False | valid/subject_11829/study_20307/pic_033568.jpg; valid/subject_11829/study_20307/pic_033569.jpg |
| 4 / FP | subject_11827/study_20615 | 0.816511 / 0.915728 / 0.921556 | True | valid/subject_11827/study_20615/pic_034073.jpg |
| 4 / FP | subject_11827/study_20349 | 0.727805 / 0.868570 / 0.912440 | True | valid/subject_11827/study_20349/pic_033633.jpg |
| 4 / FN | subject_11816/study_20414 | 0.005497 / 0.002999 / 0.003484 | True | valid/subject_11816/study_20414/pic_033741.jpg |
| 4 / FN | subject_11846/study_20044 | 0.010911 / 0.007101 / 0.009079 | True | valid/subject_11846/study_20044/pic_033126.jpg |
| 4 / FN | subject_11848/study_20799 | 0.032497 / 0.022282 / 0.014909 | True | valid/subject_11848/study_20799/pic_034371.jpg |
## E3 context and existing galleries

| Model | Class | E3 AUC | E3 FP / FN | E1 correct → E3 wrong | E1 wrong → E3 correct |
| --- | --- | ---: | ---: | ---: | ---: |
| res640 | 1 | 0.662625 | 11 / 24 | 1 | 5 |
| res640 | 4 | 0.736540 | 22 / 53 | 1 | 2 |
| res768 | 1 | 0.675150 | 14 / 22 | 0 | 4 |
| res768 | 4 | 0.746375 | 39 / 48 | 0 | 4 |
| res896 | 1 | 0.684731 | 14 / 22 | 0 | 1 |
| res896 | 4 | 0.748383 | 47 / 47 | 3 | 3 |

Task context identifies the existing two error galleries as res896 E3 high-confidence misclassifications, not errors shared across models. Their provenance is stated here without reading or changing the galleries.

The two existing galleries are not evidence of shared E1 errors. This analysis neither reads nor overwrites their image files.

## Interpretation limits and reproducibility

Sparse positive patients, particularly class 1 (16 all-view; 11 AP-only), limit precision. Absolute AUC interval overlap is not a paired significance test. These intervals condition on chosen checkpoints and this valid set; they do not capture training-seed variation or remove repeated-valid checkpoint selection bias.

Neither these probabilities nor viewing a radiograph can establish that disease is early, that a label is wrong, or why an error occurred. Clinical adjudication and longitudinal information would be needed. Mixed labels across Studies of one patient can reflect changing clinical state or annotation uncertainty; this analysis cannot distinguish these explanations.

No test, image-byte, checkpoint-byte, GPU, training, platform-submission, threshold-fitting, or declaration access was performed.

Manifest SHA-256: `c136b2d9f2646fdfa65e0db91d24b1ab5a021e7b038dcaf243855b8779bb5112`.
Preregistration SHA-256: `52609a24e2e6730d244508ff5e1e31d6a31d002fa777cc91e995248a8811ba51`.
Script SHA-256: `fd07db69a468844517e479de87bb827c581c2f97e898ce9c86dcaa60b7da2db2`.

The JSON records SHA-256 for all 18 evaluation inputs, their checkpoint identity and recorded compliance evidence. Reproduce with:

`python /home/featurize/work/cloud_project_v2/artifacts/resolution_e1_analysis_20261103/analysis/class14_diagnostic.py --output-directory /tmp/class14_diagnostic_replay`
