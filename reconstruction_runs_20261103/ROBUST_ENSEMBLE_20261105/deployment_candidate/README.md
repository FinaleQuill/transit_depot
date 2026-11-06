# Fixed Sprint + opacity ensemble

Use the official competition image with requirements_runtime.txt.

`python predict.py --image-root ROOT --manifest TEST_MANIFEST --output NEW/submission.csv`

`python cost.py submission.csv PICTURES_100 PICTURES_5 --manifest TEST_MANIFEST`

Class 1 retains Sprint exactly. Other classes use 80% Sprint and 20% opacity. Each member uses fixed FP16 identity+hflip inference; all original Study images are averaged. Cost uses the identical predictor, 100 complete Studies, five disjoint warmup Studies and three serial rounds. Official T4 timing remains pending.
