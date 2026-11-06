# Sprint inference candidate

Use the official competition image and install requirements_runtime.txt if needed. Its existing PyTorch, torchvision, NumPy, Pillow and scikit-learn are retained.

`python predict.py --image-root ROOT --manifest TEST_MANIFEST --output NEW/submission.csv`

`python cost.py submission.csv PICTURES_100 PICTURES_5 --manifest TEST_MANIFEST`

Timing inputs contain 100 measured and five disjoint warmup complete Studies. Both commands use the identical full-image FP16 hflip2 and Study aggregation. The package is selected from local validation; official T4 timing and acceptance remain pending.
