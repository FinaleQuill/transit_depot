# Exact-black trim: five fixed cases

[Five-case 768 overview](overview_5cases.png)

Experimental visual review only. Trim is computed on original grayscale pixels inside the saved first crop, before 768 letterbox. Any nonzero pixel is retained. No model, training, test, or production preprocessing.

Metrics SHA256: `2496467931184dc1faa146e542f13c6f2c465fc3ff2892eca76d642c06b4ab22`; detector SHA256: `85ec065e2b4a9bb23a0defcafc3681cd474a579563affd3501ddcb4d2426cd8d`; renderer SHA256: `b58f67148d8e4b7e61855fc66649dc59a2c1795f7c382eb0d076983634311bd0`.

## train:006758 (AP)

EXACT ZERO TRIM | `(0, 0, 3056, 1443)` -> `(8, 0, 3056, 1368)` | extra pixels `{'left': 8, 'top': 0, 'right': 0, 'bottom': 75}` | scale 1.0026x | discarded nonzero 0

[Comparison](train_006758/comparison.png) | [First original pixels](train_006758/first_raw.png) | [Second original pixels](train_006758/second_raw.png) | [New 768](train_006758/new768.png)

## train:005642 (AP)

EXACT ZERO TRIM | `(0, 0, 2539, 2990)` -> `(0, 0, 2539, 2840)` | extra pixels `{'left': 0, 'top': 0, 'right': 0, 'bottom': 150}` | scale 1.0528x | discarded nonzero 0

[Comparison](train_005642/comparison.png) | [First original pixels](train_005642/first_raw.png) | [Second original pixels](train_005642/second_raw.png) | [New 768](train_005642/new768.png)

## valid:001034 (AP)

EXACT ZERO TRIM | `(0, 743, 2544, 3056)` -> `(0, 848, 2544, 3056)` | extra pixels `{'left': 0, 'top': 105, 'right': 0, 'bottom': 0}` | scale 1.0000x | discarded nonzero 0

[Comparison](valid_001034/comparison.png) | [First original pixels](valid_001034/first_raw.png) | [Second original pixels](valid_001034/second_raw.png) | [New 768](valid_001034/new768.png)

## train:010083 (PA)

EXACT ZERO TRIM | `(0, 0, 2544, 2213)` -> `(72, 0, 2544, 2104)` | extra pixels `{'left': 72, 'top': 0, 'right': 0, 'bottom': 109}` | scale 1.0291x | discarded nonzero 0

[Comparison](train_010083/comparison.png) | [First original pixels](train_010083/first_raw.png) | [Second original pixels](train_010083/second_raw.png) | [New 768](train_010083/new768.png)

## train:009682 (Lateral)

EXACT ZERO TRIM | `(0, 0, 1681, 3056)` -> `(0, 0, 1600, 3056)` | extra pixels `{'left': 0, 'top': 0, 'right': 81, 'bottom': 0}` | scale 1.0000x | discarded nonzero 0

[Comparison](train_009682/comparison.png) | [First original pixels](train_009682/first_raw.png) | [Second original pixels](train_009682/second_raw.png) | [New 768](train_009682/new768.png)
