# Tight crop CPU latency preflight

Frozen inputs: first four `train` records for each AP, PA, Lateral, Missing group
in the prior 96-image geometry index (16 images total). These are a biased review
sample, not a population latency estimate. No valid/test image, model, GPU, or
training process was used. Original image SHA-256 values matched the frozen index
before and after measurement. `tight_crop.py` remained at SHA-256
`85ec065e2b4a9bb23a0defcafc3681cd474a579563affd3501ddcb4d2426cd8d`.

Command: `PYTHONPATH=/tmp/cxr_geometry_cv2_411 python benchmark_tight_crop_cpu.py --repeats 5 --cv2-threads 1`

The same command with `--cv2-threads 8` was the only thread comparison. Each
measurement is 16 images x 5 repeated, single-image serial. The aggregate P95
below is over 80 image/repetition observations. The Python process was CPU-only.

| Measurement | 1 thread mean / P95 ms | 8 threads mean / P95 ms |
| --- | ---: | ---: |
| Existing JPEG draft768 **decode only** | 23.526 / 32.602 | 23.539 / 32.787 |
| Full original grayscale decode | 32.410 / 44.701 | 32.904 / 44.986 |
| `detect_tight_crop` on already decoded full gray | 26.832 / 32.232 | 26.077 / 36.163 |
| Full gray decode + detector | 59.241 / 75.281 | 58.981 / 74.653 |
| Above minus draft768 decode | 35.716 / 46.565 | 35.442 / 49.481 |

The historical 84.820141ms 2080Ti complete baseline includes JPEG draft768,
model inference and all other processing. If that draft decoder must still run
for the existing model pixels, **full gray decode + detector** is the additional
CPU work: about 59.24ms here, over the 5.18ms mean allowance by 54.06ms. If
one full decode is reused for crop detection and model input, the arithmetic
increment relative to draft decode is about 35.72ms, but that changes model
input pixels and is **not an equivalent implementation** of the 84.82ms
baseline. Adding either number to 84.82ms is only a screening estimate; no
2080Ti end-to-end measurement was run. The newer 95.41ms frozen-head candidate
is a different configuration and gives no budget for this comparison.

An instrumented pass through the unchanged detector assigned mean costs at
one OpenCV thread: original-to-512 `INTER_AREA` resize 14.455ms, morphology
3.784ms, connected components 3.869ms, blur/Otsu 0.394ms, contours/hull/fill
0.383ms, and Python/other 3.347ms. These are one-pass diagnostic numbers;
instrumentation overhead means they should not be added to the uninstrumented
detector mean. Eight threads reduced resize to 11.667ms but did not materially
reduce total detector time and increased its P95. All 16 returned boxes matched
between the one-thread and eight-thread reports; eight were crop candidates and
eight returned the full frame.

The detector already works on 512-pixel analysis masks. Replacing its label
selection loops or count operations could only address a minority of measured
time. Its `INTER_AREA` resize consumes nearly three times the entire 5.18ms
allowance, and full grayscale decode adds another 32ms. A true equivalent
optimization would have to retain the same original pixels for the detector
and return identical boxes, while also preserving the existing draft768 model
pixels. A cached train-time box cannot provide a deployment path. Running the
detector on draft768 pixels, draft512 pixels, or a differently resized image
requires a separate pixel/box equivalence check and must not be called equal by
assumption. Until such a path meets the CPU budget, this prototype should not
be treated as a sub-90ms production crop.

Machine-readable per-image timings, source SHA-256 values, boxes and method
details are in `report_threads1.json` and `report_threads8.json`.
