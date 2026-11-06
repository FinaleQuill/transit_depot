# Tight-crop CPU latency and later options (2026-11-02)

The current B training uses the frozen `tight_crop.py` detector on **fully decoded
original grayscale pixels** (SHA-256 `85ec065e2b4a9bb23a0defcafc3681cd474a579563affd3501ddcb4d2426cd8d`).
The experimental `tight_crop_fast.py` is not in that training or production path.
The earlier 5.18ms incremental target has been withdrawn for this training
decision. A future sub-10ms detector would be a separate engineering candidate,
not a property of the current code.

Read-only CPU preflight used the first four **train** images per AP, PA, Lateral
and Missing group from the frozen gallery index (16 images, five serial repeats
each). Original source SHA values matched before and after. No valid/test image,
GPU, model or training process was used. The reference detector on already
decoded full grayscale cost 26.832ms mean / 32.232ms P95 with one OpenCV thread;
full grayscale decode plus detector cost 59.241ms / 75.281ms. The historical
84.820ms 2080Ti end-to-end baseline uses JPEG draft768, so these CPU numbers
cannot be directly presented as a new end-to-end result.

The independent fast experiment consumed the already decoded draft image,
created a 512-pixel analysis image, then retained the reference threshold,
morphology, components, hull, margins, safety checks and original-size coordinate
mapping. On this biased 16-image set, its returned boxes matched the full-pixel
reference 16/16, including all 8 actual crop candidates; it additionally
excluded zero nonzero original pixels relative to the reference boxes. This is
only a small-sample observation, not pixel equivalence or anatomy safety.
Draft JPEG pixels and full-decode pixels differ in principle. The final limited
8-thread pass cost **13.309ms mean / 14.845ms P95** of added CPU, including
draft-to-512 analysis 4.062ms, geometry 8.979ms and PIL crop 0.249ms. Measured
draft decode plus that increment was 36.853ms mean / 46.736ms P95. One-thread
fast-path added cost was 17.870ms / 20.816ms before a redundant `L` conversion
was removed. The small local improvement and thread results do not establish
sub-10ms performance. Detailed per-image values are under
`artifacts/tight_crop_ab_20261102/latency_preflight/`.

## Two concrete later options

1. **Share the existing draft decode and reduce 512-stage overhead.** Keep one
   draft image in memory before letterboxing; build the 512 analysis image from
   it, then execute the same geometry rules with the original dimensions for
   margins and box mapping. The 13.3ms 8-thread Python prototype is a starting
   point. A compiled C++/OpenCV implementation could avoid Python and repeated
   temporary arrays, but must reproduce the chosen draft path's boxes and
   final cropped model pixels exactly. First compare every box against the
   Python draft reference on fixed train images, then run complete valid
   accuracy and batch-1 serial end-to-end timing. Its boxes matching the old
   full-decode detector on 16 train images is insufficient to call the two
   decoders equivalent; any changed model pixels require a separate model
   validation. No sub-10ms claim is available yet.

2. **Compile the original full-pixel geometry path or add a proved fallback
   shortcut.** Preserve original JPEG decode, `INTER_AREA` pixels and all
   branch decisions; accelerate the 512-pixel morphology/components/selection
   with a native implementation only after box, fallback, mask and diagnostic
   comparisons. A cheap shortcut may return the full-frame box only when its
   condition is a sufficient condition for the reference detector's full-frame
   result, for example the exact original grayscale range `<2` branch. It must
   fall through to the reference geometry for every other image. The measured
   full-image resize alone was about 14.5ms with one OpenCV thread, so this
   route has no current evidence of reaching 10ms. Caching precomputed train
   boxes can improve training throughput but cannot stand in for online
   deployment detection.

For either option, parallel data-loader workers can improve training throughput
but cannot make a batch-1 **serial** cost test faster by hiding per-image CPU
work. Final acceptance requires the actual 2080Ti model, decoder, crop,
letterbox and output path in one end-to-end timing run; the 95.41ms frozen-head
candidate is a different configuration and is not a usable baseline here.
