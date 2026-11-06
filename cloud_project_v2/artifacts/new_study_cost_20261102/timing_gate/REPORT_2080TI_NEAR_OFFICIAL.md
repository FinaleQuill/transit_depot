# Study timing profile — near-official / RTX 2080 Ti reference

Date: 2026-11-02 (UTC). **Not official T4 Gate evidence.** The official 3.0 container ran locally on an RTX 2080 Ti; host hardware and the official T4 measurement location are different.

## Environment and candidates

- Container: `crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0`, manifest digest `sha256:125d8bcb4ce5a2a9acec6d9838b37e2ae1e2014c09b848700f9530f2f7856a0b`.
- Runtime: Python 3.12.3, PyTorch 2.10.0+cu128, CUDA runtime 12.8, cuDNN 9.10.2, TensorRT 10.9.0.34, ONNX Runtime 1.26.0, Pillow 10.4.0, `fast_lanczos` AVX2.
- Host: RTX 2080 Ti, compute capability 7.5, driver 610.57.04.
- Tiny768 041 A weights SHA-256: `fea2350a479fc65883f687ce038289666345c5b0d5fc7da44576a5f995742b5f`.
- Base768 043 B **distilled student** weights SHA-256: `903ee9b95317536b6120d399d67ddcb1e82b82f7898cc174b2fd8b55d6f8e411`; this is not an offline OOF teacher.

## Pure-forward anchors

One image, FP16, identity forward, resident `[1,3,768,768]` input, 20 warmups and 100 CUDA-event samples. This excludes image decode, resize, normalization, H2D, aggregation, output and TTA.

| Model | Median ms | P95 ms |
|---|---:|---:|
| Tiny768 041 A | 10.91 | 11.14 |
| Base768 043 B | 17.25 | 21.16 |

The segment-run script also timed its own queued-event anchor (Tiny 8.60/9.16 ms P50/P95; Base 17.59/18.53 ms). It differs from the separate anchor, which synchronizes each sample; keep both method-specific readings visible rather than merging them. Neither is a Gate number.

## Full Study segment profile

For each 1-, 2-, and 3-image stratum, 30 distinct valid Studies were sampled and timed for 3 rounds (90 observations per stratum/model); 5 other Studies per model were warmup only. All images from one Study were processed together in one FP16 model forward. There was no cross-Study batch, no hflip/TTA, no label read, no test access, and no training. Each cell below is `P50 / P95 ms` across that stratum's 90 measured Study runs.

| Model | Images in Study | Decode | Resize | Normalize | H2D | Forward | Postprocess | Aggregate | Write | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model | Images | Decode | Resize | Normalize | H2D | Forward | Postprocess | Aggregate | Write | Total | CPU proc.¹ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny768 | 1 | 44.45 / 64.17 | 10.70 / 16.29 | 0.37 / 0.59 | 0.11 / 0.18 | 10.04 / 17.28 | 0.17 / 0.39 | 0.12 / 0.19 | 0.16 / 0.40 | 68.39 / 95.34 | 164 / 185% |
| Tiny768 | 2 | 58.96 / 112.66 | 19.33 / 29.02 | 0.80 / 1.20 | 0.21 / 0.29 | 17.37 / 17.50 | 7.21 / 8.35 | 0.14 / 0.21 | 0.22 / 0.39 | 97.74 / 153.58 | 169 / 182% |
| Tiny768 | 3 | 116.71 / 162.71 | 30.53 / 44.32 | 1.16 / 1.76 | 0.30 / 0.43 | 25.40 / 25.49 | 15.21 / 16.20 | 0.15 / 0.21 | 0.25 / 0.43 | 178.19 / 228.13 | 166 / 179% |
| Base768 | 1 | 44.77 / 58.24 | 9.98 / 14.93 | 0.38 / 0.47 | 0.11 / 0.15 | 21.05 / 22.71 | 1.64 / 4.65 | 0.13 / 0.16 | 0.16 / 0.27 | 77.54 / 91.68 | 159 / 176% |
| Base768 | 2 | 51.09 / 94.40 | 17.84 / 22.50 | 0.81 / 1.28 | 0.21 / 0.30 | 38.96 / 40.59 | 17.98 / 23.09 | 0.14 / 0.19 | 0.21 / 0.43 | 108.80 / 154.53 | 158 / 168% |
| Base768 | 3 | 114.16 / 142.63 | 28.97 / 35.90 | 1.14 / 1.68 | 0.30 / 0.39 | 58.01 / 60.01 | 34.70 / 42.82 | 0.14 / 0.18 | 0.25 / 0.38 | 195.74 / 236.32 | 160 / 169% |

¹ Process CPU time divided by Study wall time, one-core-equivalent P50/P95; it can exceed 100% and is not whole-host CPU percentage.

Stage timings use a mix of CUDA events and wall timers; components do not add exactly to total because work is asynchronous and no sync is inserted after every component.

## Decision and limits

- The 2080 Ti three-image P95 is 228.13 ms for Tiny and 236.32 ms for Base, far above the 55 ms pre-rental screening threshold. **Do not rent a T4 yet** under the current two-stage rule.
- This is the requested single-forward segmentation diagnosis, not a production latency claim. It excludes hflip/TTA, and batch-versus-serial probability, Study AUC, and ranking-equivalence gates have not passed. Do not replace the existing inference path based on these numbers.
- Decode and resize dominate these measurements; H2D and aggregation are small. The next optimization work, if separately approved, should target CPU decode/resize or overlapped preprocessing, then rerun the same profile and numerical-equivalence checks.
- An initial probe-script invocation stopped before a model forward because PyTorch `torch.zeros` did not accept `memory_format`; the isolated timing helper was corrected to allocate with `torch.empty(..., memory_format=...).zero_()`. The successful run below is separate and complete; no weights or production sources changed.

## Artifacts

- Raw report: `run_official_2080ti_v2/report.json`.
- Per-Study samples and image/Study probability outputs: `run_official_2080ti_v2/`.
- Script: `bench_timing_segments.py`.
- Pure-forward anchors and scripts: `../anchor/REPORT.md` and `../anchor/*.json`.
