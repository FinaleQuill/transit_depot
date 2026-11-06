# Base/Tiny 768 pure-forward anchors (near-official reference)

Date: 2026-11-02 UTC

These numbers are a software-stack-matched reference on the local RTX 2080 Ti. They are **not** official T4 timing-Gate evidence and do not include decoding, resize, normalization, H2D, Study aggregation, or CSV writing.

## Environment

- Image: `crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0`
- Image manifest digest: `sha256:125d8bcb4ce5a2a9acec6d9838b37e2ae1e2014c09b848700f9530f2f7856a0b`
- PyTorch `2.10.0+cu128`, CUDA runtime `12.8`, cuDNN reported `91002` (9.10.2)
- TensorRT `10.9.0.34`, ONNX Runtime `1.26.0`
- Device: NVIDIA GeForce RTX 2080 Ti, compute capability 7.5, driver `610.57.04`
- The image was run with the pinned read-only project addons (`timm 1.0.30`) to load the already-exported model schemas. No network or pretrained download was used.

## Candidate identity

| Candidate | Backbone / role | Export directory | Weight SHA-256 | Config SHA-256 |
|---|---|---|---|---|
| Tiny768_041_A | ConvNeXt V1 Tiny, 041 A deployment candidate | `deliveries/044_tiny_base_class1B_draft768_20261031/model/A` | `fea2350a479fc65883f687ce038289666345c5b0d5fc7da44576a5f995742b5f` | `6a109c69ca9f15bc56fabf61002221e499e638cc09cdac9d0a7ecfba834fdffd` |
| Base768_043_B | ConvNeXt V1 Base student B, 043/045 Base member | `deliveries/044_tiny_base_class1B_draft768_20261031/model/B` | `903ee9b95317536b6120d399d67ddcb1e82b82f7898cc174b2fd8b55d6f8e411` | `ec8d155fdf1b4a8ca5de7c914ab098efa5d75c449f3a986e950a811d0c632011` |

The Base export is the distilled Base **student** from source checkpoint SHA `bf69b6e16917ab03b8f9318287a8ef660e21ebfb71095648527071377ae126eb`; it is not one of the five offline OOF teachers. The Tiny export source checkpoint SHA is `65cdb72f63a5ec0d36486d4d8dcb34dd2fde5ed85aa52d3a10be2b0dbf633f42`.

## Method

Each export was loaded through the project’s validated `ExperimentalPredictor` loader, with fixed `identity` view only for the anchor. Input was a resident zero tensor of shape `[1, 3, 768, 768]`, FP16, channels-last model, inference mode, and no gradients. cuDNN benchmark and TF32 were disabled. After 20 warmup forwards, 100 forwards were timed individually with paired CUDA events; synchronization occurred after each event pair. The reported P95 is the nearest-rank order statistic. Model load and CUDA context initialization are outside the timed region.

## Results

| Candidate | Warmup / measured | Median pure forward (ms) | P95 (ms) | Min–max (ms) |
|---|---:|---:|---:|---:|
| Tiny768_041_A | 20 / 100 | 10.9056 | 11.1428 | 10.5405–11.2005 |
| Base768_043_B | 20 / 100 | 17.2500 | 21.1569 | 16.4933–35.7836 |

Raw per-iteration samples and the exact environment/model hashes are in `Tiny768_041_A.json` and `Base768_043_B.json` in this directory. These anchors are only a forward-cost component; they cannot be added directly to the official per-Study budget without the segment table.
