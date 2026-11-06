# near-official Environment Verification — 2026-11-02

Status: software-stack and CUDA smoke test passed on the local RTX 2080 Ti. This is a near-official comparison environment only; no timing or accuracy number from this host is a competition Gate result.

## Host

- GPU: NVIDIA GeForce RTX 2080 Ti (compute capability 7.5)
- NVIDIA driver: 610.57.04
- `nvidia-smi` driver-advertised CUDA compatibility: 13.3
- Docker: 26.1.0
- NVIDIA Container Toolkit: 1.20.1-1 (installed after discovering the daemon referenced a missing `nvidia-container-runtime`)
- Existing `/etc/docker/daemon.json` backup: `/etc/docker/daemon.json.pre_nvidia_20261102`

## Official 3.0 image, probed locally

- Reference: `crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0`
- Manifest digest: `sha256:125d8bcb4ce5a2a9acec6d9838b37e2ae1e2014c09b848700f9530f2f7856a0b`
- Image config digest: `sha256:ffe75e4756ccc7a1d9c2e7e8e7726ea86ae704eabf5089b0d079c2c82a39dc3d`
- Python: 3.12.3
- PyTorch: 2.10.0+cu128; bundled CUDA runtime: 12.8
- cuDNN: 9.10.2 (PyTorch reports 91002)
- TensorRT: 10.9.0.34
- ONNX Runtime: 1.26.0; available providers include TensorRT, CUDA, CPU
- GPU container smoke test: `torch.cuda.is_available() == True`; 2080 Ti visible; FP16 CUDA matrix multiply succeeded.

The Docker manifest and blob are anonymously accessible; no registry login was required. This confirms this machine can run the 3.0 image after installing/configuring NVIDIA Container Toolkit. It does not validate T4 behavior or satisfy the T4 timing Gate. Use this image/container as the pinned near-official software stack for same-host comparisons, clearly labeled `near-official`.
