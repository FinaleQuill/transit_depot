# AVX2 native read-only audit

No source, binary, or build changes were made during this audit. No GPU or large CPU job was run.

- Wheel RECORD SHA256/size verification passed for all 10 hashed entries. Wheel includes fast_lanczos.py, native package, README, HPND license and source provenance.
- The wheel extension is byte-identical to the inplace extension used for the 7 unit tests and 105-image CPU audit.
- GCC uses generic x86_64 defaults, -O3, -pthread, -fno-fast-math, -ffp-contract=off. Its pinned CPython compiler flags include -fno-strict-overflow. No -march=native, OpenMP, CUDA, or TensorRT linkage is present.
- AVX2 instructions exist only in target("avx2") H/V integer pixel functions. __builtin_cpu_init/__builtin_cpu_supports selects them at module import; generic H/V remains callable and was independently checked against Pillow. Coefficient double/sin/normalization code stays outside the AVX2 targets.
- Integer lane regrouping preserves the final 32-bit accumulated bit pattern; the pinned compiler's no-strict-overflow semantics agree with AVX2 modulo int32 additions/multiplications. The final arithmetic >>22 and uint8 saturation match Pillow. Vertical pack operations are safe because the shifted signed int32 value is in [-512,511], inside signed-16 range.
- Coefficient allocation arithmetic now uses size_t checks before multiplication. The validated full-box dimensions use exactly Pillow's float input boundaries, coefficient order, 22-bit quantization and H uint8 intermediate before V.
- A call lock serializes concurrent callers. Each H/V task waits for all selected workers before proceeding/returning; workers sleep on condition variables between tasks. Seven workers plus the caller yield at most eight active CPU threads. No image prefetch or overlap is implemented.
- The normal project is single-process; forking after the pool starts is unsupported and documented. Workers release no Python-owned image until their call completes.
- The wheel is cp312-cp312-linux_x86_64, not a universal manylinux artifact. ELF requires GLIBC_2.34 or newer; the pinned competition image has glibc2.39. CPU dispatch removes an AVX2 hardware requirement, but does not remove Python/Pillow/glibc ABI requirements.
- Upstream Pillow10.4.0 Resample.c/Imaging.h/HPND license are retained with URLs and SHA256 in the project. Required copyright/permission notice is distributed with the wheel.

Primary agent owns subsequent GPU and end-to-end validation; this audit makes no T4 timing claim.
