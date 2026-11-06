# Exact Pillow 10.4 grayscale LANCZOS

This optional CPU extension replaces only full-box `L` image resizing. Decoding,
EXIF, color conversion, aspect-ratio rounding, letterboxing and normalization stay
in the shared project pipeline. `fast_lanczos.resize` falls back to the original
Pillow implementation for other modes, missing native builds or Pillow versions
other than 10.4.0. It does not use CUDA or change model weights.

Run from the project root inside the pinned competition image:

```sh
python native_lanczos/setup.py build_ext --inplace
python -m unittest discover -s tests -p test_native_lanczos.py -v
```

GCC and the Python 3.12 headers are provided by the pinned image. The extension
uses portable x86_64 compiler defaults, pthreads, `-O3`, `-fno-fast-math` and
`-ffp-contract=off`; it does not use `-march=native`, OpenMP, CUDA or an additional
runtime package. A cp312/Linux x86_64 wheel can be built with:

```sh
python native_lanczos/setup.py bdist_wheel --dist-dir /path/to/wheels
```

The coefficient generation and fixed-point rounding derive from upstream
Pillow 10.4.0 `Resample.c`; its complete original source, ABI header, HPND license
and source hashes are retained in `upstream/`. The image capsule prefix is used
only with exactly Pillow 10.4.0. The wrapper keeps both image owners alive during
the native call and the native entry validates their modes, dimensions and type.

The implementation preserves double-precision coefficient evaluation order,
22-bit signed integer coefficient quantization, bias `1 << 21`, uint8 saturation
after the horizontal pass, and the subsequent vertical pass. Work is split only
across independent rows within one image. Seven persistent sleeping pthread
workers plus the caller give at most eight active CPU threads. A native call
waits for both passes to finish before returning; no image prefetch or overlap is
performed. Concurrent callers serialize access to the worker pool.

The current project uses one process for serial inference. Start any process
workers before first calling the native resampler; forking a process after a
pthread pool has started is not supported. The normal `predict.py`/`cost.py`
single-process execution does not fork.

Validation must compare both resized uint8 bytes and final normalized model
inputs. The fixed 105-image CPU audit and thread sweep are diagnostic evidence,
not a T4 latency certification or a substitute for three-round end-to-end timing.
