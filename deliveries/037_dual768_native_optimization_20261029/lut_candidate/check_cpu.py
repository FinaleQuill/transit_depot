"""Meaningful CPU parity/error checks; never creates a CUDA context or model."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    import torch
    from PIL import Image
    import deploy_kd as deploy
    from lut_prototype import (CandidatePredictor, ExactU8Input, load_image_pixels,
                               member_results_to_cpu, normalization_table, normalize_pixels)

    report = {"gpu_runs": 0, "model_loads": 0, "production_files_changed": 0,
              "torch_version": torch.__version__, "numpy_version": np.__version__, "checks": []}
    for source in (Path(__file__), Path(__file__).with_name("lut_prototype.py")):
        ast.parse(source.read_text(), filename=str(source))
    table = normalization_table(deploy.MEAN, deploy.STD)
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    for pixels in (gray, np.stack((gray, gray[::-1], 255-gray), axis=-1)):
        oracle = normalize_pixels(pixels, deploy.MEAN, deploy.STD)
        for channel in range(3):
            selected = pixels if pixels.ndim == 2 else pixels[..., channel]
            assert table[channel][selected].tobytes() == oracle[channel].tobytes()
    report["checks"].append("all_3x256_values_float32_lut_byte_parity_gray_and_rgb")

    # CPU Torch validates embedding index32 support, layout, payload and control
    # flow. CUDA conversion parity remains a separate required GPU gate.
    lookup = ExactU8Input(torch, "cpu", deploy.MEAN, deploy.STD)
    rng = np.random.default_rng(20261029)
    for shape in ((16,16), (37,51), (768,768)):
        gray = rng.integers(0, 256, shape, dtype=np.uint8)
        for pixels in (gray, np.stack((gray, gray[::-1], 255-gray), axis=-1)):
            expected = torch.from_numpy(normalize_pixels(pixels, deploy.MEAN, deploy.STD)).unsqueeze(0).to(
                dtype=torch.float16, memory_format=torch.channels_last)
            actual = lookup.to_tensor(pixels)
            assert actual.is_contiguous(memory_format=torch.channels_last)
            assert actual.dtype == torch.float16 and tuple(actual.shape) == tuple(expected.shape)
            assert actual.numpy().tobytes() == expected.numpy().tobytes()
    report["checks"].append("cpu_only_embedding_int32_channels_last_and_float16_bits")

    finite_results = [torch.linspace(0, 1, 10, dtype=torch.float32),
                      torch.linspace(1, 0, 10, dtype=torch.float32)]
    expected = torch.stack(finite_results).numpy()
    actual = member_results_to_cpu(torch, finite_results, [torch.tensor(True)] * 4)
    assert actual.dtype == np.float32 and actual.tobytes() == expected.tobytes()
    for logit in (float("inf"), float("-inf"), float("nan")):
        hidden = torch.tensor(logit)
        flags = [torch.tensor(True), torch.isfinite(hidden)]
        try:
            member_results_to_cpu(torch, finite_results, flags)
        except FloatingPointError:
            pass
        else:
            raise AssertionError("Nonfinite raw logits escaped the packed flag")
    report["checks"].append("packed_20_probabilities_exact_with_nonfinite_raw_logit_rejection")

    with tempfile.TemporaryDirectory(prefix="lut-cpu-fixtures-") as temporary:
        root = Path(temporary)
        for mode in ("L", "RGB", "RGBA", "P", "1", "LA"):
            gray = rng.integers(0, 256, (37,61), dtype=np.uint8)
            rgb = np.stack((gray, gray[::-1], 255-gray), axis=-1)
            image = Image.fromarray(gray if mode == "L" else rgb)
            if mode != "L":
                image = image.convert(mode)
            for orientation in (1, 2, 6, 8):
                exif = Image.Exif(); exif[274] = orientation
                path = root / "fixture.png"
                image.save(path, exif=exif)
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                expected = deploy.load_image_arrays(path, [512,768])
                actual = load_image_pixels(path, [768,512,768], deploy._letterbox_image)
                assert list(actual) == [512,768]
                for size in expected:
                    array = normalize_pixels(actual[size], deploy.MEAN, deploy.STD)
                    assert array.tobytes() == expected[size].tobytes(), (mode, orientation, size)
                assert before == hashlib.sha256(path.read_bytes()).hexdigest()
        report["checks"].append("24_mode_exif_fixtures_two_sizes_preserve_original_float32_and_source_bytes")
        for mode, dtype in (("I", np.int32), ("F", np.float32), ("I;16", np.uint16)):
            path = root / "unsupported.tiff"
            Image.fromarray(np.arange(35, dtype=dtype).reshape(5,7), mode).save(path)
            try:
                load_image_pixels(path, [768], deploy._letterbox_image)
            except ValueError:
                pass
            else:
                raise AssertionError("Unsupported intensity policy was weakened")
        report["checks"].append("non_eight_bit_intensity_policy_rejection_preserved")
    report["status"] = "passed_cpu_checks_gpu_validation_still_required"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
