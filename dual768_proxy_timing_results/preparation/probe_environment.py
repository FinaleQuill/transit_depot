"""Record inference software and GPU properties without model execution."""
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys

import numpy
import PIL
import torch
import torchvision
import timm

expected = {"torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128", "timm": "1.0.30", "numpy": "1.26.4", "Pillow": "10.4.0"}
versions = {name: importlib.metadata.version(name) for name in expected}
assert versions == expected, (versions, expected)
assert sys.version_info[:2] == (3, 12)
assert torch.version.cuda == "12.8"
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert not any(k.startswith("CUDA_MPS_") for k in os.environ)
props = torch.cuda.get_device_properties(0)
result = {
    "python": platform.python_version(), "executable": sys.executable,
    "packages": versions, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
    "gpu": {"name": props.name, "total_memory_bytes": props.total_memory,
            "sm_count": props.multi_processor_count, "compute_capability": [props.major, props.minor]},
    "cpu_count": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
    "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_cuda_arch_list": torch.cuda.get_arch_list(),
    "default_cudnn_benchmark": torch.backends.cudnn.benchmark,
    "default_cudnn_deterministic": torch.backends.cudnn.deterministic,
    "default_allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    "default_allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    "installed_distributions": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
    "no_model_constructed": True, "no_inference_forward": True,
}
target = Path("/home/featurize/work/dual768_proxy_timing_results/preparation/environment.json")
with target.open("x") as out:
    json.dump(result, out, indent=2)
    out.write("\n")
target.with_name("environment_lock_inference.txt").write_text("\n".join(result["installed_distributions"]) + "\n")
target.with_name("torch_build_config.txt").write_text(torch.__config__.show())
print(json.dumps({k: v for k, v in result.items() if k != "installed_distributions"}, indent=2))
