import importlib.metadata
import importlib
import json
import os
import platform
import sys

report = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "packages": {},
    "probe_runs_model_inference": False,
}
for package in ("torch", "torchvision", "timm", "numpy", "Pillow", "scipy", "scikit-learn", "pandas", "opencv-python", "albumentations", "huggingface-hub", "safetensors", "PyYAML", "packaging", "pytest"):
    try:
        report["packages"][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        report["packages"][package] = None
try:
    import torch
    report["torch"] = {
        "version": torch.__version__,
        "git_version": torch.version.git_version,
        "cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "arch_list": torch.cuda.get_arch_list(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        report["torch"]["gpu"] = {
            "name": properties.name,
            "total_memory": properties.total_memory,
            "major": properties.major,
            "minor": properties.minor,
            "sm_count": properties.multi_processor_count,
        }
except Exception as error:
    report["torch_error"] = repr(error)
report["import_checks"] = {}
report["module_runtime_versions"] = {}
for module in ("torchvision", "timm", "numpy", "PIL", "scipy", "sklearn"):
    try:
        loaded = importlib.import_module(module)
        report["import_checks"][module] = "ok"
        report["module_runtime_versions"][module] = getattr(loaded, "__version__", None)
    except Exception as error:
        report["import_checks"][module] = repr(error)
print(json.dumps(report, indent=2))
