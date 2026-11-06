"""Documented seed/resource wrapper; no inference instrumentation."""
import importlib.metadata
import os
import random
import sys
from pathlib import Path

if any(key.startswith("CUDA_MPS_") for key in os.environ):
    raise RuntimeError("Unexpected MPS environment; inspect rather than hiding it")

import numpy as np
import torch

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__.split("+")[0] == "2.10.0", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
for name, expected in {"torchvision": "0.25.0", "timm": "1.0.30", "numpy": "1.26.4", "Pillow": "10.4.0"}.items():
    actual = importlib.metadata.version(name)
    assert actual.split("+")[0] == expected, (name, actual, expected)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, "Require one visible rented GPU"
random.seed(43)
np.random.seed(43)
torch.manual_seed(43)
torch.cuda.manual_seed_all(43)
actual_sm = torch.cuda.get_device_properties(0).multi_processor_count
bundle = Path("/home/featurize/work/deliveries/034_dual768_proxy_timing_20261029/bundle")
assert Path.cwd() == bundle
sys.path.insert(0, str(bundle))
from benchmark_local_resources import main

raise SystemExit(main(sys.argv[1:] + ["--baseline-sm-count", str(actual_sm)]))
