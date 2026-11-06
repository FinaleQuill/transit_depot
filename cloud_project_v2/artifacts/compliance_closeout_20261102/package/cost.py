"""Package launcher for the unchanged code/cost.py historical evaluator.

This launcher is a packaging addition, not historical execution evidence.
The preserved evaluator still uses the OLD per-image timing denominator.
It is NOT certification of the current per-Study timing Gate.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

def main(argv=None):
    code_dir = Path(__file__).resolve().parent / "code"
    sys.path.insert(0, str(code_dir))
    spec = importlib.util.spec_from_file_location("_packaged_historical_cost", code_dir / "cost.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(argv)

if __name__ == "__main__":
    raise SystemExit(main())
