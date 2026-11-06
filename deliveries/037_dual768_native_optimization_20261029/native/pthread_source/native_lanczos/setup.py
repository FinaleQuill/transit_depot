"""Build from project root: python native_lanczos/setup.py build_ext --inplace."""
from pathlib import Path
from setuptools import Extension, setup

root = Path(__file__).resolve().parent.parent
setup(
    name="cloud-native-lanczos",
    version="1.0.0",
    description="Exact Pillow 10.4 grayscale LANCZOS, parallel within one image",
    license="HPND",
    packages=["native_lanczos"],
    py_modules=["fast_lanczos"],
    ext_modules=[Extension(
        "native_lanczos._core",
        ["native_lanczos/resample.c"],
        extra_compile_args=["-O3", "-pthread", "-fno-fast-math", "-ffp-contract=off"],
        extra_link_args=["-pthread"],
    )],
    package_data={"native_lanczos": ["upstream/LICENSE", "upstream/PROVENANCE.json", "README.md"]},
)
