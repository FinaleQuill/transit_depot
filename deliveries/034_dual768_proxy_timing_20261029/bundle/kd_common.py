"""V1 蒸馏的来源登记和文件协议；导入本模块不会下载或构建模型。"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.request import urlopen

BASE_BACKBONE = "convnext_base.fb_in22k_ft_in1k_384"
TINY_BACKBONE = "convnext_tiny.fb_in22k_ft_in1k_384"
APPROVED_PRETRAINS = {
    name: {"repo_id": f"timm/{name}", "license": "apache-2.0",
           "pretrain_dataset": "ImageNet-22k", "finetune_dataset": "ImageNet-1k"}
    for name in (BASE_BACKBONE, TINY_BACKBONE)
}
UPSTREAM_LICENSE_URL = "https://raw.githubusercontent.com/facebookresearch/ConvNeXt/main/LICENSE"
APACHE_LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_pretrained_provenance(provenance: dict[str, Any], backbone_name: str) -> dict[str, Any]:
    """精确来源白名单；验证材料内容哈希，不依赖原训练机的缓存路径。"""
    if backbone_name not in APPROVED_PRETRAINS or not isinstance(provenance, dict):
        raise ValueError("Only the approved non-NC ConvNeXt V1 sources are accepted")
    expected = APPROVED_PRETRAINS[backbone_name]
    if provenance.get("backbone_name") != backbone_name:
        raise ValueError("Pretrained backbone declaration does not match the architecture")
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Unapproved pretrained provenance: {key}")
    for key in ("weights_sha256",):
        if re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(key, ""))) is None:
            raise ValueError(f"Missing original pretrained {key}")
    revision = provenance.get("revision", "")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Pretrained weights must be pinned to an immutable revision")
    base_url = f"https://huggingface.co/{expected['repo_id']}"
    if provenance.get("weights_url") != f"{base_url}/resolve/{revision}/model.safetensors":
        raise ValueError("Original pretrained URL is not the declared immutable source")
    materials = provenance.get("materials")
    if not isinstance(materials, dict):
        raise ValueError("Pretrained model card and full licenses are required")
    for name in ("model_card", "apache_license", "upstream_license"):
        item = materials.get(name, {})
        content = item.get("text")
        if not isinstance(content, str) or not content or text_sha256(content) != item.get("sha256"):
            raise ValueError(f"Missing or changed pretrained material: {name}")
    if not re.search(r"license:\s*apache-2\.0", materials["model_card"]["text"], re.I):
        raise ValueError("The exact pretrained model card does not declare Apache-2.0")
    if "MIT License" not in materials["upstream_license"]["text"]:
        raise ValueError("Upstream license does not match the approved ConvNeXt V1 source")
    if "Apache License" not in materials["apache_license"]["text"]:
        raise ValueError("Full Apache license is missing")
    return provenance


def prepare_pretrained_provenance(backbone_name: str) -> dict[str, Any]:
    """仅显式训练调用：缓存原始权重和材料，之后必须从 weights_path 加载。

    此函数不在 import、dry-run、单元测试、部署推理中调用。
    不下载 ImageNet 数据，不访问任何比赛图像。
    """
    if backbone_name not in APPROVED_PRETRAINS:
        raise ValueError("Unapproved pretrained source")
    from huggingface_hub import hf_hub_download

    approved = APPROVED_PRETRAINS[backbone_name]
    path = Path(hf_hub_download(repo_id=approved["repo_id"], filename="model.safetensors"))
    # hf_hub_download 返回 snapshots/<commit>/...；不可用可变 main 冒充固定来源。
    revision = path.parent.name
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Unable to identify the downloaded weights' immutable revision")
    card = Path(hf_hub_download(repo_id=approved["repo_id"], filename="README.md", revision=revision))
    sources = {
        "model_card": (f"https://huggingface.co/{approved['repo_id']}/raw/{revision}/README.md",
                       card.read_text(encoding="utf-8")),
    }
    for name, url in (("apache_license", APACHE_LICENSE_URL),
                      ("upstream_license", UPSTREAM_LICENSE_URL)):
        with urlopen(url, timeout=60) as response:
            sources[name] = (url, response.read().decode("utf-8"))
    value = {
        **approved, "backbone_name": backbone_name, "revision": revision,
        "weights_path": str(path), "weights_sha256": sha256_file(path),
        "weights_url": f"https://huggingface.co/{approved['repo_id']}/resolve/{revision}/model.safetensors",
        "materials": {name: {"url": url, "text": content, "sha256": text_sha256(content)}
                      for name, (url, content) in sources.items()},
    }
    return validate_pretrained_provenance(value, backbone_name)


def student_preprocessing() -> dict[str, Any]:
    return {"image_size": 512, "validation_size": 512, "training_scales": [512],
            "cache_mode": "RGB", "color_mode": "RGB", "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225], "resize": "letterboxLANCZOS",
            "exif_transpose": True, "data_source": "raw_readonly",
            "eval_transform": "EXIF+RGB+LetterboxLANCZOS+Normalize", "validation_tta": "identity"}


def require_training_runtime() -> None:
    """在下载权重之前拒绝误用旧环境；CPU逻辑测试与dry-run不调用。"""
    required = {"torch": "2.10.0", "torchvision": "0.25.0", "timm": "1.0.30"}
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use the Python 3.12 environment described in docs/KD_V1_USAGE.md")
    for package, version in required.items():
        installed = metadata.version(package).split("+")[0]
        if installed != version:
            raise RuntimeError(f"Use requirements_kd.txt: {package} must be {version}, got {installed}")
