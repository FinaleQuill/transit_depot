"""768 主线的数据公共小工具（精简版）。

只保留当前调用到的清单解析、目录定位、等比例缩放补边与必要常量。
函数主体来自当前项目；旧缓存Dataset、批次混合和通用增强均不在此包内。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = PROJECT_ROOT / "data" / "clean_v1_reviewed"
DEFAULT_TRAIN_MANIFEST = MANIFEST_ROOT / "train_clean.csv"
DEFAULT_VALID_MANIFEST = MANIFEST_ROOT / "valid_manifest.csv"
DEFAULT_TEST_MANIFEST = MANIFEST_ROOT / "test_manifest.csv"
NUM_CLASSES = 10
CLASS_IDS = tuple(range(1, NUM_CLASSES + 1))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_RAW_ROOT = Path("/home/featurize/data/cxr_first")

# 保留现有异常名称，当前用来报告原图读取、SHA或在线增强错误。
class CachedSampleError(RuntimeError):
    """原图读取、校验或在线增强失败；携带可定位的ID，保留现有异常名称。

    不自定义多参数构造函数，兼容PyTorch worker以异常消息重新构造异常。
    调用方可在串行复核时单独捕获本异常；标签及身份校验仍保持严格报错。
    """


def _split_name(split: str) -> str:
    # 把调用者常用的val统一为valid，后面的路径与清单校验才使用同一套名字。
    split = "valid" if split == "val" else split
    if split not in {"train", "valid", "test"}:
        raise ValueError("split只能是train、valid/val或test")
    return split


def read_manifest(manifest_path: str | Path, *, split: str) -> list[dict[str, Any]]:
    """只读清单，核对身份和十维标签，返回当前主线使用的图片记录。

    targets列已经是JSON十维多热，如[0,1,0,0,1,0,0,0,0,0]，
    并非[2,5]这样的类别ID列表；当前主线标签保持原始0/1，不做批次混合。
    test的空字段返回None，绝不把“未知”伪装成“十类全部阴性”。
    """
    # CSV每行对应一张图；Study可以有多张图，但每张图都携带同一份十维标签。
    # 例如[1,0,1,0,...]表示第1类和第3类同时阳性，绝不是互斥的单选答案。
    split = _split_name(split)
    path = Path(manifest_path).resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        # cache_relative_path是既有清单的格式字段，检查它不等于读取低分辨率缓存。
        required = {"image_id", "split", "study_key", "targets", "cache_relative_path"}
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("清单表头为空或存在重复列")
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"清单缺少必要列：{sorted(required - set(reader.fieldnames))}")
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        study_labels: dict[str, tuple[float, ...]] = {}
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"清单第{line_number}行列数与表头不符")
            if row["split"] != split:
                raise ValueError(f"清单第{line_number}行划分与请求的{split}不一致")
            if not row["image_id"] or row["image_id"] in seen or not row["study_key"]:
                raise ValueError(f"清单第{line_number}行图像ID重复/为空或Study为空")
            if not row["cache_relative_path"]:
                raise ValueError(f"清单第{line_number}行缺少cache_relative_path元数据字段")
            if split == "train" and row.get("training_eligible", "").lower() != "true":
                raise ValueError("训练只能读取training_eligible=True的清洗后清单")
            if row.get("status", "ok") != "ok":
                raise ValueError(f"清单第{line_number}行存在未解决的图像状态")
            raw = row["targets"].strip()
            if split == "test":
                if raw:
                    raise ValueError("测试清单应没有标签，不能读取测试答案参与训练")
                targets = None
            else:
                try:
                    values = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"清单第{line_number}行targets不是合法JSON") from exc
                if not isinstance(values, list) or len(values) != NUM_CLASSES or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or value not in (0, 1) for value in values
                ):
                    raise ValueError("原始targets必须是十维0/1多热，不能是类别ID列表/软标签")
                targets = tuple(float(value) for value in values)
                # 这里不对多视角标签投票：同次检查的标签必须一致，否则立即报错。
                old = study_labels.setdefault(row["study_key"], targets)
                if old != targets:
                    raise ValueError("同Study的多视角标签冲突；不得静默合并或改标签")
            seen.add(row["image_id"])
            records.append({**row, "targets": targets})
    if not records:
        raise ValueError("清单为空")
    return records


def resolve_native_image_root(image_root: str | Path, *, split: str) -> Path:
    """Accept a split root or its one-level cxr_first wrapper without scanning images."""
    # 只检查两种常见解压层级，不扫描整套图片，也不会搬动正在被训练读取的文件。
    split = _split_name(split)
    root = Path(image_root).resolve()
    if (root / split).is_dir():
        return root
    nested = root / "cxr_first"
    if (nested / split).is_dir():
        return nested.resolve()
    raise FileNotFoundError(f"原图目录缺少{split}划分：{root}或{nested}；请先等待解压完成")


def letterbox_rgb(image: Image.Image, image_size: int) -> Image.Image:
    """Keep the complete image and aspect ratio, with centered black padding."""
    # Pillow的image.size是(宽,高)。768除以长边得到统一缩放比例，宽高共用它。
    # 例如原图1200x900 -> 768x576，再上下各补96像素，得到768x768。
    # 直接resize到正方形会改变胸廓比例；这里缩放后的长宽比只受整数取整影响。
    if type(image_size) is not int or image_size < 1:
        raise ValueError("image_size必须是正整数")
    if image.mode != "RGB":
        raise ValueError("letterbox_rgb requires an RGB image")
    scale = image_size / max(image.size)
    size = tuple(max(1, min(image_size, round(length * scale))) for length in image.size)
    resized = image.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    # 黑边只补画布空白区域，居中粘贴，不切掉原图边缘。
    canvas.paste(resized, ((image_size - size[0]) // 2, (image_size - size[1]) // 2))
    return canvas
