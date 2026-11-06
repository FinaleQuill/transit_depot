"""胸片多标签数据接口：只读清洗清单，768输入直接读取原图并保持比例补边。

本文件只有定义，没有创建DataLoader、遍历图片或启动训练的入口。
依赖：PyTorch、NumPy、Pillow、OpenCV；构造变换时才需要
albumentations==2.0.8。没有在本次编写中安装或执行这些依赖。

数据流：CSV十维多热 -> 只读原图（高于300）或既有300RGB缓存 ->
等比例缩放和居中黑色补边 -> 单图在线增强 -> Normalize ->
Tensor -> 训练collate_fn内Mixup或CutMix -> 软标签BCEWithLogitsLoss。
valid/test始终无随机增强、无批次混合；预测后按Study平均概率。

接口依据（2026-10-17核对，使用2.x参数名，避免旧var_limit/max_holes）：
https://github.com/albumentations-team/albumentations/tree/2.0.8
https://albumentations.ai/docs/4-advanced-guides/reproducibility/
https://docs.pytorch.org/docs/2.10/data.html
"""
from __future__ import annotations

# 独立import dataset及Windows子进程也须先绕开本机Python3.14的WMI等待。
# 兼容层只改当前进程的系统信息查询路径，不读取图片、不创建worker。
from runtime_compat import prepare_windows_runtime

prepare_windows_runtime()

import csv
import hashlib
from io import BytesIO
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

# Windows的spawn子进程会重新导入本模块，主进程与每个worker都会执行本行。
# 关闭的是OpenCV内部的并行线程池；不影响PyTorch的worker数量或其他库线程。
# 多进程读图时避免再叠加OpenCV线程，减少线程争用与额外内存占用。
cv2.setNumThreads(0)

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info


PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = PROJECT_ROOT / "data" / "clean_v1_reviewed"
DEFAULT_TRAIN_MANIFEST = MANIFEST_ROOT / "train_clean.csv"
DEFAULT_VALID_MANIFEST = MANIFEST_ROOT / "valid_manifest.csv"
DEFAULT_TEST_MANIFEST = MANIFEST_ROOT / "test_manifest.csv"
NUM_CLASSES = 10
# 类别ID为1..10；targets数组的下标为0..9，不能再减一次1或做argmax。
CLASS_IDS = tuple(range(1, NUM_CLASSES + 1))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_RAW_ROOT = Path("/home/featurize/data/cxr_first")


class CachedSampleError(RuntimeError):
    """缓存读取/解码/在线增强失败；携带可定位的ID，不据此认定原图损坏。

    不自定义多参数构造函数，兼容PyTorch worker以异常消息重新构造异常。
    调用方可在串行复核时单独捕获本异常；标签及身份校验仍保持严格报错。
    """


def _split_name(split: str) -> str:
    split = "valid" if split == "val" else split
    if split not in {"train", "valid", "test"}:
        raise ValueError("split只能是train、valid/val或test")
    return split


def read_manifest(manifest_path: str | Path, *, split: str) -> list[dict[str, Any]]:
    """只读清单并严格校验标签，供Dataset和model.py正例权重共同使用。

    targets列已经是JSON十维多热，如[0,1,0,0,1,0,0,0,0,0]，
    并非[2,5]这样的类别ID列表。仅训练混合之后才允许小数软标签。
    test的空字段返回None，绝不把“未知”伪装成“十类全部阴性”。
    """
    split = _split_name(split)
    path = Path(manifest_path).resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
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
                raise ValueError(f"清单第{line_number}行缺少缓存路径；禁止退回原图")
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
    if type(image_size) is not int or image_size < 1:
        raise ValueError("image_size必须是正整数")
    if image.mode != "RGB":
        raise ValueError("letterbox_rgb requires an RGB image")
    scale = image_size / max(image.size)
    size = tuple(max(1, min(image_size, round(length * scale))) for length in image.size)
    resized = image.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    canvas.paste(resized, ((image_size - size[0]) // 2, (image_size - size[1]) // 2))
    return canvas


def build_transforms(split: str, *, image_size: int = 300, seed: int = 43) -> Any:
    """训练在线随机增强；全部划分均保持比例缩放和补边，禁止直接拉伸。

    CoarseDropout是Cutout的当前API：最多3个小遮挡块；保守参数保留胸廓。
    高斯噪声std_range按255的比例给定，而非旧版的方差像素单位。
    增强配置是待人工审查/未来实验的起点，没有声称这些参数已经提分。
    """
    split = _split_name(split)
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
        raise ValueError("image_size必须是正整数")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:
        raise ImportError("数据增强需要albumentations==2.0.8及OpenCV；本文件不自动安装依赖") from exc
    if A.__version__.split(".")[0] != "2":
        raise RuntimeError("本实现使用Albumentations 2.x接口，请先核对版本")
    transforms: list[Any] = [
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LANCZOS4),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      position="center", border_mode=cv2.BORDER_CONSTANT, fill=0),
    ]
    if split == "train":
        transforms.extend([
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
            # 胸片是灰度三通道，共用噪声，避免凭空加入彩色伪影。
            A.GaussNoise(std_range=(0.005, 0.025), mean_range=(0.0, 0.0), per_channel=False, p=0.2),
            A.Rotate(limit=5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.3),
            A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(0.04, 0.12),
                            hole_width_range=(0.04, 0.12), fill=0, p=0.25),
        ])
    transforms.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),  # HWC -> CHW；只改内存布局，不是随机增强。
    ])
    # Albumentations 2.x自持随机源，单设np.random.seed不足以控制其结果。
    return A.Compose(transforms, seed=seed)


class ChestXrayDataset(Dataset):
    """逐图样本共享Study标签；大于300输入从原图生成，禁止放大300缓存。

    复核清单路径包含../clean_v1/images300，须相对于CSV所在目录解析。
    原图逐次校验清单SHA；变换只在内存，不重写原图、缓存或清单。
    """
    def __init__(self, manifest_path: str | Path, *, split: str,
                 image_size: int = 300, seed: int = 43,
                 cache_root: str | Path = PROJECT_ROOT / "data",
                 raw_root: str | Path | None = None) -> None:
        self.split = _split_name(split)
        self.manifest_path = Path(manifest_path).resolve()
        self.records = read_manifest(self.manifest_path, split=self.split)
        self.transform = build_transforms(self.split, image_size=image_size, seed=seed)
        self.image_size = image_size
        self.use_native = raw_root is not None or image_size > 300
        self.raw_root = resolve_native_image_root(raw_root or DEFAULT_RAW_ROOT, split=self.split) \
            if self.use_native else None
        self.cache_root = Path(cache_root).resolve()
        self.paths: list[Path] = []
        for record in self.records:
            if self.use_native:
                subject, study, dicom = (record.get(key) for key in
                                         ("subject_id", "study_id", "dicom_id"))
                if any(not isinstance(value, str) or not value for value in (subject, study, dicom)):
                    raise ValueError("原图清单缺少原始患者/Study/图像身份")
                relative = Path(record.get("relative_path", ""))
                if (relative.is_absolute() or relative.drive or ".." in relative.parts
                        or len(relative.parts) != 4
                        or relative.parts[:3] != (self.split, subject, study)
                        or relative.stem != dicom or not relative.suffix
                        or record["study_key"] != f"{subject}/{study}"):
                    raise ValueError("原图路径与清单split/患者/Study/图像身份不一致")
                path = (self.raw_root / relative).resolve()
                if not path.is_relative_to(self.raw_root / self.split):
                    raise ValueError("原图路径越过允许的划分目录")
                if re.fullmatch(r"[0-9a-f]{64}", str(record.get("file_sha256", ""))) is None:
                    raise ValueError("原图清单必须包含有效file_sha256")
                self.paths.append(path)
                continue
            relative = Path(record["cache_relative_path"])
            if relative.is_absolute():
                raise ValueError("缓存路径必须相对清单目录，不能绕过只读缓存约定")
            path = (self.manifest_path.parent / relative).resolve()
            if not path.is_relative_to(self.cache_root) or path.suffix.lower() != ".png":
                raise ValueError("缓存路径不在允许的数据目录中或不是PNG")
            self.paths.append(path)

    def reseed(self, seed: int) -> None:
        """每个DataLoader worker启动时设独立随机源；不逐样本重设同一seed。"""
        self.transform.set_random_seed(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample_path = self.paths[index]
        try:
            if self.use_native:
                sample_path = sample_path.resolve()
                if not sample_path.is_relative_to(self.raw_root / self.split):
                    raise ValueError("原图路径在读取前越过允许的划分目录")
                source_bytes = sample_path.read_bytes()
                if hashlib.sha256(source_bytes).hexdigest() != record["file_sha256"]:
                    raise ValueError("原图SHA与已验收清单不一致，禁止退回300缓存")
                with Image.open(BytesIO(source_bytes)) as image:
                    image.load()
                    upright = ImageOps.exif_transpose(image)
                    if upright.mode in {"I", "F"} or upright.mode.startswith("I;"):
                        raise ValueError("Non-8-bit source requires an explicit intensity policy")
                    pixels = np.array(letterbox_rgb(upright.convert("RGB"), self.image_size),
                                      dtype=np.uint8, copy=True)
            else:
                with Image.open(sample_path, mode="r") as image:
                    image.load()
                    if image.format != "PNG" or image.mode != "RGB" or image.size != (300, 300):
                        raise ValueError("缓存格式/尺寸异常，应先排查而非重写缓存")
                    pixels = np.array(image, dtype=np.uint8, copy=True)
            tensor = self.transform(image=pixels)["image"].to(dtype=torch.float32)
        except Exception as exc:
            # worker突然退出时未必能返回此消息；训练脚本仍须捕获next(iterator)
            # 的失败并切换串行读取，才能定位真正失败的样本。这里不跳过/改写数据。
            # 仅拦普通Exception，KeyboardInterrupt/SystemExit等仍正常传递。
            raise CachedSampleError(
                f"image_id={record['image_id']}; source={sample_path.name}; "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        targets = None if record["targets"] is None else torch.tensor(record["targets"], dtype=torch.float32)
        return {"image": tensor, "targets": targets, "image_id": record["image_id"],
                "study_key": record["study_key"], "split": self.split}


@dataclass(frozen=True)
class MixConfig:
    """每个训练批次50%不混合，25% Mixup，25% CutMix；两种方法都已接入。

    mix_probability控制是否混合，cutmix_probability是在已决定混合时选CutMix
    的条件概率。不连续叠加两种混合，以便软标签含义及后续实验对照清楚。
    """
    mix_probability: float = 0.5
    cutmix_probability: float = 0.5
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0

    def __post_init__(self) -> None:
        for value in (self.mix_probability, self.cutmix_probability):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("混合概率必须在[0,1]")
        for value in (self.mixup_alpha, self.cutmix_alpha):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Beta分布的alpha必须是正数")


def _stack_batch(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples or len({s["split"] for s in samples}) != 1:
        raise ValueError("批次为空或混入不同划分")
    present = [s["targets"] is not None for s in samples]
    if any(present) and not all(present):
        raise ValueError("批次内不能同时有标注与无标注样本")
    count = len(samples)
    return {"images": torch.stack([s["image"] for s in samples]),
            "targets": torch.stack([s["targets"] for s in samples]) if all(present) else None,
            "image_ids": [s["image_id"] for s in samples],
            "study_keys": [s["study_key"] for s in samples], "split": samples[0]["split"],
            "is_mixed": False, "mix_method": "none", "mix_lambda": 1.0,
            "mix_partner_indices": torch.arange(count)}


def evaluation_collate(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """valid/test不接受混合配置；test保留targets=None供未来预测代码识别。"""
    batch = _stack_batch(samples)
    if batch["split"] not in {"valid", "test"}:
        raise ValueError("evaluation_collate只用于valid/test")
    return batch


class TrainingBatchCollator:
    """可被Windows spawn序列化的顶层collate类，所有混合只发生在内存。

    批次已经Normalize，仍可线性混合：相同仿射归一化与Mixup可交换；
    CutMix则拷贝对方已经归一化的图块。返回的targets始终是float32软标签。
    image_ids仅表示混合前样本，不可用于混合批次的Study验证指标。
    """
    def __init__(self, config: MixConfig, *, seed: int = 43) -> None:
        self.config, self.seed = config, seed
        self._rng: np.random.Generator | None = None

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        batch = _stack_batch(samples)
        if batch["split"] != "train" or batch["targets"] is None:
            raise ValueError("Mixup/CutMix只允许用于有标签的train批次")
        if self._rng is None:
            info = get_worker_info()
            seed = self.seed if info is None else (info.seed + self.seed) % (2**63)
            self._rng = np.random.default_rng(seed)
        rng = self._rng
        images, targets = batch["images"], batch["targets"]
        size = images.shape[0]
        # 尾批可能仅剩1张，保留原样，不能丢失视角或产生无意义混合。
        if size < 2 or rng.random() >= self.config.mix_probability:
            return batch
        # 无固定点的随机配对：先随机排序，再首尾相接，每张都配到另一张。
        # 只用当前worker的rng，不依赖全局torch随机源或CUDA。
        order = rng.permutation(size)
        pairing = np.empty(size, dtype=np.int64)
        pairing[order] = np.roll(order, 1)
        partner = torch.from_numpy(pairing)
        use_cutmix = rng.random() < self.config.cutmix_probability
        alpha = self.config.cutmix_alpha if use_cutmix else self.config.mixup_alpha
        lam = float(rng.beta(alpha, alpha))
        if use_cutmix:
            height, width = images.shape[-2:]
            cut_ratio = math.sqrt(1.0 - lam)
            cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
            cx, cy = int(rng.integers(width)), int(rng.integers(height))
            # 用实际裁剪后的边界算面积，不能继续使用Beta抽出的原始lam。
            x1, x2 = max(0, cx - cut_w // 2), min(width, cx + (cut_w + 1) // 2)
            y1, y2 = max(0, cy - cut_h // 2), min(height, cy + (cut_h + 1) // 2)
            mixed = images.clone()
            mixed[:, :, y1:y2, x1:x2] = images[partner, :, y1:y2, x1:x2]
            lam = 1.0 - ((x2 - x1) * (y2 - y1)) / float(width * height)
            method = "cutmix"
        else:
            mixed = lam * images + (1.0 - lam) * images[partner]
            method = "mixup"
        # 多标签每一维分别插值；不能argmax、四舍五入或硬阈值化！
        # 这里的target与图像采用同一个partner及同一个有效lam。
        soft_targets = lam * targets + (1.0 - lam) * targets[partner]
        batch.update(images=mixed, targets=soft_targets, is_mixed=True,
                     mix_method=method, mix_lambda=lam, mix_partner_indices=partner)
        return batch


def _seed_worker(worker_id: int) -> None:
    """顶层函数可被Windows序列化；不同worker不可共享相同增强随机序列。"""
    del worker_id  # torch.initial_seed已经包含worker编号，不再重复相加。
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    info = get_worker_info()
    if info is not None:
        info.dataset.reseed(seed)


def make_dataloader(manifest_path: str | Path | None = None, *, split: str,
                    batch_size: int = 16, num_workers: int = 4,
                    prefetch_factor: int = 1, pin_memory: bool = True,
                    persistent_workers: bool = False, seed: int = 43,
                    image_size: int = 300, mix_config: MixConfig | None = None,
                    shuffle: bool | None = None, timeout: float = 120.0,
                    raw_root: str | Path | None = None) -> DataLoader:
    """只在未来被显式调用时构建loader，本次没有调用。

    CPU worker并行读取/增强下一批，预取+固定内存缓解GPU等待。未来传GPU时
    还需images.to(device, non_blocking=True)；是否真正无等待须以后实测，不能
    仅靠num_workers保证。num_workers=0时不传prefetch_factor/persistent_workers。
    Windows未来主程序必须用if __name__ == '__main__'保护loader创建和迭代。
    固定seed且相同worker配置可重放；换worker数/恢复中途不承诺逐位等价。
    persistent_workers保留参数以兼容旧调用，但实际强制False，不跨epoch保留进程。
    默认每worker仅预取1批；timeout限制等待worker的秒数，0表示不设超时。
    num_workers=0时timeout强制0，避免串行DataLoader的断言错误。
    shuffle默认仅train=True；train可显式False供确定顺序诊断，valid/test禁止True。
    """
    split = _split_name(split)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size必须为正整数")
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError("num_workers必须为非负整数")
    if not isinstance(prefetch_factor, int) or isinstance(prefetch_factor, bool) or prefetch_factor < 1:
        raise ValueError("prefetch_factor必须为正整数")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
        raise ValueError("timeout必须为有限非负秒数；0表示不设超时")
    if shuffle is not None and not isinstance(shuffle, bool):
        raise ValueError("shuffle必须为布尔值或None")
    if split != "train" and shuffle is True:
        raise ValueError("验证/测试必须保持确定顺序，禁止shuffle=True")
    if split != "train" and mix_config is not None:
        raise ValueError("验证/测试不允许传入任何Mixup/CutMix配置")
    defaults = {"train": DEFAULT_TRAIN_MANIFEST, "valid": DEFAULT_VALID_MANIFEST, "test": DEFAULT_TEST_MANIFEST}
    dataset = ChestXrayDataset(manifest_path or defaults[split], split=split,
                              image_size=image_size, seed=seed, raw_root=raw_root)
    generator = torch.Generator().manual_seed(seed)
    options: dict[str, Any] = {"batch_size": batch_size,
                               "shuffle": (split == "train") if shuffle is None else shuffle,
                               "drop_last": False, "num_workers": num_workers,
                               "pin_memory": pin_memory, "generator": generator,
                               "timeout": float(timeout) if num_workers > 0 else 0,
                               "worker_init_fn": _seed_worker,
                               "collate_fn": TrainingBatchCollator(mix_config or MixConfig(), seed=seed)
                               if split == "train" else evaluation_collate}
    if num_workers > 0:
        # 不采纳旧调用传入的True：Windows夜间运行优先退出每轮worker并重新创建。
        options.update(prefetch_factor=prefetch_factor, persistent_workers=False)
    # DataLoader对象构造本身不启动worker；iter(loader)/首次取批时才spawn。
    # 因而“连主入口启动日志都没有”不能归因于这里的num_workers=8。
    return DataLoader(dataset, **options)


class StudyPredictionAggregator:
    """跨batch累积多视角概率，最终对同Study逐类求均值。

    future调用方对B3原始logits做且只做一次sigmoid，再把概率交给update；
    此处不构建/调用模型、不再sigmoid、不硬阈值化，也不平均原始logits。
    finalize要求清单中每张图恰好出现一次，避免漏尾批、重复图或分布式sampler
    补齐样本导致Study权重改变。若未来用DDP，须先收集全量并移除填充副本。
    """
    def __init__(self, manifest_path: str | Path = DEFAULT_VALID_MANIFEST, *, split: str = "valid") -> None:
        self.split = _split_name(split)
        if self.split not in {"valid", "test"}:
            raise ValueError("Study验证汇总只接收无增强的valid/test预测")
        records = read_manifest(manifest_path, split=self.split)
        self.expected = {r["image_id"]: r["study_key"] for r in records}
        # dict插入顺序保留官方清单的Study首出现顺序，不把字符串ID当整数。
        self.labels: dict[str, tuple[float, ...] | None] = {}
        self.sums: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}
        for row in records:
            study = row["study_key"]
            self.labels.setdefault(study, row["targets"])
            self.sums.setdefault(study, np.zeros(NUM_CLASSES, dtype=np.float64))
            self.counts.setdefault(study, 0)
        self.seen: set[str] = set()

    def update(self, batch: Mapping[str, Any], probabilities: torch.Tensor | np.ndarray) -> None:
        if batch.get("split") != self.split or batch.get("is_mixed", False):
            raise ValueError("Study汇总禁止使用训练混合批次或不同划分")
        ids, studies = batch["image_ids"], batch["study_keys"]
        if isinstance(probabilities, torch.Tensor):
            values = probabilities.detach().to(device="cpu", dtype=torch.float64).numpy()
        else:
            values = np.asarray(probabilities, dtype=np.float64)
        if values.shape != (len(ids), NUM_CLASSES) or len(studies) != len(ids):
            raise ValueError("预测应为[B,10]并与批次ID一一对应")
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("输入必须是有限的[0,1]概率；请在外部对logits做一次sigmoid")
        if len(set(ids)) != len(ids) or self.seen.intersection(ids):
            raise ValueError("同一张图的预测重复提交，会改变Study汇总权重")
        # 先检查整批，避免后面一条报错时前几条已经悄悄进入累积器。
        for image_id, study in zip(ids, studies):
            if image_id not in self.expected or self.expected[image_id] != study:
                raise ValueError("预测图像/Study不属于本清单，或身份关系不一致")
        for image_id, study, value in zip(ids, studies, values):
            self.sums[study] += value
            self.counts[study] += 1
            self.seen.add(image_id)

    def finalize(self) -> dict[str, Any]:
        if self.seen != set(self.expected):
            raise ValueError(f"尚缺{len(set(self.expected) - self.seen)}张图的预测，不能静默丢失视角")
        studies = list(self.labels)
        probabilities = np.stack([self.sums[s] / self.counts[s] for s in studies]).astype(np.float32)
        targets = None if self.split == "test" else np.asarray([self.labels[s] for s in studies], dtype=np.float32)
        return {"study_keys": studies, "probabilities": probabilities, "targets": targets,
                "image_counts": np.asarray([self.counts[s] for s in studies], dtype=np.int64)}
