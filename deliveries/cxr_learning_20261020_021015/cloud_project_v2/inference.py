# 【学习导航】统一命令入口：传入ensemble-plan后转到inference_overnight.py；其余大部分函数处理历史schema。当前768五折的阈值来源要按新分支理解。
# 本文件是独立学习副本；只增补解释性注释，运行语义保持源代码一致。
"""胸片多标签测试推理与Study级提交生成；本次只编写，未执行。

导入本模块只定义函数，不读清单、不创建模型、不下载权重、不启动推理。
后续须先获得用户授权，才可人工执行命令行入口。跨平台多worker由
``if __name__ == '__main__'`` 保护，不能在模块顶层创建DataLoader。
Linux云端可通过--image-root定位只读原图，通过--work-root指定输出和日志位置；
物理目录不进入预处理指纹，搬迁权重不会改变标签、阈值或Study聚合协议。

概率链路只有一条：十维logits -> 一次Sigmoid -> 同Study多视角概率均值。
十类阈值来自训练脚本的最佳验证epoch，绝不在测试集重新寻找阈值。
schema 4恢复ConvNeXt-Tiny GAP+Linear及同源EMA，并恢复验证使用的AMP精度；
schema 3从原图只读构建384输入，恢复ConvNeXt V2类别注意力模型，严格使用EMA；
schema 2的model_state_dict已经是同轮验证、阈值搜索使用的EMA或raw权重；
严格核对三者来源一致，绝不改读raw_model_state_dict。兼容schema 1历史raw权重。

输出格式特别说明：官方赛题“七、提交要求”第2项规定的是三列长表
Study_id,Predict_class,Probability，每项Study保留全部十类连续概率；
用户指定的study_id加十类宽表另存为study_probabilities.csv，同时输出
使用已保存阈值的study_binary.csv，不能用宽表覆盖正式submission.csv。
来源：https://www.ncccu.org.cn/index/Paper/case1.html （2026-10-17核对）。
--include-gate是显式选择的可选Gate列；逐类Gate的平台接受情况仍待确认，
因此默认只导出官方三列概率，绝不将低于阈值的类别删除或强制归零。
"""

from __future__ import annotations

from runtime_compat import prepare_runtime

# 在第三方库前完成跨平台启动兼容；Linux直接返回，Windows按需应用回退。
prepare_runtime()

import argparse
import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataset import (
    CLASS_IDS,
    DEFAULT_TEST_MANIFEST,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    PROJECT_ROOT,
    StudyPredictionAggregator,
    make_dataloader,
    read_manifest,
)
from model import build_model


_AGGREGATION = "study_mean_probability"
_HK_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Hong_Kong")


def _sha256(path: Path) -> str:
    """分块只读计算SHA，避免把较大的模型文件一次读进额外内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON根节点必须是对象：{path.name}")
    return value


def _class_ids(value: Any) -> None:
    """严格保留1..10的列序，避免拿到别的标签映射后仍悄悄生成CSV。"""
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError("checkpoint的class_ids必须是整数列表")
    if value != list(CLASS_IDS):
        raise ValueError("checkpoint的class_ids必须严格按1..10排列")


def _thresholds(value: Any) -> np.ndarray:
    if not isinstance(value, list) or len(value) != NUM_CLASSES:
        raise ValueError("必须保存十个独立类别阈值，不能只传一个0.5")
    if any(type(item) not in (int, float) for item in value):
        raise ValueError("阈值必须是普通数值，不能是布尔值或字符串")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
        raise ValueError("十类阈值必须都是[0,1]内的有限数值")
    return array


def _preprocessing_sha256(preprocessing: Mapping[str, Any]) -> str:
    """将预处理配置绑定到阈值，禁止384原图权重搭配300缓存或TTA阈值。"""
    encoded = json.dumps(
        dict(preprocessing), sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _highres_model_config(value: Any) -> dict[str, Any]:
    """只恢复本阶段实现的架构；配置错误不能回退到B3或下载其他骨干。"""
    # 此导入只定义模型接口，不创建模型、不下载、不执行前向；与训练复用校验。
    from model_highres import MODEL_CONFIG_KEYS, validate_model_config

    if not isinstance(value, dict) or set(value) != set(MODEL_CONFIG_KEYS):
        raise ValueError("schema 3 model_config必须完整且只能包含已实现的七个架构字段")
    if value["gradient_checkpointing"] is not True:
        raise ValueError("schema 3必须记录gradient_checkpointing=True，保持训练显存协议")
    return validate_model_config(value)


def _tiny_model_config(value: Any) -> dict[str, Any]:
    """Tiny checkpoint必须明确记录全部四项结构设置，不猜测或回退。"""
    from model_tiny import MODEL_CONFIG_KEYS, validate_model_config

    if not isinstance(value, dict) or set(value) != set(MODEL_CONFIG_KEYS):
        raise ValueError("schema 4 model_config必须完整且只能包含四个Tiny架构字段")
    return validate_model_config(value)


def _tiny_inference_precision(checkpoint: Mapping[str, Any]) -> str:
    """将CUDA推理精度绑定到产生验证概率及阈值时的精度。"""
    precision = checkpoint.get("inference_precision")
    if precision not in ("bfloat16", "float16"):
        raise ValueError("schema 4必须记录inference_precision=bfloat16或float16")
    return precision


def _checkpoint_weight_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """校验验证权重与阈值同源；历史schema 1只明确视为raw，不伪造EMA参数。"""
    schema = checkpoint.get("schema_version")
    if type(schema) is not int or schema not in (1, 2, 3, 4):
        raise ValueError("不支持该checkpoint协议版本；仅支持schema 1/2/3/4")
    source_fields = ("weights_source", "validation_weights_source", "threshold_weights_source")
    if schema == 1:
        if any(checkpoint.get(field, "raw") != "raw" for field in source_fields):
            raise ValueError("历史schema 1只支持raw权重，不能降级加载带EMA来源的checkpoint")
        legacy_ema = checkpoint.get("ema")
        if legacy_ema is not None and (
            not isinstance(legacy_ema, dict) or legacy_ema.get("enabled") is not False
        ):
            raise ValueError("历史schema 1不能携带已启用或来源不明的EMA元数据")
        return {**dict.fromkeys(source_fields, "raw"), "ema": None}

    source = checkpoint.get("weights_source")
    if source not in ("raw", "ema") or any(checkpoint.get(field) != source for field in source_fields):
        raise ValueError("schema 2/3/4必须显式记录完全一致的权重、验证及阈值来源：raw或ema")
    if schema in (3, 4) and source != "ema":
        raise ValueError("高分辨率schema 3/4只允许与验证及阈值绑定的EMA权重")
    ema = checkpoint.get("ema")
    if not isinstance(ema, dict):
        raise ValueError("schema 2/3/4必须记录完整EMA元数据，禁止缺省回退为raw")
    if type(ema.get("enabled")) is not bool or ema["enabled"] != (source == "ema"):
        raise ValueError("EMA enabled必须是布尔值，且与weights_source完全一致")
    decay = ema.get("decay")
    if type(decay) not in (int, float) or not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("EMA decay必须是[0,1)内的有限数值")
    if type(ema.get("use_warmup")) is not bool:
        raise ValueError("EMA use_warmup必须是布尔值")
    if "warmup_power" in ema and (
        type(ema["warmup_power"]) not in (int, float)
        or not math.isfinite(ema["warmup_power"]) or ema["warmup_power"] <= 0
    ):
        raise ValueError("EMA warmup_power若有记录，必须是有限正数")
    if schema in (3, 4) and "warmup_power" not in ema:
        raise ValueError("schema 3/4必须记录EMA warmup_power")
    updates = ema.get("updates")
    if type(updates) is not int or updates < 0 or (ema["enabled"] and updates == 0):
        raise ValueError("EMA updates必须为非负整数，启用EMA时必须大于0")
    if ema.get("buffers") != "copy_from_raw":
        raise ValueError("EMA buffers协议必须为copy_from_raw")
    return {**dict.fromkeys(source_fields, source), "ema": dict(ema)}


def load_checkpoint_bundle(
    checkpoint_path: str | Path,
    *,
    thresholds_path: str | Path | None = None,
) -> tuple[dict[str, Any], np.ndarray, str]:
    """仅未来显式调用时读取权重与阈值，并验证两者属于同一次最佳保存。

    权重默认包含阈值，不需要依赖外部JSON才能恢复。如果同目录存在
    thresholds.json，或调用者显式指定了外部JSON，则必须逐一核对；
    schema 2要求权重、验证和阈值均来自同一EMA/raw来源；schema 3/4仅允许
    完整验证的EMA权重，侧文件额外绑定原图预处理；schema 4还绑定AMP精度。
    历史schema 1显式按raw兼容。
    不能将新权重与旧阈值拼在一起后继续推理。weights_only=True只允许
    张量/普通容器等受限内容；本文件不会回退到不受限pickle加载。
    """
    path = Path(checkpoint_path).resolve()
    checkpoint_sha = _sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if _sha256(path) != checkpoint_sha:
        raise RuntimeError("读取过程中checkpoint发生变化，请先停止其写入再加载")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint必须是train.py保存的带协议字段字典")
    weight_metadata = _checkpoint_weight_metadata(checkpoint)
    schema = checkpoint["schema_version"]
    if schema in (3, 4):
        expected_model_name = "convnext_tiny_gap_linear" if schema == 4 else "convnextv2_study_attention"
        if checkpoint.get("model_name") != expected_model_name:
            raise ValueError(f"schema {schema}的模型必须是{expected_model_name}")
        if schema == 4:
            _tiny_model_config(checkpoint.get("model_config"))
            _tiny_inference_precision(checkpoint)
        else:
            _highres_model_config(checkpoint.get("model_config"))
        if checkpoint.get("validation_complete") is not True or checkpoint.get("inference_ready") is not True:
            raise ValueError("schema 3/4仅允许完整验证且inference_ready=True的最佳权重用于推理")
    elif checkpoint.get("model_name") != "efficientnet_b3":
        raise ValueError("历史schema 1/2的权重必须是本工程的efficientnet_b3")
    if type(checkpoint.get("num_classes")) is not int or checkpoint["num_classes"] != NUM_CLASSES:
        raise ValueError("checkpoint输出维度必须为10")
    _class_ids(checkpoint.get("class_ids"))
    epoch = checkpoint.get("epoch")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("checkpoint必须记录从1开始的最佳epoch")
    if checkpoint.get("aggregation") != _AGGREGATION:
        raise ValueError("训练验证必须与推理一致：先Sigmoid，再按Study平均概率")

    preprocessing = checkpoint.get("preprocessing")
    expected_preprocessing = {
        "image_size": 384 if schema in (3, 4) else 300,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "cache_mode": "RGB",
        "eval_transform": "EXIF+RGB+LetterboxLANCZOS+Normalize" if schema in (3, 4) else "Resize+Normalize",
    }
    if schema in (3, 4):
        expected_preprocessing.update({"data_source": "raw_readonly", "tta": "identity"})
    if not isinstance(preprocessing, dict) or any(
        preprocessing.get(key) != value for key, value in expected_preprocessing.items()
    ):
        raise ValueError("权重的尺寸、归一化、数据来源或评估预处理协议不一致")
    if schema in (3, 4) and set(preprocessing) != set(expected_preprocessing):
        raise ValueError("schema 3/4预处理包含本阶段尚未实现的额外设置")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("model_state_dict必须是非空的参数名到Tensor映射")
    for key in ("train_manifest_sha256", "valid_manifest_sha256"):
        if not isinstance(checkpoint.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", checkpoint[key]) is None:
            raise ValueError(f"checkpoint缺少有效的{key}，无法追溯训练数据")
    thresholds = _thresholds(checkpoint.get("thresholds"))

    candidates: list[Path] = []
    sibling = path.parent / "thresholds.json"
    if sibling.exists():
        candidates.append(sibling.resolve())
    if thresholds_path is not None:
        explicit = Path(thresholds_path).resolve()
        if explicit not in candidates:
            candidates.append(explicit)
    for candidate in candidates:
        metadata = _json_object(candidate)
        if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != schema:
            raise ValueError("thresholds.json协议版本与checkpoint不一致")
        if schema in (2, 3, 4) and metadata.get("weights_source") != weight_metadata["weights_source"]:
            raise ValueError("schema 2/3/4 thresholds.json必须显式绑定同源的EMA/raw验证权重")
        if schema == 1 and metadata.get("weights_source", "raw") != "raw":
            raise ValueError("历史schema 1阈值文件只允许绑定raw权重")
        if type(metadata.get("epoch")) is not int or metadata["epoch"] != epoch:
            raise ValueError("thresholds.json的epoch与checkpoint不一致")
        if metadata.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("thresholds.json绑定的权重SHA与当前checkpoint不一致")
        if metadata.get("checkpoint_filename") != path.name:
            raise ValueError("thresholds.json绑定的checkpoint文件名不一致")
        if metadata.get("aggregation") != _AGGREGATION:
            raise ValueError("thresholds.json的Study汇总方式不一致")
        if schema in (3, 4):
            if metadata.get("preprocessing_sha256") != _preprocessing_sha256(preprocessing):
                raise ValueError("schema 3/4 thresholds.json的预处理指纹与权重不一致")
            if "preprocessing" in metadata and metadata["preprocessing"] != preprocessing:
                raise ValueError("schema 3/4 thresholds.json的预处理配置与权重不一致")
        if schema == 4 and metadata.get("inference_precision") != checkpoint["inference_precision"]:
            raise ValueError("schema 4 thresholds.json的AMP精度与checkpoint不一致")
        _class_ids(metadata.get("class_ids"))
        if not np.array_equal(_thresholds(metadata.get("thresholds")), thresholds):
            raise ValueError("thresholds.json十类阈值与checkpoint内部阈值不一致")
    return checkpoint, thresholds, checkpoint_sha


def _study_identity(records: Sequence[dict[str, Any]]) -> tuple[list[str], dict[str, str]]:
    """使用原清单的Study字符串，不转整数，不根据路径猜测或删除前缀。

    Study_key携带患者维度，正式表只有Study_id，因此必须额外确认同一个
    Study_id不会映射到两个不同患者；否则不能静默合并检查。
    """
    keys: list[str] = []
    id_by_key: dict[str, str] = {}
    key_by_id: dict[str, str] = {}
    for record in records:
        key, study_id = record["study_key"], record.get("study_id")
        if not isinstance(study_id, str) or not study_id or study_id != study_id.strip():
            raise ValueError("test清单缺少合法的原始study_id字符串")
        if key in id_by_key and id_by_key[key] != study_id:
            raise ValueError("同一Study_key对应多个Study_id")
        if study_id in key_by_id and key_by_id[study_id] != key:
            raise ValueError("同一Study_id跨患者/Study_key，不能生成只按Study_id提交的表")
        if key not in id_by_key:
            keys.append(key)
        id_by_key[key], key_by_id[study_id] = study_id, key
    return keys, id_by_key


def _new_output_path(output: str | Path, work_root: str | Path = PROJECT_ROOT) -> Path:
    """只允许全新的deliveries子目录，阻止把预测写入已验收数据或图片缓存。"""
    path = Path(output).resolve()
    workspace = Path(work_root).resolve()
    allowed = (workspace / "deliveries").resolve()
    # deliveries自身若被链接到data/缓存/其他磁盘，也不能绕过只读边界。
    if allowed != workspace / "deliveries":
        raise ValueError("deliveries不能是指向数据目录或其他位置的符号链接")
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError("--output必须是work-root/deliveries下的全新子目录")
    if path.exists():
        raise FileExistsError("输出目录已存在；请使用新目录，保留旧提交原貌")
    return path


def _inference_roots(
    work_root: str | Path, raw_root: str | Path | None,
) -> tuple[Path, Path]:
    """解析可搬迁目录；运行输出与日志不能写入只读原图根目录。"""
    from dataset_highres import RAW_ROOT, resolve_image_root

    workspace = Path(work_root).resolve()
    image_root = Path(raw_root if raw_root is not None else RAW_ROOT).resolve()
    if workspace.is_relative_to(image_root):
        raise ValueError("work-root不能等于data-root或位于只读原图目录内")
    # 在首次写WORKLOG前检查可能经cxr_first链接指向外部的真实根目录。
    # 历史B3仍可只保留缓存；找不到原图时此处不要求补齐，schema 3/4另严格检查。
    try:
        effective_root = resolve_image_root(image_root, split="test")
    except FileNotFoundError:
        effective_root = None
    if effective_root is not None and workspace.is_relative_to(effective_root):
        raise ValueError("work-root不得位于解析后的只读原图目录内")
    return workspace, image_root


def _write_csv(path: Path, header: Sequence[str], rows: Any) -> None:
    """UTF-8无BOM；先写临时文件再原子替换，不修改输入清单或缓存。"""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(path)


def validate_submission(
    path: str | Path,
    *,
    study_ids: Sequence[str],
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    include_gate: bool = False,
) -> dict[str, Any]:
    """未来写出后再回读验证：表头、顺序、重复、覆盖、十类和有限概率。

    这只是本地格式/内容自检，不等价于平台已接收，也不会上传文件。
    """
    expected_header = ["Study_id", "Predict_class", "Probability"]
    if include_gate:
        expected_header.append("Gate")
    rows = 0
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != expected_header:
            raise ValueError("submission.csv表头与选择的官方长表格式不一致")
        for row in reader:
            if len(row) != len(expected_header) or rows >= len(study_ids) * NUM_CLASSES:
                raise ValueError("submission.csv列数错误或行数多于预期")
            study_index, class_index = divmod(rows, NUM_CLASSES)
            if row[0] != study_ids[study_index] or row[1] != str(CLASS_IDS[class_index]):
                raise ValueError("submission.csv的Study/类别顺序、覆盖或唯一性错误")
            probability = float(row[2])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("submission.csv存在非有限/越界概率")
            if probability != float(probabilities[study_index, class_index]):
                raise ValueError("CSV概率与内存中的Study汇总概率不一致")
            if include_gate and float(row[3]) != float(thresholds[class_index]):
                raise ValueError("CSV Gate与同epoch最佳阈值不一致")
            rows += 1
    if rows != len(study_ids) * NUM_CLASSES:
        raise ValueError("submission.csv未覆盖全部Study的十类概率")
    return {"rows": rows, "studies": len(study_ids), "classes_per_study": NUM_CLASSES,
            "header": expected_header, "passed": True}


def _validate_wide_table(
    path: Path, study_ids: Sequence[str], expected: np.ndarray, *, binary: bool
) -> None:
    """宽表仅作本地审查，回读确认列序与全部Study覆盖，防止误把它当官方表。"""
    header = ["study_id", *(f"class_{class_id}" for class_id in CLASS_IDS)]
    count = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != header:
            raise ValueError("本地宽表列名或类别顺序错误")
        for row in reader:
            if len(row) != NUM_CLASSES + 1 or count >= len(study_ids) or row[0] != study_ids[count]:
                raise ValueError("本地宽表行数、Study顺序或列数错误")
            for class_index, item in enumerate(row[1:]):
                value = float(item)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("本地宽表存在非法数值")
                if binary and item not in ("0", "1"):
                    raise ValueError("本地判定表必须是0/1，而非未经阈值的概率")
                if value != float(expected[count, class_index]):
                    raise ValueError("本地宽表数值与预测结果不一致")
            count += 1
    if count != len(study_ids):
        raise ValueError("本地宽表遗漏Study")


def _run_inference_impl(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    test_manifest: str | Path = DEFAULT_TEST_MANIFEST,
    thresholds_path: str | Path | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    prefetch_factor: int = 1,
    device: str = "cuda",
    include_gate: bool = False,
    raw_root: str | Path | None = None,
    work_root: str | Path = PROJECT_ROOT,
) -> Path:
    """未来获授权后才显式调用；现在编写该函数不代表已推理或已生成CSV。

    schema 1/2保持300缓存链路；schema 3只读原图384并且每次前向一张；
    schema 4同样使用原图384，Tiny按请求批次与已保存AMP精度推理。
    不构建损失、不读取测试标签，也不在测试集重估权重或阈值。
    """
    workspace, image_root = _inference_roots(work_root, raw_root)
    destination = _new_output_path(output_dir, workspace)
    if destination.is_relative_to(image_root) or image_root.is_relative_to(destination):
        raise ValueError("推理输出目录不得与只读原图目录重叠")
    checkpoint_file = Path(checkpoint_path).resolve()
    manifest = Path(test_manifest).resolve()
    manifest_sha = _sha256(manifest)
    records = read_manifest(manifest, split="test")
    expected_keys, id_by_key = _study_identity(records)
    checkpoint, thresholds, checkpoint_sha = load_checkpoint_bundle(
        checkpoint_file, thresholds_path=thresholds_path
    )
    weight_metadata = _checkpoint_weight_metadata(checkpoint)
    schema = checkpoint["schema_version"]
    tiny = schema == 4
    highres = schema in (3, 4)
    model_config = (
        _tiny_model_config(checkpoint.get("model_config")) if tiny
        else _highres_model_config(checkpoint.get("model_config")) if highres else None
    )
    inference_precision = _tiny_inference_precision(checkpoint) if tiny else "float16" if highres else "float32"
    effective_raw_root = None
    if highres:
        from dataset_highres import resolve_image_root

        # 兼容root/test及root/cxr_first/test两种明确布局，不扫描目录猜路径。
        # 上方仍用用户指定的外层root保护WORKLOG/输出；此处记录真实读取根目录。
        effective_raw_root = resolve_image_root(image_root, split="test")
        if workspace.is_relative_to(effective_raw_root):
            raise ValueError("work-root不得位于解析后的只读原图目录内")
        if destination.is_relative_to(effective_raw_root) or effective_raw_root.is_relative_to(destination):
            raise ValueError("推理输出目录不得与解析后的只读原图目录重叠")
    print(f"=> checkpoint schema={checkpoint['schema_version']}，"
          f"weights_source={weight_metadata['weights_source']}，"
          f"ema={json.dumps(weight_metadata['ema'], ensure_ascii=False, allow_nan=False)}", flush=True)

    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("本脚本只配置了cpu/cuda设备")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定CUDA但当前不可用；不会静默切换到其他设备")
    # 推理已有权重时明确pretrained=False，禁止为创建结构额外下载预训练权重。
    if tiny:
        from model_tiny import build_tiny_model

        model = build_tiny_model(pretrained=False, **model_config)
    elif highres:
        from model_highres import build_highres_model

        model = build_highres_model(pretrained=False, **model_config)
    else:
        model = build_model(pretrained=False)
    # 已保存的model_state_dict就是用于验证/阈值的权重，绝不改读raw备份。
    # 先在CPU恢复并释放checkpoint内的张量副本，再移动唯一推理模型到GPU。
    model.load_state_dict(checkpoint.pop("model_state_dict"), strict=True)
    for unused_tensor_payload in (
        "raw_model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "scaler_state_dict", "ema_state_dict",
    ):
        checkpoint.pop(unused_tensor_payload, None)
    model.to(target_device)
    channels_last = tiny and target_device.type == "cuda"
    if channels_last:
        model.to(memory_format=torch.channels_last)
    model.eval()
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size必须是正整数")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers必须是非负整数")
    if type(prefetch_factor) is not int or prefetch_factor < 1:
        raise ValueError("prefetch_factor必须是正整数")
    effective_batch_size = 1 if schema == 3 else batch_size
    cuda_autocast = highres and target_device.type == "cuda"
    autocast_dtype = torch.bfloat16 if inference_precision == "bfloat16" else torch.float16
    if cuda_autocast and inference_precision == "bfloat16":
        with torch.cuda.device(target_device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("该schema 4权重绑定BF16验证精度，但当前CUDA设备不支持；不会静默改为FP16")
    effective_precision = inference_precision if cuda_autocast else "float32"
    if highres:
        from dataset_highres import make_image_loader

        loader = make_image_loader(
            manifest, split="test", image_size=384, batch_size=effective_batch_size,
            num_workers=num_workers, prefetch_factor=prefetch_factor, training=False,
            pin_memory=target_device.type == "cuda", raw_root=effective_raw_root,
        )
        print(f"=> 原图只读384×384，identity TTA，实际图像batch_size={effective_batch_size}（请求{batch_size}），"
              f"workers={num_workers}，推理精度={effective_precision}，"
              f"保存的验证精度={inference_precision}", flush=True)
        if tiny and target_device.type == "cpu":
            print("=> CPU显式使用FP32；权重和阈值仍来自CUDA混合精度验证，数值可能有细微差异。", flush=True)
    else:
        loader = make_dataloader(
            manifest, split="test", batch_size=effective_batch_size, num_workers=num_workers,
            prefetch_factor=prefetch_factor, pin_memory=target_device.type == "cuda",
            persistent_workers=num_workers > 0, image_size=300,
        )
    aggregator = StudyPredictionAggregator(manifest, split="test")
    with torch.inference_mode():
        for batch in loader:
            if batch["targets"] is not None:
                raise ValueError("测试批次不得包含标签")
            if schema == 3 and batch["images"].shape[0] != 1:
                raise ValueError("schema 3显存协议要求每次推理恰好一张图")
            images = batch["images"].to(target_device, non_blocking=target_device.type == "cuda")
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=target_device.type, dtype=autocast_dtype,
                enabled=cuda_autocast,
            ):
                logits = model(images)
            if logits.ndim != 2 or logits.shape != (len(batch["image_ids"]), NUM_CLASSES):
                raise ValueError("模型必须输出[B,10]原始logits")
            if not torch.isfinite(logits).all().item():
                raise ValueError("模型输出包含NaN/Inf，禁止生成看似正常的提交")
            # 本脚本只在这一处做一次Sigmoid；Aggregator接收概率，不再做Sigmoid。
            probabilities = torch.sigmoid(logits.float())
            aggregator.update(batch, probabilities)
    result = aggregator.finalize()  # 内部要求每张测试图恰好出现一次，尾批不能遗漏。
    if result["targets"] is not None or result["study_keys"] != expected_keys:
        raise ValueError("测试Study覆盖/顺序异常，或意外出现测试标签")
    study_probabilities = np.asarray(result["probabilities"], dtype=np.float64)
    if study_probabilities.shape != (len(expected_keys), NUM_CLASSES):
        raise ValueError("Study级概率形状错误")
    # 必须先平均概率，再按每类阈值判断；不能先二值化视角再平均或多数投票。
    binary = (study_probabilities >= thresholds[None, :]).astype(np.uint8)
    study_ids = [id_by_key[key] for key in expected_keys]
    if _sha256(manifest) != manifest_sha or _sha256(checkpoint_file) != checkpoint_sha:
        raise RuntimeError("推理期间输入清单或checkpoint发生改变，停止输出")

    # 所有计算成功后才创建全新输出目录；永不覆盖验收清单、缓存或已有提交。
    destination.mkdir(parents=True, exist_ok=False)
    submission = destination / "submission.csv"
    header = ["Study_id", "Predict_class", "Probability"] + (["Gate"] if include_gate else [])

    def official_rows() -> Any:
        for study_index, study_id in enumerate(study_ids):
            for class_index, class_id in enumerate(CLASS_IDS):
                row = [study_id, class_id, repr(float(study_probabilities[study_index, class_index]))]
                if include_gate:
                    row.append(repr(float(thresholds[class_index])))
                yield row

    _write_csv(submission, header, official_rows())
    wide_header = ["study_id", *(f"class_{class_id}" for class_id in CLASS_IDS)]
    probability_path, binary_path = destination / "study_probabilities.csv", destination / "study_binary.csv"
    _write_csv(probability_path, wide_header, (
        [study_id, *(repr(float(value)) for value in study_probabilities[index])]
        for index, study_id in enumerate(study_ids)
    ))
    _write_csv(binary_path, wide_header, (
        [study_id, *(int(value) for value in binary[index])]
        for index, study_id in enumerate(study_ids)
    ))
    validation = validate_submission(
        submission, study_ids=study_ids, probabilities=study_probabilities,
        thresholds=thresholds, include_gate=include_gate,
    )
    _validate_wide_table(probability_path, study_ids, study_probabilities, binary=False)
    _validate_wide_table(binary_path, study_ids, binary, binary=True)
    provenance = {
        "schema_version": schema if highres else 2,
        "created_at": datetime.now(_HK_TIMEZONE).isoformat(timespec="seconds"),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_schema_version": checkpoint["schema_version"],
        **weight_metadata,
        "train_manifest_sha256": checkpoint["train_manifest_sha256"],
        "valid_manifest_sha256": checkpoint["valid_manifest_sha256"],
        "test_manifest_sha256": manifest_sha,
        "thresholds": thresholds.tolist(),
        "threshold_source": "best checkpoint; validation Study-level F1 search",
        "aggregation": _AGGREGATION,
        "preprocessing": checkpoint["preprocessing"],
        "preprocessing_sha256": _preprocessing_sha256(checkpoint["preprocessing"]),
        "model_name": checkpoint["model_name"],
        "model_config": model_config,
        "inference_precision": effective_precision,
        "validation_inference_precision": inference_precision,
        "work_root": str(workspace),
        "threshold_comparison": ">=; applied after Study probability averaging",
        "images": len(records), "studies": len(study_ids),
        "study_image_counts": result["image_counts"].tolist(),
        "device": str(target_device),
        "channels_last": channels_last,
        "libraries": {"torch": str(torch.__version__), "numpy": str(np.__version__)},
        "loader": {"batch_size": effective_batch_size, "requested_batch_size": batch_size,
                   "num_workers": num_workers,
                   "prefetch_factor": prefetch_factor if num_workers else None,
                   "image_size": 384 if highres else 300,
                   "data_source": "raw_readonly" if highres else "images300_cache",
                   "requested_image_root": str(image_root) if highres else None,
                   "raw_root": str(effective_raw_root) if effective_raw_root is not None else None,
                   "cuda_autocast_fp16": cuda_autocast and inference_precision == "float16",
                   "cuda_autocast_bf16": cuda_autocast and inference_precision == "bfloat16"},
        "formal_submission": "submission.csv",
        "formal_format": "Study_id,Predict_class,Probability" + (",Gate" if include_gate else ""),
        "include_gate": include_gate,
        "gate_platform_support": "per-class Gate requires platform confirmation" if include_gate else "not used",
        "local_review_tables": ["study_probabilities.csv", "study_binary.csv"],
        "submission_validation": validation,
        "files_sha256": {path.name: _sha256(path) for path in (submission, probability_path, binary_path)},
        "code_sha256": {name: _sha256(PROJECT_ROOT / name) for name in (
            ("dataset.py", "dataset_highres.py", "model_tiny.py", "inference.py") if tiny else
            ("dataset.py", "dataset_highres.py", "model_highres.py", "inference.py")
            if highres else ("dataset.py", "model.py", "inference.py")
        )},
        "platform_upload_performed": False,
    }
    with (destination / "provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return submission


def run_inference(
    *, checkpoint_path: str | Path, output_dir: str | Path,
    test_manifest: str | Path = DEFAULT_TEST_MANIFEST,
    thresholds_path: str | Path | None = None,
    batch_size: int = 64, num_workers: int = 0, prefetch_factor: int = 1,
    device: str = "cuda", include_gate: bool = False,
    raw_root: str | Path | None = None, work_root: str | Path = PROJECT_ROOT,
) -> Path:
    """未来显式调用时同步记录启动、失败和CSV结果；本次没有调用。"""
    workspace, image_root = _inference_roots(work_root, raw_root)
    destination = _new_output_path(output_dir, workspace)
    if destination.is_relative_to(image_root) or image_root.is_relative_to(destination):
        raise ValueError("推理输出目录不得与只读原图目录重叠")
    workspace.mkdir(parents=True, exist_ok=True)

    def record(message: str) -> None:
        timestamp = datetime.now(_HK_TIMEZONE).isoformat(timespec="seconds")
        with (workspace / "WORKLOG.md").open("a", encoding="utf-8") as stream:
            stream.write(f"\n### {timestamp} · 测试推理\n\n- {message}\n")

    record(f"任务：按最佳权重推理并导出完整Study概率；checkpoint={checkpoint_path}；"
           f"输出={destination}；work-root={workspace}；data-root={image_root}；"
           "训练0次，原图/缓存只读，不拟合测试阈值。")
    try:
        submission = _run_inference_impl(
            checkpoint_path=checkpoint_path, output_dir=output_dir,
            test_manifest=test_manifest, thresholds_path=thresholds_path,
            batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor,
            device=device, include_gate=include_gate,
            raw_root=image_root, work_root=workspace,
        )
    except BaseException as exc:
        record(f"失败/中断：{type(exc).__name__}: {exc}；不能将未完成文件视为通过验收，"
               "保留已有证据；下一步排查后使用新输出目录重试。")
        raise
    proof = _json_object(submission.parent / "provenance.json")
    record(f"结果：{proof['images']}图→{proof['studies']}Study，"
           f"正式表{proof['submission_validation']['rows']}行，本地格式校验通过；"
           f"weights_source={proof['weights_source']}，"
           f"ema={json.dumps(proof['ema'], ensure_ascii=False, allow_nan=False)}；"
           f"阈值来自epoch={proof['checkpoint_epoch']}，CSV SHA={proof['files_sha256']['submission.csv']}；"
           f"证据={submission.parent}/provenance.json。下一步人工核对再提交，尚未上传平台。")
    return submission


def main() -> None:
    """命令行入口仅在未来人工显式运行文件时生效；当前没有执行。"""
    # Explicit schema-5 dispatch leaves the historical schema 1..4 parser,
    # defaults, threshold binding and inference implementation unchanged.
    import sys
    # 当前768五折入口的分流点：只要指定ensemble-plan，就直接交给新集成脚本。
    # 下面那些旧schema恢复和单模型推理函数不属于此次教程的主线。
    if any(value == "--ensemble-plan" or value.startswith("--ensemble-plan=") for value in sys.argv[1:]):
        from inference_overnight import main as ensemble_main
        ensemble_main(sys.argv[1:])
        return
    parser = argparse.ArgumentParser(description="十类Study概率推理；兼容Tiny GAP、历史B3及HR384类别注意力EMA协议，须先取得运行授权")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="best.pt；schema 4为Tiny原图384/EMA，schema 3为历史类别注意力384/EMA，兼容schema 1/2 B3")
    parser.add_argument("--output", type=Path, required=True, help="全新work-root/deliveries/子目录，禁止覆盖")
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--image-root", "--data-root", "--raw-root", dest="raw_root", type=Path, default=None,
                        help="schema 3/4只读原图目录：root/test或root/cxr_first/test；默认项目cxr_first/cxr_first")
    parser.add_argument("--work-root", type=Path, default=PROJECT_ROOT,
                        help="输出和WORKLOG所在工作目录，默认源码项目目录；不得位于data-root内")
    parser.add_argument("--thresholds", type=Path, default=None,
                        help="可选外部thresholds.json；协议版本、epoch、SHA及EMA/raw来源必须与checkpoint一致")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Tiny/历史B3图像批大小；schema 3固定每次一张以降低显存")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--device", default="cuda", help="cuda、cuda:0或cpu；不静默回退")
    parser.add_argument("--include-gate", action="store_true", help="可选Gate列；逐类阈值的平台兼容性待确认")
    args = parser.parse_args()
    submission = run_inference(
        checkpoint_path=args.checkpoint, output_dir=args.output,
        test_manifest=args.test_manifest, thresholds_path=args.thresholds,
        batch_size=args.batch_size, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor, device=args.device, include_gate=args.include_gate,
        raw_root=args.raw_root, work_root=args.work_root,
    )
    print(f"已生成并通过本地格式校验：{submission}；未上传平台。")


if __name__ == "__main__":
    main()
