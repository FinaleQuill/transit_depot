"""合规 V1 Base 教师的严格折外图像级 logits 提取；默认只显示计划。

只有显式 --execute 才加载模型和图片。五个教师逐个加载，原始 train 之外
的图片不得进入缓存；旧 V2、外折早停 best、未完成训练的教师一律拒绝。
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from distill_data import (sha256_file, teacher_evidence, validate_teacher_checkpoint,
                          write_soft_labels)


def extract(*, fold_plan: str | Path, teachers_root: str | Path,
            image_root: str | Path, output: str | Path, batch_size: int = 4,
            num_workers: int = 8, amp_dtype: str = "auto") -> Path:
    """显式调用才执行前向；全部 logits 在内存完成后写一个新缓存目录。"""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if type(num_workers) is not int or not 0 <= num_workers <= 12:
        raise ValueError("num_workers must be in 0..12")
    if amp_dtype not in {"auto", "bfloat16", "float16"}:
        raise ValueError("Extraction AMP must be auto, bfloat16 or float16")
    plan_path = Path(fold_plan).resolve(strict=True)
    teacher_root = Path(teachers_root).resolve(strict=True)
    raw_root = Path(image_root).resolve(strict=True)
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"Soft-label output must be new: {destination}")
    if destination == raw_root or destination.is_relative_to(raw_root):
        raise ValueError("Soft labels must not be written into original data")
    from cv_splits import load_fold_records, read_fold_plan
    from dataset import read_manifest

    plan = read_fold_plan(plan_path)
    plan_hash = sha256_file(plan_path)
    records = read_manifest(plan["source_train_manifest"]["path"], split="train")
    image_index = {row["image_id"]: index for index, row in enumerate(records)}
    if len(image_index) != len(records):
        raise ValueError("Train manifest image IDs are duplicated")
    paths = [teacher_root / f"fold_{fold}" / "best.pt" for fold in range(5)]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("All five fixed-epoch fold_0..fold_4/best.pt files are required")
    import torch
    from dataset_overnight import make_overnight_loader
    from model_overnight import build_overnight_model

    if not torch.cuda.is_available():
        raise RuntimeError("Teacher extraction requires a CUDA GPU; no silent CPU fallback")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    bf16 = torch.cuda.is_bf16_supported()
    if amp_dtype == "bfloat16" and not bf16:
        raise ValueError("The selected GPU does not support BF16")
    precision = "bfloat16" if amp_dtype == "bfloat16" or amp_dtype == "auto" and bf16 else "float16"
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    torch.set_num_threads(4)
    values = np.empty((len(records), 10), dtype=np.float32)
    seen: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for fold, checkpoint_path in enumerate(paths):
        digest = sha256_file(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if sha256_file(checkpoint_path) != digest:
            raise RuntimeError("Teacher checkpoint changed while loading")
        validate_teacher_checkpoint(checkpoint, fold=fold, plan=plan,
                                    plan_hash=plan_hash, records=records)
        fold_evidence = teacher_evidence(checkpoint, checkpoint_path=checkpoint_path,
                                         checkpoint_sha256=digest, inference_precision=precision)
        state = checkpoint.pop("model_state_dict")
        config = dict(checkpoint["model_config"])
        del checkpoint
        model = build_overnight_model(pretrained=False, **config)
        model.load_state_dict(state, strict=True)
        del state
        model.set_grad_checkpointing(False)
        model.eval().requires_grad_(False).to(device, memory_format=torch.channels_last)
        held_out = load_fold_records(plan_path, fold, "valid")
        expected = {row["image_id"] for row in held_out}
        fold_seen: set[str] = set()
        loader = make_overnight_loader(held_out, raw_root, training=False,
            image_size=768, batch_size=batch_size, num_workers=num_workers,
            prefetch_factor=1, seed=43)
        try:
            with torch.inference_mode():
                for batch in loader:
                    ids = batch["image_ids"]
                    if (batch["split"] != "train" or len(set(ids)) != len(ids)
                            or set(ids) & seen or not set(ids) <= expected):
                        raise ValueError("Extraction batch leaks folds or original data splits")
                    inputs = batch["images"].to(device, memory_format=torch.channels_last,
                                               non_blocking=True)
                    with torch.autocast("cuda", dtype=dtype):
                        logits = model(inputs)
                    if logits.shape != (len(ids), 10) or not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError("Teacher extraction logits must be finite Nx10")
                    array = logits.float().cpu().numpy()
                    for local_index, image_id in enumerate(ids):
                        values[image_index[image_id]] = array[local_index]
                    fold_seen.update(ids)
                    seen.update(ids)
            if fold_seen != expected:
                raise RuntimeError("Held-out fold image coverage is incomplete")
            evidence.append(fold_evidence)
            print(f"OOF fold={fold} completed images={len(fold_seen)} precision={precision}", flush=True)
        finally:
            del loader, model
            gc.collect()
            torch.cuda.empty_cache()
        if sha256_file(plan_path) != plan_hash or sha256_file(checkpoint_path) != digest:
            raise RuntimeError("Fold plan or teacher changed during OOF extraction")
    if seen != set(image_index):
        raise RuntimeError("OOF extraction does not cover every original train image exactly once")
    write_soft_labels(destination, plan_path, records, values, evidence)
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-plan", type=Path, required=True)
    parser.add_argument("--teachers-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--execute", action="store_true",
                        help="Explicitly authorize loading teachers and running OOF extraction")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"execute": False, "fold_plan": str(args.fold_plan),
            "teachers": [str(args.teachers_root / f"fold_{fold}" / "best.pt") for fold in range(5)],
            "source_split": "train", "image_size": 768, "tta": "identity",
            "output": str(args.output), "next": "Add --execute only after all teachers finish."},
            ensure_ascii=False, indent=2))
        return
    result = extract(fold_plan=args.fold_plan, teachers_root=args.teachers_root,
        image_root=args.image_root, output=args.output, batch_size=args.batch_size,
        num_workers=args.num_workers, amp_dtype=args.amp_dtype)
    print(f"Strict OOF cache complete: {result}", flush=True)


if __name__ == "__main__":
    main()
