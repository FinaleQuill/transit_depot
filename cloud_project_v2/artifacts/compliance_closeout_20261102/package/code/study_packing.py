"""Batch complete Studies for forward execution; retain equal-Study objectives."""
from __future__ import annotations

import math
import time

import torch
from torch import nn


def complete_study_packs(loader, *, max_images, accumulation):
    """Never split a Study or cross a gradient-update window.

    A Study larger than the preferred image cap is emitted intact on its own.
    """
    if type(max_images) is not int or max_images < 1 or accumulation < 1:
        raise ValueError('Invalid Study forward-packing limits')
    pending, count = [], 0
    for index, batch in enumerate(loader):
        images = int(batch['images'].shape[0])
        if images < 1:
            raise ValueError('Cannot pack an empty Study')
        if pending and (count + images > max_images or
                        index // accumulation != pending[0][0] // accumulation):
            yield pending
            pending, count = [], 0
        pending.append((index, batch))
        count += images
        if count >= max_images or (index + 1) % accumulation == 0:
            yield pending
            pending, count = [], 0
    if pending:
        yield pending


def train_packed_study_epoch(model, ema, loader, optimizer, scaler, device, dtype,
                             epoch, config, progress, cache, student_weights):
    from study_student import study_distillation_losses, accumulation_window_size
    from train_distill import alpha_at
    from train_overnight import set_learning_rates

    model.train()
    expected = {}
    for row in loader.dataset.records:
        expected.setdefault(row['study_key'], set()).add(row['image_id'])
    expected_images = {row['image_id'] for row in loader.dataset.records}
    seen_images, seen_studies = set(), set()
    total = hard_total = soft_total = 0.
    total_studies = len(loader)
    windows = math.ceil(total_studies / config.accumulation_steps)
    updates_before = progress['optimizer_updates']
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    packs, alpha = 0, config.alpha
    for entries in complete_study_packs(loader, max_images=config.pack_images,
                                        accumulation=config.accumulation_steps):
        for index, batch in entries:
            ids, keys = batch['image_ids'], set(batch['study_keys'])
            if (len(keys) != 1 or len(set(ids)) != len(ids)
                    or seen_images.intersection(ids) or seen_studies.intersection(keys)):
                raise ValueError('Packed training repeated an image or mixed/repeated Studies')
            key = next(iter(keys))
            if key not in expected or set(ids) != expected[key]:
                raise ValueError('Packed training must retain every original view in the Study')
            seen_images.update(ids)
            seen_studies.add(key)
        images = torch.cat([batch['images'] for _, batch in entries]).to(
            device, memory_format=torch.channels_last, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=dtype):
            logits = model(images)
        offset, hard_terms, soft_terms, loss_terms = 0, [], [], []
        for index, batch in entries:
            count = len(batch['image_ids'])
            targets = batch['targets'].to(device, dtype=torch.float32, non_blocking=True)
            positions = [cache.index[image_id] for image_id in batch['image_ids']]
            teacher = torch.as_tensor(cache.logits[positions], device=device, dtype=torch.float32)
            teacher_weights = torch.as_tensor(cache.teacher_pos_weights[positions],
                                              device=device, dtype=torch.float32)
            hard, soft = study_distillation_losses(logits[offset:offset+count], targets,
                teacher, teacher_weights, student_weights, temperature=config.temperature,
                align=config.align_teacher_pos_weight)
            alpha = alpha_at(config, epoch, index, total_studies)
            hard_terms.append(hard)
            soft_terms.append(soft)
            loss_terms.append((1 - alpha) * hard + alpha * soft)
            offset += count
        raw_loss = torch.stack(loss_terms).sum()
        if not bool(torch.isfinite(raw_loss)):
            raise FloatingPointError('Nonfinite packed Study loss')
        denominator = accumulation_window_size(entries[0][0], total_studies, config.accumulation_steps)
        scaler.scale(raw_loss / denominator).backward()
        progress['backward_batches'] += 1
        packs += 1
        total += float(raw_loss.detach())
        hard_total += float(torch.stack(hard_terms).sum().detach())
        soft_total += float(torch.stack(soft_terms).sum().detach())
        last_index = entries[-1][0]
        if (last_index + 1) % config.accumulation_steps == 0 or last_index + 1 == total_studies:
            rates = set_learning_rates(optimizer, config, progress['optimizer_updates'], windows)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                     error_if_nonfinite=not scaler.is_enabled())
            before = progress['optimizer_updates']
            scaler.step(optimizer)
            scaler.update()
            if progress['optimizer_updates'] == before:
                progress['amp_skipped_steps'] += 1
            optimizer.zero_grad(set_to_none=True)
        first_index = entries[0][0]
        if first_index == 0 or (last_index + 1) // 100 > first_index // 100:
            print(f'packed_study epoch={epoch} studies={last_index+1}/{total_studies} '
                  f'images={len(seen_images)} packs={packs} loss={total/(last_index+1):.6f} '
                  f'alpha={alpha:.4f}', flush=True)
    if seen_images != expected_images or seen_studies != set(expected):
        raise RuntimeError('Packed epoch omitted an original image or complete Study')
    if (progress['optimizer_updates'] <= updates_before
            or progress['optimizer_updates'] != progress['ema_updates']):
        raise RuntimeError('Packed student optimizer and EMA did not advance together')
    return {'loss': total / total_studies, 'hard_bce': hard_total / total_studies,
        'soft_kl_T2': soft_total / total_studies, 'alpha_end': alpha,
        'images': len(seen_images), 'studies': len(seen_studies), 'full_manifest_coverage': True,
        'effective_batch_unit': 'Study', 'effective_batch_size': config.accumulation_steps,
        'forward_pack_preferred_images': config.pack_images, 'forward_packs': packs,
        'seconds': time.monotonic()-started, 'learning_rates': rates,
        'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30}
