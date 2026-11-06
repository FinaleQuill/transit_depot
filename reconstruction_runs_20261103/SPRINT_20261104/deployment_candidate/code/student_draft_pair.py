"""A separately declared reduced JPEG decode candidate, evaluated on valid only.

This changes pixels and must pass a new complete validation. It never replaces
the full-decode B submission or claims numerical equivalence with that version.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from deploy_kd import aggregate_member_probabilities, _letterbox_image
from student_fixed_pair import FixedStudentPair


def draft_image_pixels(path, image_sizes):
    if list(image_sizes) != [768]:
        raise ValueError('The single registered decoder candidate uses 768 input')
    with Image.open(path) as source:
        if source.format == 'JPEG' and source.mode in ('L', 'RGB'):
            source.draft(source.mode, (768, 768))
        source.load()
        ImageOps.exif_transpose(source, in_place=True)
        if source.mode in {'I', 'F'} or source.mode.startswith('I;'):
            raise ValueError('Non-8-bit image requires an explicit intensity policy')
        image = source if source.mode == 'L' else source.convert('RGB')
        mode = 'L' if image.mode == 'L' else 'RGB'
        return {768: np.asarray(_letterbox_image(image, 768, mode))}


class DraftStudentPair(FixedStudentPair):
    def __init__(self, model_dir):
        super().__init__(model_dir)
        if (self.config['candidate'] != 'AB_B100_class1'
                or self.config.get('decoder_candidate') != 'JPEG_draft_target768'):
            raise ValueError('Require the separately declared fixed draft decoder candidate')
        views = self.config.get('view_candidate')
        if views is not None:
            if views != 'A_identity_B_hflip2':
                raise ValueError('Unknown fixed per-model view candidate')
            self.members = [{**self.members[0], 'tta': 'identity'}, self.members[1]]
            self.config = {**self.config, 'tta': 'mixed', 'members': self.members}

    def _predict_image(self, path: str | Path):
        from inference_inputs import member_results_to_cpu
        torch = self.torch
        pixels = draft_image_pixels(path, [768])
        results, finite_checks = [], []
        # The following TTA/member arithmetic is the original Predictor sequence.
        # Only the image decoder above differs in this isolated candidate.
        with torch.inference_mode():
            inputs = {768: self.input_lookup.to_tensor(pixels[768])}
            transformed = {(768, 'identity'): inputs[768]}
            for member, model in zip(self.members, self.models):
                probabilities = []
                views = ('identity', 'hflip') if member['tta'] == 'hflip2' else ('identity',)
                for view in views:
                    key = (768, view)
                    if key not in transformed:
                        transformed[key] = torch.flip(inputs[768], dims=(-1,)).contiguous(
                            memory_format=torch.channels_last)
                    logits = model(transformed[key])
                    if tuple(logits.shape) != (1, 10):
                        raise FloatingPointError('Invalid single-image student logits')
                    finite_checks.append(torch.isfinite(logits).all())
                    probabilities.append(torch.sigmoid(logits.float())[0])
                results.append(torch.stack(probabilities).mean(0))
            results = member_results_to_cpu(torch, results, finite_checks)
        return aggregate_member_probabilities(results, self.model_weights)
