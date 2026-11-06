"""Train-only visual review of conservative text-like border candidates.

This detector does not establish that a region is text, a watermark, outside the
body, or safe to erase. It uses no OCR, weights or image-label fitting. It never
changes a training image or preprocessing path. Bright devices and anatomy can
be false positives; text overlapping the chest can be missed deliberately.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

DEFAULT_ROOT = Path('/home/featurize/work/reconstruction_runs_20261103')
SEED = 'watermark-review-20261103-v1'


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_training_subset(manifest, split_plan):
    """Bind the only accepted manifest to the optimization train role, never H/dev."""
    manifest, split_plan = Path(manifest).resolve(strict=True), Path(split_plan).resolve(strict=True)
    plan = json.loads(split_plan.read_text())
    role = plan['roles']['train']
    content = manifest.read_bytes()
    if manifest != Path(role['path']).resolve() or sha_bytes(content) != role['sha256']:
        raise ValueError('Only the SHA-bound optimization train.csv is accepted')
    if plan.get('train_folds') != [1, 2, 3] or plan.get('holdout_fold') != 4 or plan.get('dev_fold') != 0:
        raise ValueError('Unexpected preregistered train/dev/H partition')
    rows = list(csv.DictReader(content.decode('utf-8-sig').splitlines()))
    labels, image_ids = {}, set()
    for row in rows:
        if row['split'] != 'train' or row.get('training_eligible', '').lower() != 'true':
            raise ValueError('Only eligible original training records are accepted')
        if row['study_key'] != f"{row['subject_id']}/{row['study_id']}" or row['image_id'] in image_ids:
            raise ValueError('Invalid training identity')
        image_ids.add(row['image_id'])
        target = json.loads(row['targets'])
        if len(target) != 10 or any(type(v) not in (int, float) or v not in (0, 1) for v in target):
            raise ValueError('Original ten-class binary labels required')
        if labels.setdefault(row['study_key'], target) != target:
            raise ValueError('Conflicting Study labels')
        row['targets'] = target
    if len(rows) != role['images'] or len(labels) != role['studies']:
        raise ValueError('Training subset coverage differs from its registered role')
    return rows, {'manifest': str(manifest), 'manifest_sha256': sha_bytes(content),
                  'split_plan': str(split_plan), 'split_plan_sha256': sha_bytes(split_plan.read_bytes())}


def view_group(row):
    view = row.get('view_position', '').upper()
    if view in ('LATERAL', 'LL'):
        return 'Lateral'
    if view.startswith('PA'):
        return 'PA'
    if view == 'AP':
        return 'AP'
    return 'Missing'


def label_stratum(row):
    y = row['targets']
    return 'class1_positive' if y[0] else 'class4_positive_class1_negative' if y[3] else 'class1_and_4_negative'


def select_sample(rows):
    """48 images, globally unique patients; deterministic quotas without replacement."""
    strata = [('class1_positive', 4), ('class4_positive_class1_negative', 4), ('class1_and_4_negative', 8)]
    selected, used, shortages = [], set(), []
    # Positive quotas first across all views so negative sampling cannot consume
    # a scarce positive patient's other image. Hashes do not use pixel content.
    for stratum, quota in strata:
        for view in ('AP', 'PA', 'Lateral'):
            candidates = [r for r in rows if view_group(r) == view and label_stratum(r) == stratum]
            candidates.sort(key=lambda r: sha_bytes(f"{SEED}/{r['subject_id']}/{r['image_id']}".encode()))
            chosen = []
            for row in candidates:
                if row['subject_id'] not in used:
                    chosen.append(row); used.add(row['subject_id'])
                if len(chosen) == quota:
                    break
            if len(chosen) < quota:
                shortages.append({'view': view, 'stratum': stratum, 'requested': quota, 'selected': len(chosen)})
            selected.extend(chosen)
    selected.sort(key=lambda r: (('AP','PA','Lateral').index(view_group(r)), label_stratum(r), r['image_id']))
    return selected, shortages


def _near_edge(x, y, w, h, width, height, fraction=.18):
    return (x+w/2 < width*fraction or x+w/2 > width*(1-fraction)
            or y+h/2 < height*fraction or y+h/2 > height*(1-fraction))


def text_candidates(gray):
    """Fixed contrast/shape heuristic; outputs evidence candidates, never an erase mask.

    Thresholds apply to analysis images whose longest edge is at most 768. Only
    edge bands are screened; they are not claimed to be outside anatomy. No body
    or lung segmentation is inferred and no Otsu formula is used.
    """
    cv2.setNumThreads(1)
    if gray.dtype != np.uint8 or gray.ndim != 2 or min(gray.shape) < 8:
        raise ValueError('Expected a nontrivial uint8 grayscale analysis image')
    height, width = gray.shape
    kernel_size = max(9, int(round(max(height, width)/768*21)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
    bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    background = cv2.GaussianBlur(gray, (kernel_size*2+1,kernel_size*2+1), 0)
    # A dark glyph needs a bright surrounding field. Otherwise black-hat gaps
    # between white letters on a dark field fill the whole word into a rectangle.
    signal = (((bright >= 28) & (gray >= 160))
              | ((dark >= 40) & (gray <= 95) & (background >= 140))).astype(np.uint8)
    # Only retain edge-band contrast before components; central text is out of
    # scope for this deliberately conservative first review.
    edge = np.zeros_like(signal)
    by, bx = max(1,round(height*.20)), max(1,round(width*.20))
    edge[:by] = 1; edge[-by:] = 1; edge[:,:bx] = 1; edge[:,-bx:] = 1
    signal *= edge
    count, components, stats, centroids = cv2.connectedComponentsWithStats(signal, 8)
    scale = max(height, width)/768
    characters = []
    for component in range(1,count):
        x,y,w,h,area = map(int,stats[component])
        # Wide continuous edges, broad opacities and long catheter-like lines
        # are rejected as single components. Small wire fragments remain a
        # documented possible false positive when they form aligned groups.
        if (not _near_edge(x,y,w,h,width,height) or area < max(3,5*scale*scale)
                or area > 900*scale*scale or h < max(3,5*scale) or h > 55*scale
                or w < 2 or w > 65*scale or not .12 <= w/h <= 3.2
                or not .07 <= area/(w*h) <= .88):
            continue
        contrast = float(np.maximum(bright,dark)[components==component].mean())
        if contrast < 35:
            continue
        characters.append({'component':component,'box':[x,y,x+w,y+h], 'area':area,
                           'height':h,'width':w,'cx':float(centroids[component,0]),
                           'cy':float(centroids[component,1]),'contrast':contrast})
    # Join neighboring glyph-shaped components only when baseline/height agree.
    parent = list(range(len(characters)))
    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i,a in enumerate(characters):
        for j in range(i):
            b=characters[j]
            h=max(a['height'],b['height'])
            gap=max(0,max(a['box'][0],b['box'][0])-min(a['box'][2],b['box'][2]))
            if (min(a['height'],b['height'])/h >= .45 and abs(a['cy']-b['cy']) <= .45*h
                    and gap <= 2.3*h and abs(a['cx']-b['cx']) <= 5*h):
                parent[find(i)] = find(j)
    groups = {}
    for i,character in enumerate(characters):
        groups.setdefault(find(i),[]).append(character)
    retained, ids = [], []
    for chars in groups.values():
        if len(chars) < 3:
            continue
        box=[min(c['box'][0] for c in chars), min(c['box'][1] for c in chars),
             max(c['box'][2] for c in chars), max(c['box'][3] for c in chars)]
        x0,y0,x1,y1=box
        if (x1-x0) < (y1-y0)*1.3 or (x1-x0)*(y1-y0) > width*height*.07:
            continue
        retained.append({'box_analysis_xyxy':box,'components':len(chars),
                         'mean_local_contrast':float(np.mean([c['contrast'] for c in chars])),
                         'interpretation':'aligned edge contrast components; unconfirmed text/device/anatomy'})
        ids.extend(c['component'] for c in chars)
    retained.sort(key=lambda item:(item['box_analysis_xyxy'][1],item['box_analysis_xyxy'][0]))
    candidate_pixels=np.isin(components,ids).astype(np.uint8)*255 if ids else np.zeros_like(gray)
    return retained,candidate_pixels,{'edge_band_fraction':.20,'components_after_shape_filter':len(characters),
                                    'candidate_groups':len(retained),'candidate_pixel_fraction':float((candidate_pixels>0).mean())}


def preview(image, candidates, mask, title):
    rgb=image.convert('RGB')
    overlay=np.asarray(rgb).copy()
    selected=mask>0
    overlay[selected]=(np.asarray(overlay[selected],dtype=np.float32)*.35+np.asarray([255,70,20])*.65).astype(np.uint8)
    marked=Image.fromarray(overlay)
    draw=ImageDraw.Draw(marked)
    for item in candidates:
        draw.rectangle(tuple(item['box_analysis_xyxy']),outline=(255,140,0),width=2)
    canvas=Image.new('RGB',(rgb.width*2, rgb.height+48),'white')
    canvas.paste(rgb,(0,48));canvas.paste(marked,(rgb.width,48))
    draw=ImageDraw.Draw(canvas)
    draw.text((8,4),title,fill='black')
    draw.text((8,24),'Original',fill='black')
    draw.text((rgb.width+8,24),'Unconfirmed candidates; pixels retained',fill='black')
    return canvas


def run_review(train_manifest, split_plan, image_root, output):
    rows, binding=read_training_subset(train_manifest,split_plan)
    selected,shortages=select_sample(rows)
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new review directory; previous evidence is immutable')
    output.mkdir(parents=True,exist_ok=False)
    (output/'cases').mkdir()
    image_root=Path(image_root).resolve(strict=True)
    if not (image_root/'train').is_dir() and (image_root/'cxr_first/train').is_dir():
        image_root=image_root/'cxr_first'
    items, thumbnails=[],[]
    for i,row in enumerate(selected):
        relative=Path(row['relative_path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[:3]!=('train',row['subject_id'],row['study_id']):
            raise ValueError('Source path does not match the accepted train identity')
        path=(image_root/relative).resolve(strict=True)
        if not path.is_relative_to(image_root/'train'):
            raise ValueError('Image path leaves official training root')
        content=path.read_bytes()
        if sha_bytes(content)!=row['file_sha256']:
            raise ValueError('Source image hash differs from accepted train row')
        with Image.open(BytesIO(content)) as source:
            source.load(); source=ImageOps.exif_transpose(source)
            original_size=source.size
            gray=source.convert('L')
        gray.thumbnail((768,768),Image.Resampling.LANCZOS)
        candidates,mask,stats=text_candidates(np.asarray(gray))
        case=f'{i+1:02d}_{row["image_id"].replace(":","_")}'
        title=f'{i+1:02d} {view_group(row)} C1={row["targets"][0]} C4={row["targets"][3]} candidates={len(candidates)}'
        rendered=preview(gray,candidates,mask,title)
        rendered.save(output/'cases'/f'{case}.jpg',quality=92)
        Image.fromarray(mask).save(output/'cases'/f'{case}_candidate_pixels.png')
        thumb=rendered.copy();thumb.thumbnail((480,290),Image.Resampling.LANCZOS)
        tile=Image.new('RGB',(480,290),(230,230,230));tile.paste(thumb,((480-thumb.width)//2,0));thumbnails.append(tile)
        items.append({'number':i+1,'image_id':row['image_id'],'subject_id':row['subject_id'],
                      'study_key':row['study_key'],'view':view_group(row),'stratum':label_stratum(row),
                      'class1':row['targets'][0],'class4':row['targets'][3],'relative_path':str(relative),
                      'source_sha256':row['file_sha256'],'source_hash_matched':True,
                      'original_dimensions':list(original_size),'analysis_dimensions':list(gray.size),
                      'preview':f'cases/{case}.jpg','candidate_pixels':f'cases/{case}_candidate_pixels.png',
                      'candidates':candidates,'statistics':stats})
    pages=[]
    for page_start in range(0,len(thumbnails),12):
        page=Image.new('RGB',(1440,1200),'white');draw=ImageDraw.Draw(page)
        draw.text((12,8),'TRAIN ONLY: original | unconfirmed text-like border candidates. No pixels removed.',fill='black')
        for j,tile in enumerate(thumbnails[page_start:page_start+12]):
            page.paste(tile,((j%3)*480,40+(j//3)*290))
        name=f'contact_{page_start//12+1:02d}.jpg';page.save(output/name,quality=92);pages.append(name)
    summary={'schema':'train_border_text_review_v1', 'scope':'optimization train folds1-3 only', **binding,
             'selection_seed':SEED,'sample_images':len(items),'sample_patients':len({x['subject_id'] for x in items}),
             'sample_selection':'AP/PA/Lateral each class1+ 4, class4+ class1- 4, both- 8; unique patients',
             'shortages':shortages,'view_counts':dict(Counter(x['view'] for x in items)),
             'images_with_candidates':sum(bool(x['candidates']) for x in items),
             'candidate_groups':sum(len(x['candidates']) for x in items),
             'candidate_pixel_fraction_max':max([x['statistics']['candidate_pixel_fraction'] for x in items],default=0),
             'contact_sheets':pages,'source_code_sha256':sha_bytes(Path(__file__).read_bytes()),
             'all_source_hashes_matched':all(x['source_hash_matched'] for x in items),
             'model_or_ocr_used':False,'gpu_used':False,'training_changed':False,'images_modified':False,
             'holdout_or_dev_images_read':False,'thresholds_fitted':False,
             'limitations':['Text-like candidates are not confirmed watermarks or safe erase regions.',
                            'Devices/anatomy can resemble aligned bright/dark components; retain original pixels.',
                            'Only edge-band horizontal multi-component groups are screened; central/isolated text may be missed.',
                            'This stratified patient-balanced sample does not estimate full-train prevalence or model reliance.'],
             'next_step':'Human review first. If supported, preregister train-only low-probability border overlay/occlusion augmentation without editing source labels/images; compare against matched baseline on development patients.'}
    (output/'review.json').write_text(json.dumps({'summary':summary,'images':items},indent=2)+'\n')
    lines=['# Train-only text-candidate review','',
           'These boxes are unconfirmed contrast/shape candidates, not verified watermarks. No training pixels were removed.',
           '',f'{len(items)} images / {summary["sample_patients"]} distinct patients; {summary["images_with_candidates"]} images with candidates.',
           'The sample is stratified and does not estimate prevalence. No H/dev/test images, OCR, model weights or GPU were used.','']
    lines += [f'- [{p}]({p})' for p in pages]
    lines += ['', 'Possible future experiment: after review, add low-probability synthetic border text overlays or narrowly limited border occlusion on TRAIN only, while preserving the original image branch. Use a fixed train-selected mask policy; never infer labels from text. Broad white-pixel removal and automatic masking of these boxes are not justified.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-manifest',type=Path,default=DEFAULT_ROOT/'splits/train.csv')
    parser.add_argument('--split-plan',type=Path,default=DEFAULT_ROOT/'splits/split_plan.json')
    parser.add_argument('--image-root',type=Path,default=Path('/home/featurize/data/cxr_first'))
    parser.add_argument('--output',type=Path,default=DEFAULT_ROOT/'watermark_review')
    args=parser.parse_args()
    print(json.dumps(run_review(**vars(args)),indent=2))


if __name__=='__main__':
    main()
