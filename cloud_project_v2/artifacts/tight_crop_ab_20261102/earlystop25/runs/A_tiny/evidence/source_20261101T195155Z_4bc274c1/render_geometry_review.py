"""Render a review set from frozen geometry_candidates.csv, without rerunning detection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROJECT = Path(__file__).resolve().parent
CSV_ROOT = Path('/tmp/geometry_audit_20261102')
SOURCE_ROOT = Path('/home/featurize/data/cxr_first')
OUTPUT = PROJECT / 'artifacts/geometry_review_20261102'
VIEWS = ('AP', 'PA', 'Lateral', 'Missing')
CYAN = (24, 224, 235)
RED = (255, 72, 65)
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
SMALL = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)


def number(row, key):
    return float(row[key])


def area_cut(row):
    return 1 - (number(row, 'crop_x1') - number(row, 'crop_x0')) * (number(row, 'crop_y1') - number(row, 'crop_y0'))


def choose(rows):
    """Select distinct CSV rows for twelve complementary review roles."""
    safe = [r for r in rows if r['safe_crop'] == 'True']
    safe.sort(key=lambda r: (area_cut(r), r['image_id']))
    all_rows = sorted(rows, key=lambda r: r['image_id'])
    roles = [
        ('smallest preliminary candidate', safe[:1]),
        ('median preliminary candidate', safe[len(safe) // 2:len(safe) // 2 + 1]),
        ('largest preliminary candidate', safe[-1:]),
        ('widest black border', sorted(all_rows, key=lambda r: -number(r, 'black_margin_max'))),
        ('highest text-like proxy among preliminary candidates', sorted(safe, key=lambda r: -number(r, 'text_like_score'))),
        ('highest irregularity proxy among preliminary candidates', sorted(safe, key=lambda r: -number(r, 'irregularity_score'))),
        ('bright outer-edge risk proxy among preliminary candidates', sorted(safe, key=lambda r: -number(r, 'outer_edge_density'))),
        ('edge consistency risk proxy', sorted(all_rows, key=lambda r: -number(r, 'edge_consistency_ratio'))),
        ('largest top cut', sorted((r for r in safe if number(r, 'crop_y0') > 0), key=lambda r: -number(r, 'crop_y0'))),
        ('largest bottom cut', sorted((r for r in safe if number(r, 'crop_y1') < 1), key=lambda r: number(r, 'crop_y1'))),
        ('largest left cut', sorted((r for r in safe if number(r, 'crop_x0') > 0), key=lambda r: -number(r, 'crop_x0'))),
        ('largest right cut', sorted((r for r in safe if number(r, 'crop_x1') < 1), key=lambda r: number(r, 'crop_x1'))),
    ]
    chosen = []
    used = set()
    for role, ranked in roles:
        pick = next((r for r in ranked if r['image_id'] not in used), None)
        fallback = False
        if pick is None:
            pick = next(r for r in all_rows if r['image_id'] not in used)
            fallback = True
        used.add(pick['image_id'])
        chosen.append((role + (' (fallback: no distinct matching row)' if fallback else ''), pick))
    return chosen


def source_for(row, split):
    root = (SOURCE_ROOT / split).resolve(strict=True)
    source = Path(row['source_path']).resolve(strict=True)
    relative = Path(row['relative_path'])
    if row['split'] != split or relative.is_absolute() or relative.parts[0] != split:
        raise ValueError('CSV split/path mismatch: ' + row['image_id'])
    if not source.is_relative_to(root) or source != (SOURCE_ROOT / relative).resolve(strict=True):
        raise ValueError('Source outside requested split: ' + str(source))
    return source


def hash_file(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def dashed(draw, xy, fill, width=4, dash=14):
    x0, y0, x1, y1 = xy
    length = max(abs(x1-x0), abs(y1-y0))
    if not length:
        return
    for pos in range(0, length, dash * 2):
        end = min(pos + dash, length)
        draw.line((x0 + (x1-x0)*pos/length, y0 + (y1-y0)*pos/length,
                   x0 + (x1-x0)*end/length, y0 + (y1-y0)*end/length), fill=fill, width=width)


def render(row, reason, split, view, ordinal):
    source = source_for(row, split)
    digest = hash_file(source)
    if digest != row['source_file_sha256_manifest']:
        raise ValueError('Source SHA mismatch: ' + str(source))
    with Image.open(source) as opened:
        image = opened.convert('RGB')
    w, h = image.size
    if (w, h) != (int(row['width']), int(row['height'])):
        raise ValueError('Dimensions mismatch: ' + str(source))
    norm = [number(row, 'crop_' + k) for k in ('x0', 'y0', 'x1', 'y1')]
    projected = [norm[0]*w, norm[1]*h, norm[2]*w, norm[3]*h]
    no_cut = all(abs(a-b) < 1e-10 for a,b in zip(norm, (0,0,1,1)))
    csv_box = [int(row['crop_original_' + k]) for k in ('x0', 'y0', 'x1', 'y1')]
    delta = [round(csv_box[i] - projected[i], 3) for i in range(4)]
    if not (0 <= projected[0] < projected[2] <= w and 0 <= projected[1] < projected[3] <= h):
        raise ValueError('Invalid normalized crop: ' + row['image_id'])

    size = (min(650, round(w * min(650/w, 760/h))), min(760, round(h * min(650/w, 760/h))))
    # Each display panel uses the same scale; the full original frame remains visible.
    original = image.resize(size, Image.Resampling.LANCZOS)
    edge = image.convert('L').resize(size, Image.Resampling.LANCZOS).filter(ImageFilter.FIND_EDGES).convert('RGB')
    overlay = original.copy()
    d = ImageDraw.Draw(overlay)
    dw, dh = size
    d.rectangle((1, 1, dw-2, dh-2), outline=CYAN, width=4)
    x0, y0, x1, y1 = [round(v * (dw/w if i % 2 == 0 else dh/h)) for i, v in enumerate(projected)]
    x0, x1 = max(0, x0), min(dw-1, x1)
    y0, y1 = max(0, y0), min(dh-1, y1)
    for segment in ((x0,y0,x1,y0), (x1,y0,x1,y1), (x1,y1,x0,y1), (x0,y1,x0,y0)):
        if (segment[0] <= 4 and segment[2] <= 4 or segment[0] >= dw-5 and segment[2] >= dw-5 or
            segment[1] <= 4 and segment[3] <= 4 or segment[1] >= dh-5 and segment[3] >= dh-5):
            dashed(d, segment, RED)
        else:
            d.line(segment, fill=RED, width=4)
    # External brackets keep coincident full-frame and candidate edges legible.
    margin, top, gap = 16, 130, 20
    canvas_w = 3*dw + 4*margin + 2*gap
    canvas = Image.new('RGB', (canvas_w, top+dh+100), (26, 30, 35))
    cd = ImageDraw.Draw(canvas)
    title = f'{split} / {view} / {ordinal:02d}  {row["image_id"]}  ' + ('NO CUT / full-frame fallback' if no_cut else 'preliminary candidate')
    cd.text((margin, 10), title, font=FONT, fill='white')
    cd.text((margin, 40), reason, font=SMALL, fill='white')
    cd.text((margin, 65), f'Original {w}x{h}  normalized {tuple(round(v,6) for v in norm)}', font=SMALL, fill='white')
    cd.text((margin, 87), f'CSV original pixels {tuple(csv_box)}  CSV - normalized*size {tuple(delta)} px', font=SMALL, fill=(255, 225, 55))
    labels = ('Original: full frame', 'Display-only full-decode Pillow FIND_EDGES (not Canny)', 'CYAN original boundary / RED candidate')
    for i, panel in enumerate((original, edge, overlay)):
        px = margin + i*(dw+gap)
        canvas.paste(panel, (px, top))
        cd.text((px, top+dh+10), labels[i], font=SMALL, fill='white')
    x = margin + 2*(dw+gap)
    cd.line((x-8,top-6,x-8,top+24), fill=CYAN, width=3)
    cd.line((x-8,top-6,x+16,top-6), fill=CYAN, width=3)
    cd.line((x-8,top+dh+6,x-8,top+dh-24), fill=CYAN, width=3)
    cd.line((x-8,top+dh+6,x+16,top+dh+6), fill=CYAN, width=3)
    stem = f'{split}_{view}_{ordinal:02d}_{row["image_id"].replace(":", "_")}'
    out_dir = OUTPUT / split / view
    out_dir.mkdir(parents=True, exist_ok=True)
    main = out_dir / (stem + '.jpg')
    canvas.save(main, quality=93, subsampling=0)
    strips = []
    cuts = {'left': round(projected[0]), 'top': round(projected[1]),
            'right': w-round(projected[2]), 'bottom': h-round(projected[3])}
    for side, cut in cuts.items():
        if cut <= 0:
            continue
        reach = min(160, (w if side in ('left','right') else h)-cut)
        if side == 'left':
            box, boundary = (0,0,cut+reach,h), cut
        elif side == 'right':
            box, boundary = (w-cut-reach,0,w,h), reach
        elif side == 'top':
            box, boundary = (0,0,w,cut+reach), cut
        else:
            box, boundary = (0,h-cut-reach,w,h), reach
        raw = image.crop(box).convert('RGB')
        strip = raw.convert('RGBA')
        shade = Image.new('RGBA', strip.size, (0, 0, 0, 0))
        md = ImageDraw.Draw(shade)
        sw, sh = strip.size
        if side == 'left': md.rectangle((0,0,boundary,sh), fill=RED + (65,))
        if side == 'right': md.rectangle((boundary,0,sw,sh), fill=RED + (65,))
        if side == 'top': md.rectangle((0,0,sw,boundary), fill=RED + (65,))
        if side == 'bottom': md.rectangle((0,boundary,sw,sh), fill=RED + (65,))
        strip = Image.alpha_composite(strip, shade)
        sd = ImageDraw.Draw(strip)
        if side in ('left','right'): sd.line((boundary,0,boundary,sh), fill=RED, width=4)
        else: sd.line((0,boundary,sw,boundary), fill=RED, width=4)
        strip_path = out_dir / (stem + '_' + side + '_1to1.png')
        raw_path = out_dir / (stem + '_' + side + '_raw_1to1.png')
        raw.save(raw_path)
        strip.convert('RGB').save(strip_path)
        strips.append({'side': side, 'cut_px_rounded_from_normalized': cut,
                       'source_box_1to1': box, 'file': str(strip_path.relative_to(OUTPUT)),
                       'raw_file': str(raw_path.relative_to(OUTPUT))})
    if hash_file(source) != digest:
        raise ValueError('Source changed during render: ' + str(source))
    return {'image_id':row['image_id'], 'split':split, 'view':view, 'selection_reason':reason,
            'source_path':str(source), 'relative_path':row['relative_path'], 'source_sha256':digest,
            'original_size':[w,h], 'normalized_box':norm, 'projected_original_box':projected,
            'csv_original_pixel_box':csv_box, 'csv_minus_projected_px':delta,
            'safe_crop_csv':row['safe_crop']=='True', 'no_cut_full_frame':no_cut,
            'crop_reason_csv':row['crop_reason'],
            'main':str(main.relative_to(OUTPUT)), 'strips':strips}


def write_index(records, complete, input_shas):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance = {'input_csv_sha256': input_shas, 'renderer_sha256': hash_file(Path(__file__))}
    (OUTPUT / 'index.json').write_text(json.dumps({'complete':complete, 'count':len(records),
                                                    'provenance':provenance, 'items':records}, indent=2)+'\n')
    lines = ['# Geometry review: frozen preliminary candidates', '',
             'Originals are full frame. Cyan marks the original boundary; red marks the CSV normalized candidate.',
             'The edge panel is display-only full-decode Pillow FIND_EDGES, not Canny or a pixel-identical replay of audit JPEG draft. Red tinted 1:1 strips show removed source pixels.',
             'CSV pixel coordinates and their difference from normalized coordinates are printed on each image.',
             'These are visual review candidates, not medical safety judgments.', '']
    lines.extend(['NO CUT / full-frame fallback means the frozen CSV proposes no removal; such images remain useful for comparison.', ''])
    lines.extend([f'- `{key}` CSV SHA256: `{sha}`' for key, sha in input_shas.items()])
    lines.extend([f'- Renderer SHA256: `{provenance["renderer_sha256"]}`', ''])
    for item in records:
        lines.extend([f'## {item["split"]} / {item["view"]} / {item["image_id"]}', '',
                      f'- Reason: {item["selection_reason"]}',
                      f'- Frame status: {"NO CUT / full-frame fallback" if item["no_cut_full_frame"] else "preliminary candidate with cut"}',
                      f'- Original: {item["original_size"]}; SHA256: `{item["source_sha256"]}`',
                      f'- Normalized box: `{item["normalized_box"]}`',
                      f'- CSV original pixel box: `{item["csv_original_pixel_box"]}`; error px: `{item["csv_minus_projected_px"]}`',
                      f'- [Review image]({item["main"]})',
                      *[f'- [{s["side"]} 1:1 strip]({s["file"]}) / [raw pixels]({s["raw_file"]})' for s in item['strips']], ''])
    (OUTPUT / 'index.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-six', action='store_true')
    args = parser.parse_args()
    groups = {}
    input_shas = {}
    for split in ('train','valid'):
        csv_path = CSV_ROOT / split / 'geometry_candidates.csv'
        input_shas[split] = hash_file(csv_path)
        with csv_path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        for view in VIEWS:
            subset = [r for r in rows if r['view_group'] == view]
            if len(subset) < 12:
                raise ValueError(f'Insufficient {split}/{view}: {len(subset)}')
            groups[(split,view)] = choose(subset)
    records = []
    for split, view in groups:
        for ordinal, (reason, row) in enumerate(groups[(split,view)], 1):
            if args.first_six and (split != 'train' or view not in ('AP','PA','Lateral') or ordinal not in (2,3)):
                continue
            records.append(render(row, reason, split, view, ordinal))
    if len({r['source_path'] for r in records}) != len(records):
        raise ValueError('Duplicate source images in review set')
    write_index(records, not args.first_six, input_shas)
    print(f'{len(records)} review images: {OUTPUT / "index.md"}')


if __name__ == '__main__':
    main()
