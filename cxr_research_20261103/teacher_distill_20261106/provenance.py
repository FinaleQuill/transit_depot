"""Anonymous, pinned and resumable public initialization asset acquisition.

No model is created or run. Does not read Hugging Face tokens, use an
authenticated endpoint, or access any competition images.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import requests


ASSET_ROOT = Path('/home/featurize/data/kd_rebuild_20261106/assets')
MODELS = (
    dict(key='dinov3_l', model_id='timm/vit_large_patch16_dinov3.lvd1689m',
         revision='30c1109559f65dea34316b0d4842d35c5771fe11',
         parameters=303079424, expected_bytes=1212347640,
         license='dinov3-license',
         license_notes='Non-standard DINOv3 License; preserve full text for event review',
         license_files={'LICENSE.md': 'LICENSE.md'}),
    dict(key='convnext_large', model_id='timm/convnext_large.fb_in22k_ft_in1k_384',
         revision='1c3c222eca50028b09e888e12746ac10f2403a36',
         parameters=197767336, expected_bytes=791103952,
         license='apache-2.0',
         license_notes='Apache-2.0 per timm weight model card; upstream ConvNeXt code MIT',
         license_files={}),
)


def now():
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, record):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def session():
    # requests does not automatically load HF credentials. Disable .netrc auth
    # explicitly while retaining the environment's required network proxy.
    class AnonymousAuth(requests.auth.AuthBase):
        def __call__(self, request):
            request.headers.pop('Authorization', None)
            return request
    result = requests.Session()
    result.auth = AnonymousAuth()
    return result


def fetch_small(client, url, path):
    response = client.get(url, timeout=(20, 90))
    response.raise_for_status()
    path.write_bytes(response.content)
    return dict(url=url, bytes=path.stat().st_size, sha256=digest(path),
                response_status=response.status_code)


def download_weights(client, url, destination, expected_bytes, status_path, record):
    partial = destination.with_suffix(destination.suffix + '.part')
    if destination.exists():
        if destination.stat().st_size != expected_bytes:
            raise ValueError(f'Existing asset size differs: {destination}')
        return dict(url=url, bytes=expected_bytes, sha256=digest(destination),
                    completed_at=now(), resumed_existing_complete=True)
    failures = []
    for attempt in range(1, 7):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        try:
            with client.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                if response.status_code == 416 and offset == expected_bytes:
                    partial.replace(destination)
                    break
                response.raise_for_status()
                if offset and response.status_code != 206:
                    # A server ignoring Range supplies the full payload: safely
                    # start this partial file over rather than append twice.
                    offset = 0
                if offset and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected Content-Range for resumption')
                mode = 'ab' if offset else 'wb'
                last_update = time.monotonic()
                with partial.open(mode) as output:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        if time.monotonic() - last_update >= 10:
                            output.flush()
                            record.update(state='downloading', received_bytes=output.tell(),
                                          expected_bytes=expected_bytes, updated_at=now(), attempt=attempt)
                            write_json(status_path, record)
                            print(json.dumps(dict(model=record['model_id'], bytes=output.tell(), total=expected_bytes)), flush=True)
                            last_update = time.monotonic()
                if partial.stat().st_size != expected_bytes:
                    raise ValueError(f'Incomplete download: {partial.stat().st_size}/{expected_bytes}')
                partial.replace(destination)
                break
        except Exception as error:
            failures.append(f'{type(error).__name__}: {error}')
            record.update(state='download_retry', failures=failures, updated_at=now())
            write_json(status_path, record)
            if attempt == 6:
                raise
            time.sleep(min(attempt * 2, 10))
    return dict(url=url, bytes=destination.stat().st_size, sha256=digest(destination),
                completed_at=now(), failures=failures)


def acquire(model):
    directory = ASSET_ROOT / model['key']
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / 'source.json'
    record = dict(model, repo=model['model_id'], state='checking_public_source', started_at=now(),
                  anonymous=True, authentication_used=False, resources={})
    write_json(status_path, record)
    client = session()
    api_url = f'https://huggingface.co/api/models/{model["model_id"]}/revision/{model["revision"]}'
    record['resources']['api_snapshot.json'] = fetch_small(client, api_url, directory / 'api_snapshot.json')
    api = json.loads((directory / 'api_snapshot.json').read_text())
    if api['id'] != model['model_id'] or api['sha'] != model['revision']:
        raise ValueError('Public source identity/revision differs from pinned model')
    if api.get('private') or api.get('gated'):
        raise ValueError('Only public nongated anonymous sources are permitted')
    record.update(private=api.get('private'), gated=api.get('gated'),
                  upstream_training_data=api.get('cardData', {}).get('datasets'),
                  pretraining_dataset=api.get('cardData', {}).get('datasets'),
                  parameter_metadata=api.get('safetensors'))
    tree_url = f'https://huggingface.co/api/models/{model["model_id"]}/tree/{model["revision"]}?recursive=true&expand=false'
    record['resources']['tree_snapshot.json'] = fetch_small(client, tree_url, directory / 'tree_snapshot.json')
    entries = json.loads((directory / 'tree_snapshot.json').read_text())
    weight_entry = next(item for item in entries if item.get('path') == 'model.safetensors')
    record['public_lfs_sha256'] = weight_entry['lfs']['oid']
    if weight_entry['size'] != model['expected_bytes']:
        raise ValueError('Pinned public weight size differs from expected')
    base = f'https://huggingface.co/{model["model_id"]}/raw/{model["revision"]}'
    for remote, local in {'README.md': 'README.md', 'config.json': 'config.json', **model['license_files']}.items():
        record['resources'][local] = fetch_small(client, f'{base}/{remote}', directory / local)
    if model['key'] == 'convnext_large':
        license_root = Path(__file__).resolve().parents[1] / 'licenses'
        for source, target, url in (
            (license_root / 'convnext_base/LICENSE-Apache-2.0.txt', 'LICENSE-Apache-2.0.txt', 'https://www.apache.org/licenses/LICENSE-2.0.txt'),
            (license_root / 'convnext_upstream/LICENSE-MIT.txt', 'UPSTREAM-LICENSE-MIT.txt', 'https://github.com/facebookresearch/ConvNeXt/blob/main/LICENSE'),
        ):
            (directory / target).write_bytes(source.read_bytes())
            record['resources'][target] = dict(url=url, source='Existing preserved public license text',
                                               bytes=(directory / target).stat().st_size,
                                               sha256=digest(directory / target))
    write_json(status_path, record)
    weights_url = f'https://huggingface.co/{model["model_id"]}/resolve/{model["revision"]}/model.safetensors'
    record['resources']['model.safetensors'] = download_weights(client, weights_url,
        directory / 'model.safetensors', model['expected_bytes'], status_path, record)
    if record['resources']['model.safetensors']['sha256'] != record['public_lfs_sha256']:
        raise ValueError('Downloaded weight SHA256 differs from pinned public LFS object')
    record['sha256'] = record['resources']['model.safetensors']['sha256']
    record.update(state='complete', completed_at=now())
    write_json(status_path, record)
    print(json.dumps(dict(model=model['model_id'], state='complete',
                         sha256=record['resources']['model.safetensors']['sha256'])), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if not args.download:
        print(json.dumps(MODELS, ensure_ascii=False, indent=2))
        return
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(acquire, MODELS))
    write_json(ASSET_ROOT / 'teacher_source_manifest.json',
               dict(created_at=now(), models=records, scope='Public initializations only; no inference/training'))


if __name__ == '__main__':
    main()
