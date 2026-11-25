"""Download uncompressed Amazon 2023 files with atomic completion markers."""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path


def download(url, destination):
    temporary = destination.with_suffix(destination.suffix + '.partial')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                expected = response.headers.get('Content-Length')
                written = 0
                with temporary.open('wb') as stream:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        written += len(block)
            if not written or (expected is not None and written != int(expected)):
                raise IOError('Incomplete download')
            os.replace(temporary, destination)
            print(f'Saved {destination}: {written} bytes', flush=True)
            return
        except (OSError, ValueError) as error:
            if attempt == 2:
                raise
            print(f'Retrying {url}: {error}', flush=True)
            time.sleep(2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', default='Musical_Instruments',
                        help='Amazon category; suite domains: Musical_Instruments, Amazon_Fashion, Appliances')
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--endpoint', default='https://huggingface.co')
    parser.add_argument('--revision', default='main')
    args = parser.parse_args(argv)
    if not args.category.replace('_', '').isalnum():
        parser.error('category must contain only letters, numbers and underscores')
    args.output.mkdir(parents=True, exist_ok=True)
    source = f'{args.endpoint.rstrip("/")}/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/{args.revision}/raw'
    urls = {
        f'{args.category}.jsonl': f'{source}/review_categories/{args.category}.jsonl?download=true',
        f'meta_{args.category}.jsonl': f'{source}/meta_categories/meta_{args.category}.jsonl?download=true',
    }
    for name, url in urls.items():
        download(url, args.output / name)
    manifest_path = args.output / 'download_sources.json'
    sources = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    sources.update(urls)
    manifest_path.write_text(json.dumps(sources, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
