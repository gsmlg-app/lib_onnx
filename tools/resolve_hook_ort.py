#!/usr/bin/env python3
"""Locate the exact checksummed ORT asset produced by the preceding hook run."""
import argparse
import hashlib
import json
from pathlib import Path


def resolve(root):
    outputs = sorted((root / '.dart_tool/hooks_runner/lib_onnx').glob('*/output.json'),
                     key=lambda path: path.stat().st_mtime, reverse=True)
    for output in outputs:
        document = json.loads(output.read_text())
        for asset in document.get('assets', []):
            encoded = asset.get('encoding', {})
            if encoded.get('id') != 'package:lib_onnx/onnxruntime':
                continue
            binary = Path(encoded['file'])
            receipt_file = binary.parent / 'artifacts_manifest.json'
            if not binary.is_file() or not receipt_file.is_file():
                continue
            receipt = json.loads(receipt_file.read_text())
            if receipt.get('ort_version') != '1.31.1' or receipt.get('bridge_abi_version') != 2:
                continue
            checksum = hashlib.sha256(binary.read_bytes()).hexdigest()
            recorded = next((item['sha256'] for item in receipt.get('binaries', []) if item['name'] == binary.name), None)
            lock = json.loads((root / 'native/ort_artifacts.json').read_text())
            artifact = lock['targets'].get(receipt.get('target'), {})
            if recorded == checksum and receipt.get('archive_sha256') == artifact.get('sha256'):
                return binary.resolve()
    raise SystemExit('No verified ORT hook asset. Run dart test first; native tests never use a system fallback.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--github-env', type=Path)
    args = parser.parse_args()
    binary = resolve(Path.cwd())
    if args.github_env:
        with args.github_env.open('a') as env:
            env.write(f'LIB_ONNX_TEST_ORT={binary}\n')
    else:
        print(binary)


if __name__ == '__main__':
    main()
