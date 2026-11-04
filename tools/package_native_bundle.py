#!/usr/bin/env python3
"""Package checksummed native hook outputs for use with native_bundle.

Run dart test first to compile and exercise the bridge on the release host.
Packaging does not establish hardware or production platform acceptance.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from release_gate import source_fingerprint
from resolve_hook_ort import resolve


def package_bundle(root, target, output):
    runtime = resolve(root)
    manifest = json.loads((runtime.parent / 'artifacts_manifest.json').read_text())
    if manifest.get('target') != target:
        raise ValueError(f'Native hook target {manifest.get("target")} does not match {target}')
    if manifest.get('status', {}).get('compiled') != 'passed':
        raise ValueError('Native hook compilation did not pass')

    version = next(line.split(':', 1)[1].strip() for line in (root / 'pubspec.yaml').read_text().splitlines()
                   if line.startswith('version:'))
    suffix = 'dylib' if target == 'macos_arm64' else 'so'
    bridge = runtime.parent / f'liblib_onnx_bridge.{suffix}'
    binaries = []
    for binary, role in [(bridge, 'bridge'), (runtime, 'runtime')]:
        checksum = hashlib.sha256(binary.read_bytes()).hexdigest()
        recorded = next((item['sha256'] for item in manifest.get('binaries', [])
                         if item['name'] == binary.name), None)
        if checksum != recorded:
            raise ValueError(f'SHA256 mismatch for {binary.name}')
        binaries.append({'name': binary.name, 'sha256': checksum, 'role': role})

    files = [(binary, binary.name) for binary in (bridge, runtime)]
    files.append((root / 'LICENSE', 'lib_onnx-LICENSE'))
    for name in ('onnxruntime-LICENSE', 'onnxruntime-ThirdPartyNotices.txt'):
        files.append((runtime.parent / name, name))
    for binary, _ in files:
        if not binary.is_file():
            raise ValueError(f'Missing native bundle file: {binary}')

    manifest.update({'package': 'lib_onnx', 'package_version': version,
                     'source_fingerprint': source_fingerprint(root), 'binaries': binaries})
    manifest['status']['packaged'] = 'passed'
    payload = (json.dumps(manifest, indent=2) + '\n').encode()
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / f'lib_onnx-native-{target}-{version}.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for binary, name in files:
            archive.add(binary, arcname=name, recursive=False)
        entry = tarfile.TarInfo('artifacts_manifest.json')
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
    return archive_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True)
    parser.add_argument('--output', type=Path, default=Path('build/release'))
    args = parser.parse_args()
    print(package_bundle(Path.cwd(), args.target, args.output))


if __name__ == '__main__':
    main()
