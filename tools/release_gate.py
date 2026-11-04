#!/usr/bin/env python3
"""Reject publication without current, platform-specific acceptance receipts."""
import argparse
import json
import hashlib
from pathlib import Path
import re


def source_fingerprint(root=Path('.')):
    paths = {root / 'pubspec.yaml', root / 'native/ort_artifacts.json'}
    for folder, pattern in [('lib', '*.dart'), ('hook', '*.dart'), ('native/src', '*.c'), ('native/include', '*.h')]:
        paths.update((root / folder).rglob(pattern))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def check(manifest, version, stable):
    errors = []
    if manifest.get('package_version') != version:
        errors.append('manifest package_version does not match release version')
    if manifest.get('bridge_abi_version') != 2 or manifest.get('ort_version') != '1.31.1':
        errors.append('manifest must describe ABI2 and ORT 1.31.1')
    software = manifest.get('software', {})
    if software.get('status') != 'passed' or not software.get('evidence'):
        errors.append('software repair acceptance has no passing evidence')
    if stable:
        targets = manifest.get('claimed_supported_targets', [])
        if not targets:
            errors.append('stable release needs explicit supported targets')
        for target in targets:
            item = manifest.get('targets', {}).get(target, {})
            for stage in ('compiled', 'packaged', 'clean_consumer', 'device'):
                if item.get('status', {}).get(stage) != 'passed':
                    errors.append(f'{target}: {stage} acceptance is not passed')
            if not item.get('evidence'):
                errors.append(f'{target}: missing reproducible acceptance evidence')
            binaries = item.get('binaries', [])
            if not binaries or any(not re.fullmatch(r'[a-f0-9]{64}', b.get('sha256', '')) for b in binaries):
                errors.append(f'{target}: missing actual binary hashes')
            if item.get('acceleration_claimed') and item.get('hardware_observed') != 'passed':
                errors.append(f'{target}: accelerated hardware was not observed')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('artifacts_manifest.json'))
    parser.add_argument('--version')
    args = parser.parse_args()
    version = args.version
    if not version:
        version = next(line.split(':', 1)[1].strip() for line in Path('pubspec.yaml').read_text().splitlines() if line.startswith('version:'))
    version = version.removeprefix('v')
    manifest = json.loads(args.manifest.read_text())
    errors = check(manifest, version, '-' not in version)
    if manifest.get('source_fingerprint') != source_fingerprint():
        errors.append('artifact/source fingerprint is missing or stale; regenerate and revalidate')
    if errors:
        raise SystemExit('Release acceptance gate rejected:\n- ' + '\n- '.join(errors))
    print(f'Acceptance gate passed for {version}; publication remains an explicit action.')


if __name__ == '__main__':
    main()
