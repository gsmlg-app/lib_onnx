#!/usr/bin/env python3
"""Record hashes and actual compiled providers from an explicitly built bridge.

Compilation/provider enumeration does not establish packaging or device success.
Only replace status fields when independently evidenced; defaults remain not-run.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from release_gate import source_fingerprint


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def inspect(bridge):
    library = ctypes.CDLL(str(bridge.resolve()))
    library.lib_onnx_abi_version.restype = ctypes.c_uint32
    library.lib_onnx_ort_version_string.restype = ctypes.c_char_p
    library.lib_onnx_runtime_create.argtypes = [ctypes.c_int32]
    library.lib_onnx_runtime_create.restype = ctypes.c_void_p
    library.lib_onnx_runtime_destroy.argtypes = [ctypes.c_void_p]
    library.lib_onnx_runtime_get_available_providers.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    abi = library.lib_onnx_abi_version()
    version = library.lib_onnx_ort_version_string().decode()
    if abi != 2 or version != '1.31.1':
        raise ValueError(f'Cannot describe mismatched ABI {abi} / ORT {version}')
    runtime = library.lib_onnx_runtime_create(1)
    if not runtime:
        raise ValueError('Bridge failed to create runtime for provider enumeration')
    try:
        buffer = ctypes.create_string_buffer(8192)
        count = library.lib_onnx_runtime_get_available_providers(runtime, buffer, len(buffer))
        if count < 0:
            raise ValueError('Provider enumeration failed')
        return abi, version, buffer.value.decode().split(',') if count else []
    finally:
        library.lib_onnx_runtime_destroy(runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge', required=True, type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--target', required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts_manifest.json'))
    parser.add_argument('--evidence', required=True)
    args = parser.parse_args()
    manifest = json.loads(args.output.read_text()) if args.output.exists() else {}
    abi, version, providers = inspect(args.bridge)
    package_version = next(line.split(':', 1)[1].strip() for line in Path('pubspec.yaml').read_text().splitlines() if line.startswith('version:'))
    manifest.update({
        'manifest_version': 2, 'package': 'lib_onnx', 'package_version': package_version,
        'bridge_abi_version': abi, 'ort_version': version, 'source_fingerprint': source_fingerprint(),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'license': 'MIT; ORT third-party notices are retained from the pinned upstream archive',
    })
    manifest.setdefault('claimed_supported_targets', [])
    manifest.setdefault('software', {'status': 'not-run', 'evidence': []})
    item = manifest.setdefault('targets', {}).setdefault(args.target, {})
    item.update({
        'providers_compiled': providers,
        'providers_source': 'lib_onnx_runtime_get_available_providers / ORT GetAvailableProviders',
        'acceleration_claimed': False,
        'hardware_observed': 'not-run',
        'binaries': [
            {'name': path.name, 'sha256': digest(path), 'role': role, 'checksum_status': 'verified-local-file'}
            for path, role in [(args.bridge, 'bridge'), (args.runtime, 'runtime')]
        ],
        'evidence': [args.evidence],
    })
    item.setdefault('status', {}).update({'configured': 'passed', 'compiled': 'passed'})
    for stage in ['packaged', 'clean_consumer', 'device']:
        item['status'].setdefault(stage, 'not-run')
    args.output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Recorded {args.target}: ABI {abi}, ORT {version}, compiled providers {providers}')


if __name__ == '__main__':
    main()
