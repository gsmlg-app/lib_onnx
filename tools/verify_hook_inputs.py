#!/usr/bin/env python3
"""Negative native-hook input checks; no native artifact or device is simulated."""
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    cases = []
    with tempfile.TemporaryDirectory(prefix='lib-onnx-hook-negative-') as directory:
        work = Path(directory)
        corrupted = work / 'corrupt.tgz'
        corrupted.write_bytes(b'not the pinned ORT release')
        bundle = work / 'wrong-abi-bundle'
        bundle.mkdir()
        (bundle / 'artifacts_manifest.json').write_text(json.dumps({
            'target': 'macos_arm64', 'bridge_abi_version': 1, 'ort_version': '1.31.1'}))
        checks = [
            ('corrupt-archive', {'ort_archive': str(corrupted), 'offline': True}, 'macos',
             {'macos': {'target_version': 14}}, 'SHA256 mismatch'),
            ('missing-offline-archive', {'ort_archive': str(work / 'absent.tgz'), 'offline': True},
             'macos', {'macos': {'target_version': 14}}, 'Pinned ORT archive missing'),
            ('ios-never-uses-macos-library', {}, 'ios',
             {'ios': {'target_sdk': 'iphoneos', 'target_version': 15}}, 'separately built device/simulator'),
            ('android-needs-qnn-inputs', {}, 'android',
             {'android': {'target_ndk_api': 29}}, 'matching Qualcomm runtime libraries'),
            ('old-bundle-abi', {'native_bundle': str(bundle)}, 'macos',
             {'macos': {'target_version': 14}}, 'Native bundle target/SDK/ABI/ORT mismatch'),
        ]
        for name, defines, os_name, platform, message in checks:
            target = work / name
            target.mkdir()
            configuration = {
                'assets': [], 'package_name': 'lib_onnx', 'package_root': str(ROOT) + '/',
                'out_dir_shared': str(target / 'shared'), 'out_file': str(target / 'output.json'),
                'config': {'build_asset_types': ['code_assets/code'], 'linking_enabled': False,
                           'extensions': {'code_assets': {'link_mode_preference': 'dynamic',
                             'target_os': os_name, 'target_architecture': 'arm64', **platform}}},
                'user_defines': {'workspace_pubspec': {'base_path': str(ROOT) + '/', 'defines': defines}},
            }
            path = target / 'input.json'
            path.write_text(json.dumps(configuration))
            result = subprocess.run(['dart', '--packages=.dart_tool/package_config.json',
                                     'hook/build.dart', f'--config={path}'], cwd=ROOT,
                                    capture_output=True, text=True, timeout=60)
            output = result.stdout + result.stderr
            if result.returncode == 0 or message not in output:
                raise RuntimeError(f'{name}: wrong rejection (exit {result.returncode})\n{output}')
            cases.append({'case': name, 'status': 'passed', 'rejection': message})
    print(json.dumps({'status': 'passed', 'cases': cases}, indent=2))


if __name__ == '__main__':
    main()
