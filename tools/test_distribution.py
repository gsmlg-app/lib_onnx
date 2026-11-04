"""Tooling regressions; these synthetic structures are not Android acceptance."""
import importlib.util
import hashlib
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import zipfile
import unittest

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


audit = module('audit_android_pages')
gate = module('release_gate')
native_bundle = module('package_native_bundle')


class DistributionTests(unittest.TestCase):
    def elf(self, alignment=16384):
        data = bytearray(120)
        data[:6] = b'\x7fELF\x02\x01'
        struct.pack_into('<H', data, 18, 183)
        struct.pack_into('<Q', data, 32, 64)
        struct.pack_into('<HH', data, 54, 56, 1)
        struct.pack_into('<IIQQQQQQ', data, 64, 1, 5, 0, 0, 0, 120, 120, alignment)
        return data

    def test_real_elf_layout_offsets_are_read(self):
        self.assertEqual(audit.elf_check(self.elf(), 'fixture')['load_alignments'], [16384])

    def test_4kb_only_segment_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incompatible with 16 KB'):
            audit.elf_check(self.elf(4096), 'fixture')

    def test_wrong_architecture_rejected(self):
        data = self.elf()
        struct.pack_into('<H', data, 18, 62)
        with self.assertRaisesRegex(ValueError, 'AArch64'):
            audit.elf_check(data, 'fixture')

    def test_every_packaged_elf_is_checked_even_without_so_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'app.apk'
            with zipfile.ZipFile(package, 'w') as archive:
                archive.writestr('assets/dsp/htp.skel', self.elf(4096))
            with self.assertRaisesRegex(ValueError, 'incompatible with 16 KB'):
                audit.audit(package)

    def test_stable_requires_every_platform_stage(self):
        manifest = {'package_version': '0.2.0', 'bridge_abi_version': 2, 'ort_version': '1.31.1',
                    'software': {'status': 'passed', 'evidence': ['regression-log']},
                    'claimed_supported_targets': ['android_arm64'],
                    'targets': {'android_arm64': {'status': {'compiled': 'passed'}}}}
        errors = gate.check(manifest, '0.2.0', stable=True)
        self.assertTrue(any('device' in message for message in errors))
        self.assertTrue(any('clean_consumer' in message for message in errors))

    def test_candidate_does_not_imply_hardware_acceptance(self):
        manifest = {'package_version': '0.2.0-dev.1', 'bridge_abi_version': 2, 'ort_version': '1.31.1',
                    'software': {'status': 'passed', 'evidence': ['regression-log']}}
        self.assertEqual(gate.check(manifest, '0.2.0-dev.1', stable=False), [])


class NativeBundleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'native').mkdir()
        (self.root / 'pubspec.yaml').write_text('version: 0.2.0-dev.1\n')
        (self.root / 'LICENSE').write_text('lib_onnx license')
        (self.root / 'native/ort_artifacts.json').write_text(json.dumps({
            'targets': {'linux_x64': {'sha256': 'a' * 64}}}))
        self.build = self.root / 'hook-output'
        self.build.mkdir()
        for name in ('liblib_onnx_bridge.so', 'libonnxruntime.so.1',
                     'onnxruntime-LICENSE', 'onnxruntime-ThirdPartyNotices.txt'):
            (self.build / name).write_bytes(name.encode())
        self.manifest = {
            'manifest_version': 2, 'target': 'linux_x64',
            'bridge_abi_version': 2, 'ort_version': '1.31.1', 'archive_sha256': 'a' * 64,
            'status': {'compiled': 'passed', 'packaged': 'not-run', 'device': 'not-run'},
            'binaries': [{'name': name, 'sha256': hashlib.sha256((self.build / name).read_bytes()).hexdigest()}
                         for name in ('liblib_onnx_bridge.so', 'libonnxruntime.so.1')],
        }
        self.receipt = self.build / 'artifacts_manifest.json'
        self.receipt.write_text(json.dumps(self.manifest))
        output = self.root / '.dart_tool/hooks_runner/lib_onnx/test/output.json'
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps({'assets': [{'encoding': {
            'id': 'package:lib_onnx/onnxruntime', 'file': str(self.build / 'libonnxruntime.so.1')}}]}))

    def test_bundle_has_verified_binaries_roles_and_notices(self):
        result = native_bundle.package_bundle(self.root, 'linux_x64', self.root / 'dist')
        self.assertEqual(result.name, 'lib_onnx-native-linux_x64-0.2.0-dev.1.tar.gz')
        with tarfile.open(result) as archive:
            self.assertEqual(set(archive.getnames()), {
                'liblib_onnx_bridge.so', 'libonnxruntime.so.1', 'artifacts_manifest.json',
                'lib_onnx-LICENSE', 'onnxruntime-LICENSE', 'onnxruntime-ThirdPartyNotices.txt'})
            self.assertTrue(all(item.isfile() for item in archive.getmembers()))
            manifest = json.load(archive.extractfile('artifacts_manifest.json'))
            self.assertEqual(manifest['package_version'], '0.2.0-dev.1')
            self.assertEqual(manifest['source_fingerprint'], gate.source_fingerprint(self.root))
            self.assertEqual(manifest['status']['packaged'], 'passed')
            self.assertEqual(manifest['status']['device'], 'not-run')
            self.assertEqual({item['role'] for item in manifest['binaries']}, {'bridge', 'runtime'})
            for item in manifest['binaries']:
                self.assertEqual(hashlib.sha256(archive.extractfile(item['name']).read()).hexdigest(), item['sha256'])

    def test_wrong_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'does not match macos_arm64'):
            native_bundle.package_bundle(self.root, 'macos_arm64', self.root / 'dist')

    def test_modified_bridge_is_rejected(self):
        (self.build / 'liblib_onnx_bridge.so').write_bytes(b'corrupted bridge')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            native_bundle.package_bundle(self.root, 'linux_x64', self.root / 'dist')

    def test_modified_runtime_is_rejected(self):
        (self.build / 'libonnxruntime.so.1').write_bytes(b'corrupted runtime')
        with self.assertRaisesRegex(SystemExit, 'No verified ORT hook asset'):
            native_bundle.package_bundle(self.root, 'linux_x64', self.root / 'dist')

    def test_missing_notices_are_rejected(self):
        (self.build / 'onnxruntime-ThirdPartyNotices.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing native bundle file'):
            native_bundle.package_bundle(self.root, 'linux_x64', self.root / 'dist')


if __name__ == '__main__':
    unittest.main()
