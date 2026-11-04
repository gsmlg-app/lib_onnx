"""Tooling regressions; these synthetic structures are not Android acceptance."""
import importlib.util
from pathlib import Path
import struct
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


if __name__ == '__main__':
    unittest.main()
