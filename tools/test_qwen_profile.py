#!/usr/bin/env python3
"""Dependency-free regression tests for model/external-data inventory safety."""
import json
from pathlib import Path
import tempfile
import unittest

from export_embedding import REVISION, inventory, model_inventory, checked_file, verify_manifest


class InventoryTest(unittest.TestCase):
    def test_external_data_changes_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.onnx").write_bytes(b"test-only-model-content")
            (root / "weights.data").write_bytes(b"weights-v1")
            _, first = inventory(root, ["model.onnx", "weights.data"])
            (root / "weights.data").write_bytes(b"weights-v2")
            _, second = inventory(root, ["weights.data", "model.onnx"])
            self.assertNotEqual(first, second)
            self.assertEqual(second, inventory(root, ["model.onnx", "weights.data"])[1])

    def test_rejects_escaping_paths_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            root.mkdir()
            outside = Path(temporary) / "outside"
            outside.write_bytes(b"outside")
            (root / "link").symlink_to(outside)
            for name in ("../outside", "/outside", "link", "a/../b", "a\\b", "file:outside"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    checked_file(root, name)

    def test_verifies_entire_manifest_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.onnx").write_bytes(b"test-only-model-content")
            (root / "weights.data").write_bytes(b"test-only-external-content")
            files, identity = model_inventory(root, "model.onnx", ["weights.data"])
            digests = {record["path"]: record["sha256"] for record in files}
            manifest = {"checkpoint_revision": REVISION, "tokenizer_revision": REVISION,
                        "onnx_file_name": "model.onnx", "external_data_files": ["weights.data"],
                        "sha256": digests["model.onnx"],
                        "external_data_sha256": {"weights.data": digests["weights.data"]},
                        "files": files, "model_identity_sha256": identity}
            verify_manifest(manifest, root)
            (root / "weights.data").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_manifest(manifest, root)


if __name__ == "__main__":
    unittest.main()
