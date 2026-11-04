#!/usr/bin/env python3
"""Acquire the pinned official checkpoint or inventory an existing real ONNX export.

This does not manufacture a small synthetic graph and call it Qwen. Importing an
ONNX export records declared producer provenance; only validate_qwen.dart can
record a matching actual Dart/native inference against the checkpoint reference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import sys

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
OPSET = 17
TOOLCHAIN = {"numpy": "2.2.6", "onnx": "1.18.0", "huggingface-hub": "0.33.4",
             "transformers": "4.53.3", "tokenizers": "0.21.2",
             "safetensors": "0.5.3", "torch": "2.7.1"}


def verify_toolchain(names: list[str]) -> None:
    if sys.version_info[:2] != (3, 11):
        raise ValueError("Qwen profile toolchain requires Python 3.11 (see qwen-validation-requirements.txt)")
    for name in names:
        actual = importlib.metadata.version(name)
        if actual != TOOLCHAIN[name]:
            raise ValueError(f"Toolchain mismatch: {name} {actual}; expected {TOOLCHAIN[name]}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or "\\" in relative or ":" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError(f"Unsafe model-relative path: {relative!r}")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()) or not result.is_file():
        raise ValueError(f"Missing or escaping model file: {relative}")
    return result


def inventory(root: Path, names: list[str]) -> tuple[list[dict], str]:
    if len(set(names)) != len(names):
        raise ValueError("Duplicate inventory entries")
    records = []
    for name in sorted(names):
        path = checked_file(root, name)
        records.append({"path": name, "size": path.stat().st_size, "sha256": sha256(path)})
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return records, hashlib.sha256(canonical).hexdigest()


def model_inventory(root: Path, model_name: str, external: list[str]) -> tuple[list[dict], str]:
    records, _ = inventory(root, [model_name, *external])
    by_name = {record["path"]: record for record in records}
    ordered = []
    for name in [model_name, *sorted(external)]:
        record = by_name[name]
        ordered.append({"path": "model.onnx" if name == model_name else name,
                        "size": record["size"], "sha256": record["sha256"]})
    canonical = json.dumps(ordered, separators=(",", ":"), ensure_ascii=False).encode()
    return records, hashlib.sha256(canonical).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_manifest(manifest: dict, root: Path) -> None:
    if manifest["checkpoint_revision"] != REVISION or manifest["tokenizer_revision"] != REVISION:
        raise ValueError("Checkpoint/tokenizer revision does not match the pinned Qwen profile")
    records, identity = model_inventory(root, manifest["onnx_file_name"], manifest["external_data_files"])
    if records != manifest["files"] or identity != manifest["model_identity_sha256"]:
        raise ValueError("Model/external-data inventory checksum mismatch")
    by_name = {record["path"]: record["sha256"] for record in records}
    if (manifest["sha256"] != by_name[manifest["onnx_file_name"]]
            or manifest["external_data_sha256"] != {
                name: by_name[name] for name in manifest["external_data_files"]}):
        raise ValueError("Manifest file checksums disagree")


def acquire(args: argparse.Namespace) -> None:
    verify_toolchain(["huggingface-hub"])
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(
        repo_id=MODEL_ID, revision=REVISION, local_dir=args.output,
        local_files_only=args.offline,
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "LICENSE", "README.md"],
    ))
    names = sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                   if path.is_file() and ".cache" not in path.relative_to(root).parts
                   and path.name != "snapshot_metadata.json")
    records, identity = inventory(root, names)
    write_json(root / "snapshot_metadata.json", {
        "model_id": MODEL_ID, "revision": REVISION, "files": records,
        "inventory_sha256": identity,
        "source": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
    })
    print(f"Acquired {MODEL_ID}@{REVISION}; {len(records)} files; inventory {identity}")


def graph_tensors(graph):
    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    yield from node_tensors(graph.node)


def attribute_tensors(attributes):
    import onnx
    for attribute in attributes:
        if attribute.type == onnx.AttributeProto.TENSOR:
            yield attribute.t
        elif attribute.type == onnx.AttributeProto.TENSORS:
            yield from attribute.tensors
        elif attribute.type == onnx.AttributeProto.SPARSE_TENSOR:
            yield attribute.sparse_tensor.values
            yield attribute.sparse_tensor.indices
        elif attribute.type == onnx.AttributeProto.SPARSE_TENSORS:
            for sparse in attribute.sparse_tensors:
                yield sparse.values
                yield sparse.indices
        elif attribute.type == onnx.AttributeProto.GRAPH:
            yield from graph_tensors(attribute.g)
        elif attribute.type == onnx.AttributeProto.GRAPHS:
            for child in attribute.graphs:
                yield from graph_tensors(child)


def node_tensors(nodes):
    for node in nodes:
        yield from attribute_tensors(node.attribute)


def tensor_spec(value) -> dict:
    import onnx
    if not value.type.HasField("tensor_type"):
        raise ValueError(f"Unsupported non-tensor graph value: {value.name}")
    tensor = value.type.tensor_type
    types = {onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.FLOAT16: "float16",
             onnx.TensorProto.INT64: "int64", onnx.TensorProto.INT32: "int32",
             onnx.TensorProto.UINT8: "uint8", onnx.TensorProto.INT8: "int8",
             onnx.TensorProto.BOOL: "bool"}
    if tensor.elem_type not in types:
        raise ValueError(f"Unsupported graph dtype {tensor.elem_type} for {value.name}")
    return {"name": value.name, "dtype": types[tensor.elem_type], "shape": [
        dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param or "unknown"
        for dimension in tensor.shape.dim]}


def import_export(args: argparse.Namespace) -> None:
    verify_toolchain(["onnx"])
    import onnx
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise ValueError(f"Missing exported ONNX file: {model_path}")
    provenance = json.loads(args.provenance.read_text())
    if (provenance.get("checkpoint_revision") != REVISION
            or provenance.get("tokenizer_revision") != REVISION
            or not provenance.get("toolchain") or not provenance.get("command")):
        raise ValueError("Producer provenance must declare pinned revisions, exact toolchain and export command")
    model = onnx.load(model_path, load_external_data=False)
    default_opsets = [entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")]
    if default_opsets != [OPSET]:
        raise ValueError(f"Profile requires ONNX opset {OPSET}; found {default_opsets}")
    external = set()
    tensors = list(graph_tensors(model.graph))
    for function in model.functions:
        tensors.extend(node_tensors(function.node))
        tensors.extend(attribute_tensors(function.attribute_proto))
    for tensor in tensors:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            entries = {entry.key: entry.value for entry in tensor.external_data}
            name = entries.get("location", "")
            file = checked_file(model_path.parent, name)
            offset, length = int(entries.get("offset", "0")), entries.get("length")
            if offset < 0 or (length is not None and int(length) < 0):
                raise ValueError(f"Invalid external tensor bounds: {name}")
            if offset > file.stat().st_size or (length is not None and offset + int(length) > file.stat().st_size):
                raise ValueError(f"External tensor exceeds file: {name}")
            external.add(name)
    # Path-based checking permits protobufs with external tensors >2GB.
    onnx.checker.check_model(str(model_path))
    initializers = {tensor.name for tensor in model.graph.initializer}
    inputs = [tensor_spec(value) for value in model.graph.input if value.name not in initializers]
    outputs = [tensor_spec(value) for value in model.graph.output]
    output = next((value for value in outputs if value["name"] == args.output_name), None)
    expected_rank = 2 if args.embedded_pooling else 3
    if output is None or len(output["shape"]) != expected_rank or output["shape"][-1] != 1024:
        raise ValueError("Selected Qwen output must match rank and hidden dimension 1024")
    supported = {"input_ids", "attention_mask", "position_ids", "token_type_ids"}
    if not {"input_ids", "attention_mask"}.issubset({value["name"] for value in inputs}):
        raise ValueError("Profile requires input_ids and attention_mask for reproducible token pooling")
    for value in inputs:
        if value["name"] not in supported or value["dtype"] not in ("int32", "int64"):
            raise ValueError(f"Unsupported exported input contract (not silently omitted): {value}")
        if len(value["shape"]) != 2:
            raise ValueError(f"Expected rank-two text input: {value}")
        for declared, profile in zip(value["shape"], (1, args.sequence_length)):
            if isinstance(declared, int) and declared != profile:
                raise ValueError(f"Input shape differs from selected batch1/sequence profile: {value}")
    records, identity = model_inventory(model_path.parent, model_path.name, sorted(external))
    by_name = {record["path"]: record["sha256"] for record in records}
    manifest = {
        "schema_version": 2, "model_name": MODEL_ID, "checkpoint_revision": REVISION,
        "tokenizer_revision": REVISION, "onnx_file_name": model_path.name,
        "sha256": by_name[model_path.name], "external_data_files": sorted(external),
        "external_data_sha256": {name: by_name[name] for name in sorted(external)},
        "model_identity_sha256": identity, "files": records, "opset_version": OPSET,
        "inputs": inputs, "outputs": outputs, "embedding_output": args.output_name,
        "has_embedded_pooling": args.embedded_pooling, "hidden_dim": 1024,
        "producer_provenance": provenance,
        "provenance_status": "declared-by-exporter; checkpoint-reference validation required",
        "import_toolchain": {name: importlib.metadata.version(name) for name in ("onnx",)},
        "profiles": [{"profile_id": args.profile_id, "target_backend": args.target_backend,
                      "precision": args.precision, "batch_size": 1,
                      "max_sequence_length": args.sequence_length, "verified": False,
                      "notes": "Inventory checked; actual inference and hardware acceptance not yet performed"}],
    }
    write_json(args.output, manifest)
    print(f"Inventoried ONNX export; identity {identity}; validation remains not-run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    acquire_parser = commands.add_parser("acquire", help="Download exact official checkpoint and tokenizer revision")
    acquire_parser.add_argument("--output", type=Path, required=True)
    acquire_parser.add_argument("--offline", action="store_true")
    acquire_parser.set_defaults(action=acquire)
    importer = commands.add_parser("import", help="Validate and inventory an already exported real Qwen ONNX model")
    importer.add_argument("--model", type=Path, required=True)
    importer.add_argument("--provenance", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    importer.add_argument("--output-name", required=True)
    importer.add_argument("--embedded-pooling", action="store_true")
    importer.add_argument("--profile-id", required=True)
    importer.add_argument("--target-backend", choices=("cpu", "coreml", "qnn_htp"), required=True)
    importer.add_argument("--precision", choices=("fp32", "fp16", "int8"), required=True)
    importer.add_argument("--sequence-length", type=int, choices=(128, 256, 512), default=128)
    importer.set_defaults(action=import_export)
    args = parser.parse_args()
    try:
        args.action(args)
    except (ImportError, ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Blocked model preparation: {error}\n")


if __name__ == "__main__":
    main()
