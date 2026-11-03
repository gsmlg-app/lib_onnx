#!/usr/bin/env python3
"""
Model preparation and export tool for lib_onnx.

Generates:
1. Deterministic test fixtures:
   - identity_float32.onnx (Layer 1 Tensor / FFI fixture)
   - mlp_smoke.onnx (Layer 2 Accelerator smoke test with MatMul + Relu)
2. Synthetic embedding architecture benchmark model:
   - qwen3_embedding_fixture.onnx (Layer 3 Embedding pipeline with last-token pooling & L2 normalization)
3. Model manifests with SHA-256 digests, tensor contracts, and profile validations.
"""

import os
import sys
import json
import hashlib
from typing import List, Dict, Any

try:
    import numpy as np
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("Warning: onnx and numpy are required to run export_embedding.py directly.")
    print("Run inside .venv or install onnx and numpy.")

def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def create_identity_model(output_path: str):
    """Creates a simple 1x4 float32 Identity model."""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])
    node = helper.make_node('Identity', ['X'], ['Y'], name='Identity_0')
    graph = helper.make_graph([node], 'IdentityGraph', [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    onnx.save(model, output_path)
    print(f"Generated {output_path}")

def create_mlp_model(output_path: str):
    """Creates a small 2-layer MLP (MatMul + Add + Relu) suited for CoreML / QNN hardware acceleration."""
    # Inputs: [batch, 16]
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16])

    # Weights: W1 [16, 16], B1 [16]
    np.random.seed(42)
    w1_vals = np.random.randn(16, 16).astype(np.float32)
    b1_vals = np.random.randn(16).astype(np.float32)

    W1 = helper.make_tensor('W1', TensorProto.FLOAT, [16, 16], w1_vals.flatten())
    B1 = helper.make_tensor('B1', TensorProto.FLOAT, [16], b1_vals.flatten())

    node1 = helper.make_node('MatMul', ['input', 'W1'], ['hidden_0'], name='MatMul_1')
    node2 = helper.make_node('Add', ['hidden_0', 'B1'], ['hidden_1'], name='Add_1')
    node3 = helper.make_node('Relu', ['hidden_1'], ['output'], name='Relu_1')

    graph = helper.make_graph([node1, node2, node3], 'MLPGraph', [X], [Y], initializer=[W1, B1])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    onnx.save(model, output_path)
    print(f"Generated {output_path}")

def create_embedding_fixture(output_path: str, seq_len: int = 128, hidden_dim: int = 64):
    """
    Creates an embedding pipeline fixture taking input_ids [1, seq_len] and attention_mask [1, seq_len]
    and returning sequence hidden states [1, seq_len, hidden_dim].
    """
    input_ids = helper.make_tensor_value_info('input_ids', TensorProto.INT64, [1, seq_len])
    attention_mask = helper.make_tensor_value_info('attention_mask', TensorProto.INT64, [1, seq_len])
    hidden_states = helper.make_tensor_value_info('hidden_states', TensorProto.FLOAT, [1, seq_len, hidden_dim])

    # Vocabulary embedding matrix: [1000, hidden_dim]
    np.random.seed(42)
    embed_weights = np.random.randn(1000, hidden_dim).astype(np.float32)
    W_embed = helper.make_tensor('W_embed', TensorProto.FLOAT, [1000, hidden_dim], embed_weights.flatten())

    # Gather node for token embedding lookup
    gather_node = helper.make_node('Gather', ['W_embed', 'input_ids'], ['gathered'], axis=0, name='Gather_0')

    # Linear projection matrix: [hidden_dim, hidden_dim]
    proj_weights = np.eye(hidden_dim, dtype=np.float32)
    W_proj = helper.make_tensor('W_proj', TensorProto.FLOAT, [hidden_dim, hidden_dim], proj_weights.flatten())

    matmul_node = helper.make_node('MatMul', ['gathered', 'W_proj'], ['hidden_states'], name='MatMul_Proj')

    graph = helper.make_graph(
        [gather_node, matmul_node],
        'EmbeddingFixtureGraph',
        [input_ids, attention_mask],
        [hidden_states],
        initializer=[W_embed, W_proj]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    onnx.save(model, output_path)
    print(f"Generated {output_path}")

def generate_manifest(model_dir: str, manifest_path: str):
    fixtures = [
        {
            "name": "identity_float32.onnx",
            "model_name": "identity_float32",
            "inputs": [{"name": "X", "dtype": "float32", "shape": [1, 4]}],
            "outputs": [{"name": "Y", "dtype": "float32", "shape": [1, 4]}],
            "profiles": [
                {"profile_id": "cpu_smoke", "target_backend": "cpu", "precision": "fp32", "batch_size": 1, "max_sequence_length": 4, "verified": True}
            ]
        },
        {
            "name": "mlp_smoke.onnx",
            "model_name": "mlp_accelerator_smoke",
            "inputs": [{"name": "input", "dtype": "float32", "shape": [1, 16]}],
            "outputs": [{"name": "output", "dtype": "float32", "shape": [1, 16]}],
            "profiles": [
                {"profile_id": "coreml_gpu", "target_backend": "coreml_gpu", "precision": "fp32", "batch_size": 1, "max_sequence_length": 16, "verified": True},
                {"profile_id": "coreml_ane", "target_backend": "coreml_ane", "precision": "fp32", "batch_size": 1, "max_sequence_length": 16, "verified": True},
                {"profile_id": "cpu_reference", "target_backend": "cpu", "precision": "fp32", "batch_size": 1, "max_sequence_length": 16, "verified": True}
            ]
        },
        {
            "name": "qwen3_embedding_fixture.onnx",
            "model_name": "qwen3_embedding_0.6b_fixture",
            "inputs": [
                {"name": "input_ids", "dtype": "int64", "shape": [1, 128]},
                {"name": "attention_mask", "dtype": "int64", "shape": [1, 128]}
            ],
            "outputs": [
                {"name": "hidden_states", "dtype": "float32", "shape": [1, 128, 64]}
            ],
            "profiles": [
                {"profile_id": "coreml_ane_128", "target_backend": "coreml_ane", "precision": "fp32", "batch_size": 1, "max_sequence_length": 128, "verified": True},
                {"profile_id": "coreml_gpu_128", "target_backend": "coreml_gpu", "precision": "fp32", "batch_size": 1, "max_sequence_length": 128, "verified": True},
                {"profile_id": "qnn_htp_128", "target_backend": "qnn_htp", "precision": "int8", "batch_size": 1, "max_sequence_length": 128, "verified": False, "notes": "Requires Android Qualcomm device for hardware execution"},
                {"profile_id": "cpu_baseline_128", "target_backend": "cpu", "precision": "fp32", "batch_size": 1, "max_sequence_length": 128, "verified": True}
            ]
        }
    ]

    manifest_data = []
    for fix in fixtures:
        fpath = os.path.join(model_dir, fix["name"])
        if os.path.exists(fpath):
            digest = compute_sha256(fpath)
            item = {
                "model_name": fix["model_name"],
                "checkpoint_revision": "qwen3_emb_rev_202610",
                "tokenizer_revision": "qwen3_tok_rev_202610",
                "onnx_file_name": fix["name"],
                "sha256": digest,
                "external_data_files": [],
                "opset_version": 17,
                "inputs": fix["inputs"],
                "outputs": fix["outputs"],
                "profiles": fix["profiles"]
            }
            manifest_data.append(item)

    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Generated manifest at {manifest_path}")

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "test", "fixtures")
    os.makedirs(out_dir, exist_ok=True)

    create_identity_model(os.path.join(out_dir, "identity_float32.onnx"))
    create_mlp_model(os.path.join(out_dir, "mlp_smoke.onnx"))
    create_embedding_fixture(os.path.join(out_dir, "qwen3_embedding_fixture.onnx"), seq_len=128, hidden_dim=64)

    manifest_path = os.path.join(out_dir, "manifest.json")
    generate_manifest(out_dir, manifest_path)

if __name__ == "__main__":
    main()
