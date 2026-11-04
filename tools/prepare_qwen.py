#!/usr/bin/env python3
"""Prepare real text/tokenizer inputs and PyTorch references for validate_qwen.dart."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from export_embedding import MODEL_ID, REVISION, TOOLCHAIN, inventory, verify_manifest, verify_toolchain, write_json

TASK = "Given a web search query, retrieve relevant passages that answer the query"
QUERIES = ["What is the capital of China?", "Explain gravity"]
DOCUMENTS = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]


def prepare(args: argparse.Namespace) -> None:
    verify_toolchain(list(TOOLCHAIN))
    manifest = json.loads(args.manifest.read_text())
    verify_manifest(manifest, args.model_directory)
    metadata = json.loads((args.checkpoint / "snapshot_metadata.json").read_text())
    if metadata["model_id"] != MODEL_ID or metadata["revision"] != REVISION:
        raise ValueError("Checkpoint snapshot metadata revision mismatch")
    records, identity = inventory(args.checkpoint, [record["path"] for record in metadata["files"]])
    if records != metadata["files"] or identity != metadata["inventory_sha256"]:
        raise ValueError("Checkpoint/tokenizer snapshot checksum mismatch")

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, padding_side=args.padding_side,
        local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        args.checkpoint, torch_dtype=torch.float32,
        attn_implementation="eager", local_files_only=True,
        trust_remote_code=False).eval()
    if model.config.hidden_size != manifest["hidden_dim"]:
        raise ValueError("Checkpoint and ONNX hidden dimensions differ")
    sequence = manifest["profiles"][0]["max_sequence_length"]
    texts = [f"Instruct: {TASK}\nQuery: {query}" for query in QUERIES] + DOCUMENTS
    cases = []
    names = {value["name"] for value in manifest["inputs"]}
    for index, text in enumerate(texts):
        encoded = tokenizer(text, truncation=False, return_tensors="np")
        if encoded["input_ids"].shape[1] > sequence:
            raise ValueError(f"Text case {index} exceeds sequence profile {sequence}; truncation is prohibited")
        batch = tokenizer(text, padding="max_length", max_length=sequence,
                          truncation=False, return_tensors="np")
        mask = batch["attention_mask"]
        if not np.isin(mask, (0, 1)).all() or not mask.any():
            raise ValueError("Tokenizer returned an invalid/all-padding mask")
        position_ids = np.maximum(np.cumsum(mask, axis=1) - 1, 0)
        available = {**batch, "position_ids": position_ids}
        if "token_type_ids" in names and "token_type_ids" not in available:
            # Qwen has no token type embeddings. Do not fabricate input values
            # for an export that declares a different tokenizer contract.
            raise ValueError("Export requires token_type_ids but pinned tokenizer does not produce them")
        inputs = {}
        for spec in manifest["inputs"]:
            if spec["name"] not in available:
                raise ValueError(f"Cannot construct actual model input: {spec['name']}")
            values = np.asarray(available[spec["name"]], dtype=spec["dtype"])
            if list(values.shape) != [1, sequence]:
                raise ValueError(f"Tokenizer shape mismatch for {spec['name']}: {values.shape}")
            inputs[spec["name"]] = {"dtype": spec["dtype"], "shape": list(values.shape),
                                    "values": values.reshape(-1).tolist()}
        # Compute the reference from the actual pinned checkpoint, with the
        # same position IDs if they are an explicit graph input.
        reference_inputs = {"input_ids": torch.from_numpy(batch["input_ids"]),
                            "attention_mask": torch.from_numpy(mask)}
        if "position_ids" in names:
            reference_inputs["position_ids"] = torch.from_numpy(position_ids)
        with torch.inference_mode():
            hidden = model(**reference_inputs).last_hidden_state
        last = int(np.flatnonzero(mask[0])[-1])
        vector = hidden[0, last].float().numpy()
        if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("Pinned checkpoint produced a nonfinite or zero reference")
        vector = vector / np.linalg.norm(vector)
        cases.append({"text": text, "inputs": inputs, "reference_embedding": vector.tolist()})
    vectors = np.asarray([case["reference_embedding"] for case in cases], dtype=np.float64)
    scores = vectors[:2] @ vectors[2:].T
    if list(np.argmax(scores, axis=1)) != [0, 1]:
        raise ValueError(f"Reference retrieval smoke check failed: {scores.tolist()}")
    write_json(args.output, {
        "model_identity_sha256": manifest["model_identity_sha256"],
        "checkpoint_revision": REVISION, "tokenizer_revision": REVISION,
        "checkpoint_inventory_sha256": identity, "padding_side": args.padding_side,
        "reference_backend": "PyTorch CPU fp32 eager; actual checkpoint weights",
        "toolchain": {name: importlib.metadata.version(name)
                      for name in ("torch", "transformers", "tokenizers", "numpy")},
        "cases": cases, "reference_retrieval_scores": scores.tolist(),
        "expected_top_document": [0, 1],
        "scope": "four real texts; smoke correctness, not MTEB or hardware verification",
    })
    print(f"Prepared {len(cases)} actual tokenizer/reference cases; native validation remains not-run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--padding-side", choices=("left", "right"), default="left")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        prepare(args)
    except (ImportError, ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Blocked Qwen reference preparation: {error}\n")


if __name__ == "__main__":
    main()
