# Changelog

## 0.2.0 (2026-11-05)

- Stable CPU inference on macOS arm64 with bridge ABI2 and ONNX Runtime 1.31.1. CoreML/QNN, iOS/Android and full Qwen acceptance remain experimental or incomplete.
- Release workflows precompile macOS arm64 and Linux x64/arm64 bundles, retain license notices and binary manifests, and publish source archives with SHA-256 checksums. Linux bundles remain CPU correctness inputs outside the stable platform claim.
- Updated package, example and native bridge versions together; stable publication retains the platform acceptance gate.

## 0.2.0-dev.1 (2026-11-04)

- **Breaking:** bridge ABI2 and Dart3.10+ native-assets packaging; pure Dart core no longer depends on Flutter. Old bridges/ORT mismatches are rejected.
- Checked tensor sizes, scalars/empty tensors/arbitrary ranks, complete partial-result cleanup, task-owned input copies and retained result handles.
- Real asynchronous initialization/loading/run/close, per-task termination, bounded scheduling, monotonic timeouts, removable cancellation listeners and independent errors/diagnostics.
- Enforced ORT CPU fallback constraint, explicit unsupported whitelist/hardware verification results, correct CoreML options/content caches, QNN HTP selection independent of Apple hardware.
- Removed inferred hardware observations and simulated example results; model identity includes external data. Added adapter safety/fp16 support, real small-model example and pinned genuine Qwen validation workflow.
- Added sanitizer/lifecycle/clean-consumer checks and release acceptance gates. iOS/Android accelerator and full Qwen acceptance remain open; no stable release or production acceleration claim.

## 0.1.0

Historical release notes below are retained. The2026-11-04 review found that async, platform readiness,16KB and hardware-observation claims were not established by the original evidence. They are superseded by the remediation report and current device matrix.


- Initial release of `lib_onnx`: Hardware-accelerated ONNX Runtime inference library for Dart and Flutter.
- Supported execution providers:
  - Apple CoreML Execution Provider (Apple Silicon GPU & Apple Neural Engine / ANE) on macOS (arm64) and iOS.
  - Qualcomm Snapdragon QNN HTP Execution Provider (Hexagon NPU) on Android (arm64-v8a) with 16 KB page compatibility.
  - ORT CPU Execution Provider for numeric baseline and explicit fallback.
- Pure C ABI bridge with opaque handles, non-blocking native worker thread pool for Flutter UI isolate safety.
- Three-dimensional `ExecutionPolicy`: Provider priority list, CPU fallback policy (`strict`, `allowed`, `whitelistedOps`), and hardware requirement (`any`, `gpu`, `npu`).
- Transparent 6-level hardware evidence system (`DiagnosticReport` and `EvidenceLevel`).
- Numerical tensors: `float32`, `float16`, `int64`, `int32`, `int8`, `uint8`, `bool`.
- Cooperative per-request cancellation (`CancellationToken`).
- `EmbeddingAdapter` implementing official Qwen3-Embedding Last-token pooling and L2 normalization.
- Interactive Flutter example app in `example/`.
- Automated test suite covering tensor operations, execution policy, embedding adapter, lifecycle, and 1,000 continuous inferences.
