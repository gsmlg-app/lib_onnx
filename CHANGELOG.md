# Changelog

## 0.1.0

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
