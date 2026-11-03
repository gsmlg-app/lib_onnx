/// lib_onnx: Hardware-accelerated ONNX Runtime inference library for Dart and Flutter.
///
/// Features:
/// - CoreML execution on Apple Silicon (GPU & Neural Engine / ANE)
/// - Qualcomm Snapdragon QNN HTP (Hexagon NPU)
/// - Numerical tensors: float32, float16, int64, int32, int8, uint8, bool
/// - Explicit CPU fallback policy & transparent hardware evidence diagnostics
/// - Cooperative per-request cancellation
/// - Embedding adapter with last-token pooling & L2 normalization
library lib_onnx;

// Runtime
export 'src/runtime/onnx_runtime.dart';

// Session
export 'src/session/inference_session.dart';
export 'src/session/session_options.dart';
export 'src/session/run_options.dart';

// Tensor
export 'src/tensor/onnx_tensor.dart';
export 'src/tensor/tensor_type.dart';

// Providers and Execution Policy
export 'src/provider/execution_provider.dart';
export 'src/provider/execution_policy.dart';

// Diagnostics and Hardware Evidence
export 'src/diagnostics/diagnostic_report.dart';
export 'src/diagnostics/hardware_evidence.dart';

// Adapters
export 'src/adapter/embedding_adapter.dart';
export 'src/adapter/model_manifest.dart';

// FFI Bindings (optional advanced usage)
export 'src/ffi/bindings.dart' show LibOnnxBindings;
