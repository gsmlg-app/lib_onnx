import 'dart:ffi' as ffi;

// Native opaque pointers
final class LibOnnxRuntime extends ffi.Opaque {}

final class LibOnnxSession extends ffi.Opaque {}

final class LibOnnxTensor extends ffi.Opaque {}

final class LibOnnxDiagnosticReport extends ffi.Opaque {}

// Completion callback signature
typedef LibOnnxCompletionCallbackNative = ffi.Void Function(
  ffi.Int64 runId,
  ffi.Int32 statusCode,
  ffi.Pointer<ffi.Char> errorMsg,
  ffi.Pointer<ffi.Pointer<LibOnnxTensor>> outputs,
  ffi.Int32 numOutputs,
  ffi.Pointer<LibOnnxDiagnosticReport> diagnostic,
  ffi.Pointer<ffi.Void> userData,
);

typedef LibOnnxCompletionCallbackDart = void Function(
  int runId,
  int statusCode,
  ffi.Pointer<ffi.Char> errorMsg,
  ffi.Pointer<ffi.Pointer<LibOnnxTensor>> outputs,
  int numOutputs,
  ffi.Pointer<LibOnnxDiagnosticReport> diagnostic,
  ffi.Pointer<ffi.Void> userData,
);
