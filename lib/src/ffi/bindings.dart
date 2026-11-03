import 'dart:ffi' as ffi;
import 'c_structures.dart';
import 'native_library.dart';

class LibOnnxBindings {
  final ffi.DynamicLibrary library;

  late final int Function() versionMajor;
  late final int Function() versionMinor;
  late final int Function() versionPatch;
  late final ffi.Pointer<ffi.Char> Function() versionString;
  late final ffi.Pointer<ffi.Char> Function() ortVersionString;

  late final ffi.Pointer<ffi.Char> Function() getLastError;
  late final void Function() clearLastError;

  late final ffi.Pointer<LibOnnxRuntime> Function(int maxConcurrency)
      runtimeCreate;
  late final void Function(ffi.Pointer<LibOnnxRuntime>) runtimeDestroy;
  late final int Function(ffi.Pointer<LibOnnxRuntime>, ffi.Pointer<ffi.Char>)
      runtimeIsProviderSupported;
  late final int Function(
          ffi.Pointer<LibOnnxRuntime>, ffi.Pointer<ffi.Char>, int)
      runtimeGetAvailableProviders;

  late final ffi.Pointer<LibOnnxTensor> Function(
      int dtype,
      ffi.Pointer<ffi.Int64> shape,
      int rank,
      ffi.Pointer<ffi.Void> data,
      int byteLength,
      int copyData) tensorCreate;
  late final void Function(ffi.Pointer<LibOnnxTensor>) tensorDestroy;
  late final int Function(ffi.Pointer<LibOnnxTensor>) tensorGetDtype;
  late final int Function(ffi.Pointer<LibOnnxTensor>) tensorGetRank;
  late final int Function(
      ffi.Pointer<LibOnnxTensor>, ffi.Pointer<ffi.Int64>, int) tensorGetShape;
  late final ffi.Pointer<ffi.Void> Function(ffi.Pointer<LibOnnxTensor>)
      tensorGetData;
  late final int Function(ffi.Pointer<LibOnnxTensor>) tensorGetByteLength;
  late final int Function(ffi.Pointer<LibOnnxTensor>) tensorGetElementCount;

  late final ffi.Pointer<LibOnnxSession> Function(
      ffi.Pointer<LibOnnxRuntime> runtime,
      ffi.Pointer<ffi.Char> modelPath,
      ffi.Pointer<ffi.Void> modelBytes,
      int modelBytesLen,
      ffi.Pointer<ffi.Pointer<ffi.Char>> providers,
      int numProviders,
      ffi.Pointer<ffi.Pointer<ffi.Char>> optionKeys,
      ffi.Pointer<ffi.Pointer<ffi.Char>> optionValues,
      int numOptions,
      int threadCount,
      int fallbackPolicy,
      int hardwareRequirement) sessionCreate;
  late final void Function(ffi.Pointer<LibOnnxSession>) sessionDestroy;

  late final int Function(ffi.Pointer<LibOnnxSession>) sessionGetInputCount;
  late final ffi.Pointer<ffi.Char> Function(ffi.Pointer<LibOnnxSession>, int)
      sessionGetInputName;
  late final int Function(ffi.Pointer<LibOnnxSession>, int) sessionGetInputType;
  late final int Function(
          ffi.Pointer<LibOnnxSession>, int, ffi.Pointer<ffi.Int64>, int)
      sessionGetInputShape;

  late final int Function(ffi.Pointer<LibOnnxSession>) sessionGetOutputCount;
  late final ffi.Pointer<ffi.Char> Function(ffi.Pointer<LibOnnxSession>, int)
      sessionGetOutputName;
  late final int Function(ffi.Pointer<LibOnnxSession>, int)
      sessionGetOutputType;
  late final int Function(
          ffi.Pointer<LibOnnxSession>, int, ffi.Pointer<ffi.Int64>, int)
      sessionGetOutputShape;

  late final int Function(
          ffi.Pointer<LibOnnxSession> session,
          int runId,
          ffi.Pointer<ffi.Pointer<ffi.Char>> inputNames,
          ffi.Pointer<ffi.Pointer<LibOnnxTensor>> inputTensors,
          int numInputs,
          ffi.Pointer<ffi.Pointer<ffi.Char>> outputNames,
          ffi.Pointer<ffi.Pointer<LibOnnxTensor>> outOutputTensors,
          int numOutputs,
          ffi.Pointer<ffi.Pointer<LibOnnxDiagnosticReport>> outDiagnostic)
      sessionRunSync;

  late final int Function(
      ffi.Pointer<LibOnnxSession> session,
      int runId,
      ffi.Pointer<ffi.Pointer<ffi.Char>> inputNames,
      ffi.Pointer<ffi.Pointer<LibOnnxTensor>> inputTensors,
      int numInputs,
      ffi.Pointer<ffi.Pointer<ffi.Char>> outputNames,
      int numOutputs,
      int timeoutMs,
      ffi.Pointer<ffi.NativeFunction<LibOnnxCompletionCallbackNative>> callback,
      ffi.Pointer<ffi.Void> userData) sessionRunAsync;

  late final int Function(ffi.Pointer<LibOnnxSession>, int runId)
      sessionCancelRun;

  late final ffi.Pointer<ffi.Char> Function(
      ffi.Pointer<LibOnnxDiagnosticReport>) diagnosticGetProvider;
  late final int Function(ffi.Pointer<LibOnnxDiagnosticReport>)
      diagnosticGetEvidenceLevel;
  late final int Function(ffi.Pointer<LibOnnxDiagnosticReport>)
      diagnosticGetDurationUs;
  late final ffi.Pointer<ffi.Char> Function(
      ffi.Pointer<LibOnnxDiagnosticReport>) diagnosticGetHardwareDetails;
  late final ffi.Pointer<ffi.Char> Function(
      ffi.Pointer<LibOnnxDiagnosticReport>) diagnosticGetModelHash;
  late final int Function(ffi.Pointer<LibOnnxDiagnosticReport>)
      diagnosticGetFallbackOccurred;
  late final void Function(ffi.Pointer<LibOnnxDiagnosticReport>)
      diagnosticDestroy;

  LibOnnxBindings(this.library) {
    versionMajor = library.lookupFunction<ffi.Int32 Function(), int Function()>(
        'lib_onnx_version_major');
    versionMinor = library.lookupFunction<ffi.Int32 Function(), int Function()>(
        'lib_onnx_version_minor');
    versionPatch = library.lookupFunction<ffi.Int32 Function(), int Function()>(
        'lib_onnx_version_patch');
    versionString = library.lookupFunction<ffi.Pointer<ffi.Char> Function(),
        ffi.Pointer<ffi.Char> Function()>('lib_onnx_version_string');
    ortVersionString = library.lookupFunction<ffi.Pointer<ffi.Char> Function(),
        ffi.Pointer<ffi.Char> Function()>('lib_onnx_ort_version_string');

    getLastError = library.lookupFunction<ffi.Pointer<ffi.Char> Function(),
        ffi.Pointer<ffi.Char> Function()>('lib_onnx_get_last_error');
    clearLastError =
        library.lookupFunction<ffi.Void Function(), void Function()>(
            'lib_onnx_clear_last_error');

    runtimeCreate = library.lookupFunction<
        ffi.Pointer<LibOnnxRuntime> Function(ffi.Int32),
        ffi.Pointer<LibOnnxRuntime> Function(int)>('lib_onnx_runtime_create');
    runtimeDestroy = library.lookupFunction<
        ffi.Void Function(ffi.Pointer<LibOnnxRuntime>),
        void Function(ffi.Pointer<LibOnnxRuntime>)>('lib_onnx_runtime_destroy');
    runtimeIsProviderSupported = library.lookupFunction<
        ffi.Int32 Function(ffi.Pointer<LibOnnxRuntime>, ffi.Pointer<ffi.Char>),
        int Function(ffi.Pointer<LibOnnxRuntime>,
            ffi.Pointer<ffi.Char>)>('lib_onnx_runtime_is_provider_supported');
    runtimeGetAvailableProviders = library.lookupFunction<
        ffi.Int32 Function(
            ffi.Pointer<LibOnnxRuntime>, ffi.Pointer<ffi.Char>, ffi.Size),
        int Function(ffi.Pointer<LibOnnxRuntime>, ffi.Pointer<ffi.Char>,
            int)>('lib_onnx_runtime_get_available_providers');

    tensorCreate = library.lookupFunction<
        ffi.Pointer<LibOnnxTensor> Function(ffi.Int32, ffi.Pointer<ffi.Int64>,
            ffi.Int32, ffi.Pointer<ffi.Void>, ffi.Size, ffi.Int32),
        ffi.Pointer<LibOnnxTensor> Function(int, ffi.Pointer<ffi.Int64>, int,
            ffi.Pointer<ffi.Void>, int, int)>('lib_onnx_tensor_create');
    tensorDestroy = library.lookupFunction<
        ffi.Void Function(ffi.Pointer<LibOnnxTensor>),
        void Function(ffi.Pointer<LibOnnxTensor>)>('lib_onnx_tensor_destroy');
    tensorGetDtype = library.lookupFunction<
        ffi.Int32 Function(ffi.Pointer<LibOnnxTensor>),
        int Function(ffi.Pointer<LibOnnxTensor>)>('lib_onnx_tensor_get_dtype');
    tensorGetRank = library.lookupFunction<
        ffi.Int32 Function(ffi.Pointer<LibOnnxTensor>),
        int Function(ffi.Pointer<LibOnnxTensor>)>('lib_onnx_tensor_get_rank');
    tensorGetShape = library.lookupFunction<
        ffi.Int32 Function(
            ffi.Pointer<LibOnnxTensor>, ffi.Pointer<ffi.Int64>, ffi.Int32),
        int Function(ffi.Pointer<LibOnnxTensor>, ffi.Pointer<ffi.Int64>,
            int)>('lib_onnx_tensor_get_shape');
    tensorGetData = library.lookupFunction<
        ffi.Pointer<ffi.Void> Function(ffi.Pointer<LibOnnxTensor>),
        ffi.Pointer<ffi.Void> Function(
            ffi.Pointer<LibOnnxTensor>)>('lib_onnx_tensor_get_data');
    tensorGetByteLength = library.lookupFunction<
        ffi.Size Function(ffi.Pointer<LibOnnxTensor>),
        int Function(
            ffi.Pointer<LibOnnxTensor>)>('lib_onnx_tensor_get_byte_length');
    tensorGetElementCount = library.lookupFunction<
        ffi.Int64 Function(ffi.Pointer<LibOnnxTensor>),
        int Function(
            ffi.Pointer<LibOnnxTensor>)>('lib_onnx_tensor_get_element_count');

    sessionCreate = library.lookupFunction<
        ffi.Pointer<LibOnnxSession> Function(
            ffi.Pointer<LibOnnxRuntime>,
            ffi.Pointer<ffi.Char>,
            ffi.Pointer<ffi.Void>,
            ffi.Size,
            ffi.Pointer<ffi.Pointer<ffi.Char>>,
            ffi.Int32,
            ffi.Pointer<ffi.Pointer<ffi.Char>>,
            ffi.Pointer<ffi.Pointer<ffi.Char>>,
            ffi.Int32,
            ffi.Int32,
            ffi.Int32,
            ffi.Int32),
        ffi.Pointer<LibOnnxSession> Function(
            ffi.Pointer<LibOnnxRuntime>,
            ffi.Pointer<ffi.Char>,
            ffi.Pointer<ffi.Void>,
            int,
            ffi.Pointer<ffi.Pointer<ffi.Char>>,
            int,
            ffi.Pointer<ffi.Pointer<ffi.Char>>,
            ffi.Pointer<ffi.Pointer<ffi.Char>>,
            int,
            int,
            int,
            int)>('lib_onnx_session_create');
    sessionDestroy = library.lookupFunction<
        ffi.Void Function(ffi.Pointer<LibOnnxSession>),
        void Function(ffi.Pointer<LibOnnxSession>)>('lib_onnx_session_destroy');

    sessionGetInputCount = library.lookupFunction<
        ffi.Int32 Function(ffi.Pointer<LibOnnxSession>),
        int Function(
            ffi.Pointer<LibOnnxSession>)>('lib_onnx_session_get_input_count');
    sessionGetInputName = library.lookupFunction<
        ffi.Pointer<ffi.Char> Function(ffi.Pointer<LibOnnxSession>, ffi.Int32),
        ffi.Pointer<ffi.Char> Function(ffi.Pointer<LibOnnxSession>,
            int)>('lib_onnx_session_get_input_name');
    sessionGetInputType = library.lookupFunction<
        ffi.Int32 Function(ffi.Pointer<LibOnnxSession>, ffi.Int32),
        int Function(ffi.Pointer<LibOnnxSession>,
            int)>('lib_onnx_session_get_input_type');
    sessionGetInputShape = library.lookupFunction<
        ffi.Int32 Function(ffi.Pointer<LibOnnxSession>, ffi.Int32,
            ffi.Pointer<ffi.Int64>, ffi.Int32),
        int Function(ffi.Pointer<LibOnnxSession>, int, ffi.Pointer<ffi.Int64>,
            int)>('lib_onnx_session_get_input_shape');

    sessionGetOutputCount = library.lookupFunction<
        ffi.Int32 Function(ffi.Pointer<LibOnnxSession>),
        int Function(
            ffi.Pointer<LibOnnxSession>)>('lib_onnx_session_get_output_count');
    sessionGetOutputName = library.lookupFunction<
        ffi.Pointer<ffi.Char> Function(ffi.Pointer<LibOnnxSession>, ffi.Int32),
        ffi.Pointer<ffi.Char> Function(ffi.Pointer<LibOnnxSession>,
            int)>('lib_onnx_session_get_output_name');
    sessionGetOutputType = library.lookupFunction<
        ffi.Int32 Function(ffi.Pointer<LibOnnxSession>, ffi.Int32),
        int Function(ffi.Pointer<LibOnnxSession>,
            int)>('lib_onnx_session_get_output_type');
    sessionGetOutputShape = library.lookupFunction<
        ffi.Int32 Function(ffi.Pointer<LibOnnxSession>, ffi.Int32,
            ffi.Pointer<ffi.Int64>, ffi.Int32),
        int Function(ffi.Pointer<LibOnnxSession>, int, ffi.Pointer<ffi.Int64>,
            int)>('lib_onnx_session_get_output_shape');

    sessionRunSync = library.lookupFunction<
            ffi.Int32 Function(
                ffi.Pointer<LibOnnxSession>,
                ffi.Int64,
                ffi.Pointer<ffi.Pointer<ffi.Char>>,
                ffi.Pointer<ffi.Pointer<LibOnnxTensor>>,
                ffi.Int32,
                ffi.Pointer<ffi.Pointer<ffi.Char>>,
                ffi.Pointer<ffi.Pointer<LibOnnxTensor>>,
                ffi.Int32,
                ffi.Pointer<ffi.Pointer<LibOnnxDiagnosticReport>>),
            int Function(
                ffi.Pointer<LibOnnxSession>,
                int,
                ffi.Pointer<ffi.Pointer<ffi.Char>>,
                ffi.Pointer<ffi.Pointer<LibOnnxTensor>>,
                int,
                ffi.Pointer<ffi.Pointer<ffi.Char>>,
                ffi.Pointer<ffi.Pointer<LibOnnxTensor>>,
                int,
                ffi.Pointer<ffi.Pointer<LibOnnxDiagnosticReport>>)>(
        'lib_onnx_session_run_sync');

    sessionRunAsync = library.lookupFunction<
        ffi.Int32 Function(
            ffi.Pointer<LibOnnxSession>,
            ffi.Int64,
            ffi.Pointer<ffi.Pointer<ffi.Char>>,
            ffi.Pointer<ffi.Pointer<LibOnnxTensor>>,
            ffi.Int32,
            ffi.Pointer<ffi.Pointer<ffi.Char>>,
            ffi.Int32,
            ffi.Int64,
            ffi.Pointer<ffi.NativeFunction<LibOnnxCompletionCallbackNative>>,
            ffi.Pointer<ffi.Void>),
        int Function(
            ffi.Pointer<LibOnnxSession>,
            int,
            ffi.Pointer<ffi.Pointer<ffi.Char>>,
            ffi.Pointer<ffi.Pointer<LibOnnxTensor>>,
            int,
            ffi.Pointer<ffi.Pointer<ffi.Char>>,
            int,
            int,
            ffi.Pointer<ffi.NativeFunction<LibOnnxCompletionCallbackNative>>,
            ffi.Pointer<ffi.Void>)>('lib_onnx_session_run_async');

    sessionCancelRun = library.lookupFunction<
        ffi.Int32 Function(ffi.Pointer<LibOnnxSession>, ffi.Int64),
        int Function(
            ffi.Pointer<LibOnnxSession>, int)>('lib_onnx_session_cancel_run');

    diagnosticGetProvider =
        library.lookupFunction<
                ffi.Pointer<ffi.Char> Function(
                    ffi.Pointer<LibOnnxDiagnosticReport>),
                ffi.Pointer<ffi.Char> Function(
                    ffi.Pointer<LibOnnxDiagnosticReport>)>(
            'lib_onnx_diagnostic_get_provider');
    diagnosticGetEvidenceLevel = library.lookupFunction<
            ffi.Int32 Function(ffi.Pointer<LibOnnxDiagnosticReport>),
            int Function(ffi.Pointer<LibOnnxDiagnosticReport>)>(
        'lib_onnx_diagnostic_get_evidence_level');
    diagnosticGetDurationUs = library.lookupFunction<
            ffi.Int64 Function(ffi.Pointer<LibOnnxDiagnosticReport>),
            int Function(ffi.Pointer<LibOnnxDiagnosticReport>)>(
        'lib_onnx_diagnostic_get_duration_us');
    diagnosticGetHardwareDetails =
        library.lookupFunction<
                ffi.Pointer<ffi.Char> Function(
                    ffi.Pointer<LibOnnxDiagnosticReport>),
                ffi.Pointer<ffi.Char> Function(
                    ffi.Pointer<LibOnnxDiagnosticReport>)>(
            'lib_onnx_diagnostic_get_hardware_details');
    diagnosticGetModelHash =
        library.lookupFunction<
                ffi.Pointer<ffi.Char> Function(
                    ffi.Pointer<LibOnnxDiagnosticReport>),
                ffi.Pointer<ffi.Char> Function(
                    ffi.Pointer<LibOnnxDiagnosticReport>)>(
            'lib_onnx_diagnostic_get_model_hash');
    diagnosticGetFallbackOccurred = library.lookupFunction<
            ffi.Int32 Function(ffi.Pointer<LibOnnxDiagnosticReport>),
            int Function(ffi.Pointer<LibOnnxDiagnosticReport>)>(
        'lib_onnx_diagnostic_get_fallback_occurred');
    diagnosticDestroy = library.lookupFunction<
            ffi.Void Function(ffi.Pointer<LibOnnxDiagnosticReport>),
            void Function(ffi.Pointer<LibOnnxDiagnosticReport>)>(
        'lib_onnx_diagnostic_destroy');
  }

  static LibOnnxBindings? _instance;
  static LibOnnxBindings get instance {
    _instance ??= LibOnnxBindings(NativeLibrary.load());
    return _instance!;
  }

  static void init({String? libraryPath}) {
    _instance = LibOnnxBindings(NativeLibrary.load(overridePath: libraryPath));
  }
}
