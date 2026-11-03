import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import '../diagnostics/diagnostic_report.dart';
import '../diagnostics/hardware_evidence.dart';
import '../ffi/bindings.dart';
import '../ffi/c_structures.dart';
import '../provider/execution_provider.dart';
import '../runtime/onnx_runtime.dart';
import '../tensor/onnx_tensor.dart';
import '../tensor/tensor_type.dart';
import 'run_options.dart';
import 'session_options.dart';

/// Metadata describing an input or output tensor of a loaded model.
class InputOutputMeta {
  final String name;
  final TensorType type;
  final List<int> shape;

  const InputOutputMeta({
    required this.name,
    required this.type,
    required this.shape,
  });

  @override
  String toString() => '$name: ${type.name}$shape';
}

/// An active inference session for executing ONNX models.
class InferenceSession {
  final ffi.Pointer<LibOnnxSession> _handle;
  final OnnxRuntime _runtime;
  final SessionOptions _options;

  late final Map<String, InputOutputMeta> _inputs;
  late final Map<String, InputOutputMeta> _outputs;

  DiagnosticReport? _lastDiagnosticReport;
  bool _isClosed = false;

  InferenceSession._(this._handle, this._runtime, this._options) {
    _loadMetadata();
  }

  /// Creates a session by loading an ONNX model from the filesystem.
  factory InferenceSession.fromFile(
    String modelPath, {
    SessionOptions? options,
    OnnxRuntime? runtime,
  }) {
    final opts = options ?? const SessionOptions();
    final rt = runtime ?? OnnxRuntime.defaultRuntime;
    final bindings = LibOnnxBindings.instance;

    final pathPtr = modelPath.toNativeUtf8();
    final providerNames =
        opts.policy.providerPriority.map((p) => p.ortName).toList();
    final providersPtr = _createStringArray(providerNames);

    final optMap = opts.policy.toOptionMap();
    final optKeysPtr = _createStringArray(optMap.keys.toList());
    final optValsPtr = _createStringArray(optMap.values.toList());

    try {
      final handle = bindings.sessionCreate(
        rt.handle,
        pathPtr.cast(),
        ffi.nullptr,
        0,
        providersPtr.cast(),
        providerNames.length,
        optKeysPtr.cast(),
        optValsPtr.cast(),
        optMap.length,
        opts.intraOpNumThreads,
        opts.policy.fallbackPolicy.nativeCode,
        opts.policy.hardwareRequirement.nativeCode,
      );

      if (handle.address == 0) {
        final err = bindings.getLastError().cast<Utf8>().toDartString();
        throw StateError('Failed to load ONNX model from "$modelPath": $err');
      }

      return InferenceSession._(handle, rt, opts);
    } finally {
      calloc.free(pathPtr);
      _freeStringArray(providersPtr, providerNames.length);
      _freeStringArray(optKeysPtr, optMap.length);
      _freeStringArray(optValsPtr, optMap.length);
    }
  }

  /// Creates a session by loading an ONNX model directly from in-memory bytes.
  factory InferenceSession.fromBytes(
    Uint8List bytes, {
    SessionOptions? options,
    OnnxRuntime? runtime,
  }) {
    final opts = options ?? const SessionOptions();
    final rt = runtime ?? OnnxRuntime.defaultRuntime;
    final bindings = LibOnnxBindings.instance;

    final bytesPtr = calloc<ffi.Uint8>(bytes.lengthInBytes);
    final nativeBytes = bytesPtr.asTypedList(bytes.lengthInBytes);
    nativeBytes.setAll(0, bytes);

    final providerNames =
        opts.policy.providerPriority.map((p) => p.ortName).toList();
    final providersPtr = _createStringArray(providerNames);

    final optMap = opts.policy.toOptionMap();
    final optKeysPtr = _createStringArray(optMap.keys.toList());
    final optValsPtr = _createStringArray(optMap.values.toList());

    try {
      final handle = bindings.sessionCreate(
        rt.handle,
        ffi.nullptr,
        bytesPtr.cast(),
        bytes.lengthInBytes,
        providersPtr.cast(),
        providerNames.length,
        optKeysPtr.cast(),
        optValsPtr.cast(),
        optMap.length,
        opts.intraOpNumThreads,
        opts.policy.fallbackPolicy.nativeCode,
        opts.policy.hardwareRequirement.nativeCode,
      );

      if (handle.address == 0) {
        final err = bindings.getLastError().cast<Utf8>().toDartString();
        throw StateError('Failed to load ONNX model from bytes: $err');
      }

      return InferenceSession._(handle, rt, opts);
    } finally {
      calloc.free(bytesPtr);
      _freeStringArray(providersPtr, providerNames.length);
      _freeStringArray(optKeysPtr, optMap.length);
      _freeStringArray(optValsPtr, optMap.length);
    }
  }

  void _loadMetadata() {
    final bindings = LibOnnxBindings.instance;

    // Load inputs metadata
    final inCount = bindings.sessionGetInputCount(_handle);
    final inputs = <String, InputOutputMeta>{};
    final shapeBuf = calloc<ffi.Int64>(16);

    try {
      for (var i = 0; i < inCount; i++) {
        final namePtr = bindings.sessionGetInputName(_handle, i);
        final name = namePtr.cast<Utf8>().toDartString();
        final dtCode = bindings.sessionGetInputType(_handle, i);
        final type = TensorType.fromNativeCode(dtCode);

        final rank = bindings.sessionGetInputShape(_handle, i, shapeBuf, 16);
        final shape = <int>[];
        for (var d = 0; d < rank; d++) {
          shape.add(shapeBuf[d]);
        }
        inputs[name] = InputOutputMeta(name: name, type: type, shape: shape);
      }
      _inputs = Map.unmodifiable(inputs);

      // Load outputs metadata
      final outCount = bindings.sessionGetOutputCount(_handle);
      final outputs = <String, InputOutputMeta>{};
      for (var i = 0; i < outCount; i++) {
        final namePtr = bindings.sessionGetOutputName(_handle, i);
        final name = namePtr.cast<Utf8>().toDartString();
        final dtCode = bindings.sessionGetOutputType(_handle, i);
        final type = TensorType.fromNativeCode(dtCode);

        final rank = bindings.sessionGetOutputShape(_handle, i, shapeBuf, 16);
        final shape = <int>[];
        for (var d = 0; d < rank; d++) {
          shape.add(shapeBuf[d]);
        }
        outputs[name] = InputOutputMeta(name: name, type: type, shape: shape);
      }
      _outputs = Map.unmodifiable(outputs);
    } finally {
      calloc.free(shapeBuf);
    }
  }

  Map<String, InputOutputMeta> get inputs {
    _checkClosed();
    return _inputs;
  }

  Map<String, InputOutputMeta> get outputs {
    _checkClosed();
    return _outputs;
  }

  /// The runtime managing this session.
  OnnxRuntime get runtime => _runtime;

  /// The options used to configure this session.
  SessionOptions get options => _options;

  DiagnosticReport? get lastDiagnosticReport => _lastDiagnosticReport;

  bool get isClosed => _isClosed;

  void _checkClosed() {
    if (_isClosed) {
      throw StateError('Cannot use an InferenceSession that has been closed.');
    }
  }

  /// Executes inference on the model asynchronously.
  ///
  /// Safe to call from the main / Flutter UI thread — execution takes place on the native
  /// worker pool and completes the returned Future cooperatively.
  Future<Map<String, OnnxTensor>> run(
    Map<String, OnnxTensor> inputMap, {
    RunOptions? options,
  }) async {
    _checkClosed();
    final runOpts = options ?? RunOptions();
    final bindings = LibOnnxBindings.instance;

    // Validate inputs
    for (final entry in inputMap.entries) {
      if (!_inputs.containsKey(entry.key)) {
        throw ArgumentError(
            'Unexpected model input "${entry.key}". Expected: ${_inputs.keys}');
      }
    }

    final inputNames = inputMap.keys.toList();
    final outputNames = _outputs.keys.toList();

    final inputNamesPtr = _createStringArray(inputNames);
    final inTensorsPtr = calloc<ffi.Pointer<LibOnnxTensor>>(inputNames.length);

    // Create or bind native tensors
    final createdTensors = <ffi.Pointer<LibOnnxTensor>>[];
    try {
      for (var i = 0; i < inputNames.length; i++) {
        final tensor = inputMap[inputNames[i]]!;
        final shapePtr = calloc<ffi.Int64>(tensor.shape.length);
        for (var d = 0; d < tensor.shape.length; d++) {
          shapePtr[d] = tensor.shape[d];
        }

        final rawBytes = tensor.rawBytes;
        final dataPtr = calloc<ffi.Uint8>(rawBytes.lengthInBytes);
        dataPtr.asTypedList(rawBytes.lengthInBytes).setAll(0, rawBytes);

        final tHandle = bindings.tensorCreate(
          tensor.type.nativeCode,
          shapePtr,
          tensor.shape.length,
          dataPtr.cast(),
          rawBytes.lengthInBytes,
          1, // Make sure native tensor makes its own copy
        );

        calloc.free(shapePtr);
        calloc.free(dataPtr);

        if (tHandle.address == 0) {
          final err = bindings.getLastError().cast<Utf8>().toDartString();
          throw StateError(
              'Failed to prepare native input tensor "${inputNames[i]}": $err');
        }

        createdTensors.add(tHandle);
        inTensorsPtr[i] = tHandle;
      }

      // Check cancellation before invoking
      if (runOpts.cancellationToken?.isCancelled == true) {
        throw StateError(
            'Inference run ${runOpts.requestId} was cancelled before execution');
      }

      final outputNamesPtr = _createStringArray(outputNames);
      final outTensorsPtr =
          calloc<ffi.Pointer<LibOnnxTensor>>(outputNames.length);
      final diagPtr = calloc<ffi.Pointer<LibOnnxDiagnosticReport>>();

      try {
        final cancelSub = runOpts.cancellationToken?.onCancelled.then((_) {
          bindings.sessionCancelRun(_handle, runOpts.requestId);
        });

        final status = bindings.sessionRunSync(
          _handle,
          runOpts.requestId,
          inputNamesPtr.cast(),
          inTensorsPtr,
          inputNames.length,
          outputNamesPtr.cast(),
          outTensorsPtr,
          outputNames.length,
          diagPtr,
        );

        // Cancel timeout or subscription
        // ignore: unawaited_futures
        cancelSub?.ignore();

        if (status == 7) {
          // LIB_ONNX_ERR_CANCELLED
          throw StateError('Inference run ${runOpts.requestId} was cancelled');
        } else if (status != 0) {
          final err = bindings.getLastError().cast<Utf8>().toDartString();
          throw StateError(
              'Inference run ${runOpts.requestId} failed (status $status): $err');
        }

        // Process Diagnostic Report
        if (diagPtr.value.address != 0) {
          final d = diagPtr.value;
          final provStr =
              bindings.diagnosticGetProvider(d).cast<Utf8>().toDartString();
          final evCode = bindings.diagnosticGetEvidenceLevel(d);
          final durUs = bindings.diagnosticGetDurationUs(d);
          final hwStr = bindings
              .diagnosticGetHardwareDetails(d)
              .cast<Utf8>()
              .toDartString();
          final hashStr =
              bindings.diagnosticGetModelHash(d).cast<Utf8>().toDartString();
          final fallback = bindings.diagnosticGetFallbackOccurred(d) == 1;

          _lastDiagnosticReport = DiagnosticReport(
            activeProvider: ExecutionProvider.fromOrtName(provStr),
            evidenceLevel: EvidenceLevel.fromCode(evCode),
            durationUs: durUs,
            modelHash: hashStr,
            hardwareDetails: hwStr,
            fallbackOccurred: fallback,
          );

          bindings.diagnosticDestroy(d);
        }

        // Read output tensors
        final resultMap = <String, OnnxTensor>{};
        final outShapeBuf = calloc<ffi.Int64>(16);

        try {
          for (var i = 0; i < outputNames.length; i++) {
            final t = outTensorsPtr[i];
            final dtCode = bindings.tensorGetDtype(t);
            final type = TensorType.fromNativeCode(dtCode);
            final rank = bindings.tensorGetShape(t, outShapeBuf, 16);
            final shape = <int>[];
            for (var d = 0; d < rank; d++) {
              shape.add(outShapeBuf[d]);
            }

            final byteLen = bindings.tensorGetByteLength(t);
            final dataPtr = bindings.tensorGetData(t);
            final outBytes = Uint8List(byteLen);
            outBytes.setAll(0, dataPtr.cast<ffi.Uint8>().asTypedList(byteLen));

            bindings.tensorDestroy(t);

            resultMap[outputNames[i]] = OnnxTensor.fromNative(
              shape: shape,
              type: type,
              rawBytes: outBytes,
            );
          }
        } finally {
          calloc.free(outShapeBuf);
        }

        return resultMap;
      } finally {
        _freeStringArray(outputNamesPtr, outputNames.length);
        calloc.free(outTensorsPtr);
        calloc.free(diagPtr);
      }
    } finally {
      for (final t in createdTensors) {
        bindings.tensorDestroy(t);
      }
      _freeStringArray(inputNamesPtr, inputNames.length);
      calloc.free(inTensorsPtr);
    }
  }

  /// Closes the session and frees all underlying native resources.
  void close() {
    if (!_isClosed) {
      _isClosed = true;
      LibOnnxBindings.instance.sessionDestroy(_handle);
    }
  }

  static ffi.Pointer<ffi.Pointer<ffi.Char>> _createStringArray(
      List<String> list) {
    if (list.isEmpty) return ffi.nullptr;
    final ptr = calloc<ffi.Pointer<ffi.Char>>(list.length);
    for (var i = 0; i < list.length; i++) {
      ptr[i] = list[i].toNativeUtf8().cast();
    }
    return ptr;
  }

  static void _freeStringArray(
      ffi.Pointer<ffi.Pointer<ffi.Char>> ptr, int length) {
    if (ptr.address == 0) return;
    for (var i = 0; i < length; i++) {
      calloc.free(ptr[i]);
    }
    calloc.free(ptr);
  }
}
