import 'dart:async';
import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:ffi/ffi.dart';
import 'package:path/path.dart' as p;
import '../diagnostics/diagnostic_report.dart';
import '../diagnostics/hardware_evidence.dart';
import '../diagnostics/onnx_exception.dart';
import '../ffi/bindings.dart';
import '../ffi/c_structures.dart';
import '../provider/execution_provider.dart';
import '../runtime/onnx_runtime.dart';
import '../tensor/onnx_tensor.dart';
import '../tensor/tensor_type.dart';
import 'model_identity.dart';
import 'run_options.dart';
import 'session_options.dart';

class InputOutputMeta {
  final String name;
  final TensorType type;

  /// Model constraints; negative dimensions are dynamic, unlike actual tensors.
  final List<int> shape;
  const InputOutputMeta({
    required this.name,
    required this.type,
    required this.shape,
  });
  @override
  String toString() => '$name: ${type.name}$shape';
}

class InferenceResult {
  final Map<String, OnnxTensor> outputs;
  final DiagnosticReport diagnostic;
  const InferenceResult({required this.outputs, required this.diagnostic});
}

/// A Session owns accepted tasks until their native completion is acknowledged.
class InferenceSession {
  final ffi.Pointer<LibOnnxSession> _handle;
  final OnnxRuntime _runtime;
  final SessionOptions _options;
  final String _modelHash;
  final Map<String, InputOutputMeta> _inputs, _outputs;
  bool _isClosed = false;
  Future<void>? _closing;
  final _pending = <Future<void>>{};
  DiagnosticReport? _lastDiagnosticReport;
  InferenceSession._(
    this._handle,
    this._runtime,
    this._options,
    this._modelHash,
    this._inputs,
    this._outputs,
  );

  /// Recommended: all file IO, hashing, ORT initialization and EP compilation
  /// happen on an execution isolate, with the UI/control isolate responsive.
  static Future<InferenceSession> loadFile(
    String modelPath, {
    SessionOptions? options,
    OnnxRuntime? runtime,
  }) async => _load(
    modelPath,
    null,
    options ?? const SessionOptions(),
    runtime ?? await OnnxRuntime.defaultRuntimeAsync,
  );
  static Future<InferenceSession> loadBytes(
    Uint8List bytes, {
    SessionOptions? options,
    OnnxRuntime? runtime,
  }) async => _load(
    null,
    Uint8List.fromList(bytes),
    options ?? const SessionOptions(),
    runtime ?? await OnnxRuntime.defaultRuntimeAsync,
  );
  static Future<InferenceSession> _load(
    String? path,
    Uint8List? bytes,
    SessionOptions opts,
    OnnxRuntime rt,
  ) async {
    _validateOptions(opts);
    final address = rt.handle.address, libPath = rt.libraryPath;
    rt.beginLoad();
    try {
      final loaded = await Isolate.run(
        () => _createNative(address, libPath, path, bytes, opts),
      );
      final s = _fromLoaded(loaded, rt, opts);
      if (rt.isDisposed) {
        await s.close();
        throw StateError('Runtime closed during model loading');
      }
      rt.registerSession(s, s.close);
      return s;
    } finally {
      rt.endLoad();
    }
  }

  @Deprecated('Blocks the calling isolate; use loadFile()')
  factory InferenceSession.fromFile(
    String modelPath, {
    SessionOptions? options,
    OnnxRuntime? runtime,
  }) => InferenceSession.fromFileSync(
    modelPath,
    options: options,
    runtime: runtime,
  );
  factory InferenceSession.fromFileSync(
    String modelPath, {
    SessionOptions? options,
    OnnxRuntime? runtime,
  }) {
    final rt = runtime ?? OnnxRuntime.defaultRuntime,
        opts = options ?? const SessionOptions();
    _validateOptions(opts);
    final s = _fromLoaded(
      _createNative(rt.handle.address, rt.libraryPath, modelPath, null, opts),
      rt,
      opts,
    );
    rt.registerSession(s, s.close);
    return s;
  }
  @Deprecated('Blocks the calling isolate; use loadBytes()')
  factory InferenceSession.fromBytes(
    Uint8List bytes, {
    SessionOptions? options,
    OnnxRuntime? runtime,
  }) =>
      InferenceSession.fromBytesSync(bytes, options: options, runtime: runtime);
  factory InferenceSession.fromBytesSync(
    Uint8List bytes, {
    SessionOptions? options,
    OnnxRuntime? runtime,
  }) {
    final rt = runtime ?? OnnxRuntime.defaultRuntime,
        opts = options ?? const SessionOptions();
    _validateOptions(opts);
    final s = _fromLoaded(
      _createNative(rt.handle.address, rt.libraryPath, null, bytes, opts),
      rt,
      opts,
    );
    rt.registerSession(s, s.close);
    return s;
  }
  static InferenceSession _fromLoaded(
    _Loaded l,
    OnnxRuntime rt,
    SessionOptions o,
  ) => InferenceSession._(
    ffi.Pointer<LibOnnxSession>.fromAddress(l.address),
    rt,
    o,
    l.hash,
    l.inputs,
    l.outputs,
  );
  static void _validateOptions(SessionOptions o) {
    o.policy.validate();
    if (o.maxQueueDepth < 0 ||
        o.maxQueueDepth > 65536 ||
        o.maxConcurrentRuns < 1 ||
        o.maxConcurrentRuns > 64 ||
        o.intraOpNumThreads < 1 ||
        o.interOpNumThreads < 1)
      throw ArgumentError('Invalid Session queue/concurrency/thread options');
  }

  Map<String, InputOutputMeta> get inputs {
    _checkOpen();
    return _inputs;
  }

  Map<String, InputOutputMeta> get outputs {
    _checkOpen();
    return _outputs;
  }

  OnnxRuntime get runtime => _runtime;
  SessionOptions get options => _options;

  /// Last successfully decoded completion on this Dart isolate. Concurrent callers
  /// must use runWithDiagnostics() to obtain their own task's report.
  DiagnosticReport? get lastDiagnosticReport => _lastDiagnosticReport;
  bool get isClosed => _isClosed;
  void _checkOpen() {
    if (_isClosed || _runtime.isDisposed)
      throw StateError('Session or runtime is closed');
  }

  Future<Map<String, OnnxTensor>> run(
    Map<String, OnnxTensor> inputs, {
    RunOptions? options,
  }) async => (await runWithDiagnostics(inputs, options: options)).outputs;
  Future<InferenceResult> runWithDiagnostics(
    Map<String, OnnxTensor> inputMap, {
    RunOptions? options,
  }) async {
    _checkOpen();
    final receivedAt = LibOnnxBindings.instance.monotonicUs();
    final opts = options ?? RunOptions();
    if (opts.cancellationToken?.isCancelled == true)
      throw OnnxException(
        'Cancelled before submission',
        code: 7,
        stage: 'submit',
        requestId: opts.requestId,
      );
    if (opts.timeout != null && opts.timeout!.inMicroseconds <= 0)
      throw ArgumentError('timeout must be positive');
    if (inputMap.length != _inputs.length ||
        !_inputs.keys.every(inputMap.containsKey))
      throw ArgumentError(
        'Expected exactly ${_inputs.keys}, received ${inputMap.keys}',
      );
    final copied = <String, OnnxTensor>{};
    // Copy and prepare potentially large inputs off the UI isolate. Snapshotting
    // bytes here preserves the public tensor disposal/mutation boundary.
    for (final entry in inputMap.entries) {
      final t = entry.value, m = _inputs[entry.key]!;
      if (t.type != m.type ||
          t.shape.length != m.shape.length ||
          List.generate(
            m.shape.length,
            (i) => m.shape[i] >= 0 && m.shape[i] != t.shape[i],
          ).any((x) => x)) {
        throw ArgumentError('Input ${entry.key} does not match $m');
      }
      copied[entry.key] = OnnxTensor.fromNative(
        shape: t.shape,
        type: t.type,
        rawBytes: t.rawBytes,
      );
    }
    final drained = Completer<void>();
    _pending.add(drained.future);
    ffi.Pointer<LibOnnxTask>? task;
    var active = true;
    void Function()? removeCancel;
    final submittedAt = Stopwatch()..start();
    try {
      final address = _handle.address, libPath = _runtime.libraryPath;
      final outputNames = _outputs.keys.toList();
      final timeout = opts.timeout == null
              ? 0
              : receivedAt + opts.timeout!.inMicroseconds,
          requestId = opts.requestId;
      final taskAddress = await Isolate.run(
        () => _submitNative(
          address,
          libPath,
          copied,
          outputNames,
          requestId,
          timeout,
        ),
      );
      task = ffi.Pointer<LibOnnxTask>.fromAddress(taskAddress);
      final b = LibOnnxBindings.instance, ownedTask = task;
      removeCancel = opts.cancellationToken?.addListener(() {
        if (active) b.taskCancel(ownedTask, 7);
      });
      if (_isClosed || _runtime.isDisposed) b.taskCancel(task, 7);
      // A native monotonic deadline governs queue+run. Account for preparation
      // before native acceptance too, without requiring the Run isolate to poll.
      if (opts.timeout != null && submittedAt.elapsed >= opts.timeout!)
        b.taskCancel(task, 8);
      Timer? timer;
      if (opts.timeout != null) {
        final remaining = opts.timeout! - submittedAt.elapsed;
        timer = Timer(remaining.isNegative ? Duration.zero : remaining, () {
          if (active) b.taskCancel(ownedTask, 8);
        });
      }
      try {
        while (b.taskState(task) != 2) {
          await Future<void>.delayed(const Duration(milliseconds: 1));
        }
        active = false;
        removeCancel?.call();
        removeCancel = null;
        final status = b.taskStatus(task), taskId = b.taskId(task);
        if (status != 0)
          throw OnnxException(
            b.taskError(task).cast<Utf8>().toDartString(),
            code: status,
            ortCode: b.taskOrtCode(task),
            stage: b.taskStage(task).cast<Utf8>().toDartString(),
            taskId: taskId,
            requestId: requestId,
            providers: List.unmodifiable(
              _options.policy.providerPriority.map((p) => p.ortName),
            ),
          );
        final outputs = <String, OnnxTensor>{};
        try {
          for (var i = 0; i < outputNames.length; i++)
            outputs[outputNames[i]] = _decodeTensor(b, b.taskOutput(task, i));
          final d = b.taskDiagnostic(task);
          if (d == ffi.nullptr)
            throw StateError('Successful task has no diagnostic');
          final registered = b
              .diagnosticGetRegisteredProviders(d)
              .cast<Utf8>()
              .toDartString()
              .split(',')
              .where((x) => x.isNotEmpty)
              .map(ExecutionProvider.fromOrtName)
              .toList();
          final actualName = b
              .diagnosticGetProvider(d)
              .cast<Utf8>()
              .toDartString();
          final actual = actualName.isEmpty
              ? null
              : ExecutionProvider.fromOrtName(actualName);
          final fallback = b.diagnosticGetFallbackOccurred(d),
              version = OnnxRuntime.ortVersion;
          final report = DiagnosticReport(
            activeProvider: actual,
            registeredProviders: List.unmodifiable(registered),
            actualProviders: List.unmodifiable(
              actual == null ? <ExecutionProvider>[] : [actual],
            ),
            evidenceLevel: EvidenceLevel.fromCode(
              b.diagnosticGetEvidenceLevel(d),
            ),
            durationUs: b.diagnosticGetDurationUs(d),
            modelHash: _modelHash,
            hardwareDetails: b
                .diagnosticGetHardwareDetails(d)
                .cast<Utf8>()
                .toDartString(),
            fallbackOccurred: fallback < 0 ? null : fallback == 1,
            taskId: taskId,
            requestId: requestId,
            runtimeVersion: version,
            evidence: List.unmodifiable([
              HardwareEvidence(
                kind: 'epRegistration',
                source:
                    'ORT SessionOptionsAppendExecutionProvider + CreateSession',
                scope: 'session',
                details:
                    'Registered: ${registered.map((p) => p.ortName).join(',')}; graph assignment, compute plan and physical usage unknown',
                runtimeVersion: version,
                modelHash: _modelHash,
              ),
              HardwareEvidence(
                kind: 'runCompletion',
                source: 'ORT Run returned',
                scope: 'run',
                details: 'Completion does not identify physical compute units',
                runtimeVersion: version,
                modelHash: _modelHash,
                taskId: taskId,
              ),
            ]),
          );
          _lastDiagnosticReport = report;
          return InferenceResult(
            outputs: Map.unmodifiable(outputs),
            diagnostic: report,
          );
        } catch (_) {
          for (final t in outputs.values) t.dispose();
          rethrow;
        }
      } finally {
        timer?.cancel();
      }
    } finally {
      active = false;
      removeCancel?.call();
      if (task != null) LibOnnxBindings.instance.taskRelease(task);
      for (final t in copied.values) t.dispose();
      _pending.remove(drained.future);
      drained.complete();
    }
  }

  /// Stops admission, requests cancellation, awaits every native result, then
  /// destroys on an execution isolate. Calling again returns the same Future.
  Future<void> close() => _closing ??= _close();
  Future<void> _close() async {
    _isClosed = true;
    LibOnnxBindings.instance.sessionBeginClose(_handle);
    await Future.wait(_pending.toList());
    final address = _handle.address, path = _runtime.libraryPath;
    await Isolate.run(() {
      if (path != null) LibOnnxBindings.init(libraryPath: path);
      LibOnnxBindings.instance.sessionDestroy(
        ffi.Pointer<LibOnnxSession>.fromAddress(address),
      );
    });
    _runtime.unregisterSession(this);
  }
}

class _Loaded {
  final int address;
  final String hash;
  final Map<String, InputOutputMeta> inputs, outputs;
  _Loaded(this.address, this.hash, this.inputs, this.outputs);
}

_Loaded _createNative(
  int rtAddress,
  String? libPath,
  String? path,
  Uint8List? bytes,
  SessionOptions options,
) {
  if (libPath != null) LibOnnxBindings.init(libraryPath: libPath);
  final b = LibOnnxBindings.instance;
  final model = bytes ?? File(path!).readAsBytesSync();
  final hash = modelIdentity(model, modelPath: path);
  final opts = <String, String>{
    ...options.policy.toOptionMap(),
    'bridge.max_queue': '${options.maxQueueDepth}',
    'bridge.max_active': '${options.maxConcurrentRuns}',
    'session.inter_threads': '${options.interOpNumThreads}',
    'session.optimization': '${options.graphOptimizationLevel.nativeLevel}',
    'session.memory_pattern': options.enableMemoryPattern ? '1' : '0',
    if (options.logId != null) 'session.log_id': options.logId!,
  };
  final providerNames = options.policy.providerPriority
      .map((p) => p.ortName)
      .toList();
  if (opts['coreml.ModelCacheDirectory'] case final cache?) {
    // ORT's own fallback hash may only cover model URL/IO names. Namespace the
    // entire cache by actual content+external data+all compilation options.
    final config = Map<String, String>.from(opts)
      ..remove('coreml.ModelCacheDirectory');
    final sorted = config.keys.toList()..sort();
    final key = sha256.convert(
      utf8.encode(
        '$hash${jsonEncode({
          'providers': providerNames,
          'options': {for (final k in sorted) k: config[k]},
        })}',
      ),
    );
    opts['coreml.ModelCacheDirectory'] = p.join(cache, key.toString());
  }
  final providers = _strings(providerNames),
      keys = _strings(opts.keys.toList()),
      values = _strings(opts.values.toList());
  final pathPtr = path?.toNativeUtf8();
  final data = path == null
      ? calloc<ffi.Uint8>(model.isEmpty ? 1 : model.length)
      : ffi.nullptr.cast<ffi.Uint8>();
  ffi.Pointer<LibOnnxSession>? handle;
  try {
    if (path == null) data.asTypedList(model.length).setAll(0, model);
    handle = b.sessionCreate(
      ffi.Pointer<LibOnnxRuntime>.fromAddress(rtAddress),
      pathPtr?.cast() ?? ffi.nullptr,
      data.cast(),
      path == null ? model.length : 0,
      providers,
      providerNames.length,
      keys,
      values,
      opts.length,
      options.intraOpNumThreads,
      options.policy.fallbackPolicy.nativeCode,
      options.policy.hardwareRequirement.nativeCode,
    );
    if (handle == ffi.nullptr)
      throw nativeException(b, providers: providerNames);
    final inputs = _metadata(b, handle, false),
        outputs = _metadata(b, handle, true);
    if (path != null &&
        modelIdentity(File(path).readAsBytesSync(), modelPath: path) != hash)
      throw StateError('Model/external data changed during load');
    return _Loaded(handle.address, hash, inputs, outputs);
  } catch (_) {
    if (handle != null && handle != ffi.nullptr) b.sessionDestroy(handle);
    rethrow;
  } finally {
    if (pathPtr != null) calloc.free(pathPtr);
    if (data != ffi.nullptr) calloc.free(data);
    _freeStrings(providers, providerNames.length);
    _freeStrings(keys, opts.length);
    _freeStrings(values, opts.length);
  }
}

Map<String, InputOutputMeta> _metadata(
  LibOnnxBindings b,
  ffi.Pointer<LibOnnxSession> h,
  bool output,
) {
  final count = output ? b.sessionGetOutputCount(h) : b.sessionGetInputCount(h);
  final result = <String, InputOutputMeta>{};
  for (var i = 0; i < count; i++) {
    final name =
        (output ? b.sessionGetOutputName(h, i) : b.sessionGetInputName(h, i))
            .cast<Utf8>()
            .toDartString();
    final code = output
        ? b.sessionGetOutputType(h, i)
        : b.sessionGetInputType(h, i);
    final rank = output
        ? b.sessionGetOutputShape(h, i, ffi.nullptr, 0)
        : b.sessionGetInputShape(h, i, ffi.nullptr, 0);
    if (rank < 0 || code == 0) {
      throw OnnxException(
        rank < 0
            ? 'Non-tensor model IO: $name'
            : 'Unsupported model IO dtype: $name',
        code: 11,
        stage: rank < 0 ? 'metadata.kind' : 'metadata.type',
      );
    }
    final type = TensorType.fromNativeCode(code);
    final shape = calloc<ffi.Int64>(rank == 0 ? 1 : rank);
    try {
      final got = output
          ? b.sessionGetOutputShape(h, i, shape, rank)
          : b.sessionGetInputShape(h, i, shape, rank);
      if (got != rank) throw StateError('Metadata rank changed');
      result[name] = InputOutputMeta(
        name: name,
        type: type,
        shape: List.unmodifiable(shape.asTypedList(rank)),
      );
    } finally {
      calloc.free(shape);
    }
  }
  return Map.unmodifiable(result);
}

int _submitNative(
  int address,
  String? libPath,
  Map<String, OnnxTensor> input,
  List<String> outputs,
  int request,
  int timeout,
) {
  if (libPath != null) LibOnnxBindings.init(libraryPath: libPath);
  final b = LibOnnxBindings.instance,
      names = input.keys.toList(),
      inNames = _strings(input.keys.toList()),
      outNames = _strings(outputs);
  final handles = calloc<ffi.Pointer<LibOnnxTensor>>(
    input.isEmpty ? 1 : input.length,
  );
  try {
    for (var i = 0; i < names.length; i++) {
      final t = input[names[i]]!,
          shape = calloc<ffi.Int64>(t.shape.isEmpty ? 1 : t.shape.length),
          data = calloc<ffi.Uint8>(t.byteLength == 0 ? 1 : t.byteLength);
      try {
        shape.asTypedList(t.shape.length).setAll(0, t.shape);
        data.asTypedList(t.byteLength).setAll(0, t.rawBytes);
        handles[i] = b.tensorCreate(
          t.type.nativeCode,
          shape,
          t.shape.length,
          data.cast(),
          t.byteLength,
          1,
        );
        if (handles[i] == ffi.nullptr)
          throw nativeException(b, requestId: request);
      } finally {
        calloc.free(shape);
        calloc.free(data);
      }
    }
    final remaining = timeout == 0 ? 0 : timeout - b.monotonicUs();
    if (timeout != 0 && remaining <= 0) {
      throw OnnxException(
        'Timeout during input preparation',
        code: 8,
        stage: 'submit.timeout',
        requestId: request,
      );
    }
    final task = b.sessionSubmit(
      ffi.Pointer<LibOnnxSession>.fromAddress(address),
      request,
      inNames,
      handles,
      names.length,
      outNames,
      outputs.length,
      remaining,
    );
    if (task == ffi.nullptr) throw nativeException(b, requestId: request);
    return task.address;
  } finally {
    for (var i = 0; i < names.length; i++)
      if (handles[i] != ffi.nullptr) b.tensorDestroy(handles[i]);
    calloc.free(handles);
    _freeStrings(inNames, names.length);
    _freeStrings(outNames, outputs.length);
  }
}

OnnxTensor _decodeTensor(LibOnnxBindings b, ffi.Pointer<LibOnnxTensor> t) {
  if (t == ffi.nullptr) throw StateError('Missing native output');
  final type = TensorType.fromNativeCode(b.tensorGetDtype(t)),
      rank = b.tensorGetRank(t);
  if (rank < 0) throw StateError('Invalid output rank');
  final shapePtr = calloc<ffi.Int64>(rank == 0 ? 1 : rank);
  try {
    if (b.tensorGetShape(t, shapePtr, rank) != rank)
      throw StateError('Invalid output shape');
    final shape = shapePtr.asTypedList(rank).toList(),
        bytes = b.tensorGetByteLength(t);
    if (checkedElementCount(shape) * type.elementSizeInBytes != bytes)
      throw StateError('Output buffer length mismatch');
    final data = b.tensorGetData(t);
    if (bytes > 0 && data == ffi.nullptr)
      throw StateError('Null output buffer');
    return OnnxTensor.fromNative(
      shape: shape,
      type: type,
      rawBytes: bytes == 0
          ? Uint8List(0)
          : Uint8List.fromList(data.cast<ffi.Uint8>().asTypedList(bytes)),
    );
  } finally {
    calloc.free(shapePtr);
  }
}

ffi.Pointer<ffi.Pointer<ffi.Char>> _strings(List<String> values) {
  if (values.isEmpty) return ffi.nullptr;
  final ptr = calloc<ffi.Pointer<ffi.Char>>(values.length);
  try {
    for (var i = 0; i < values.length; i++)
      ptr[i] = values[i].toNativeUtf8().cast();
    return ptr;
  } catch (_) {
    _freeStrings(ptr, values.length);
    rethrow;
  }
}

void _freeStrings(ffi.Pointer<ffi.Pointer<ffi.Char>> ptr, int length) {
  if (ptr == ffi.nullptr) return;
  for (var i = 0; i < length; i++)
    if (ptr[i] != ffi.nullptr) calloc.free(ptr[i]);
  calloc.free(ptr);
}
