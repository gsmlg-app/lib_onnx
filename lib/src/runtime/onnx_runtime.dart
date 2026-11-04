import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:isolate';
import 'package:ffi/ffi.dart';
import 'package:meta/meta.dart';
import '../diagnostics/onnx_exception.dart';
import '../ffi/bindings.dart';
import '../ffi/c_structures.dart';
import '../provider/execution_provider.dart';

/// Owns a bounded native worker pool. Sessions retain its OrtEnv natively.
class OnnxRuntime {
  final ffi.Pointer<LibOnnxRuntime> _handle;
  final int maxConcurrency;
  final String? libraryPath;
  bool _isDisposed = false;
  Future<void>? _closing;
  int _loads = 0;
  Completer<void>? _loadsDone;
  final _sessions = <Object, Future<void> Function()>{};
  static OnnxRuntime? _defaultRuntime;
  static Future<OnnxRuntime>? _defaultInitializing;
  OnnxRuntime._(this._handle, this.maxConcurrency, this.libraryPath);
  static OnnxRuntime get defaultRuntime {
    if (_defaultInitializing != null)
      throw StateError(
        'Default runtime is initializing; await defaultRuntimeAsync',
      );
    return _defaultRuntime ??= OnnxRuntime.create();
  }

  static Future<OnnxRuntime> get defaultRuntimeAsync => _defaultRuntime != null
      ? Future.value(_defaultRuntime!)
      : _defaultInitializing ??= _initializeDefault();
  static Future<OnnxRuntime> _initializeDefault() async {
    try {
      return _defaultRuntime = await createAsync();
    } finally {
      _defaultInitializing = null;
    }
  }

  /// Recommended initialization; native dependency loading and OrtEnv creation
  /// run on an execution isolate too.
  static Future<OnnxRuntime> createAsync({
    int maxConcurrency = 4,
    String? libraryPath,
  }) async {
    if (maxConcurrency < 1 || maxConcurrency > 64)
      throw RangeError.range(maxConcurrency, 1, 64, 'maxConcurrency');
    final address = await Isolate.run(() {
      if (libraryPath != null) LibOnnxBindings.init(libraryPath: libraryPath);
      final b = LibOnnxBindings.instance;
      final h = b.runtimeCreate(maxConcurrency);
      if (h == ffi.nullptr) throw nativeException(b);
      return h.address;
    });
    if (libraryPath != null) LibOnnxBindings.init(libraryPath: libraryPath);
    return OnnxRuntime._(
      ffi.Pointer<LibOnnxRuntime>.fromAddress(address),
      maxConcurrency,
      libraryPath,
    );
  }

  factory OnnxRuntime.create({int maxConcurrency = 4, String? libraryPath}) {
    if (maxConcurrency < 1 || maxConcurrency > 64)
      throw RangeError.range(maxConcurrency, 1, 64, 'maxConcurrency');
    if (libraryPath != null) LibOnnxBindings.init(libraryPath: libraryPath);
    final b = LibOnnxBindings.instance;
    final handle = b.runtimeCreate(maxConcurrency);
    if (handle == ffi.nullptr) throw nativeException(b);
    return OnnxRuntime._(handle, maxConcurrency, libraryPath);
  }
  ffi.Pointer<LibOnnxRuntime> get handle {
    checkOpen();
    return _handle;
  }

  static String get bridgeVersion =>
      LibOnnxBindings.instance.versionString().cast<Utf8>().toDartString();
  static String get ortVersion =>
      LibOnnxBindings.instance.ortVersionString().cast<Utf8>().toDartString();
  bool get isDisposed => _isDisposed;
  @internal
  void checkOpen() {
    if (_isDisposed) throw StateError('Runtime is closed');
  }

  @internal
  void beginLoad() {
    checkOpen();
    if (_loads++ == 0) _loadsDone = Completer<void>();
  }

  @internal
  void endLoad() {
    if (--_loads == 0) _loadsDone!.complete();
  }

  @internal
  void registerSession(Object session, Future<void> Function() close) {
    checkOpen();
    _sessions[session] = close;
  }

  @internal
  void unregisterSession(Object session) {
    _sessions.remove(session);
  }

  bool isProviderSupported(ExecutionProvider provider) {
    checkOpen();
    final ptr = provider.ortName.toNativeUtf8();
    try {
      final b = LibOnnxBindings.instance;
      final supported = b.runtimeIsProviderSupported(_handle, ptr.cast());
      if (supported < 0) throw nativeException(b);
      return supported == 1;
    } finally {
      calloc.free(ptr);
    }
  }

  /// Build capability only. This does not prove backend/device initialization.
  List<String> getAvailableProviders() {
    checkOpen();
    final buffer = calloc<ffi.Char>(4096);
    try {
      if (LibOnnxBindings.instance.runtimeGetAvailableProviders(
            _handle,
            buffer,
            4096,
          ) <
          0)
        throw nativeException(LibOnnxBindings.instance);
      return List.unmodifiable(
        buffer
            .cast<Utf8>()
            .toDartString()
            .split(',')
            .where((s) => s.isNotEmpty),
      );
    } finally {
      calloc.free(buffer);
    }
  }

  /// Idempotent cancellation-and-drain shutdown; waits never block the caller isolate.
  Future<void> close() => _closing ??= _close();
  Future<void> _close() async {
    _isDisposed = true;
    if (identical(this, _defaultRuntime)) _defaultRuntime = null;
    final b = LibOnnxBindings.instance;
    b.runtimeBeginClose(_handle);
    if (_loads > 0) await _loadsDone!.future;
    await Future.wait(_sessions.values.toList().map((close) => close()));
    final address = _handle.address, path = libraryPath;
    await Isolate.run(() {
      if (path != null) LibOnnxBindings.init(libraryPath: path);
      LibOnnxBindings.instance.runtimeDestroy(
        ffi.Pointer<LibOnnxRuntime>.fromAddress(address),
      );
    });
  }

  @Deprecated('Await close() to observe shutdown completion')
  Future<void> dispose() => close();
}

@internal
OnnxException nativeException(
  LibOnnxBindings b, {
  int? requestId,
  List<String> providers = const [],
}) => OnnxException(
  b.getLastError().cast<Utf8>().toDartString(),
  code: b.getLastErrorCode(),
  ortCode: b.getLastOrtCode(),
  stage: b.getLastErrorStage().cast<Utf8>().toDartString(),
  requestId: requestId,
  providers: List.unmodifiable(providers),
);
