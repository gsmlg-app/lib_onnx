import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import '../ffi/bindings.dart';
import '../ffi/c_structures.dart';
import '../provider/execution_provider.dart';

/// Central runtime managing native ONNX Runtime initialization,
/// version verification, and provider availability.
class OnnxRuntime {
  final ffi.Pointer<LibOnnxRuntime> _handle;
  final int maxConcurrency;
  bool _isDisposed = false;

  static OnnxRuntime? _defaultRuntime;

  OnnxRuntime._(this._handle, this.maxConcurrency);

  /// Acquires or creates the shared default runtime instance.
  static OnnxRuntime get defaultRuntime {
    _defaultRuntime ??= OnnxRuntime.create();
    return _defaultRuntime!;
  }

  /// Creates a new isolated OnnxRuntime instance with a dedicated worker pool.
  factory OnnxRuntime.create({int maxConcurrency = 4, String? libraryPath}) {
    if (libraryPath != null) {
      LibOnnxBindings.init(libraryPath: libraryPath);
    }
    final bindings = LibOnnxBindings.instance;
    final handle = bindings.runtimeCreate(maxConcurrency);
    if (handle.address == 0) {
      final err = bindings.getLastError().cast<Utf8>().toDartString();
      throw StateError('Failed to create OnnxRuntime: $err');
    }
    return OnnxRuntime._(handle, maxConcurrency);
  }

  ffi.Pointer<LibOnnxRuntime> get handle {
    _checkDisposed();
    return _handle;
  }

  /// Version of the lib_onnx native bridge (e.g. 0.1.0).
  static String get bridgeVersion {
    final bindings = LibOnnxBindings.instance;
    return bindings.versionString().cast<Utf8>().toDartString();
  }

  /// Version of the underlying ONNX Runtime engine (e.g. 1.31.1).
  static String get ortVersion {
    final bindings = LibOnnxBindings.instance;
    return bindings.ortVersionString().cast<Utf8>().toDartString();
  }

  /// Returns whether a given execution provider is compiled and supported in this build.
  bool isProviderSupported(ExecutionProvider provider) {
    _checkDisposed();
    final bindings = LibOnnxBindings.instance;
    final namePtr = provider.ortName.toNativeUtf8();
    try {
      return bindings.runtimeIsProviderSupported(_handle, namePtr.cast()) == 1;
    } finally {
      calloc.free(namePtr);
    }
  }

  /// Returns a list of all available execution providers reported by ONNX Runtime.
  List<String> getAvailableProviders() {
    _checkDisposed();
    final bindings = LibOnnxBindings.instance;
    final buffer = calloc<ffi.Char>(1024);
    try {
      final count =
          bindings.runtimeGetAvailableProviders(_handle, buffer, 1024);
      if (count <= 0) return const [];
      final str = buffer.cast<Utf8>().toDartString();
      return str.split(',').where((s) => s.isNotEmpty).toList();
    } finally {
      calloc.free(buffer);
    }
  }

  bool get isDisposed => _isDisposed;

  void _checkDisposed() {
    if (_isDisposed) {
      throw StateError('Cannot use an OnnxRuntime that has been disposed.');
    }
  }

  /// Shuts down the runtime worker pool and releases the ORT environment.
  void dispose() {
    if (!_isDisposed) {
      _isDisposed = true;
      LibOnnxBindings.instance.runtimeDestroy(_handle);
      if (identical(this, _defaultRuntime)) {
        _defaultRuntime = null;
      }
    }
  }
}
