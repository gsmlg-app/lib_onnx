import 'dart:ffi' as ffi;
import 'dart:io' show Platform;
import 'package:ffi/ffi.dart';

@ffi.Native<ffi.Pointer<ffi.Char> Function()>(
  assetId: 'package:lib_onnx/src/ffi/native_library.dart',
  symbol: 'lib_onnx_library_path',
)
external ffi.Pointer<ffi.Char> _bundledLibraryPath();

/// Loads the SDK-bundled native asset or an explicitly supplied development
/// library. No CWD, system-library, or Homebrew search is performed.
class NativeLibrary {
  static const abiVersion = 2;
  static const ortVersion = '1.31.1';

  /// Official iOS ORT archives lag desktop/Android releases; the newest
  /// published iOS archive is pinned instead (see native/ort_artifacts.json).
  static const iosOrtVersion = '1.30.0';

  /// The ORT version the build hook pins for the running platform.
  static String get expectedOrtVersion =>
      Platform.isIOS ? iosOrtVersion : ortVersion;
  static ffi.DynamicLibrary? _cached;

  static ffi.DynamicLibrary load({String? overridePath}) {
    if (overridePath == null && _cached != null) return _cached!;
    late final ffi.DynamicLibrary library;
    try {
      if (overridePath != null) {
        library = ffi.DynamicLibrary.open(overridePath);
      } else {
        final ffi.Pointer<ffi.Char> path;
        try {
          path = _bundledLibraryPath();
        } on ArgumentError catch (error) {
          throw UnsupportedError(
            'lib_onnx has no native runtime for this platform '
            '(${Platform.operatingSystem}): the build hook found no pinned '
            'ONNX Runtime artifact and no native_bundle was configured. '
            '$error',
          );
        }
        if (path == ffi.nullptr) {
          throw StateError('Native asset could not identify its library path.');
        }
        library = ffi.DynamicLibrary.open(path.cast<Utf8>().toDartString());
      }
      validate(library);
    } on UnsupportedError {
      rethrow;
    } catch (error) {
      throw StateError(
        'Unable to load compatible lib_onnx ABI $abiVersion '
        '(ORT $expectedOrtVersion). Build with Dart native-assets hooks, or provide '
        'an explicit libraryPath for development. Loader error: $error',
      );
    }
    if (overridePath == null) _cached = library;
    return library;
  }

  /// The platform loader rejects incompatible Mach-O/ELF architectures first;
  /// these checks then reject old bridge ABIs and mismatched ORT engines.
  static void validate(ffi.DynamicLibrary library, {String? expectedOrt}) {
    final expected = expectedOrt ?? expectedOrtVersion;
    final abi = library.lookupFunction<ffi.Uint32 Function(), int Function()>(
      'lib_onnx_abi_version',
    )();
    if (abi != abiVersion) {
      throw StateError(
        'Bridge ABI mismatch: expected $abiVersion, found $abi.',
      );
    }
    final ort = library
        .lookupFunction<
          ffi.Pointer<ffi.Char> Function(),
          ffi.Pointer<ffi.Char> Function()
        >('lib_onnx_ort_version_string')();
    if (ort == ffi.nullptr) {
      throw StateError('ORT version is unavailable.');
    }
    final version = ort.cast<Utf8>().toDartString();
    if (version != expected) {
      throw StateError('ORT mismatch: expected $expected, found $version.');
    }
  }
}
