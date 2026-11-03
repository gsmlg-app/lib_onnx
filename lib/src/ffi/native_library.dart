import 'dart:ffi' as ffi;
import 'dart:io' show Platform, Directory, File;
import 'package:path/path.dart' as p;

/// Resolves and loads the native lib_onnx dynamic library and its runtime dependencies.
class NativeLibrary {
  static ffi.DynamicLibrary? _cached;
  static bool _dependencyPreloaded = false;

  static ffi.DynamicLibrary load({String? overridePath}) {
    if (_cached != null && overridePath == null) {
      return _cached!;
    }

    _preloadDependencies();

    if (overridePath != null) {
      final lib = ffi.DynamicLibrary.open(overridePath);
      _cached = lib;
      return lib;
    }

    final candidates = _getCandidatePaths();
    for (final candidate in candidates) {
      if (candidate.isNotEmpty && File(candidate).existsSync()) {
        try {
          final lib = ffi.DynamicLibrary.open(candidate);
          _cached = lib;
          return lib;
        } catch (_) {
          // Continue searching candidate paths
        }
      }
    }

    // Fallback: try loading by base name
    final libName = _getPlatformLibraryName();
    try {
      final lib = ffi.DynamicLibrary.open(libName);
      _cached = lib;
      return lib;
    } catch (e) {
      throw StateError(
        'Failed to load lib_onnx native library ($libName). Checked paths: $candidates. Error: $e',
      );
    }
  }

  static void _preloadDependencies() {
    if (_dependencyPreloaded) return;
    _dependencyPreloaded = true;

    final cwd = Directory.current.path;
    final ortCandidates = <String>[
      p.join(cwd, 'libonnxruntime.1.dylib'),
      p.join(cwd, 'libonnxruntime.dylib'),
      p.join(cwd, 'native', 'build', 'libonnxruntime.dylib'),
      p.join(cwd, 'native', 'third_party', 'onnxruntime', 'lib', 'macos_arm64',
          'libonnxruntime.dylib'),
      p.join(cwd, 'native', 'third_party', 'onnxruntime', 'lib', 'macos_arm64',
          'libonnxruntime.1.dylib'),
      p.join(cwd, 'Frameworks', 'libonnxruntime.dylib'),
    ];

    for (final cand in ortCandidates) {
      if (File(cand).existsSync()) {
        try {
          ffi.DynamicLibrary.open(cand);
          break;
        } catch (_) {}
      }
    }
  }

  static String _getPlatformLibraryName() {
    if (Platform.isMacOS || Platform.isIOS) {
      return 'liblib_onnx_bridge.dylib';
    } else if (Platform.isWindows) {
      return 'lib_onnx_bridge.dll';
    } else {
      return 'liblib_onnx_bridge.so';
    }
  }

  static List<String> _getCandidatePaths() {
    final list = <String>[];
    final libName = _getPlatformLibraryName();

    final cwd = Directory.current.path;
    list.add(p.join(cwd, 'native', 'build', libName));
    list.add(p.join(cwd, 'build', libName));
    list.add(p.join(cwd, libName));

    if (Platform.isMacOS) {
      list.add(p.join(cwd, 'Frameworks', libName));
      list.add(p.join('/usr/local/lib', libName));
      list.add(p.join('/opt/homebrew/lib', libName));
    }

    return list;
  }
}
