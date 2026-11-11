import 'dart:ffi';
import 'dart:io';
import 'package:lib_onnx/src/ffi/native_library.dart';
import 'package:test/test.dart';

void main() {
  test('missing native override fails with actionable ABI/version context', () {
    expect(
      () => NativeLibrary.load(overridePath: '/lib_onnx-test-missing-file'),
      throwsA(
        isA<StateError>().having(
          (e) => e.message,
          'context',
          contains('ABI 2'),
        ),
      ),
    );
  });
  test('loader rejects old bridge ABI and different ORT version', () async {
    final temp = await Directory.systemTemp.createTemp('lib_onnx_abi_test_');
    try {
      for (final (abi, version) in [(1, '1.31.1'), (2, '1.30.0')]) {
        final c = File('${temp.path}/stub$abi.c');
        await c.writeAsString(
          '#include <stdint.h>\nuint32_t lib_onnx_abi_version(void){return $abi;}\nconst char* lib_onnx_ort_version_string(void){return "$version";}\n',
        );
        final out =
            '${temp.path}/stub$abi.${Platform.isMacOS ? 'dylib' : 'so'}';
        final result = await Process.run('clang', [
          '-shared',
          '-fPIC',
          c.path,
          '-o',
          out,
        ]);
        expect(
          result.exitCode,
          0,
          reason: '${result.stdout}\n${result.stderr}',
        );
        expect(
          () => NativeLibrary.load(overridePath: out),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'mismatch',
              contains(abi == 1 ? 'ABI mismatch' : 'ORT mismatch'),
            ),
          ),
        );
      }
    } finally {
      await temp.delete(recursive: true);
    }
  });
  test('loader accepts the platform-pinned ORT version only', () async {
    final temp = await Directory.systemTemp.createTemp('lib_onnx_ort_test_');
    try {
      final c = File('${temp.path}/stub.c');
      await c.writeAsString(
        '#include <stdint.h>\nuint32_t lib_onnx_abi_version(void){return 2;}\nconst char* lib_onnx_ort_version_string(void){return "${NativeLibrary.iosOrtVersion}";}\n',
      );
      final out = '${temp.path}/stub.${Platform.isMacOS ? 'dylib' : 'so'}';
      final result = await Process.run('clang', [
        '-shared',
        '-fPIC',
        c.path,
        '-o',
        out,
      ]);
      expect(result.exitCode, 0, reason: '${result.stderr}');
      final library = DynamicLibrary.open(out);
      // The iOS archive version is accepted only where it is pinned.
      NativeLibrary.validate(library, expectedOrt: NativeLibrary.iosOrtVersion);
      expect(
        () => NativeLibrary.validate(library, expectedOrt: '1.31.1'),
        throwsA(isA<StateError>()),
      );
      expect(NativeLibrary.expectedOrtVersion, NativeLibrary.ortVersion);
    } finally {
      await temp.delete(recursive: true);
    }
  });
}
