import 'dart:io';
import 'dart:typed_data';
import 'package:lib_onnx/lib_onnx.dart';
import 'package:test/test.dart';

void main() {
  test(
    '100 load/run/await-close cycles and 1000 native runs retain exact results',
    () async {
      final rt = OnnxRuntime.create(
        libraryPath: Platform.environment['LIB_ONNX_TEST_BRIDGE'],
      );
      try {
        final tensor = OnnxTensor.fromFloat32([
          1,
          4,
        ], Float32List.fromList([1, 2, 3, 4]));
        for (var i = 0; i < 100; i++) {
          final s = await InferenceSession.loadFile(
            'test/fixtures/identity_float32.onnx',
            runtime: rt,
            options: SessionOptions.cpuOnly(),
          );
          final out = await s.run({'X': tensor});
          expect(out['Y']!.rawBytes, tensor.rawBytes);
          await s.close();
          await s.close();
        }
        final s = await InferenceSession.loadFile(
          'test/fixtures/identity_float32.onnx',
          runtime: rt,
          options: SessionOptions.cpuOnly(),
        );
        for (var i = 0; i < 1000; i++) {
          final out = await s.run({'X': tensor});
          expect(out['Y']!.rawBytes, tensor.rawBytes);
        }
      } finally {
        await rt.close();
      }
    },
    timeout: const Timeout(Duration(minutes: 3)),
  );
}
