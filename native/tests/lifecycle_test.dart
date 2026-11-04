import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';
import 'package:lib_onnx/lib_onnx.dart';
import 'package:test/test.dart';

void main() {
  late OnnxRuntime rt;
  setUp(() {
    rt = OnnxRuntime.create(
      libraryPath: Platform.environment['LIB_ONNX_TEST_BRIDGE'],
    );
  });
  tearDown(() async {
    await rt.close();
  });
  String model(String name) => 'test/fixtures/$name.onnx';
  final samples = {
    'float32': OnnxTensor.fromFloat32([1, 4], Float32List(4)),
  };
  test(
    'native control remains responsive in dequeue gap; timeout and bounded queue',
    () async {
      final lib = LibOnnxBindings.instance.library;
      // Test-only symbols are required by this specific native test executable.
      final hold = lib
          .lookupFunction<ffi.Void Function(ffi.Int32), void Function(int)>(
            'lib_onnx_test_barrier',
          );
      final s = await InferenceSession.loadFile(
        model('identity_float32'),
        runtime: rt,
        options: SessionOptions(
          policy: ExecutionPolicy.cpuOnly(),
          maxQueueDepth: 0,
        ),
      );
      hold(1);
      addTearDown(() => hold(0));
      final first = s.run({
        'X': samples['float32']!,
      }, options: RunOptions(timeout: const Duration(milliseconds: 20)));
      final firstAssertion = expectLater(
        first,
        throwsA(isA<OnnxException>().having((e) => e.code, 'code', 8)),
      );
      final waiting = lib.lookupFunction<ffi.Int32 Function(), int Function()>(
        'lib_onnx_test_barrier_waiting',
      );
      final limit = Stopwatch()..start();
      while (waiting() == 0 && limit.elapsed < const Duration(seconds: 10)) {
        await Future<void>.delayed(const Duration(milliseconds: 1));
      }
      expect(waiting(), greaterThan(0));
      await expectLater(
        s.run({'X': samples['float32']!}),
        throwsA(isA<OnnxException>().having((e) => e.code, 'code', 9)),
      );
      // Event-loop and timeout callbacks keep running while the C task is latched.
      await Future<void>.delayed(const Duration(milliseconds: 25));
      hold(0);
      await firstAssertion;
      await s.close();
    },
  );
}
