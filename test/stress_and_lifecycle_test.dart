import 'dart:io';
import 'dart:typed_data';
import 'package:lib_onnx/lib_onnx.dart';
import 'package:path/path.dart' as p;
import 'package:test/test.dart';

void main() {
  final fixturesDir = p.join(Directory.current.path, 'test', 'fixtures');
  final identityPath = p.join(fixturesDir, 'identity_float32.onnx');
  final mlpPath = p.join(fixturesDir, 'mlp_smoke.onnx');

  group('Stress and Lifecycle Robustness', () {
    test('100 consecutive load / run / close cycles without leaks or crashes',
        () async {
      final inTensor = OnnxTensor.fromFloat32(
          [1, 4], Float32List.fromList([1.0, 2.0, 3.0, 4.0]));

      for (var i = 0; i < 100; i++) {
        final session = InferenceSession.fromFile(identityPath,
            options: SessionOptions.cpuOnly());
        final out = await session.run({'X': inTensor});
        expect(out['Y']!.elementCount, equals(4));
        session.close();
        expect(session.isClosed, isTrue);
      }
    });

    test('1000 continuous inferences throughput stress test', () async {
      final session = InferenceSession.fromFile(
        mlpPath,
        options: Platform.isMacOS
            ? SessionOptions.appleSilicon(
                preferAne: true, allowCpuFallback: true)
            : SessionOptions.cpuOnly(),
      );

      final inTensor = OnnxTensor.fromFloat32(
          [1, 4], Float32List.fromList([1.0, 2.0, 3.0, 4.0]));
      final stopwatch = Stopwatch()..start();

      for (var i = 0; i < 1000; i++) {
        final out = await session.run({'input': inTensor});
        expect(out['output']!.elementCount, equals(4));
      }

      stopwatch.stop();
      final totalMs = stopwatch.elapsedMilliseconds;
      final avgUs = (totalMs * 1000) / 1000;
      print(
          '1,000 inferences completed in ${totalMs}ms (average ${avgUs.toStringAsFixed(1)} µs/run)');

      final report = session.lastDiagnosticReport;
      expect(report, isNotNull);
      session.close();
    });

    test('Double close is idempotent and does not crash or throw', () {
      final session = InferenceSession.fromFile(identityPath,
          options: SessionOptions.cpuOnly());
      expect(session.isClosed, isFalse);

      session.close();
      expect(session.isClosed, isTrue);

      // Second close must be a no-op
      expect(() => session.close(), returnsNormally);
      expect(session.isClosed, isTrue);
    });

    test('Strict fallback policy rejects unavailable accelerator', () {
      expect(
        () => InferenceSession.fromFile(
          identityPath,
          options: const SessionOptions(
            policy: ExecutionPolicy(
              providerPriority: [ExecutionProvider.cuda],
              fallbackPolicy: FallbackPolicy.strict,
            ),
          ),
        ),
        throwsA(isA<StateError>()),
      );
    });
  });
}
