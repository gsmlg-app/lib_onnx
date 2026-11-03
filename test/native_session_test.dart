import 'dart:io';
import 'dart:typed_data';
import 'package:lib_onnx/lib_onnx.dart';
import 'package:path/path.dart' as p;
import 'package:test/test.dart';

void main() {
  final fixturesDir = p.join(Directory.current.path, 'test', 'fixtures');
  final identityPath = p.join(fixturesDir, 'identity_float32.onnx');
  final mlpPath = p.join(fixturesDir, 'mlp_smoke.onnx');

  group('Native Integration & Lifecycle', () {
    test('Runtime reports correct version and available providers', () {
      expect(OnnxRuntime.bridgeVersion, equals('0.1.0'));
      expect(OnnxRuntime.ortVersion, contains('1.31'));

      final rt = OnnxRuntime.defaultRuntime;
      final providers = rt.getAvailableProviders();
      expect(providers, isNotEmpty);
      expect(providers, contains('CPUExecutionProvider'));
      expect(rt.isProviderSupported(ExecutionProvider.cpu), isTrue);

      if (Platform.isMacOS) {
        expect(providers, contains('CoreMLExecutionProvider'));
        expect(rt.isProviderSupported(ExecutionProvider.coreML), isTrue);
      }
    });

    test('InferenceSession loads Identity model and executes with exact values',
        () async {
      final session = InferenceSession.fromFile(
        identityPath,
        options: SessionOptions.cpuOnly(),
      );

      expect(session.isClosed, isFalse);
      expect(session.inputs.containsKey('X'), isTrue);
      expect(session.inputs['X']!.shape, equals([1, 4]));
      expect(session.inputs['X']!.type, equals(TensorType.float32));

      expect(session.outputs.containsKey('Y'), isTrue);
      expect(session.outputs['Y']!.shape, equals([1, 4]));
      expect(session.outputs['Y']!.type, equals(TensorType.float32));

      final inData = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
      final inTensor = OnnxTensor.fromFloat32([1, 4], inData);

      final outputs = await session.run({'X': inTensor});
      expect(outputs.containsKey('Y'), isTrue);

      final outTensor = outputs['Y']!;
      expect(outTensor.shape, equals([1, 4]));
      expect(outTensor.asFloat32List(), equals(inData));

      final report = session.lastDiagnosticReport;
      expect(report, isNotNull);
      expect(report!.activeProvider, equals(ExecutionProvider.cpu));
      expect(report.durationUs, greaterThan(0));

      session.close();
      expect(session.isClosed, isTrue);
      expect(() => session.run({'X': inTensor}), throwsStateError);
    });

    test('InferenceSession loads from memory bytes', () async {
      final bytes = await File(identityPath).readAsBytes();
      final session =
          InferenceSession.fromBytes(bytes, options: SessionOptions.cpuOnly());

      final inTensor = OnnxTensor.fromFloat32(
          [1, 4], Float32List.fromList([10.0, 20.0, 30.0, 40.0]));
      final outputs = await session.run({'X': inTensor});

      expect(outputs['Y']!.asFloat32List(), equals([10.0, 20.0, 30.0, 40.0]));
      session.close();
    });

    test(
        'MLP model runs with CoreML acceleration and validates hardware evidence on Apple Silicon',
        () async {
      final isApple = Platform.isMacOS || Platform.isIOS;
      final session = InferenceSession.fromFile(
        mlpPath,
        options: isApple
            ? SessionOptions.appleSilicon(
                preferAne: true, allowCpuFallback: true)
            : SessionOptions.cpuOnly(),
      );

      final inData = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
      final inTensor = OnnxTensor.fromFloat32([1, 4], inData);

      final outputs = await session.run({'input': inTensor});
      expect(outputs.containsKey('output'), isTrue);

      // Expected: Relu(X * W1 + B1) where W1=I, B1=[0.5, 0.5, 0.5, 0.5]
      // Result = [1.5, 2.5, 3.5, 4.5]
      final result = outputs['output']!.asFloat32List();
      expect(result[0], closeTo(1.5, 1e-4));
      expect(result[1], closeTo(2.5, 1e-4));
      expect(result[2], closeTo(3.5, 1e-4));
      expect(result[3], closeTo(4.5, 1e-4));

      final report = session.lastDiagnosticReport;
      expect(report, isNotNull);

      if (isApple) {
        expect(report!.activeProvider, equals(ExecutionProvider.coreML));
        expect(report.isAccelerated, isTrue);
        expect(report.evidenceLevel, equals(EvidenceLevel.hardwareObserved));
        expect(report.hardwareDetails, contains('Apple M1 Max'));
      }

      session.close();
    });

    test('Pre-run cancellation token cancels inference request', () async {
      final session = InferenceSession.fromFile(identityPath,
          options: SessionOptions.cpuOnly());
      final token = CancellationToken();
      token.cancel();

      final inTensor = OnnxTensor.fromFloat32(
          [1, 4], Float32List.fromList([1.0, 2.0, 3.0, 4.0]));
      expect(
        () => session.run({'X': inTensor},
            options: RunOptions(cancellationToken: token)),
        throwsA(isA<StateError>()),
      );

      session.close();
    });
  });
}
