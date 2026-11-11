import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:lib_onnx/lib_onnx.dart';
import 'package:test/test.dart';

void main() {
  String model(String name) => 'test/fixtures/$name.onnx';
  late OnnxRuntime rt;
  setUp(() {
    rt = OnnxRuntime.create(
      libraryPath: Platform.environment['LIB_ONNX_TEST_BRIDGE'],
    );
  });
  tearDown(() async {
    await rt.close();
  });
  test(
    'async runtime initialization leaves the event loop responsive',
    () async {
      var ticks = 0;
      final heartbeat = Timer.periodic(Duration.zero, (_) {
        ticks++;
      });
      final created = await OnnxRuntime.createAsync(
        libraryPath: Platform.environment['LIB_ONNX_TEST_BRIDGE'],
      );
      heartbeat.cancel();
      expect(ticks, greaterThan(0));
      await created.close();
    },
  );
  test(
    'bridge ABI and build capability are separate from execution evidence',
    () {
      expect(OnnxRuntime.bridgeVersion, '0.3.0');
      expect(OnnxRuntime.ortVersion, '1.31.1');
      expect(rt.getAvailableProviders(), contains('CPUExecutionProvider'));
      expect(rt.isProviderSupported(ExecutionProvider.cpu), isTrue);
    },
  );
  test('async loading heartbeat, inference and per-task diagnostics', () async {
    var ticks = 0;
    final heartbeat = Timer.periodic(Duration.zero, (_) {
      ticks++;
    });
    final session = await InferenceSession.loadFile(
      model('identity_float32'),
      runtime: rt,
      options: SessionOptions.cpuOnly(),
    );
    expect(ticks, greaterThan(0));
    ticks = 0;
    final t = OnnxTensor.fromFloat32([
      1,
      4,
    ], Float32List.fromList([1, 2, 3, 4]));
    final result = await session.runWithDiagnostics({'X': t});
    heartbeat.cancel();
    expect(ticks, greaterThan(0));
    expect(result.outputs['Y']!.asFloat32List(), [1, 2, 3, 4]);
    expect(result.diagnostic.activeProvider, ExecutionProvider.cpu);
    expect(result.diagnostic.fallbackOccurred, isFalse);
    expect(result.diagnostic.modelHash, matches(RegExp(r'^[0-9a-f]{64}$')));
    expect(result.diagnostic.nodeDistribution, isEmpty);
    await session.close();
    await session.close();
    await expectLater(session.run({'X': t}), throwsStateError);
  });
  final samples = <String, OnnxTensor>{
    'float32': OnnxTensor.fromFloat32([
      1,
      4,
    ], Float32List.fromList([1, 2, 3, 4])),
    'float16': OnnxTensor.fromFloat16Bytes([
      1,
      4,
    ], Uint8List.fromList([0, 60, 0, 64, 0, 66, 0, 68])),
    'int64': OnnxTensor.fromInt64([1, 4], Int64List.fromList([1, 2, 3, 4])),
    'int32': OnnxTensor.fromInt32([1, 4], Int32List.fromList([1, 2, 3, 4])),
    'int8': OnnxTensor.fromInt8([1, 4], Int8List.fromList([-1, 0, 1, 127])),
    'uint8': OnnxTensor.fromUint8([1, 4], Uint8List.fromList([0, 1, 128, 255])),
    'bool': OnnxTensor.fromBool([1, 4], [false, true, true, false]),
  };
  for (final entry in samples.entries) {
    test('real ORT ${entry.key} round trip retains byte length', () async {
      final s = await InferenceSession.loadFile(
        model('identity_${entry.key}'),
        runtime: rt,
        options: SessionOptions.cpuOnly(),
      );
      final out = (await s.run({'X': entry.value}))['Y']!;
      expect(out.rawBytes, entry.value.rawBytes);
      expect(out.byteLength, entry.value.byteLength);
      await s.close();
    });
  }
  test('scalar, empty and rank20 metadata are preserved', () async {
    for (final suffix in ['_scalar', '_empty', '_rank20']) {
      final s = await InferenceSession.loadFile(
        model('identity_uint8$suffix'),
        runtime: rt,
        options: SessionOptions.cpuOnly(),
      );
      final shape = s.inputs['X']!.shape;
      final count = shape.contains(0) ? 0 : 1;
      final out = (await s.run({
        'X': OnnxTensor.fromUint8(shape, Uint8List(count)),
      }))['Y']!;
      expect(out.shape, shape);
      expect(out.elementCount, count);
      await s.close();
    }
  });
  test(
    'late/repeated token cancellation after Session and Runtime close is safe',
    () async {
      final s = await InferenceSession.loadBytes(
        await File(model('identity_float32')).readAsBytes(),
        runtime: rt,
        options: SessionOptions.cpuOnly(),
      );
      final token = CancellationToken();
      await s.run({
        'X': samples['float32']!,
      }, options: RunOptions(cancellationToken: token));
      await s.close();
      await rt.close();
      token.cancel();
      token.cancel();
    },
  );
  test('duplicate request IDs have different native task identities', () async {
    final a = await InferenceSession.loadFile(
      model('identity_float32'),
      runtime: rt,
      options: SessionOptions.cpuOnly(),
    );
    final b = await InferenceSession.loadFile(
      model('identity_float32'),
      runtime: rt,
      options: SessionOptions.cpuOnly(),
    );
    final results = await Future.wait(
      [a, b].map(
        (s) => s.runWithDiagnostics({
          'X': samples['float32']!,
        }, options: RunOptions(requestId: 42)),
      ),
    );
    expect(results[0].diagnostic.taskId, isNot(results[1].diagnostic.taskId));
    expect(results.every((r) => r.diagnostic.requestId == 42), isTrue);
  });
  test('pre-cancellation never submits native work', () async {
    final s = await InferenceSession.loadFile(
      model('identity_float32'),
      runtime: rt,
      options: SessionOptions.cpuOnly(),
    );
    final token = CancellationToken()..cancel();
    await expectLater(
      s.run({
        'X': samples['float32']!,
      }, options: RunOptions(cancellationToken: token)),
      throwsA(isA<OnnxException>().having((e) => e.code, 'code', 7)),
    );
  });
  test(
    'load rejects unsupported non-tensor/type metadata without leaking session',
    () async {
      for (final name in ['partial_sequence', 'unsupported_double']) {
        await expectLater(
          InferenceSession.loadFile(
            model(name),
            runtime: rt,
            options: SessionOptions.cpuOnly(),
          ),
          throwsA(isA<OnnxException>().having((e) => e.code, 'code', 11)),
        );
      }
    },
  );
  test(
    'force hardware verification fails explicitly rather than trusting Apple name',
    () async {
      await expectLater(
        InferenceSession.loadFile(
          model('mlp_smoke'),
          runtime: rt,
          options: const SessionOptions(
            policy: ExecutionPolicy(
              providerPriority: [ExecutionProvider.coreML],
              hardwareRequirement: HardwareRequirement.npu,
            ),
          ),
        ),
        throwsA(
          isA<OnnxException>()
              .having((e) => e.code, 'code', 6)
              .having(
                (e) => e.message,
                'message',
                contains('HardwareVerificationUnavailable'),
              ),
        ),
      );
    },
  );
  test(
    'CoreML CPUOnly never upgrades registration to physical observation',
    () async {
      final options = const SessionOptions(
        policy: ExecutionPolicy(
          providerPriority: [ExecutionProvider.coreML],
          coreMLComputeUnits: CoreMLComputeUnits.cpuOnly,
        ),
      );
      if (!rt.isProviderSupported(ExecutionProvider.coreML)) {
        final strict = const SessionOptions(
          policy: ExecutionPolicy(
            providerPriority: [ExecutionProvider.coreML],
            fallbackPolicy: FallbackPolicy.strict,
          ),
        );
        await expectLater(
          InferenceSession.loadFile(
            model('mlp_smoke'),
            runtime: rt,
            options: strict,
          ),
          throwsStateError,
        );
        return;
      }
      final s = await InferenceSession.loadFile(
        model('mlp_smoke'),
        runtime: rt,
        options: options,
      );
      final result = await s.runWithDiagnostics({'input': samples['float32']!});
      expect(result.outputs['output']!.asFloat32List(), [1.5, 2.5, 3.5, 4.5]);
      expect(
        result.diagnostic.registeredProviders,
        contains(ExecutionProvider.coreML),
      );
      expect(
        result.diagnostic.evidenceLevel,
        isNot(EvidenceLevel.hardwareObserved),
      );
      expect(result.diagnostic.isAccelerated, isFalse);
      expect(result.diagnostic.nodeDistribution, isEmpty);
      expect(result.diagnostic.actualProviders, isEmpty);
      expect(result.diagnostic.fallbackOccurred, isNull);
    },
  );
  test(
    'strict CoreML rejects an unsupported Loop graph instead of CPU fallback',
    () async {
      await expectLater(
        InferenceSession.loadFile(
          model('long_loop'),
          runtime: rt,
          options: const SessionOptions(
            policy: ExecutionPolicy(
              providerPriority: [ExecutionProvider.coreML],
              fallbackPolicy: FallbackPolicy.strict,
            ),
          ),
        ),
        throwsA(
          isA<OnnxException>().having(
            (e) => [4, 5].contains(e.code),
            'provider or graph fallback failure',
            isTrue,
          ),
        ),
      );
    },
  );
  test(
    'QNN unavailable reports backend/provider failure without Apple ANE probe',
    () async {
      if (rt.isProviderSupported(ExecutionProvider.qnn))
        return; // device-specific acceptance runs independently.
      await expectLater(
        InferenceSession.loadFile(
          model('mlp_smoke'),
          runtime: rt,
          options: SessionOptions.qualcommHtp(),
        ),
        throwsA(
          isA<OnnxException>()
              .having((e) => e.stage, 'stage', 'qnn.initialize')
              .having((e) => e.message, 'cause', isNot(contains('has_ane'))),
        ),
      );
    },
  );
  test('runtime close safely races async loading', () async {
    final loading = InferenceSession.loadFile(
      model('identity_float32'),
      runtime: rt,
      options: SessionOptions.cpuOnly(),
    );
    final assertion = expectLater(loading, throwsStateError);
    await rt.close();
    await assertion;
  });
  test('independent isolates do not reuse native IDs', () async {
    final path = Platform.environment['LIB_ONNX_TEST_BRIDGE'];
    Future<int?> run() => Isolate.run(() async {
      final r = OnnxRuntime.create(libraryPath: path);
      try {
        final s = await InferenceSession.loadFile(
          'test/fixtures/identity_float32.onnx',
          runtime: r,
          options: SessionOptions.cpuOnly(),
        );
        return (await s.runWithDiagnostics({
          'X': OnnxTensor.fromFloat32([1, 4], Float32List(4)),
        }, options: RunOptions(requestId: 1))).diagnostic.taskId;
      } finally {
        await r.close();
      }
    });
    final results = await Future.wait([run(), run()]);
    expect(results[0], isNot(results[1]));
  });
}
