import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:lib_onnx/lib_onnx.dart';

/// Small-model software route check. No profiling/physical utilization claim.
Future<void> main(List<String> arguments) async {
  final settings = <String, String>{};
  for (final argument in arguments) {
    final split = argument.indexOf('=');
    if (!argument.startsWith('--') || split < 0)
      throw ArgumentError(
        'Use --provider=CPU|CoreML|QNN --units=ALL --fallback=strict|allowed --model=path',
      );
    settings[argument.substring(2, split)] = argument.substring(split + 1);
  }
  final provider = ExecutionProvider.fromOrtName(
    settings['provider'] ?? 'CoreML',
  );
  final units = CoreMLComputeUnits.values.firstWhere(
    (u) => u.value == (settings['units'] ?? 'ALL'),
  );
  final fallback = switch (settings['fallback'] ?? 'strict') {
    'strict' => FallbackPolicy.strict,
    'allowed' => FallbackPolicy.allowed,
    _ => throw ArgumentError('Invalid fallback'),
  };
  final model = settings['model'] ?? 'test/fixtures/mlp_smoke.onnx';
  final runtime = await OnnxRuntime.createAsync();
  try {
    final session = await InferenceSession.loadFile(
      model,
      runtime: runtime,
      options: SessionOptions(
        policy: ExecutionPolicy(
          providerPriority: [provider],
          fallbackPolicy: fallback,
          coreMLComputeUnits: units,
          qnnBackendPath: 'libQnnHtp.so',
        ),
      ),
    );
    try {
      final result = await session.runWithDiagnostics({
        'input': OnnxTensor.fromFloat32([
          1,
          4,
        ], Float32List.fromList([1, 2, 3, 4])),
      });
      final values = result.outputs['output']!.asFloat32List();
      if (List.generate(
        4,
        (i) => (values[i] - (i + 1.5)).abs() > 0.0001,
      ).any((x) => x))
        throw StateError('Actual model numerical mismatch: $values');
      final d = result.diagnostic;
      print(
        jsonEncode({
          'status': 'passed',
          'model': model,
          'model_file_sha256': sha256
              .convert(await File(model).readAsBytes())
              .toString(),
          'model_identity': d.modelHash,
          'task_id': d.taskId,
          'runtime_version': d.runtimeVersion,
          'bridge': OnnxRuntime.bridgeVersion,
          'requested_provider': provider.ortName,
          'compute_units': units.value,
          'fallback_policy': fallback.label,
          'registered_providers': d.registeredProviders
              .map((p) => p.ortName)
              .toList(),
          'actual_providers': d.actualProviders.map((p) => p.ortName).toList(),
          'node_distribution': d.nodeDistribution,
          'fallback_occurred': d.fallbackOccurred,
          'duration_us': d.durationUs,
          'evidence': d.evidenceLevel.name,
          'details': d.hardwareDetails,
          'output': values.toList(),
          'hardware_acceptance':
              'not-run; no physical utilization trace collected',
        }),
      );
    } finally {
      await session.close();
    }
  } on OnnxException catch (error) {
    print(
      jsonEncode({
        'status': 'failed',
        'model': model,
        'provider': provider.ortName,
        'units': units.value,
        'fallback_policy': fallback.label,
        'code': error.code,
        'ort_code': error.ortCode,
        'stage': error.stage,
        'error': error.message,
        'hardware_acceptance': 'not-run',
      }),
    );
    exitCode = 1;
  } finally {
    await runtime.close();
  }
}
