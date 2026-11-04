import 'package:lib_onnx/lib_onnx.dart';
import 'package:test/test.dart';

void main() {
  test('Apple preference is distinct from mandatory observation', () {
    final p = ExecutionPolicy.appleSilicon();
    expect(p.hardwareRequirement, HardwareRequirement.any);
    expect(p.coreMLComputeUnits, CoreMLComputeUnits.cpuAndNeuralEngine);
    expect(p.toOptionMap()['coreml.MLComputeUnits'], 'CPUAndNeuralEngine');
    expect(
      ExecutionPolicy.appleSilicon(verifyHardware: true).hardwareRequirement,
      HardwareRequirement.npu,
    );
  });
  test('CoreML real option names and cache mapping', () {
    final options = ExecutionPolicy.appleSilicon(
      modelCacheDir: '/tmp/cache',
    ).toOptionMap();
    expect(options['coreml.ModelCacheDirectory'], '/tmp/cache');
    expect(options, isNot(contains('coreml.ModelCacheDir')));
  });
  test('QNN HTP preference does not depend on Apple hardware', () {
    final p = ExecutionPolicy.qualcommHtp();
    expect(p.hardwareRequirement, HardwareRequirement.any);
    expect(p.fallbackPolicy, FallbackPolicy.strict);
    expect(p.toOptionMap()['qnn.backend_path'], 'libQnnHtp.so');
  });
  test('whitelisted ops unsupported rather than permitted', () {
    expect(
      () => const ExecutionPolicy(
        providerPriority: [ExecutionProvider.coreML],
        fallbackPolicy: FallbackPolicy.whitelistedOps,
      ).toOptionMap(),
      throwsUnsupportedError,
    );
  });
  test('extras cannot override enforced policy/backend', () {
    for (final key in [
      'fallback_policy',
      'session.disable_cpu_ep_fallback',
      'bridge.max_active',
      'qnn.backend_path',
      'coreml.MLComputeUnits',
    ]) {
      expect(
        () => ExecutionPolicy(
          providerPriority: const [ExecutionProvider.qnn],
          extraProviderOptions: {key: '0'},
        ).toOptionMap(),
        throwsArgumentError,
      );
    }
  });
  test('CPUOnly hardware conflicts fail before execution', () {
    expect(
      () => const ExecutionPolicy(
        providerPriority: [ExecutionProvider.coreML],
        coreMLComputeUnits: CoreMLComputeUnits.cpuOnly,
        hardwareRequirement: HardwareRequirement.npu,
      ).toOptionMap(),
      throwsArgumentError,
    );
  });
  test('CPU priority before an accelerator is explicitly rejected', () {
    expect(
      () => const ExecutionPolicy(
        providerPriority: [ExecutionProvider.cpu, ExecutionProvider.coreML],
      ).toOptionMap(),
      throwsArgumentError,
    );
  });
  test('cancellation listener removes finished operation references', () {
    final token = CancellationToken();
    var count = 0;
    final remove = token.addListener(() => count++);
    remove();
    remove();
    token.cancel();
    expect(count, 0);
  });
}
