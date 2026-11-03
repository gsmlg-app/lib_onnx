import 'package:lib_onnx/lib_onnx.dart';
import 'package:test/test.dart';

void main() {
  group('ExecutionPolicy', () {
    test('Apple Silicon factory sets CoreML with ANE and strict defaults', () {
      final policy = ExecutionPolicy.appleSilicon(
          preferAne: true, allowCpuFallback: false);

      expect(policy.providerPriority, equals([ExecutionProvider.coreML]));
      expect(policy.fallbackPolicy, equals(FallbackPolicy.strict));
      expect(policy.hardwareRequirement, equals(HardwareRequirement.npu));
      expect(policy.coreMLComputeUnits, equals(CoreMLComputeUnits.all));
      expect(policy.requireAne, isTrue);

      final map = policy.toOptionMap();
      expect(map['coreml.MLComputeUnits'], equals('ALL'));
      expect(map['coreml.RequireANE'], equals('1'));
      expect(map['fallback_policy'], equals('1')); // strict
      expect(map['hardware_requirement'], equals('2')); // npu
    });

    test('Apple Silicon factory with CPU fallback allowed', () {
      final policy = ExecutionPolicy.appleSilicon(
          preferAne: false, allowCpuFallback: true);

      expect(policy.providerPriority,
          equals([ExecutionProvider.coreML, ExecutionProvider.cpu]));
      expect(policy.fallbackPolicy, equals(FallbackPolicy.allowed));
      expect(policy.hardwareRequirement, equals(HardwareRequirement.gpu));
      expect(policy.coreMLComputeUnits, equals(CoreMLComputeUnits.cpuAndGpu));
      expect(policy.requireAne, isFalse);
    });

    test('Qualcomm Snapdragon HTP factory sets QNN options', () {
      final policy = ExecutionPolicy.qualcommHtp(
        socModel: 68,
        allowCpuFallback: false,
        backendPath: 'libQnnHtp.so',
      );

      expect(policy.providerPriority, equals([ExecutionProvider.qnn]));
      expect(policy.hardwareRequirement, equals(HardwareRequirement.npu));
      expect(policy.fallbackPolicy, equals(FallbackPolicy.strict));

      final map = policy.toOptionMap();
      expect(map['qnn.htp_performance_mode'], equals('high_performance'));
      expect(map['qnn.soc_model'], equals('68'));
      expect(map['qnn.backend_path'], equals('libQnnHtp.so'));
    });

    test('CPU only policy', () {
      final policy = ExecutionPolicy.cpuOnly();
      expect(policy.providerPriority, equals([ExecutionProvider.cpu]));
      expect(policy.fallbackPolicy, equals(FallbackPolicy.allowed));
      expect(policy.hardwareRequirement, equals(HardwareRequirement.any));
    });
  });
}
