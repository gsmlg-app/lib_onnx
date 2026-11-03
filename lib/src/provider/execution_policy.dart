import 'execution_provider.dart';

/// Comprehensive execution policy controlling provider order, CPU fallback, and hardware requirements.
class ExecutionPolicy {
  /// Ordered list of execution providers to attempt.
  final List<ExecutionProvider> providerPriority;

  /// CPU fallback behavior.
  final FallbackPolicy fallbackPolicy;

  /// Hardware verification requirement.
  final HardwareRequirement hardwareRequirement;

  /// Whitelisted ops permitted on CPU if fallbackPolicy is whitelistedOps.
  final List<String> whitelistedOps;

  /// Apple CoreML compute units configuration.
  final CoreMLComputeUnits coreMLComputeUnits;

  /// Whether to strictly require Apple Neural Engine in CoreML.
  final bool requireAne;

  /// Optional CoreML compiled model cache directory.
  final String? coreMLModelCacheDir;

  /// Qualcomm QNN HTP performance mode.
  final QnnHtpPerformanceMode qnnPerformanceMode;

  /// Qualcomm QNN custom backend library path (e.g. libQnnHtp.so).
  final String? qnnBackendPath;

  /// Target Qualcomm SoC model number.
  final int? qnnSocModel;

  /// Optional extra key-value provider options.
  final Map<String, String> extraProviderOptions;

  const ExecutionPolicy({
    required this.providerPriority,
    this.fallbackPolicy = FallbackPolicy.allowed,
    this.hardwareRequirement = HardwareRequirement.any,
    this.whitelistedOps = const [],
    this.coreMLComputeUnits = CoreMLComputeUnits.all,
    this.requireAne = false,
    this.coreMLModelCacheDir,
    this.qnnPerformanceMode = QnnHtpPerformanceMode.highPerformance,
    this.qnnBackendPath,
    this.qnnSocModel,
    this.extraProviderOptions = const {},
  });

  /// Factory for Apple Silicon (macOS / iOS) with GPU and ANE support.
  factory ExecutionPolicy.appleSilicon({
    bool preferAne = true,
    bool allowCpuFallback = false,
    String? modelCacheDir,
  }) {
    return ExecutionPolicy(
      providerPriority: allowCpuFallback
          ? [ExecutionProvider.coreML, ExecutionProvider.cpu]
          : [ExecutionProvider.coreML],
      fallbackPolicy:
          allowCpuFallback ? FallbackPolicy.allowed : FallbackPolicy.strict,
      hardwareRequirement:
          preferAne ? HardwareRequirement.npu : HardwareRequirement.gpu,
      coreMLComputeUnits:
          preferAne ? CoreMLComputeUnits.all : CoreMLComputeUnits.cpuAndGpu,
      requireAne: preferAne,
      coreMLModelCacheDir: modelCacheDir,
    );
  }

  /// Factory for Qualcomm Snapdragon (Android) with HTP NPU support.
  factory ExecutionPolicy.qualcommHtp({
    int? socModel,
    bool allowCpuFallback = false,
    String? backendPath,
  }) {
    return ExecutionPolicy(
      providerPriority: allowCpuFallback
          ? [ExecutionProvider.qnn, ExecutionProvider.cpu]
          : [ExecutionProvider.qnn],
      fallbackPolicy:
          allowCpuFallback ? FallbackPolicy.allowed : FallbackPolicy.strict,
      hardwareRequirement: HardwareRequirement.npu,
      qnnPerformanceMode: QnnHtpPerformanceMode.highPerformance,
      qnnSocModel: socModel,
      qnnBackendPath: backendPath,
    );
  }

  /// Factory for strictly CPU-based execution (benchmarks and explicit fallback).
  factory ExecutionPolicy.cpuOnly() {
    return const ExecutionPolicy(
      providerPriority: [ExecutionProvider.cpu],
      fallbackPolicy: FallbackPolicy.allowed,
      hardwareRequirement: HardwareRequirement.any,
    );
  }

  /// Exports options to key-value pairs for the native bridge.
  Map<String, String> toOptionMap() {
    final map = <String, String>{};
    map['fallback_policy'] = fallbackPolicy.nativeCode.toString();
    map['hardware_requirement'] = hardwareRequirement.nativeCode.toString();

    // CoreML options
    map['coreml.MLComputeUnits'] = coreMLComputeUnits.value;
    map['coreml.RequireANE'] = requireAne ? '1' : '0';
    if (coreMLModelCacheDir != null && coreMLModelCacheDir!.isNotEmpty) {
      map['coreml.ModelCacheDir'] = coreMLModelCacheDir!;
    }

    // QNN options
    map['qnn.htp_performance_mode'] = qnnPerformanceMode.value;
    if (qnnBackendPath != null && qnnBackendPath!.isNotEmpty) {
      map['qnn.backend_path'] = qnnBackendPath!;
    }
    if (qnnSocModel != null) {
      map['qnn.soc_model'] = qnnSocModel.toString();
    }

    // Whitelisted ops
    if (whitelistedOps.isNotEmpty) {
      map['whitelisted_ops'] = whitelistedOps.join(',');
    }

    map.addAll(extraProviderOptions);
    return map;
  }
}
