import 'execution_provider.dart';

/// Comprehensive execution policy controlling provider order, CPU fallback, and hardware requirements.
class ExecutionPolicy {
  /// Graph allocation priority. All requested accelerators are registered in order.
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
    bool verifyHardware = false,
  }) {
    return ExecutionPolicy(
      providerPriority: allowCpuFallback
          ? [ExecutionProvider.coreML, ExecutionProvider.cpu]
          : [ExecutionProvider.coreML],
      fallbackPolicy: allowCpuFallback
          ? FallbackPolicy.allowed
          : FallbackPolicy.strict,
      hardwareRequirement: verifyHardware
          ? (preferAne ? HardwareRequirement.npu : HardwareRequirement.gpu)
          : HardwareRequirement.any,
      coreMLComputeUnits: preferAne
          ? CoreMLComputeUnits.cpuAndNeuralEngine
          : CoreMLComputeUnits.cpuAndGpu,
      coreMLModelCacheDir: modelCacheDir,
    );
  }

  /// Factory for Qualcomm Snapdragon (Android) with HTP NPU support.
  factory ExecutionPolicy.qualcommHtp({
    int? socModel,
    bool allowCpuFallback = false,
    String? backendPath,
    bool verifyHardware = false,
  }) {
    return ExecutionPolicy(
      providerPriority: allowCpuFallback
          ? [ExecutionProvider.qnn, ExecutionProvider.cpu]
          : [ExecutionProvider.qnn],
      fallbackPolicy: allowCpuFallback
          ? FallbackPolicy.allowed
          : FallbackPolicy.strict,
      hardwareRequirement: verifyHardware
          ? HardwareRequirement.npu
          : HardwareRequirement.any,
      qnnPerformanceMode: QnnHtpPerformanceMode.highPerformance,
      qnnSocModel: socModel,
      qnnBackendPath: backendPath ?? 'libQnnHtp.so',
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
    validate();
    final map = <String, String>{};
    if (providerPriority.contains(ExecutionProvider.coreML)) {
      map['coreml.MLComputeUnits'] = coreMLComputeUnits.value;
      map['coreml.ModelFormat'] = 'MLProgram';
      map['coreml.RequireStaticInputShapes'] = '0';
      map['coreml.EnableOnSubgraphs'] = '0';
      if (coreMLModelCacheDir case final dir? when dir.isNotEmpty) {
        map['coreml.ModelCacheDirectory'] = dir;
      }
    }
    if (providerPriority.contains(ExecutionProvider.qnn)) {
      map['qnn.htp_performance_mode'] = qnnPerformanceMode.value;
      map['qnn.backend_path'] = qnnBackendPath ?? 'libQnnHtp.so';
      if (qnnSocModel != null) map['qnn.soc_model'] = qnnSocModel.toString();
    }
    map.addAll(extraProviderOptions);
    return Map.unmodifiable(map);
  }

  /// Reject guarantees the selected ORT API cannot safely enforce.
  void validate() {
    if (providerPriority.contains(ExecutionProvider.cpu) &&
        providerPriority.last != ExecutionProvider.cpu) {
      throw ArgumentError(
        'ORT CPU EP must be last in graph allocation priority',
      );
    }
    if (providerPriority.toSet().length != providerPriority.length) {
      throw ArgumentError('Duplicate providers');
    }
    if (fallbackPolicy == FallbackPolicy.whitelistedOps) {
      throw UnsupportedError(
        'UnsupportedPolicy: fused CPU nodes and subgraphs cannot be reliably whitelisted',
      );
    }
    if (requireAne) {
      throw UnsupportedError(
        'RequireANE is unsupported. CPUAndNeuralEngine permits CPU and ANE; use hardwareRequirement for verification',
      );
    }
    if (fallbackPolicy == FallbackPolicy.strict &&
        providerPriority.contains(ExecutionProvider.cpu) &&
        providerPriority.length > 1) {
      throw ArgumentError(
        'Strict accelerator policy conflicts with explicit CPU EP',
      );
    }
    if (hardwareRequirement != HardwareRequirement.any &&
        (providerPriority.isEmpty ||
            providerPriority.every((p) => p == ExecutionProvider.cpu) ||
            (providerPriority.contains(ExecutionProvider.coreML) &&
                coreMLComputeUnits == CoreMLComputeUnits.cpuOnly))) {
      throw ArgumentError(
        'CPU-only configuration conflicts with physical hardware requirement',
      );
    }
    for (final key in extraProviderOptions.keys) {
      // Advanced options stay EP-specific and cannot override enforced settings.
      if ((!key.startsWith('coreml.') && !key.startsWith('qnn.')) ||
          const {
            'coreml.MLComputeUnits',
            'coreml.ModelFormat',
            'coreml.ModelCacheDirectory',
            'coreml.ModelCacheDir',
            'coreml.RequireANE',
            'qnn.backend_path',
            'qnn.htp_performance_mode',
            'qnn.soc_model',
          }.contains(key)) {
        throw ArgumentError('Reserved/unsafe extra provider option: $key');
      }
    }
  }
}
