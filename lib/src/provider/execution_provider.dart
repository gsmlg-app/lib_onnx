/// Execution provider identifiers.
enum ExecutionProvider {
  coreML('CoreML', 'Apple CoreML Execution Provider (GPU / ANE)'),
  qnn('QNN', 'Qualcomm Snapdragon QNN Execution Provider (HTP / GPU)'),
  cuda('CUDA', 'NVIDIA CUDA Execution Provider'),
  directML('DirectML', 'DirectML Execution Provider'),
  cpu('CPU', 'Default CPU Execution Provider');

  final String ortName;
  final String description;

  const ExecutionProvider(this.ortName, this.description);

  static ExecutionProvider fromOrtName(String name) {
    for (final p in ExecutionProvider.values) {
      if (p.ortName.toLowerCase() == name.toLowerCase()) {
        return p;
      }
    }
    throw ArgumentError('Unknown execution provider name: $name');
  }
}

/// Fallback policy specifying whether and how CPU nodes are allowed.
enum FallbackPolicy {
  /// Strictly prohibits CPU execution when an accelerator is requested.
  /// If any node falls back to CPU, execution or creation fails.
  strict(1, 'strict'),

  /// Allows ORT to fall back to CPU execution if acceleration fails or is incomplete.
  allowed(0, 'allowed'),

  /// Allows CPU fallback only for a whitelisted set of control/shape ops.
  whitelistedOps(2, 'whitelisted_ops');

  final int nativeCode;
  final String label;

  const FallbackPolicy(this.nativeCode, this.label);
}

/// Hardware requirement specifying whether physical accelerator hardware must be verified.
enum HardwareRequirement {
  /// Any available hardware (accelerator preferred, or fallback allowed per policy).
  any(0, 'any'),

  /// Requires physical GPU involvement.
  gpu(1, 'gpu'),

  /// Requires physical NPU / ANE involvement.
  npu(2, 'npu');

  final int nativeCode;
  final String label;

  const HardwareRequirement(this.nativeCode, this.label);
}

/// CoreML specific compute units.
enum CoreMLComputeUnits {
  /// Allows all compute units (CPU, GPU, and Apple Neural Engine).
  all('ALL'),

  /// Restricts to CPU and GPU (disables ANE).
  cpuAndGpu('CPUAndGPU'),

  /// Restricts to CPU only.
  cpuOnly('CPUOnly'),

  /// Restricts to CPU and Neural Engine.
  cpuAndNeuralEngine('CPUAndNE');

  final String value;
  const CoreMLComputeUnits(this.value);
}

/// QNN HTP performance profile.
enum QnnHtpPerformanceMode {
  burst('burst'),
  sustainedHighPerformance('sustained_high_performance'),
  highPerformance('high_performance'),
  powerSaver('power_saver'),
  balanced('balanced');

  final String value;
  const QnnHtpPerformanceMode(this.value);
}
