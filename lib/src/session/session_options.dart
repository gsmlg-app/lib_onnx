import '../provider/execution_policy.dart';

/// Graph optimization levels.
enum GraphOptimizationLevel {
  disableAll(0),
  basic(1),
  extended(2),
  all(99);

  final int nativeLevel;
  const GraphOptimizationLevel(this.nativeLevel);
}

/// Configuration options for configuring an InferenceSession.
class SessionOptions {
  /// Execution policy defining providers, fallback, and hardware constraints.
  final ExecutionPolicy policy;

  /// Thread count for intra-op parallelism (inside an operator).
  final int intraOpNumThreads;

  /// Thread count for inter-op parallelism (across independent operators).
  final int interOpNumThreads;

  /// Graph optimization level.
  final GraphOptimizationLevel graphOptimizationLevel;

  /// Whether memory pattern optimization is enabled.
  final bool enableMemoryPattern;

  /// Maximum pending tasks in the session's execution queue (bounded queue).
  final int maxQueueDepth;

  /// Maximum concurrent in-flight native executions allowed for this session.
  /// Section 4.4 defaults to 1 per session until backend concurrency is validated.
  final int maxConcurrentRuns;

  /// Custom logging identifier.
  final String? logId;

  const SessionOptions({
    this.policy = const ExecutionPolicy(providerPriority: []),
    this.intraOpNumThreads = 1,
    this.interOpNumThreads = 1,
    this.graphOptimizationLevel = GraphOptimizationLevel.all,
    this.enableMemoryPattern = true,
    this.maxQueueDepth = 16,
    this.maxConcurrentRuns = 1,
    this.logId,
  });

  /// Factory configured for Apple Silicon acceleration.
  factory SessionOptions.appleSilicon({
    bool preferAne = true,
    bool allowCpuFallback = false,
    String? modelCacheDir,
  }) {
    return SessionOptions(
      policy: ExecutionPolicy.appleSilicon(
        preferAne: preferAne,
        allowCpuFallback: allowCpuFallback,
        modelCacheDir: modelCacheDir,
      ),
    );
  }

  /// Factory configured for Qualcomm Snapdragon HTP acceleration.
  factory SessionOptions.qualcommHtp({
    int? socModel,
    bool allowCpuFallback = false,
    String? backendPath,
  }) {
    return SessionOptions(
      policy: ExecutionPolicy.qualcommHtp(
        socModel: socModel,
        allowCpuFallback: allowCpuFallback,
        backendPath: backendPath,
      ),
    );
  }

  /// Factory configured for strict CPU evaluation.
  factory SessionOptions.cpuOnly({int numThreads = 1}) {
    return SessionOptions(
      policy: ExecutionPolicy.cpuOnly(),
      intraOpNumThreads: numThreads,
    );
  }
}
