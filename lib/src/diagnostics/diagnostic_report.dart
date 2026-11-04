import '../provider/execution_provider.dart';
import 'hardware_evidence.dart';

/// A statement with its provenance; registration and compute plans are not usage.
class HardwareEvidence {
  final String kind, source, scope, details;
  final String? modelHash;
  final String runtimeVersion;
  final int? taskId;
  const HardwareEvidence({
    required this.kind,
    required this.source,
    required this.scope,
    required this.details,
    required this.runtimeVersion,
    this.modelHash,
    this.taskId,
  });
}

/// Immutable diagnostics belonging to one native task.
class DiagnosticReport {
  /// Null when actual graph assignment was not measured.
  final ExecutionProvider? activeProvider;
  final List<ExecutionProvider> registeredProviders;

  /// Multiple providers can execute a graph. Empty means unobserved.
  final List<ExecutionProvider> actualProviders;
  final EvidenceLevel evidenceLevel;
  final int durationUs;

  /// Digest of a reproducible file manifest, including external tensor data.
  final String? modelHash;
  final String hardwareDetails;

  /// Null means unknown; an explicit CPU baseline is false.
  final bool? fallbackOccurred;
  final Map<String, int> nodeDistribution;
  final int? taskId, requestId;
  final String? runtimeVersion;
  final List<HardwareEvidence> evidence;
  const DiagnosticReport({
    required this.activeProvider,
    required this.evidenceLevel,
    required this.durationUs,
    required this.modelHash,
    required this.hardwareDetails,
    required this.fallbackOccurred,
    this.nodeDistribution = const {},
    this.registeredProviders = const [],
    this.actualProviders = const [],
    this.taskId,
    this.requestId,
    this.runtimeVersion,
    this.evidence = const [],
  });
  double get durationMs => durationUs / 1000;

  /// Only a run observation establishes physical acceleration.
  bool get isAccelerated => evidenceLevel == EvidenceLevel.hardwareObserved;
  @override
  String toString() =>
      'DiagnosticReport(task: $taskId, actual: '
      '${actualProviders.map((p) => p.ortName).toList()}, registered: '
      '${registeredProviders.map((p) => p.ortName).toList()}, evidence: '
      '${evidenceLevel.description}, duration: $durationMs ms, fallback: $fallbackOccurred)';
}
