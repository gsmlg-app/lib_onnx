import '../provider/execution_provider.dart';
import 'hardware_evidence.dart';

/// Diagnostics report providing transparency into model execution, provider assignment,
/// hardware evidence, and CPU fallbacks.
class DiagnosticReport {
  /// The active execution provider that executed the model.
  final ExecutionProvider activeProvider;

  /// Level of evidence verifying hardware accelerator involvement.
  final EvidenceLevel evidenceLevel;

  /// Duration of the inference run in microseconds.
  final int durationUs;

  /// SHA-256 or digest hash of the model.
  final String modelHash;

  /// Detailed hardware descriptions or backend specifics.
  final String hardwareDetails;

  /// Whether any nodes or fallback to CPU occurred during execution.
  final bool fallbackOccurred;

  /// Optional breakdown of assigned nodes per provider.
  final Map<String, int> nodeDistribution;

  const DiagnosticReport({
    required this.activeProvider,
    required this.evidenceLevel,
    required this.durationUs,
    required this.modelHash,
    required this.hardwareDetails,
    required this.fallbackOccurred,
    this.nodeDistribution = const {},
  });

  /// Duration in milliseconds.
  double get durationMs => durationUs / 1000.0;

  /// Whether physical hardware acceleration was observed or graph partitioned.
  bool get isAccelerated =>
      evidenceLevel == EvidenceLevel.graphPartitioned ||
      evidenceLevel == EvidenceLevel.hardwareObserved;

  @override
  String toString() {
    return 'DiagnosticReport('
        'provider: ${activeProvider.ortName}, '
        'evidence: ${evidenceLevel.description}, '
        'duration: ${durationMs.toStringAsFixed(2)} ms, '
        'fallback: $fallbackOccurred, '
        'details: "$hardwareDetails")';
  }
}
