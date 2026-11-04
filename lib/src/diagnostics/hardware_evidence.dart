/// Hierarchical levels of hardware acceleration evidence.
enum EvidenceLevel {
  /// Unknown or unprobed state.
  unknown(0, 'Unknown'),

  /// No acceleration evidence; executing purely on CPU or accelerator rejected.
  none(1, 'None (CPU)'),

  /// The Execution Provider library was successfully compiled and linked into the binary.
  epCompiled(2, 'EP Compiled'),

  /// The model graph was partitioned into subgraphs assigned to the accelerator EP.
  graphPartitioned(3, 'Graph Partitioned to EP'),

  /// Physical accelerator engagement confirmed by per-run telemetry (not a compute plan).
  hardwareObserved(4, 'Hardware Observed (GPU/ANE/NPU)'),

  /// Requested accelerator, but hardware activity cannot be reliably verified on this OS/driver.
  unverifiable(5, 'Unverifiable');

  final int code;
  final String description;

  const EvidenceLevel(this.code, this.description);

  static EvidenceLevel fromCode(int code) {
    for (final level in EvidenceLevel.values) {
      if (level.code == code) {
        return level;
      }
    }
    return EvidenceLevel.unknown;
  }
}
