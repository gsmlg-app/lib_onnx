import 'dart:convert';

/// Specification for an input or output tensor.
class TensorSpec {
  final String name;
  final String dtype;
  final List<dynamic> shape; // int or String for dynamic dimension

  const TensorSpec({
    required this.name,
    required this.dtype,
    required this.shape,
  });

  Map<String, dynamic> toJson() => {
    'name': name,
    'dtype': dtype,
    'shape': shape,
  };

  factory TensorSpec.fromJson(Map<String, dynamic> json) => TensorSpec(
    name: json['name'] as String,
    dtype: json['dtype'] as String,
    shape: List<dynamic>.from(json['shape'] as List),
  );
}

/// Profile describing target hardware, quantization, and verified shape tier.
class ModelProfile {
  final String profileId;
  final String
  targetBackend; // e.g. "coreml_gpu", "coreml_ane", "qnn_htp", "cpu"
  final String precision; // e.g. "fp16", "int8", "fp32"
  final int batchSize;
  final int maxSequenceLength;
  final bool verified;
  final String? notes;

  const ModelProfile({
    required this.profileId,
    required this.targetBackend,
    required this.precision,
    required this.batchSize,
    required this.maxSequenceLength,
    required this.verified,
    this.notes,
  });

  Map<String, dynamic> toJson() => {
    'profile_id': profileId,
    'target_backend': targetBackend,
    'precision': precision,
    'batch_size': batchSize,
    'max_sequence_length': maxSequenceLength,
    'verified': verified,
    if (notes != null) 'notes': notes,
  };

  factory ModelProfile.fromJson(Map<String, dynamic> json) => ModelProfile(
    profileId: json['profile_id'] as String,
    targetBackend: json['target_backend'] as String,
    precision: json['precision'] as String,
    batchSize: json['batch_size'] as int,
    maxSequenceLength: json['max_sequence_length'] as int,
    verified: json['verified'] as bool,
    notes: json['notes'] as String?,
  );
}

/// Complete model manifest tracking file integrity, revision, contracts, and verified profiles.
class ModelManifest {
  final String modelName;
  final String checkpointRevision;
  final String tokenizerRevision;
  final String onnxFileName;
  final String sha256;
  final List<String> externalDataFiles;

  /// SHA-256 of every external data file, keyed by model-relative path.
  final Map<String, String> externalDataSha256;

  /// Digest of the reproducible ONNX + external data file inventory.
  final String? modelIdentitySha256;
  final int opsetVersion;
  final List<TensorSpec> inputs;
  final List<TensorSpec> outputs;
  final List<ModelProfile> profiles;

  const ModelManifest({
    required this.modelName,
    required this.checkpointRevision,
    required this.tokenizerRevision,
    required this.onnxFileName,
    required this.sha256,
    this.externalDataFiles = const [],
    this.externalDataSha256 = const {},
    this.modelIdentitySha256,
    required this.opsetVersion,
    required this.inputs,
    required this.outputs,
    required this.profiles,
  });

  Map<String, dynamic> toJson() => {
    'model_name': modelName,
    'checkpoint_revision': checkpointRevision,
    'tokenizer_revision': tokenizerRevision,
    'onnx_file_name': onnxFileName,
    'sha256': sha256,
    'external_data_files': externalDataFiles,
    'external_data_sha256': externalDataSha256,
    'model_identity_sha256': modelIdentitySha256,
    'opset_version': opsetVersion,
    'inputs': inputs.map((e) => e.toJson()).toList(),
    'outputs': outputs.map((e) => e.toJson()).toList(),
    'profiles': profiles.map((e) => e.toJson()).toList(),
  };

  String toPrettyJson() => const JsonEncoder.withIndent('  ').convert(toJson());

  factory ModelManifest.fromJson(Map<String, dynamic> json) {
    final manifest = ModelManifest(
      modelName: json['model_name'] as String,
      checkpointRevision: json['checkpoint_revision'] as String,
      tokenizerRevision: json['tokenizer_revision'] as String,
      onnxFileName: json['onnx_file_name'] as String,
      sha256: json['sha256'] as String,
      externalDataFiles:
          (json['external_data_files'] as List<dynamic>?)?.cast<String>() ??
          const [],
      externalDataSha256:
          (json['external_data_sha256'] as Map<String, dynamic>?)
              ?.cast<String, String>() ??
          const {},
      modelIdentitySha256: json['model_identity_sha256'] as String?,
      opsetVersion: json['opset_version'] as int,
      inputs: (json['inputs'] as List<dynamic>)
          .map((e) => TensorSpec.fromJson(e as Map<String, dynamic>))
          .toList(),
      outputs: (json['outputs'] as List<dynamic>)
          .map((e) => TensorSpec.fromJson(e as Map<String, dynamic>))
          .toList(),
      profiles: (json['profiles'] as List<dynamic>)
          .map((e) => ModelProfile.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
    manifest.validate();
    return manifest;
  }

  /// Validates identifiers and file inventory; actual file contents must still
  /// be hashed by the caller before loading. [sha256] covers the ONNX file only.
  void validate() {
    final digest = RegExp(r'^[a-f0-9]{64}$');
    bool safePath(String value) =>
        value.isNotEmpty &&
        !value.startsWith('/') &&
        !value.contains('\\') &&
        !value.contains(':') &&
        value
            .split('/')
            .every((part) => part.isNotEmpty && part != '.' && part != '..');
    if (!digest.hasMatch(sha256) ||
        !safePath(onnxFileName) ||
        checkpointRevision.isEmpty ||
        tokenizerRevision.isEmpty ||
        opsetVersion <= 0 ||
        (modelIdentitySha256 != null &&
            !digest.hasMatch(modelIdentitySha256!))) {
      throw const FormatException('Invalid model identifier, SHA-256 or opset');
    }
    if (externalDataFiles.toSet().length != externalDataFiles.length ||
        externalDataFiles.any(
          (path) =>
              !safePath(path) ||
              path == onnxFileName ||
              !digest.hasMatch(externalDataSha256[path] ?? ''),
        ) ||
        externalDataSha256.keys.any(
          (path) => !externalDataFiles.contains(path),
        )) {
      throw const FormatException(
        'External data requires safe, unique paths and per-file SHA-256',
      );
    }
  }
}
