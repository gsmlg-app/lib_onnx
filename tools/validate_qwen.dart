// Runs a real imported Qwen ONNX export through the Dart/native library.
// Inputs/reference vectors must come from prepare_qwen.py, never hand-written
// hidden states. A passing report is model correctness evidence for this run;
// physical GPU/NPU acceptance requires separate hardware observations.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:lib_onnx/lib_onnx.dart';
import 'package:path/path.dart' as path;

const revision = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3';

Future<String> digestFile(File file) async =>
    (await sha256.bind(file.openRead()).first).toString();

Future<String> verifyFiles(
  Map<String, dynamic> manifest,
  String directory,
) async {
  final root = await Directory(directory).resolveSymbolicLinks();
  final files = (manifest['files'] as List).cast<Map<String, dynamic>>()
    ..sort((a, b) => (a['path'] as String).compareTo(b['path'] as String));
  final seen = <String>{};
  final canonical = <Map<String, dynamic>>[];
  for (final entry in files) {
    final relative = entry['path'] as String;
    if (path.isAbsolute(relative) ||
        relative.contains('\\') ||
        relative.contains(':') ||
        relative
            .split('/')
            .any((part) => part.isEmpty || part == '.' || part == '..') ||
        !seen.add(relative)) {
      throw FormatException('Unsafe or duplicate model path: $relative');
    }
    final file = File(path.join(root, relative));
    final resolved = await file.resolveSymbolicLinks();
    if (!path.isWithin(root, resolved)) {
      throw FormatException('Model file escapes inventory root: $relative');
    }
    final hash = await digestFile(file);
    final size = await file.length();
    if (entry['sha256'] != hash || entry['size'] != size) {
      throw FormatException('Model file checksum/size mismatch: $relative');
    }
    // Key order path,size,sha256; UTF-8 compact JSON, matching session identity.
    canonical.add({'path': relative, 'size': size, 'sha256': hash});
  }
  final required = {
    manifest['onnx_file_name'] as String,
    ...(manifest['external_data_files'] as List).cast<String>(),
  };
  if (seen.length != required.length || !seen.containsAll(required)) {
    throw const FormatException(
      'Inventory omits ONNX/external data or adds unrelated files',
    );
  }
  final primary = manifest['onnx_file_name'] as String;
  final externalChecksums =
      manifest['external_data_sha256'] as Map<String, dynamic>;
  for (final record in canonical) {
    final name = record['path'] as String;
    final declared = name == primary
        ? manifest['sha256']
        : externalChecksums[name];
    if (declared != record['sha256']) {
      throw FormatException('Manifest checksums disagree for $name');
    }
  }
  final primaryRecord = canonical.singleWhere(
    (record) => record['path'] == primary,
  );
  final ordered = [
    {
      'path': 'model.onnx',
      'size': primaryRecord['size'],
      'sha256': primaryRecord['sha256'],
    },
    ...canonical.where((record) => record['path'] != primary),
  ];
  final identity = sha256.convert(utf8.encode(jsonEncode(ordered))).toString();
  if (identity != manifest['model_identity_sha256']) {
    throw const FormatException(
      'Combined model/external data identity mismatch',
    );
  }
  return identity;
}

OnnxTensor inputTensor(Map<String, dynamic> spec) {
  final shape = (spec['shape'] as List).cast<int>();
  final values = (spec['values'] as List).cast<int>();
  return switch (spec['dtype']) {
    'int64' => OnnxTensor.fromInt64(shape, Int64List.fromList(values)),
    'int32' => OnnxTensor.fromInt32(shape, Int32List.fromList(values)),
    _ => throw FormatException(
      'Unsupported prepared input dtype: ${spec['dtype']}',
    ),
  };
}

Future<void> main(List<String> arguments) async {
  if (arguments.contains('--help')) {
    stdout.writeln(
      'dart run tools/validate_qwen.dart '
      '--manifest FILE --model-directory DIR --reference FILE '
      '--backend cpu|coreml|qnn --report FILE '
      '[--minimum-cosine 0.999] [--maximum-absolute-error 0.01]',
    );
    return;
  }
  final args = <String, String>{};
  for (var i = 0; i < arguments.length; i += 2) {
    if (!arguments[i].startsWith('--') || i + 1 >= arguments.length) {
      throw const FormatException('Expected --name value argument pairs');
    }
    args[arguments[i].substring(2)] = arguments[i + 1];
  }
  for (final key in [
    'manifest',
    'model-directory',
    'reference',
    'backend',
    'report',
  ]) {
    if (!args.containsKey(key)) throw ArgumentError('Missing --$key');
  }
  final minimumCosine = double.parse(args['minimum-cosine'] ?? '0.999');
  final maximumError = double.parse(args['maximum-absolute-error'] ?? '0.01');
  if (!minimumCosine.isFinite ||
      minimumCosine < -1 ||
      minimumCosine > 1 ||
      !maximumError.isFinite ||
      maximumError < 0) {
    throw ArgumentError('Invalid correctness thresholds');
  }
  final manifest =
      jsonDecode(await File(args['manifest']!).readAsString())
          as Map<String, dynamic>;
  ModelManifest.fromJson(manifest).validate();
  final identity = await verifyFiles(manifest, args['model-directory']!);
  final reference =
      jsonDecode(await File(args['reference']!).readAsString())
          as Map<String, dynamic>;
  if (reference['model_identity_sha256'] != identity ||
      reference['checkpoint_revision'] != revision ||
      reference['tokenizer_revision'] != revision) {
    throw const FormatException('Reference/model/revision mismatch');
  }
  final provider = switch (args['backend']) {
    'cpu' => ExecutionProvider.cpu,
    'coreml' => ExecutionProvider.coreML,
    'qnn' => ExecutionProvider.qnn,
    _ => throw ArgumentError('Unsupported --backend ${args['backend']}'),
  };
  final runtime = OnnxRuntime.create();
  InferenceSession? session;
  final reports = <Map<String, dynamic>>[];
  final embeddings = <Float32List>[];
  try {
    session = await InferenceSession.loadFile(
      path.join(args['model-directory']!, manifest['onnx_file_name'] as String),
      runtime: runtime,
      options: SessionOptions(
        policy: ExecutionPolicy(
          providerPriority: [provider],
          fallbackPolicy: FallbackPolicy.strict,
          hardwareRequirement: HardwareRequirement.any,
        ),
      ),
    );
    final adapter = EmbeddingAdapter(
      hiddenDim: manifest['hidden_dim'] as int,
      hasEmbeddedPooling: manifest['has_embedded_pooling'] as bool,
    );
    final cases = reference['cases'] as List;
    if (cases.length != 4)
      throw const FormatException('Expected four real text/reference cases');
    for (var index = 0; index < cases.length; index++) {
      final item = cases[index] as Map<String, dynamic>;
      final tensors = <String, OnnxTensor>{
        for (final entry in (item['inputs'] as Map<String, dynamic>).entries)
          entry.key: inputTensor(entry.value as Map<String, dynamic>),
      };
      Map<String, OnnxTensor>? outputs;
      try {
        if (tensors.keys
                .toSet()
                .difference(session.inputs.keys.toSet())
                .isNotEmpty ||
            session.inputs.keys
                .toSet()
                .difference(tensors.keys.toSet())
                .isNotEmpty) {
          throw const FormatException(
            'Prepared inputs do not match actual ONNX metadata',
          );
        }
        final result = await session.runWithDiagnostics(tensors);
        outputs = result.outputs;
        final output = outputs[manifest['embedding_output']];
        if (output == null)
          throw const FormatException('Missing selected embedding output');
        final vector = adapter
            .extractEmbeddings(
              modelOutput: output,
              attentionMask: adapter.hasEmbeddedPooling
                  ? null
                  : tensors['attention_mask'],
            )
            .single;
        final expected = Float32List.fromList(
          (item['reference_embedding'] as List)
              .map((value) => (value as num).toDouble())
              .toList(),
        );
        EmbeddingAdapter.normalizeL2(expected);
        final cosine = EmbeddingAdapter.cosineSimilarity(vector, expected);
        var error = 0.0;
        for (var i = 0; i < vector.length; i++) {
          error = math.max(error, (vector[i] - expected[i]).abs());
        }
        if (cosine < minimumCosine || error > maximumError) {
          throw StateError(
            'Case $index differs from actual checkpoint: '
            'cosine=$cosine, maximum_absolute_error=$error',
          );
        }
        embeddings.add(vector);
        final diagnostic = result.diagnostic;
        if (diagnostic.modelHash != identity) {
          throw StateError(
            'Native run model identity differs from imported inventory',
          );
        }
        reports.add({
          'case': index,
          'cosine_to_checkpoint': cosine,
          'maximum_absolute_error': error,
          'duration_us': diagnostic.durationUs,
          'actual_provider': diagnostic.activeProvider?.ortName,
          'evidence': diagnostic.evidenceLevel.description,
          'hardware_details': diagnostic.hardwareDetails,
          'fallback_occurred': diagnostic.fallbackOccurred,
          'native_model_hash': diagnostic.modelHash,
          'node_distribution': diagnostic.nodeDistribution,
        });
      } finally {
        for (final tensor in tensors.values) {
          tensor.dispose();
        }
        for (final tensor in outputs?.values ?? <OnnxTensor>[]) {
          tensor.dispose();
        }
      }
    }
    final scores = [
      for (var query = 0; query < 2; query++)
        [
          for (var document = 2; document < 4; document++)
            EmbeddingAdapter.cosineSimilarity(
              embeddings[query],
              embeddings[document],
            ),
        ],
    ];
    final expected = (reference['expected_top_document'] as List).cast<int>();
    for (var query = 0; query < 2; query++) {
      final top = scores[query][0] > scores[query][1] ? 0 : 1;
      if (top != expected[query])
        throw StateError('Retrieval smoke ranking failed: $scores');
    }
    await File(args['report']!).writeAsString(
      const JsonEncoder.withIndent('  ').convert({
        'status': 'passed',
        'model_identity_sha256': identity,
        'checkpoint_revision': revision,
        'tokenizer_revision': revision,
        'device_os':
            '${Platform.operatingSystem} ${Platform.operatingSystemVersion}',
        'dart_version': Platform.version,
        'bridge_version': OnnxRuntime.bridgeVersion,
        'ort_version': OnnxRuntime.ortVersion,
        'requested_backend': args['backend'],
        'padding_side': reference['padding_side'],
        'cases': reports,
        'retrieval_scores': scores,
        'minimum_cosine': minimumCosine,
        'maximum_absolute_error': maximumError,
        'hardware_acceptance':
            'not established by numerical correctness or EP registration',
        'scope':
            'four real texts; not MTEB and not a production-readiness claim',
      }),
    );
    stdout.writeln(
      'Passed actual Qwen Dart/native correctness and retrieval smoke check. '
      'Report: ${args['report']}',
    );
  } catch (error, stack) {
    await File(args['report']!).writeAsString(
      const JsonEncoder.withIndent('  ').convert({
        'status': 'failed',
        'model_identity_sha256': identity,
        'requested_backend': args['backend'],
        'completed_cases': reports,
        'error': error.toString(),
        'stack': stack.toString(),
        'hardware_acceptance': 'not established',
      }),
    );
    rethrow;
  } finally {
    await session?.close();
    await runtime.close();
  }
}
