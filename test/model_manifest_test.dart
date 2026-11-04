import 'dart:convert';
import 'dart:io';
import 'package:crypto/crypto.dart';
import 'package:lib_onnx/src/adapter/model_manifest.dart';
import 'package:lib_onnx/src/session/model_identity.dart';
import 'package:test/test.dart';
import '../tools/validate_qwen.dart' as qwen;

void main() {
  group('ModelManifest', () {
    test('Serializes and deserializes model manifest JSON accurately', () {
      const manifest = ModelManifest(
        modelName: 'serialization_fixture_only',
        checkpointRevision: 'local-test-fixture',
        tokenizerRevision: 'not-applicable',
        onnxFileName: 'fixture.onnx',
        sha256:
            'd3dc6ebdfff0c3b32a814eb77868bda8f4cdb9dc1538b6d1878560613bfb21cf',
        opsetVersion: 17,
        inputs: [
          TensorSpec(name: 'input_ids', dtype: 'int64', shape: [1, 128]),
          TensorSpec(name: 'attention_mask', dtype: 'int64', shape: [1, 128]),
        ],
        outputs: [
          TensorSpec(
            name: 'hidden_states',
            dtype: 'float32',
            shape: [1, 128, 64],
          ),
        ],
        profiles: [
          ModelProfile(
            profileId: 'coreml_ane_128',
            targetBackend: 'coreml_ane',
            precision: 'fp32',
            batchSize: 1,
            maxSequenceLength: 128,
            verified: false,
            notes: 'Serialization fixture; no hardware validation',
          ),
          ModelProfile(
            profileId: 'qnn_htp_128',
            targetBackend: 'qnn_htp',
            precision: 'int8',
            batchSize: 1,
            maxSequenceLength: 128,
            verified: false,
            notes: 'Requires Qualcomm Snapdragon device',
          ),
        ],
      );

      final jsonStr = manifest.toPrettyJson();
      final decodedMap = jsonDecode(jsonStr) as Map<String, dynamic>;
      final restored = ModelManifest.fromJson(decodedMap);

      expect(restored.modelName, equals(manifest.modelName));
      expect(restored.checkpointRevision, equals(manifest.checkpointRevision));
      expect(restored.sha256, equals(manifest.sha256));
      expect(restored.inputs.length, equals(2));
      expect(restored.outputs.length, equals(1));
      expect(restored.profiles.length, equals(2));
      expect(restored.profiles[0].targetBackend, equals('coreml_ane'));
      expect(restored.profiles[0].verified, isFalse);
      expect(restored.profiles[1].verified, isFalse);
    });

    test('rejects malformed file identity and unsafe external data paths', () {
      final json = {
        'model_name': 'invalid',
        'checkpoint_revision': 'local-test-fixture',
        'tokenizer_revision': 'not-applicable',
        'onnx_file_name': 'fixture.onnx',
        'sha256': 'uncomputed',
        'external_data_files': ['../outside.data'],
        'opset_version': 17,
        'inputs': <Object>[],
        'outputs': <Object>[],
        'profiles': <Object>[],
      };
      expect(() => ModelManifest.fromJson(json), throwsFormatException);
    });

    test('rejects an external data manifest without per-file checksums', () {
      final json = {
        'model_name': 'invalid',
        'checkpoint_revision': 'local-test-fixture',
        'tokenizer_revision': 'not-applicable',
        'onnx_file_name': 'fixture.onnx',
        'sha256': 'a' * 64,
        'external_data_files': ['weights.data'],
        'opset_version': 17,
        'inputs': <Object>[],
        'outputs': <Object>[],
        'profiles': <Object>[],
      };
      expect(() => ModelManifest.fromJson(json), throwsFormatException);
    });

    test(
      'Qwen verifier matches runtime identity after renaming the model',
      () async {
        final directory = await Directory.systemTemp.createTemp(
          'lib-onnx-qwen-inventory-',
        );
        try {
          final bytes = await File(
            'test/fixtures/mlp_smoke.onnx',
          ).readAsBytes();
          await File('${directory.path}/renamed.onnx').writeAsBytes(bytes);
          final identity = modelIdentity(bytes);
          final manifest = <String, dynamic>{
            'onnx_file_name': 'renamed.onnx',
            'sha256': sha256.convert(bytes).toString(),
            'external_data_files': <String>[],
            'external_data_sha256': <String, String>{},
            'model_identity_sha256': identity,
            'files': [
              {
                'path': 'renamed.onnx',
                'size': bytes.length,
                'sha256': sha256.convert(bytes).toString(),
              },
            ],
          };
          expect(await qwen.verifyFiles(manifest, directory.path), identity);
        } finally {
          await directory.delete(recursive: true);
        }
      },
    );
  });
}
