import 'dart:convert';
import 'package:lib_onnx/lib_onnx.dart';
import 'package:test/test.dart';

void main() {
  group('ModelManifest', () {
    test('Serializes and deserializes model manifest JSON accurately', () {
      const manifest = ModelManifest(
        modelName: 'qwen3_embedding_0.6b_fixture',
        checkpointRevision: 'rev_20261031',
        tokenizerRevision: 'tok_20261031',
        onnxFileName: 'qwen3_embedding_fixture.onnx',
        sha256:
            'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
        opsetVersion: 17,
        inputs: [
          TensorSpec(name: 'input_ids', dtype: 'int64', shape: [1, 128]),
          TensorSpec(name: 'attention_mask', dtype: 'int64', shape: [1, 128]),
        ],
        outputs: [
          TensorSpec(
              name: 'hidden_states', dtype: 'float32', shape: [1, 128, 64]),
        ],
        profiles: [
          ModelProfile(
            profileId: 'coreml_ane_128',
            targetBackend: 'coreml_ane',
            precision: 'fp32',
            batchSize: 1,
            maxSequenceLength: 128,
            verified: true,
            notes: 'Verified on Apple Silicon M-series ANE',
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
      expect(restored.profiles[0].verified, isTrue);
      expect(restored.profiles[1].verified, isFalse);
    });
  });
}
