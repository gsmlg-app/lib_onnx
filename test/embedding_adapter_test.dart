import 'dart:math' as math;
import 'dart:typed_data';
import 'package:lib_onnx/lib_onnx.dart';
import 'package:test/test.dart';

void main() {
  group('EmbeddingAdapter', () {
    test('L2 Normalization produces unit vectors', () {
      final vec = Float32List.fromList([3.0, 4.0]);
      EmbeddingAdapter.normalizeL2(vec);

      expect(vec[0], closeTo(0.6, 1e-6));
      expect(vec[1], closeTo(0.8, 1e-6));

      final norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
      expect(norm, closeTo(1.0, 1e-6));
    });

    test('Cosine similarity of identical and orthogonal vectors', () {
      final a = Float32List.fromList([1.0, 0.0]);
      final b = Float32List.fromList([1.0, 0.0]);
      final c = Float32List.fromList([0.0, 1.0]);

      expect(EmbeddingAdapter.cosineSimilarity(a, b), closeTo(1.0, 1e-6));
      expect(EmbeddingAdapter.cosineSimilarity(a, c), closeTo(0.0, 1e-6));
    });

    test('Last-token pooling extracts the last non-padding token per sequence',
        () {
      // Shape: [batch=2, seq_len=3, dim=2]
      // Batch 0: tokens 0, 1 are valid, token 2 is padding (mask: [1, 1, 0])
      // Batch 1: all 3 tokens are valid (mask: [1, 1, 1])
      final hiddenStates = Float32List.fromList([
        // Batch 0
        1.0, 0.0, // Token 0
        2.0, 0.0, // Token 1 (last valid token for batch 0!)
        9.0, 9.0, // Token 2 (padding)
        // Batch 1
        1.0, 1.0, // Token 0
        2.0, 2.0, // Token 1
        0.0, 5.0, // Token 2 (last valid token for batch 1!)
      ]);

      final outTensor = OnnxTensor.fromFloat32([2, 3, 2], hiddenStates);
      final maskTensor =
          OnnxTensor.fromInt64([2, 3], Int64List.fromList([1, 1, 0, 1, 1, 1]));

      const adapter = EmbeddingAdapter(hiddenDim: 2, hasEmbeddedPooling: false);
      final embeddings = adapter.extractEmbeddings(
        modelOutput: outTensor,
        attentionMask: maskTensor,
      );

      expect(embeddings.length, equals(2));

      // Batch 0 should be normalized [2.0, 0.0] -> [1.0, 0.0]
      expect(embeddings[0][0], closeTo(1.0, 1e-6));
      expect(embeddings[0][1], closeTo(0.0, 1e-6));

      // Batch 1 should be normalized [0.0, 5.0] -> [0.0, 1.0]
      expect(embeddings[1][0], closeTo(0.0, 1e-6));
      expect(embeddings[1][1], closeTo(1.0, 1e-6));
    });

    test('Query formatting follows Qwen instruction convention', () {
      final formatted = EmbeddingAdapter.formatQuery(
          'Retrieve relevant documents', 'What is lib_onnx?');
      expect(
          formatted,
          equals(
              'Instruct: Retrieve relevant documents\nQuery: What is lib_onnx?'));
    });
  });
}
