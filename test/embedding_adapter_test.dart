import 'dart:math' as math;
import 'dart:typed_data';
import 'package:lib_onnx/src/adapter/embedding_adapter.dart';
import 'package:lib_onnx/src/tensor/onnx_tensor.dart';
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

    test(
      'Last-token pooling extracts the last non-padding token per sequence',
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
        final maskTensor = OnnxTensor.fromInt64([
          2,
          3,
        ], Int64List.fromList([1, 1, 0, 1, 1, 1]));

        const adapter = EmbeddingAdapter(
          hiddenDim: 2,
          hasEmbeddedPooling: false,
        );
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
      },
    );

    test('Query formatting follows Qwen instruction convention', () {
      final formatted = EmbeddingAdapter.formatQuery(
        'Retrieve relevant documents',
        'What is lib_onnx?',
      );
      expect(
        formatted,
        equals(
          'Instruct: Retrieve relevant documents\nQuery: What is lib_onnx?',
        ),
      );
    });

    test('rejects a hidden dimension that differs from the profile', () {
      final output = OnnxTensor.fromFloat32([
        1,
        2,
      ], Float32List.fromList([3, 4]));
      expect(
        () => const EmbeddingAdapter(
          hiddenDim: 3,
          hasEmbeddedPooling: true,
        ).extractEmbeddings(modelOutput: output),
        throwsArgumentError,
      );
    });

    test('embedded pooling requires rank two', () {
      final output = OnnxTensor.fromFloat32([
        1,
        1,
        2,
      ], Float32List.fromList([3, 4]));
      expect(
        () => const EmbeddingAdapter(
          hiddenDim: 2,
          hasEmbeddedPooling: true,
        ).extractEmbeddings(modelOutput: output),
        throwsArgumentError,
      );
    });

    test('token pooling requires a correctly shaped binary mask', () {
      final output = OnnxTensor.fromFloat32([
        1,
        2,
        2,
      ], Float32List.fromList([3, 4, 4, 3]));
      const adapter = EmbeddingAdapter(hiddenDim: 2);
      for (final mask in [
        null,
        OnnxTensor.fromInt64([2], Int64List.fromList([1, 1])),
        OnnxTensor.fromInt64([1, 2], Int64List.fromList([1, 2])),
        OnnxTensor.fromFloat32([1, 2], Float32List.fromList([1, 1])),
      ]) {
        expect(
          () => adapter.extractEmbeddings(
            modelOutput: output,
            attentionMask: mask,
          ),
          throwsArgumentError,
        );
      }
    });

    test('rejects all-padding rows instead of pooling token zero', () {
      final output = OnnxTensor.fromFloat32([
        1,
        2,
        2,
      ], Float32List.fromList([3, 4, 4, 3]));
      final mask = OnnxTensor.fromInt64([1, 2], Int64List(2));
      expect(
        () => const EmbeddingAdapter(
          hiddenDim: 2,
        ).extractEmbeddings(modelOutput: output, attentionMask: mask),
        throwsArgumentError,
      );
    });

    test('last valid token works for left and right padding', () {
      final output = OnnxTensor.fromFloat32([
        2,
        3,
        2,
      ], Float32List.fromList([9, 9, 3, 4, 4, 3, 3, 4, 4, 3, 9, 9]));
      final mask = OnnxTensor.fromBool(
        [2, 3],
        [false, true, true, true, true, false],
      );
      final vectors = const EmbeddingAdapter(
        hiddenDim: 2,
      ).extractEmbeddings(modelOutput: output, attentionMask: mask);
      expect(vectors[0], orderedEquals(vectors[1]));
      expect(vectors[0][0], closeTo(.8, 1e-6));
      expect(vectors[0][1], closeTo(.6, 1e-6));
    });

    test('fp16 outputs are converted rather than read as float32', () {
      // IEEE binary16: 3.0 = 0x4200, 4.0 = 0x4400 (little endian).
      final output = OnnxTensor.fromFloat16Bytes([
        1,
        2,
      ], Uint8List.fromList([0, 0x42, 0, 0x44]));
      final vectors = const EmbeddingAdapter(
        hiddenDim: 2,
        hasEmbeddedPooling: true,
      ).extractEmbeddings(modelOutput: output);
      expect(vectors[0][0], closeTo(.6, 1e-6));
      expect(vectors[0][1], closeTo(.8, 1e-6));
    });

    test('rejects zero norm and nonfinite values', () {
      for (final data in [
        [0.0, 0.0],
        [double.nan, 1.0],
        [double.infinity, 1.0],
        [double.negativeInfinity, 1.0],
      ]) {
        expect(
          () => EmbeddingAdapter.normalizeL2(Float32List.fromList(data)),
          throwsArgumentError,
        );
      }
    });

    test('normalization remains stable for very small finite vectors', () {
      final vector = Float32List.fromList([3e-30, 4e-30]);
      EmbeddingAdapter.normalizeL2(vector);
      expect(vector[0], closeTo(.6, 1e-6));
      expect(vector[1], closeTo(.8, 1e-6));
    });

    test('rejects nonfinite values outside the selected token as well', () {
      final output = OnnxTensor.fromFloat32([
        1,
        2,
        2,
      ], Float32List.fromList([double.nan, 0, 3, 4]));
      final mask = OnnxTensor.fromInt32([1, 2], Int32List.fromList([0, 1]));
      expect(
        () => const EmbeddingAdapter(
          hiddenDim: 2,
        ).extractEmbeddings(modelOutput: output, attentionMask: mask),
        throwsArgumentError,
      );
    });
  });
}
