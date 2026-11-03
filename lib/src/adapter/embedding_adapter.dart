import 'dart:math' as math;
import 'dart:typed_data';
import '../tensor/onnx_tensor.dart';

/// Adapter for text embedding models such as Qwen3-Embedding.
///
/// Implements last-token pooling (as required by Qwen3-Embedding) and L2 normalization.
/// Does not depend on Python or PyTorch.
class EmbeddingAdapter {
  final int hiddenDim;
  final bool hasEmbeddedPooling;

  const EmbeddingAdapter({
    this.hiddenDim = 1024,
    this.hasEmbeddedPooling = false,
  });

  /// Extracts normalized embedding vectors from the model output tensor.
  ///
  /// If [hasEmbeddedPooling] is true, [modelOutput] is expected to have shape [batch, hiddenDim].
  /// Otherwise, [modelOutput] is expected to have shape [batch, seqLen, hiddenDim],
  /// and [attentionMask] (shape [batch, seqLen]) is used to locate the last non-padding token.
  List<Float32List> extractEmbeddings({
    required OnnxTensor modelOutput,
    OnnxTensor? attentionMask,
  }) {
    final rawData = modelOutput.asFloat32List();
    final shape = modelOutput.shape;

    if (hasEmbeddedPooling || shape.length == 2) {
      // Shape is [batchSize, hiddenDim]
      final batchSize = shape[0];
      final dim = shape[1];
      final results = <Float32List>[];

      for (var b = 0; b < batchSize; b++) {
        final vec = Float32List(dim);
        final offset = b * dim;
        for (var i = 0; i < dim; i++) {
          vec[i] = rawData[offset + i];
        }
        normalizeL2(vec);
        results.add(vec);
      }
      return results;
    }

    if (shape.length != 3) {
      throw ArgumentError(
        'Expected modelOutput with shape [batch, seqLen, hiddenDim] (3D) or [batch, hiddenDim] (2D), got $shape',
      );
    }

    final batchSize = shape[0];
    final seqLen = shape[1];
    final dim = shape[2];

    final maskData = attentionMask != null
        ? _extractMask(attentionMask, batchSize, seqLen)
        : null;
    final results = <Float32List>[];

    for (var b = 0; b < batchSize; b++) {
      // Locate index of last valid token
      var lastTokenIdx = seqLen - 1;
      if (maskData != null) {
        lastTokenIdx = 0;
        for (var s = seqLen - 1; s >= 0; s--) {
          if (maskData[b][s] > 0) {
            lastTokenIdx = s;
            break;
          }
        }
      }

      // Extract hidden state for lastTokenIdx
      final vec = Float32List(dim);
      final offset = (b * seqLen + lastTokenIdx) * dim;
      for (var i = 0; i < dim; i++) {
        vec[i] = rawData[offset + i];
      }

      normalizeL2(vec);
      results.add(vec);
    }

    return results;
  }

  static List<List<int>> _extractMask(
      OnnxTensor maskTensor, int batch, int seqLen) {
    final list = <List<int>>[];
    if (maskTensor.type.name == 'int64') {
      final data = maskTensor.asInt64List();
      for (var b = 0; b < batch; b++) {
        final row = <int>[];
        final offset = b * seqLen;
        for (var s = 0; s < seqLen; s++) {
          row.add(data[offset + s]);
        }
        list.add(row);
      }
    } else if (maskTensor.type.name == 'int32') {
      final data = maskTensor.asInt32List();
      for (var b = 0; b < batch; b++) {
        final row = <int>[];
        final offset = b * seqLen;
        for (var s = 0; s < seqLen; s++) {
          row.add(data[offset + s]);
        }
        list.add(row);
      }
    } else {
      final data = maskTensor.asUint8List();
      for (var b = 0; b < batch; b++) {
        final row = <int>[];
        final offset = b * seqLen;
        for (var s = 0; s < seqLen; s++) {
          row.add(data[offset + s]);
        }
        list.add(row);
      }
    }
    return list;
  }

  /// In-place L2 normalization: v = v / ||v||_2
  static void normalizeL2(Float32List vector) {
    var sumSq = 0.0;
    for (var i = 0; i < vector.length; i++) {
      sumSq += vector[i] * vector[i];
    }
    final norm = math.sqrt(sumSq);
    if (norm > 1e-12) {
      for (var i = 0; i < vector.length; i++) {
        vector[i] /= norm;
      }
    }
  }

  /// Calculates cosine similarity between two normalized vectors.
  static double cosineSimilarity(Float32List a, Float32List b) {
    if (a.length != b.length) {
      throw ArgumentError(
          'Vector dimensions do not match: ${a.length} vs ${b.length}');
    }
    var dot = 0.0;
    for (var i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  }

  /// Formats a query with instruction per Qwen3-Embedding standard prompt.
  static String formatQuery(String instruction, String query) {
    if (instruction.isEmpty) return query;
    return 'Instruct: $instruction\nQuery: $query';
  }
}
