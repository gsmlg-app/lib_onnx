import 'dart:math' as math;
import 'dart:typed_data';
import '../tensor/onnx_tensor.dart';
import '../tensor/tensor_type.dart';

/// Last-valid-token pooling and L2 normalization for text embedding models.
///
/// This adapter does not tokenize text or claim that a particular model/backend
/// has been validated. Use the actual model contract to choose [hiddenDim] and
/// [hasEmbeddedPooling].
class EmbeddingAdapter {
  final int hiddenDim;
  final bool hasEmbeddedPooling;

  const EmbeddingAdapter({
    this.hiddenDim = 1024,
    this.hasEmbeddedPooling = false,
  });

  /// Converts fp32 or IEEE binary16 output into normalized fp32 vectors.
  ///
  /// Embedded pooling requires `[batch, hiddenDim]`. Token pooling requires
  /// `[batch, sequence, hiddenDim]` and a binary mask `[batch, sequence]`.
  /// Both left and right padding are supported. An all-padding row is invalid.
  List<Float32List> extractEmbeddings({
    required OnnxTensor modelOutput,
    OnnxTensor? attentionMask,
  }) {
    final shape = modelOutput.shape;
    final expectedRank = hasEmbeddedPooling ? 2 : 3;
    if (hiddenDim <= 0 ||
        shape.length != expectedRank ||
        shape.any((dimension) => dimension <= 0) ||
        shape.last != hiddenDim) {
      throw ArgumentError(
        'Expected ${hasEmbeddedPooling ? '[batch, $hiddenDim]' : '[batch, sequence, $hiddenDim]'} '
        'with positive dimensions, got $shape (hiddenDim=$hiddenDim)',
      );
    }

    final data = _floatData(modelOutput);
    if (data.any((value) => !value.isFinite)) {
      throw ArgumentError('Model output contains NaN or infinity');
    }
    final batch = shape[0];
    final sequence = hasEmbeddedPooling ? 1 : shape[1];
    final List<int>? mask;
    if (hasEmbeddedPooling) {
      if (attentionMask != null) {
        throw ArgumentError(
          'An embedded-pooling output does not accept a mask',
        );
      }
      mask = null;
    } else {
      if (attentionMask == null) {
        throw ArgumentError('Last-token pooling requires an attention mask');
      }
      mask = _maskData(attentionMask, batch, sequence);
    }

    final results = <Float32List>[];
    for (var b = 0; b < batch; b++) {
      var token = sequence - 1;
      if (mask != null) {
        while (token >= 0 && mask[b * sequence + token] == 0) {
          token--;
        }
        if (token < 0) {
          throw ArgumentError('Attention mask row $b contains no valid token');
        }
      }
      final offset = (b * sequence + token) * hiddenDim;
      final vector = Float32List.fromList(
        data.sublist(offset, offset + hiddenDim),
      );
      normalizeL2(vector);
      results.add(vector);
    }
    return results;
  }

  static Float32List _floatData(OnnxTensor tensor) {
    if (tensor.type == TensorType.float32) return tensor.asFloat32List();
    if (tensor.type != TensorType.float16) {
      throw ArgumentError('Embedding output must be float32 or float16');
    }
    final bytes = tensor.rawBytes;
    final data = ByteData.sublistView(bytes);
    final result = Float32List(tensor.elementCount);
    for (var i = 0; i < result.length; i++) {
      final bits = data.getUint16(i * 2, Endian.little);
      final sign = (bits & 0x8000) == 0 ? 1.0 : -1.0;
      final exponent = (bits >> 10) & 0x1f;
      final fraction = bits & 0x3ff;
      if (exponent == 0) {
        result[i] = sign * fraction * math.pow(2, -24);
      } else if (exponent == 31) {
        result[i] = fraction == 0 ? sign * double.infinity : double.nan;
      } else {
        result[i] = sign * (1 + fraction / 1024) * math.pow(2, exponent - 15);
      }
    }
    return result;
  }

  static List<int> _maskData(OnnxTensor tensor, int batch, int sequence) {
    if (tensor.shape.length != 2 ||
        tensor.shape[0] != batch ||
        tensor.shape[1] != sequence) {
      throw ArgumentError(
        'Expected attention mask [$batch, $sequence], got ${tensor.shape}',
      );
    }
    final List<int> data;
    switch (tensor.type) {
      case TensorType.int64:
        data = tensor.asInt64List();
      case TensorType.int32:
        data = tensor.asInt32List();
      case TensorType.uint8:
      case TensorType.boolType:
        data = tensor.asUint8List();
      default:
        throw ArgumentError(
          'Attention mask must be int64, int32, uint8 or bool',
        );
    }
    if (data.any((value) => value != 0 && value != 1)) {
      throw ArgumentError('Attention mask values must be zero or one');
    }
    return data;
  }

  /// Normalizes a finite, nonzero vector in place, using scaling to avoid
  /// overflow/underflow. Zero-norm, empty, NaN and infinite vectors are invalid.
  static void normalizeL2(Float32List vector) {
    if (vector.isEmpty || vector.any((value) => !value.isFinite)) {
      throw ArgumentError('Expected a nonempty finite embedding vector');
    }
    var scale = 0.0;
    for (final value in vector) {
      scale = math.max(scale, value.abs());
    }
    if (scale == 0) throw ArgumentError('Embedding vector has zero norm');
    var sum = 0.0;
    for (final value in vector) {
      final scaled = value / scale;
      sum += scaled * scaled;
    }
    final scaledNorm = math.sqrt(sum);
    for (var i = 0; i < vector.length; i++) {
      vector[i] = (vector[i] / scale) / scaledNorm;
    }
  }

  /// Dot product of two normalized vectors (their cosine similarity).
  static double cosineSimilarity(Float32List a, Float32List b) {
    if (a.isEmpty || a.length != b.length) {
      throw ArgumentError('Expected matching, nonempty vector dimensions');
    }
    var dot = 0.0;
    for (var i = 0; i < a.length; i++) {
      if (!a[i].isFinite || !b[i].isFinite) {
        throw ArgumentError('Cosine similarity requires finite vectors');
      }
      dot += a[i] * b[i];
    }
    return dot;
  }

  /// Qwen3-Embedding instruction format. Documents normally have no instruction.
  static String formatQuery(String instruction, String query) =>
      instruction.isEmpty ? query : 'Instruct: $instruction\nQuery: $query';
}
