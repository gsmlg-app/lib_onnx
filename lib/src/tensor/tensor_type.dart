/// Supported ONNX tensor types.
///
/// In accordance with lib_onnx specification:
/// Supports: float32, float16, int64, int32, int8, uint8, bool.
/// Explicitly rejects: string, sequence, map, sparse, bfloat16.
enum TensorType {
  float32(1, 'float32', 4),
  float16(2, 'float16', 2),
  int64(3, 'int64', 8),
  int32(4, 'int32', 4),
  int8(5, 'int8', 1),
  uint8(6, 'uint8', 1),
  boolType(7, 'bool', 1);

  final int nativeCode;
  final String name;
  final int elementSizeInBytes;

  const TensorType(this.nativeCode, this.name, this.elementSizeInBytes);

  static TensorType fromNativeCode(int code) {
    for (final type in TensorType.values) {
      if (type.nativeCode == code) {
        return type;
      }
    }
    throw UnsupportedError(
        'Unsupported or unrecognized ONNX native data type code: $code');
  }

  static TensorType fromName(String name) {
    final lower = name.toLowerCase();
    switch (lower) {
      case 'float32':
      case 'float':
        return TensorType.float32;
      case 'float16':
      case 'half':
        return TensorType.float16;
      case 'int64':
      case 'long':
        return TensorType.int64;
      case 'int32':
      case 'int':
        return TensorType.int32;
      case 'int8':
        return TensorType.int8;
      case 'uint8':
      case 'byte':
        return TensorType.uint8;
      case 'bool':
      case 'boolean':
        return TensorType.boolType;
      case 'string':
      case 'str':
      case 'sequence':
      case 'map':
      case 'sparse_tensor':
      case 'sparse':
      case 'bfloat16':
      case 'bf16':
      case 'complex64':
      case 'complex128':
        throw UnsupportedError(
          'ONNX type "$name" is intentionally unsupported in lib_onnx v0.1. '
          'Only numerical tensors (float32, float16, int64, int32, int8, uint8, bool) are supported.',
        );
      default:
        throw UnsupportedError(
            'Unknown or unsupported ONNX tensor data type: "$name"');
    }
  }
}
