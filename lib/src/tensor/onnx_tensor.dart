import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:meta/meta.dart';
import 'tensor_type.dart';

/// Represents a multidimensional tensor for ONNX inference.
class OnnxTensor {
  final List<int> shape;
  final TensorType type;
  final int elementCount;
  final int byteLength;

  Uint8List _data;
  ffi.Pointer<ffi.Void>? _nativeHandle;
  bool _isDisposed = false;

  OnnxTensor._({
    required this.shape,
    required this.type,
    required this.elementCount,
    required this.byteLength,
    required Uint8List data,
    ffi.Pointer<ffi.Void>? nativeHandle,
  })  : _data = data,
        _nativeHandle = nativeHandle;

  /// Creates a tensor from float32 data.
  factory OnnxTensor.fromFloat32(List<int> shape, Float32List data) {
    _validateShapeAndLength(shape, data.length);
    final byteData =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    return OnnxTensor._(
      shape: List.unmodifiable(shape),
      type: TensorType.float32,
      elementCount: data.length,
      byteLength: byteData.lengthInBytes,
      data: Uint8List.fromList(byteData),
    );
  }

  /// Creates a tensor from float16 raw bytes.
  factory OnnxTensor.fromFloat16Bytes(List<int> shape, Uint8List bytes) {
    final expectedCount = _calculateElementCount(shape);
    if (bytes.lengthInBytes != expectedCount * 2) {
      throw ArgumentError(
        'Expected ${expectedCount * 2} bytes for float16 tensor, got ${bytes.lengthInBytes}',
      );
    }
    return OnnxTensor._(
      shape: List.unmodifiable(shape),
      type: TensorType.float16,
      elementCount: expectedCount,
      byteLength: bytes.lengthInBytes,
      data: Uint8List.fromList(bytes),
    );
  }

  /// Creates a tensor from int64 data.
  factory OnnxTensor.fromInt64(List<int> shape, Int64List data) {
    _validateShapeAndLength(shape, data.length);
    final byteData =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    return OnnxTensor._(
      shape: List.unmodifiable(shape),
      type: TensorType.int64,
      elementCount: data.length,
      byteLength: byteData.lengthInBytes,
      data: Uint8List.fromList(byteData),
    );
  }

  /// Creates a tensor from int32 data.
  factory OnnxTensor.fromInt32(List<int> shape, Int32List data) {
    _validateShapeAndLength(shape, data.length);
    final byteData =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    return OnnxTensor._(
      shape: List.unmodifiable(shape),
      type: TensorType.int32,
      elementCount: data.length,
      byteLength: byteData.lengthInBytes,
      data: Uint8List.fromList(byteData),
    );
  }

  /// Creates a tensor from uint8 data.
  factory OnnxTensor.fromUint8(List<int> shape, Uint8List data) {
    _validateShapeAndLength(shape, data.length);
    return OnnxTensor._(
      shape: List.unmodifiable(shape),
      type: TensorType.uint8,
      elementCount: data.length,
      byteLength: data.lengthInBytes,
      data: Uint8List.fromList(data),
    );
  }

  /// Creates a tensor from int8 data.
  factory OnnxTensor.fromInt8(List<int> shape, Int8List data) {
    _validateShapeAndLength(shape, data.length);
    final byteData =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    return OnnxTensor._(
      shape: List.unmodifiable(shape),
      type: TensorType.int8,
      elementCount: data.length,
      byteLength: byteData.lengthInBytes,
      data: Uint8List.fromList(byteData),
    );
  }

  /// Creates a tensor from boolean values.
  factory OnnxTensor.fromBool(List<int> shape, List<bool> data) {
    _validateShapeAndLength(shape, data.length);
    final bytes = Uint8List(data.length);
    for (var i = 0; i < data.length; i++) {
      bytes[i] = data[i] ? 1 : 0;
    }
    return OnnxTensor._(
      shape: List.unmodifiable(shape),
      type: TensorType.boolType,
      elementCount: data.length,
      byteLength: bytes.length,
      data: bytes,
    );
  }

  /// Internal factory for native outputs.
  @internal
  factory OnnxTensor.fromNative({
    required List<int> shape,
    required TensorType type,
    required Uint8List rawBytes,
    ffi.Pointer<ffi.Void>? nativeHandle,
  }) {
    final count = _calculateElementCount(shape);
    return OnnxTensor._(
      shape: List.unmodifiable(shape),
      type: type,
      elementCount: count,
      byteLength: rawBytes.lengthInBytes,
      data: rawBytes,
      nativeHandle: nativeHandle,
    );
  }

  static int _calculateElementCount(List<int> shape) {
    if (shape.isEmpty) return 0;
    var count = 1;
    for (final dim in shape) {
      if (dim <= 0) {
        throw ArgumentError('Invalid shape dimension: $dim in shape $shape');
      }
      count *= dim;
    }
    return count;
  }

  static void _validateShapeAndLength(List<int> shape, int actualCount) {
    final expected = _calculateElementCount(shape);
    if (expected != actualCount) {
      throw ArgumentError(
        'Shape $shape expects $expected elements, but received data with $actualCount elements.',
      );
    }
  }

  bool get isDisposed => _isDisposed;

  void _checkDisposed() {
    if (_isDisposed) {
      throw StateError('Cannot access a disposed OnnxTensor.');
    }
  }

  /// Read data as Float32List.
  Float32List asFloat32List() {
    _checkDisposed();
    if (type != TensorType.float32) {
      throw StateError('Tensor is of type $type, cannot view as Float32List');
    }
    return _data.buffer.asFloat32List(_data.offsetInBytes, elementCount);
  }

  /// Read data as Int64List.
  Int64List asInt64List() {
    _checkDisposed();
    if (type != TensorType.int64) {
      throw StateError('Tensor is of type $type, cannot view as Int64List');
    }
    return _data.buffer.asInt64List(_data.offsetInBytes, elementCount);
  }

  /// Read data as Int32List.
  Int32List asInt32List() {
    _checkDisposed();
    if (type != TensorType.int32) {
      throw StateError('Tensor is of type $type, cannot view as Int32List');
    }
    return _data.buffer.asInt32List(_data.offsetInBytes, elementCount);
  }

  /// Read data as Uint8List.
  Uint8List asUint8List() {
    _checkDisposed();
    if (type != TensorType.uint8 && type != TensorType.boolType) {
      throw StateError('Tensor is of type $type, cannot view as Uint8List');
    }
    return _data;
  }

  /// Get raw bytes.
  Uint8List get rawBytes {
    _checkDisposed();
    return _data;
  }

  @internal
  ffi.Pointer<ffi.Void>? get nativeHandle => _nativeHandle;

  @internal
  void setNativeHandle(ffi.Pointer<ffi.Void> handle) {
    _nativeHandle = handle;
  }

  /// Explicitly releases resources associated with this tensor.
  void dispose() {
    if (!_isDisposed) {
      _isDisposed = true;
      _data = Uint8List(0);
      _nativeHandle = null;
    }
  }

  @override
  String toString() {
    return 'OnnxTensor(type: ${type.name}, shape: $shape, elements: $elementCount)';
  }
}
