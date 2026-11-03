import 'dart:typed_data';
import 'package:lib_onnx/lib_onnx.dart';
import 'package:test/test.dart';

void main() {
  group('OnnxTensor & TensorType', () {
    test('Float32 tensor creation and data view', () {
      final data = Float32List.fromList([1.0, 2.5, -3.0, 4.25]);
      final tensor = OnnxTensor.fromFloat32([2, 2], data);

      expect(tensor.shape, equals([2, 2]));
      expect(tensor.type, equals(TensorType.float32));
      expect(tensor.elementCount, equals(4));
      expect(tensor.byteLength, equals(16));
      expect(tensor.asFloat32List(), equals(data));
      expect(tensor.isDisposed, isFalse);

      tensor.dispose();
      expect(tensor.isDisposed, isTrue);
      expect(() => tensor.asFloat32List(), throwsStateError);
    });

    test('Int64 tensor creation and data view', () {
      final data = Int64List.fromList([100, 200, 300]);
      final tensor = OnnxTensor.fromInt64([1, 3], data);

      expect(tensor.shape, equals([1, 3]));
      expect(tensor.type, equals(TensorType.int64));
      expect(tensor.elementCount, equals(3));
      expect(tensor.byteLength, equals(24));
      expect(tensor.asInt64List(), equals(data));
    });

    test('Bool tensor creation and raw byte conversion', () {
      final tensor = OnnxTensor.fromBool([2, 2], [true, false, false, true]);
      expect(tensor.shape, equals([2, 2]));
      expect(tensor.type, equals(TensorType.boolType));
      expect(tensor.elementCount, equals(4));
      expect(tensor.asUint8List(), equals(Uint8List.fromList([1, 0, 0, 1])));
    });

    test('Shape mismatch throws ArgumentError', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0]);
      expect(
        () => OnnxTensor.fromFloat32([2, 2], data),
        throwsArgumentError,
      );
    });

    test('Type validation explicitly rejects unsupported types per spec', () {
      expect(() => TensorType.fromName('string'), throwsUnsupportedError);
      expect(() => TensorType.fromName('sequence'), throwsUnsupportedError);
      expect(() => TensorType.fromName('map'), throwsUnsupportedError);
      expect(() => TensorType.fromName('sparse'), throwsUnsupportedError);
      expect(() => TensorType.fromName('bfloat16'), throwsUnsupportedError);
    });

    test('Type validation accepts supported types', () {
      expect(TensorType.fromName('float32'), equals(TensorType.float32));
      expect(TensorType.fromName('float16'), equals(TensorType.float16));
      expect(TensorType.fromName('int64'), equals(TensorType.int64));
      expect(TensorType.fromName('int32'), equals(TensorType.int32));
      expect(TensorType.fromName('int8'), equals(TensorType.int8));
      expect(TensorType.fromName('uint8'), equals(TensorType.uint8));
      expect(TensorType.fromName('bool'), equals(TensorType.boolType));
    });
  });
}
