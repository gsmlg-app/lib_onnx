import 'dart:typed_data';
import 'package:lib_onnx/src/session/model_identity.dart';
import 'package:test/test.dart';

List<int> _v(int n) {
  final b = <int>[];
  while (n > 127) {
    b.add((n & 127) | 128);
    n >>= 7;
  }
  return [...b, n];
}

List<int> _f(int n, List<int> b) => [
  ..._v((n << 3) | 2),
  ..._v(b.length),
  ...b,
];
void main() {
  final external = _f(13, [
    ..._f(1, 'location'.codeUnits),
    ..._f(2, 'weights.data'.codeUnits),
  ]);
  test(
    'external data in local function node attributes must not be omitted',
    () {
      final attr = _f(5, external), node = _f(5, _f(5, external));
      final model = Uint8List.fromList(
        _f(25, [..._f(7, node), ..._f(11, attr)]),
      );
      expect(() => modelIdentity(model), throwsUnsupportedError);
    },
  );
  test(
    'external data in training initialization graph must not be omitted',
    () {
      final model = Uint8List.fromList(_f(20, _f(1, _f(5, external))));
      expect(() => modelIdentity(model), throwsUnsupportedError);
    },
  );
}
