import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:path/path.dart' as p;

/// Minimal protobuf traversal of ONNX tensor locations, including nested graphs.
/// No tokenizer/model-package dependency enters the runtime library.
Iterable<(int, int, Object)> _fields(Uint8List data) sync* {
  var i = 0;
  int varint() {
    var value = 0, shift = 0;
    while (i < data.length && shift < 70) {
      final b = data[i++];
      value |= (b & 127) << shift;
      if (b < 128) return value;
      shift += 7;
    }
    throw FormatException('Malformed ONNX protobuf varint');
  }

  while (i < data.length) {
    final tag = varint(), field = tag >> 3, wire = tag & 7;
    if (field == 0) throw FormatException('Invalid ONNX field');
    if (wire == 0) {
      yield (field, wire, varint());
      continue;
    }
    final size = wire == 2
        ? varint()
        : wire == 1
        ? 8
        : wire == 5
        ? 4
        : -1;
    if (size < 0 || size > data.length - i)
      throw FormatException('Malformed ONNX field size');
    if (wire == 2)
      yield (field, wire, Uint8List.sublistView(data, i, i + size));
    i += size;
  }
}

Set<String> _externalLocations(Uint8List model) {
  final locations = <String>{};
  void visit(Uint8List bytes, String kind) {
    for (final (f, w, v) in _fields(bytes)) {
      if (w != 2) continue;
      final b = v as Uint8List;
      switch (kind) {
        case 'model':
          if (f == 7) visit(b, 'graph');
          if (f == 25) visit(b, 'function');
          if (f == 20) visit(b, 'training');
        case 'function':
          if (f == 7) visit(b, 'node');
          if (f == 11) visit(b, 'attribute');
        case 'training':
          if (f == 1 || f == 2) visit(b, 'graph');
        case 'graph':
          if (f == 5) visit(b, 'tensor');
          if (f == 1) visit(b, 'node');
          if (f == 15) visit(b, 'sparse');
        case 'node':
          if (f == 5) visit(b, 'attribute');
        case 'attribute':
          if (f == 5 || f == 10) visit(b, 'tensor');
          if (f == 6 || f == 11) visit(b, 'graph');
          if (f == 22 || f == 23) visit(b, 'sparse');
        case 'sparse':
          if (f == 1 || f == 2) visit(b, 'tensor');
        case 'tensor':
          if (f == 13) {
            String? key, value;
            for (final (ef, ew, ev) in _fields(b)) {
              if (ew == 2 && ef == 1) key = utf8.decode(ev as Uint8List);
              if (ew == 2 && ef == 2) value = utf8.decode(ev as Uint8List);
            }
            if (key == 'location' && value != null) locations.add(value);
          }
      }
    }
  }

  visit(model, 'model');
  return locations;
}

/// SHA256 over UTF8 JSON array of ordered {path,size,sha256} records.
/// The primary model record always uses "model.onnx"; paths use forward slashes.
String modelIdentity(Uint8List model, {String? modelPath}) {
  final records = <Map<String, Object>>[
    {
      'path': 'model.onnx',
      'size': model.length,
      'sha256': sha256.convert(model).toString(),
    },
  ];
  final locations = _externalLocations(model).toList()..sort();
  for (final location in locations) {
    if (modelPath == null)
      throw UnsupportedError(
        'External tensor data requires loadFile and colocated files',
      );
    final normalized = p.posix.normalize(location.replaceAll('\\', '/'));
    if (p.posix.isAbsolute(normalized) ||
        normalized == '..' ||
        normalized.startsWith('../') ||
        normalized.contains(':')) {
      throw FormatException(
        'External tensor data path must stay within model directory: $location',
      );
    }
    final dir = Directory(p.dirname(modelPath)).resolveSymbolicLinksSync();
    final file = File(p.join(dir, normalized));
    final resolved = file.resolveSymbolicLinksSync();
    if (!p.isWithin(dir, resolved))
      throw FormatException('External data symlink escapes model directory');
    final data = file.readAsBytesSync();
    records.add({
      'path': normalized,
      'size': data.length,
      'sha256': sha256.convert(data).toString(),
    });
  }
  return sha256.convert(utf8.encode(jsonEncode(records))).toString();
}
