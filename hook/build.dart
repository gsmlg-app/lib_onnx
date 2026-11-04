import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:code_assets/code_assets.dart';
import 'package:crypto/crypto.dart';
import 'package:hooks/hooks.dart';
import 'package:native_toolchain_c/native_toolchain_c.dart';

const _assetName = 'src/ffi/native_library.dart';

void main(List<String> arguments) async {
  await build(arguments, (input, output) async {
    if (!input.config.buildCodeAssets) return;
    final code = input.config.code;
    final target = '${code.targetOS.name}_${code.targetArchitecture.name}';
    final bundle = input.userDefines.path('native_bundle');
    if (bundle != null) {
      await _useBundle(input, output, bundle, target);
      return;
    }
    final lockFile = File.fromUri(
      input.packageRoot.resolve('native/ort_artifacts.json'),
    );
    final lock =
        jsonDecode(await lockFile.readAsString()) as Map<String, dynamic>;
    output.dependencies.add(lockFile.uri);
    final artifact =
        (lock['targets'] as Map<String, dynamic>)[target]
            as Map<String, dynamic>?;
    if (artifact == null) {
      final details = code.targetOS == OS.iOS
          ? 'iOS ${code.iOS.targetSdk}: supply a separately built device/simulator '
                'ORT CoreML bundle; a macOS dylib is never linked into iOS.'
          : code.targetOS == OS.android
          ? 'Android requires an arm64-v8a NDK build, ORT QNN HTP and legally '
                'obtained matching Qualcomm runtime libraries, all verified for 16 KB pages.'
          : 'No pinned native artifact exists for $target.';
      throw UnsupportedError(
        '$details See docs/distribution.md; configure '
        'hooks.user_defines.lib_onnx.native_bundle for an explicitly verified bundle.',
      );
    }
    final archiveName = artifact['archive'] as String;
    final explicitArchive = input.userDefines.path('ort_archive');
    final cache =
        input.userDefines.path('ort_cache') ??
        input.outputDirectoryShared.resolve('ort-cache/');
    await Directory.fromUri(cache).create(recursive: true);
    final archive = File.fromUri(explicitArchive ?? cache.resolve(archiveName));
    if (!await archive.exists()) {
      if (explicitArchive != null || input.userDefines['offline'] == true) {
        throw StateError(
          'Pinned ORT archive missing: ${archive.path}. '
          'Provide ort_archive or populate ort_cache; offline mode never downloads.',
        );
      }
      final url = Uri.parse(
        'https://github.com/microsoft/onnxruntime/releases/'
        'download/v${lock['ort_version']}/$archiveName',
      );
      final client = HttpClient()
        ..connectionTimeout = const Duration(seconds: 30)
        ..findProxy = HttpClient.findProxyFromEnvironment;
      final temporary = File('${archive.path}.download');
      try {
        final response = await (await client.getUrl(url)).close();
        if (response.statusCode != 200) {
          throw HttpException(
            'ORT download returned HTTP ${response.statusCode}',
            uri: url,
          );
        }
        await response
            .pipe(temporary.openWrite())
            .timeout(const Duration(minutes: 10));
        await _verify(temporary, artifact['sha256'] as String);
        await temporary.rename(archive.path);
      } finally {
        client.close(force: true);
        if (await temporary.exists()) await temporary.delete();
      }
    }
    await _verify(archive, artifact['sha256'] as String);
    output.dependencies.add(archive.uri);
    final unpack = input.outputDirectory.resolve('ort/');
    await Directory.fromUri(unpack).create(recursive: true);
    await _run('tar', ['-xzf', archive.path, '-C', unpack.toFilePath()]);
    final root = unpack.resolve('${artifact['root']}/');
    final libraryName = artifact['library'] as String;
    final library = await File.fromUri(
      root.resolve('lib/$libraryName'),
    ).copy(input.outputDirectory.resolve(libraryName).toFilePath());
    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'onnxruntime',
        linkMode: DynamicLoadingBundled(),
        file: library.uri,
      ),
    );
    final isMac = code.targetOS == OS.macOS;
    await CBuilder.library(
      name: 'lib_onnx_bridge',
      assetName: _assetName,
      sources: ['native/src/lib_onnx_bridge.c', 'native/src/library_locator.c'],
      includes: ['native/include', 'native/third_party/onnxruntime/include'],
      flags: [
        '-pthread',
        '-Wall',
        '-Wextra',
        // The pinned official ORT artifact itself has LC_BUILD_VERSION minos14.
        // Dart JIT's generic hook target defaults12, so set the actual bridge
        // target explicitly and document the effective native minimum14.
        if (isMac) '--target=arm64-apple-macos14.0',
        library.path,
        if (isMac) '-Wl,-rpath,@loader_path',
        if (!isMac) '-Wl,-rpath,\$ORIGIN',
        if (!isMac) '-ldl',
      ],
      std: 'c11',
      linkModePreference: LinkModePreference.dynamic,
    ).run(input: input, output: output);
    for (final filename in ['LICENSE', 'ThirdPartyNotices.txt']) {
      final source = File.fromUri(root.resolve(filename));
      if (await source.exists()) {
        await source.copy(
          input.outputDirectory.resolve('onnxruntime-$filename').toFilePath(),
        );
      }
    }
    // This receipt describes produced files. Device acceptance is never inferred.
    final bridge = File.fromUri(
      input.outputDirectory.resolve(
        code.targetOS.dylibFileName('lib_onnx_bridge'),
      ),
    );
    await File.fromUri(
      input.outputDirectory.resolve('artifacts_manifest.json'),
    ).writeAsString(
      const JsonEncoder.withIndent('  ').convert({
        'manifest_version': 2,
        'target': target,
        'bridge_abi_version': 2,
        'ort_version': lock['ort_version'],
        'providers_expected_from_release_metadata': artifact['providers'],
        'providers_compiled': null,
        'provider_verification': 'not-run',
        'archive_sha256': artifact['sha256'],
        'license':
            'MIT; see onnxruntime-LICENSE and onnxruntime-ThirdPartyNotices.txt',
        'status': {
          'configured': 'passed',
          'compiled': 'passed',
          'packaged': 'not-run',
          'device': 'not-run',
        },
        'binaries': [
          for (final file in [bridge, library])
            {'name': file.uri.pathSegments.last, 'sha256': await _digest(file)},
        ],
      }),
    );
  });
}

Future<void> _useBundle(
  BuildInput input,
  BuildOutputBuilder output,
  Uri bundle,
  String target,
) async {
  final directory = Directory.fromUri(bundle);
  final manifestFile = File('${directory.path}/artifacts_manifest.json');
  final manifest =
      jsonDecode(await manifestFile.readAsString()) as Map<String, dynamic>;
  final sdk = input.config.code.targetOS == OS.iOS
      ? input.config.code.iOS.targetSdk.type
      : null;
  if (manifest['target'] != target ||
      manifest['bridge_abi_version'] != 2 ||
      manifest['ort_version'] != '1.31.1' ||
      (sdk != null && manifest['ios_sdk'] != sdk)) {
    throw StateError(
      'Native bundle target/SDK/ABI/ORT mismatch for $target ${sdk ?? ''}.',
    );
  }
  if (input.config.code.targetOS == OS.android &&
      manifest['page_size_check'] != 'passed') {
    throw StateError(
      'Android bundle needs a passing actual ELF 16 KB page audit.',
    );
  }
  output.dependencies.add(manifestFile.uri);
  var bridgeCount = 0;
  var hasRuntime = false;
  for (final binary in manifest['binaries'] as List<dynamic>) {
    final entry = binary as Map<String, dynamic>;
    final name = entry['name'] as String;
    if (name.contains('/') ||
        name.contains('\\') ||
        name == '.' ||
        name == '..') {
      throw FormatException('Bundle binary must be a file name: $name');
    }
    final file = File('${directory.path}/$name');
    await _verify(file, entry['sha256'] as String);
    output.dependencies.add(file.uri);
    final isBridge = entry['role'] == 'bridge';
    if (isBridge) bridgeCount++;
    hasRuntime |= entry['role'] == 'runtime';
    if (input.config.code.targetOS == OS.android) {
      await _verifyAndroidElf(file);
    }
    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: isBridge ? _assetName : name,
        linkMode: DynamicLoadingBundled(),
        file: file.uri,
      ),
    );
  }
  if (bridgeCount != 1 || !hasRuntime) {
    throw StateError(
      'Native bundle needs exactly one ABI2 bridge and its ORT runtime.',
    );
  }
}

Future<String> _digest(File file) async =>
    (await sha256.bind(file.openRead()).first).toString();

Future<void> _verify(File file, String expected) async {
  if (!RegExp(r'^[0-9a-f]{64}$').hasMatch(expected))
    throw FormatException('Invalid pinned SHA256: $expected');
  final actual = await _digest(file);
  if (actual != expected)
    throw StateError(
      'SHA256 mismatch for ${file.path}: expected $expected, found $actual.',
    );
}

Future<void> _run(String executable, List<String> arguments) async {
  final result = await Process.run(executable, arguments);
  if (result.exitCode != 0)
    throw ProcessException(
      executable,
      arguments,
      '${result.stdout}\n${result.stderr}',
      result.exitCode,
    );
}

// Check every supplied Android library, rather than trusting manifest claims.
Future<void> _verifyAndroidElf(File file) async {
  final bytes = await file.readAsBytes();
  final data = ByteData.sublistView(bytes);
  if (bytes.length < 64 ||
      bytes[0] != 0x7f ||
      bytes[1] != 0x45 ||
      bytes[2] != 0x4c ||
      bytes[3] != 0x46 ||
      bytes[4] != 2 ||
      bytes[5] != 1 ||
      data.getUint16(18, Endian.little) != 183) {
    throw StateError('${file.path}: expected little-endian arm64 ELF64.');
  }
  final offset = data.getUint64(32, Endian.little);
  final size = data.getUint16(54, Endian.little);
  final count = data.getUint16(56, Endian.little);
  if (size < 56 || count == 0 || offset + count * size > bytes.length) {
    throw StateError('${file.path}: invalid ELF program headers.');
  }
  var loads = 0;
  for (var i = 0; i < count; i++) {
    final pos = offset + size * i;
    if (data.getUint32(pos, Endian.little) != 1) continue;
    loads++;
    final fileOffset = data.getUint64(pos + 8, Endian.little);
    final address = data.getUint64(pos + 16, Endian.little);
    final alignment = data.getUint64(pos + 48, Endian.little);
    if (alignment < 16384 || (address - fileOffset) % 16384 != 0) {
      throw StateError('${file.path}: PT_LOAD[$i] is not 16 KB compatible.');
    }
  }
  if (loads == 0) throw StateError('${file.path}: no ELF PT_LOAD segments.');
}
