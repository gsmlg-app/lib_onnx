#!/usr/bin/env python3
"""Exercise the real pub archive through a local hosted-package repository.

The consumer lives outside this checkout, has a fresh pub cache, and uses no path
reference or pre-copied native library. The SDK's native assets build and bundle
ORT. Only the small ONNX fixture is embedded; this is not Qwen/hardware proof.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import threading
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def pubspec_metadata(text):
    # Read the scalar fields used by pub's solver. The actual archived pubspec is
    # installed intact; this metadata parser deliberately rejects nested deps.
    result, section = {}, None
    for line in text.splitlines():
        if not line or line.lstrip().startswith('#'):
            continue
        if not line.startswith(' '):
            key, _, value = line.partition(':')
            section = key if not value.strip() else None
            if value.strip():
                result[key] = value.strip().strip("'\"")
            elif key in ('dependencies', 'environment'):
                result[key] = {}
        elif section in ('dependencies', 'environment') and line.startswith('  ') and not line.startswith('    '):
            key, _, value = line.strip().partition(':')
            if not value.strip():
                raise ValueError('Consumer server metadata only supports scalar package dependencies')
            result[section][key] = value.strip().strip("'\"")
    return {key: result[key] for key in ('name', 'version', 'description', 'environment', 'dependencies')}


def invoke(command, cwd, environment, log, timeout=600):
    log.write(f'\n$ {" ".join(map(str, command))}\n[cwd: {cwd}]\n')
    log.flush()
    try:
        result = subprocess.run(list(map(str, command)), cwd=cwd, env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        captured = error.output or b''
        log.write(captured.decode(errors='replace') if isinstance(captured, bytes) else captured)
        log.write(f'\nTIMEOUT after {timeout}s; owned child terminated by subprocess.run\n')
        log.flush()
        raise
    log.write(result.stdout)
    log.write(f'\nexit_code={result.returncode}\n')
    log.flush()
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {command}; see {log.name}')
    return result.stdout


def dart_smoke(model):
    encoded = base64.b64encode(model).decode()
    return f'''import 'dart:convert';
import 'dart:typed_data';
import 'package:lib_onnx/lib_onnx.dart';

Future<Map<String, Object?>> smoke() async {{
  final runtime = await OnnxRuntime.createAsync();
  final session = await InferenceSession.loadBytes(
    base64Decode('{encoded}'), runtime: runtime,
    options: SessionOptions.cpuOnly(),
  );
  final input = OnnxTensor.fromFloat32([1, 4], Float32List.fromList([1, 2, 3, 4]));
  try {{
    final result = await session.runWithDiagnostics({{'X': input}});
    try {{
      final output = result.outputs['Y']!.asFloat32List().toList();
      if (output.toString() != '[1.0, 2.0, 3.0, 4.0]') throw StateError('Unexpected actual ORT output: $output');
      if (result.diagnostic.activeProvider != ExecutionProvider.cpu ||
          result.diagnostic.evidenceLevel != EvidenceLevel.none ||
          result.diagnostic.fallbackOccurred != false) throw StateError('CPU baseline diagnostics are inaccurate');
      return {{'status': 'passed', 'fixture': 'identity_float32', 'output': output,
        'modelHash': result.diagnostic.modelHash,
        'actualProviders': result.diagnostic.actualProviders.map((p) => p.ortName).toList(),
        'hardwareObserved': result.diagnostic.evidenceLevel == EvidenceLevel.hardwareObserved,
        'bridge': OnnxRuntime.bridgeVersion, 'ort': OnnxRuntime.ortVersion,
        'compiledProviders': runtime.getAvailableProviders(),
        'diagnostic': result.diagnostic.toString(),
        'hardware_acceptance': 'not-run', 'qwen_acceptance': 'not-run'}};
    }} finally {{
      for (final tensor in result.outputs.values) {{ tensor.dispose(); }}
    }}
  }} finally {{
    input.dispose();
    await session.close();
    await runtime.close();
  }}
}}
'''


MOBILE_MAIN = """import 'dart:io';
import 'package:flutter/material.dart';
void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MaterialApp(home: Scaffold(body: Text('Real ORT consumer smoke'))));
  WidgetsBinding.instance.addPostFrameCallback((_) async {
    Map<String, Object?> result;
    try {
      result = await smoke();
    } catch (error, stack) {
      result = {'status': 'failed', 'error': '$error', 'stack': '$stack'};
    }
    // Android routes print() to logcat; iOS release consoles capture stdout.
    final line = 'LIB_ONNX_CONSUMER ${jsonEncode(result)}';
    print(line);
    stdout.writeln(line);
    await stdout.flush();
  });
}
"""


def mobile_consumer(platform, device, team, work, output, dependency, configuration, smoke,
                    environment, unrelated, log):
    """Build the archive consumer for a mobile device, install it, and read its result."""
    import time
    app = work / f'{platform}-consumer'
    invoke(['flutter', 'create', f'--platforms={platform}', '--org=dev.libonnx',
            '--project-name=archive_flutter_consumer', '--empty', app], work, environment, log)
    (app / 'pubspec.yaml').write_text(
        "name: archive_flutter_consumer\npublish_to: none\nenvironment:\n  sdk: '>=3.10.0 <4.0.0'\n"
        "dependencies:\n  flutter:\n    sdk: flutter\n" + dependency + configuration +
        '\nflutter:\n  uses-material-design: true\n')
    # Merge the smoke's and the app's imports; Dart requires them before declarations.
    lines = (smoke + '\n' + MOBILE_MAIN).splitlines()
    imports = dict.fromkeys(line for line in lines if line.startswith('import '))
    body = [line for line in lines if not line.startswith('import ')]
    (app / 'lib/main.dart').write_text('\n'.join([*imports, *body]) + '\n')
    invoke(['flutter', 'pub', 'get'], app, environment, log)
    marker = 'LIB_ONNX_CONSUMER '
    if platform == 'android':
        invoke(['flutter', 'build', 'apk', '--release'], app, environment, log, timeout=1800)
        apk = app / 'build/app/outputs/flutter-apk/app-release.apk'
        audit = load_tool('audit_android_pages')
        with zipfile.ZipFile(apk) as package:
            libraries = {name: package.read(name) for name in package.namelist()
                         if name.startswith('lib/arm64-v8a/') and name.endswith('.so')}
        pages = [audit.elf_check(data, name) for name, data in sorted(libraries.items())]
        log.write(f'\n[APK arm64-v8a ELF 16 KB audit]\n{json.dumps(pages, indent=2)}\n')
        bundled = [{'name': name, 'sha256': hashlib.sha256(data).hexdigest()}
                   for name, data in sorted(libraries.items()) if 'onnx' in name]
        gradle = next(app.glob('android/app/build.gradle*')).read_text()
        package_id = re.search(r'applicationId\s*=?\s*"([^"]+)"', gradle).group(1)
        adb = ['adb', '-s', device]
        invoke(adb + ['install', '-r', apk], unrelated, environment, log, timeout=600)
        invoke(adb + ['logcat', '-c'], unrelated, environment, log)
        invoke(adb + ['shell', 'am', 'start', '-n', f'{package_id}/.MainActivity'], unrelated, environment, log)
        emitted = None
        for _ in range(60):
            time.sleep(2)
            logcat = subprocess.run(adb + ['logcat', '-d', '-s', 'flutter'], capture_output=True, text=True).stdout
            emitted = next((line[line.index(marker) + len(marker):] for line in logcat.splitlines()
                            if marker in line), None)
            if emitted:
                break
        invoke(adb + ['shell', 'am', 'force-stop', package_id], unrelated, environment, log)
        device_info = {key: subprocess.run(adb + ['shell', 'getprop', key], capture_output=True,
                                           text=True).stdout.strip()
                       for key in ('ro.product.model', 'ro.build.version.release', 'ro.product.cpu.abi')}
        extra = {'apk_elf_16kb': 'passed', 'apk_arm64_libraries': [p['name'] for p in pages]}
    else:
        for project in (app / 'ios').rglob('project.pbxproj'):
            text = re.sub(r'IPHONEOS_DEPLOYMENT_TARGET = [0-9.]+;', 'IPHONEOS_DEPLOYMENT_TARGET = 15.1;',
                          project.read_text())
            text = text.replace('buildSettings = {', f'buildSettings = {{\n\t\t\t\tDEVELOPMENT_TEAM = {team};')
            project.write_text(text)
        invoke(['flutter', 'build', 'ios', '--release', '--config-only'], app, environment, log, timeout=1200)
        derived = work / 'ios-derived'
        invoke(['xcodebuild', '-quiet', '-workspace', 'Runner.xcworkspace', '-scheme', 'Runner', '-configuration', 'Release',
                '-destination', f'id={device}', '-allowProvisioningUpdates', '-allowProvisioningDeviceRegistration',
                '-derivedDataPath', derived, 'build'], app / 'ios', environment, log, timeout=1800)
        app_bundle = derived / 'Build/Products/Release-iphoneos/Runner.app'
        invoke(['codesign', '--verify', '--deep', '--strict', '--verbose=2', app_bundle], unrelated, environment, log)
        bridge = app_bundle / 'Frameworks/lib_onnx_bridge.framework/lib_onnx_bridge'
        invoke(['otool', '-L', bridge], unrelated, environment, log)
        bundled = [{'name': 'lib_onnx_bridge.framework/lib_onnx_bridge',
                    'sha256': hashlib.sha256(bridge.read_bytes()).hexdigest()}]
        bundle_id = subprocess.run(['/usr/libexec/PlistBuddy', '-c', 'Print :CFBundleIdentifier',
                                    app_bundle / 'Info.plist'], capture_output=True, text=True).stdout.strip()
        invoke(['xcrun', 'devicectl', 'device', 'install', 'app', '--device', device, app_bundle],
               unrelated, environment, log, timeout=600)
        try:
            console = subprocess.run(['xcrun', 'devicectl', 'device', 'process', 'launch', '--console',
                                      '--terminate-existing', '--device', device, bundle_id],
                                     capture_output=True, text=True, timeout=90).stdout
        except subprocess.TimeoutExpired as error:
            captured = error.stdout or b''
            console = captured.decode(errors='replace') if isinstance(captured, bytes) else captured
        log.write(f'\n[devicectl console]\n{console}\n')
        emitted = next((line[line.index(marker) + len(marker):] for line in console.splitlines()
                        if marker in line), None)
        listing = subprocess.run(['xcrun', 'devicectl', 'list', 'devices'], capture_output=True, text=True).stdout
        device_info = {'devicectl': next((line.strip() for line in listing.splitlines() if device in line), device)}
        extra = {}
    if not emitted:
        raise ValueError(f'{platform} consumer emitted no result')
    result = json.loads(emitted)
    log.write(f'\n[{platform} consumer result]\n{emitted}\n')
    if result.get('status') != 'passed':
        raise ValueError(f'{platform} consumer reported inference failure: {result}')
    (output / f'{platform}-launch.json').write_text(json.dumps(result, indent=2) + '\n')
    return {'result': result, 'device': device_info, 'bundled_binaries': bundled, **extra}


def load_tool(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/consumer-evidence')
    parser.add_argument('--ort-archive', type=Path, help='Explicit pinned ORT archive, used offline by the hook')
    parser.add_argument('--ort-cache', type=Path,
                        help='Directory of pinned ORT archives (named as in native/ort_artifacts.json), used offline by the hook')
    parser.add_argument('--flutter', action='store_true', help='Also build and launch a clean macOS Flutter release app')
    parser.add_argument('--android', metavar='ADB_SERIAL',
                        help='Also build a clean Flutter release APK, audit its ELF pages, install and launch it on this adb device')
    parser.add_argument('--ios', metavar='DEVICE_ID',
                        help='Also build a clean signed Flutter release app, install and launch it on this iOS device (devicectl)')
    parser.add_argument('--ios-team', help='Apple development team for --ios automatic signing')
    parser.add_argument('--keep', action='store_true', help='Keep the external consumer workspace for inspection')
    args = parser.parse_args()
    if args.ios and not args.ios_team:
        parser.error('--ios requires --ios-team')
    if args.ort_archive and args.ort_cache:
        parser.error('use either --ort-archive or --ort-cache')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='lib-onnx-consumer-'))
    archive = output / 'lib_onnx-candidate.tar.gz'
    receipt = {'status': 'running', 'workspace': str(work), 'started_at': datetime.now(timezone.utc).isoformat(),
               'source': 'dart pub publish --to-archive (exact publish payload)', 'hardware_acceptance': 'not-run'}
    environment = os.environ.copy()
    # No user native-library override is used. Fresh cache isolates consumer
    # dependency resolution from the workspace's package and native build dirs.
    environment['PUB_CACHE'] = str(work / 'pub-cache')
    environment['NO_PROXY'] = environment.get('NO_PROXY', '') + ',127.0.0.1,localhost'
    environment['no_proxy'] = environment.get('no_proxy', '') + ',127.0.0.1,localhost'
    server = None
    thread = None
    try:
        with (output / 'commands.log').open('w') as log:
            # Archive using the package manager's own publish selection rules.
            # Warnings remain visible in the log; no validations are skipped.
            invoke(['dart', 'pub', 'publish', f'--to-archive={archive}'], ROOT, os.environ.copy(), log)
            with tarfile.open(archive) as package:
                names = package.getnames()
                if any(name.endswith(('.dylib', '.so', '.dll', '.a')) for name in names):
                    raise ValueError('Candidate archive unexpectedly contains developer native binaries')
                metadata = pubspec_metadata(package.extractfile('pubspec.yaml').read().decode())
                model = package.extractfile('test/fixtures/identity_float32.onnx').read()
                fingerprint = hashlib.sha256()
                for name in sorted(names):
                    if name in ('pubspec.yaml', 'native/ort_artifacts.json') or (
                        name.endswith('.dart') and name.startswith(('lib/', 'hook/'))) or (
                        name.endswith('.c') and name.startswith('native/src/')) or (
                        name.endswith('.h') and name.startswith('native/include/')):
                        fingerprint.update(name.encode() + b'\0' + package.extractfile(name).read() + b'\0')
                receipt['source_fingerprint'] = fingerprint.hexdigest()
            payload = archive.read_bytes()
            archive_hash = hashlib.sha256(payload).hexdigest()
            receipt['archive_sha256'] = archive_hash
            version = metadata['version']

            class HostedHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    address = f'http://127.0.0.1:{self.server.server_port}'
                    item = {'version': version, 'pubspec': metadata, 'archive_url': f'{address}/archive.tar.gz',
                            'archive_sha256': archive_hash, 'published': '2026-11-04T00:00:00.000Z'}
                    if self.path == '/api/packages/lib_onnx':
                        data, kind = json.dumps({'name': 'lib_onnx', 'latest': item, 'versions': [item]}).encode(), 'application/json'
                    elif self.path == f'/api/packages/lib_onnx/versions/{version}':
                        data, kind = json.dumps(item).encode(), 'application/json'
                    elif self.path == '/archive.tar.gz':
                        data, kind = payload, 'application/gzip'
                    else:
                        self.send_error(404)
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', kind)
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                def log_message(self, *unused):
                    pass

            server = ThreadingHTTPServer(('127.0.0.1', 0), HostedHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            address = f'http://127.0.0.1:{server.server_port}'
            dependency = f'  lib_onnx:\n    hosted: {address}\n    version: {version}\n'
            configuration = ''
            if args.ort_archive:
                artifact_input = work / 'inputs' / args.ort_archive.name
                artifact_input.parent.mkdir()
                shutil.copyfile(args.ort_archive.resolve(), artifact_input)
                receipt['ort_archive_sha256'] = hashlib.sha256(artifact_input.read_bytes()).hexdigest()
                configuration = ('\nhooks:\n  user_defines:\n    lib_onnx:\n'
                                 f'      ort_archive: {json.dumps(str(artifact_input))}\n      offline: true\n')
            if args.ort_cache:
                cache_input = work / 'inputs' / 'ort-cache'
                shutil.copytree(args.ort_cache.resolve(), cache_input)
                receipt['ort_cache_sha256'] = {
                    file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                    for file in sorted(cache_input.iterdir()) if file.is_file()}
                configuration = ('\nhooks:\n  user_defines:\n    lib_onnx:\n'
                                 f'      ort_cache: {json.dumps(str(cache_input) + "/")}\n      offline: true\n')
            consumer = work / 'dart-consumer'
            (consumer / 'bin').mkdir(parents=True)
            (consumer / 'pubspec.yaml').write_text(
                "name: archive_consumer\nenvironment:\n  sdk: '>=3.10.0 <4.0.0'\ndependencies:\n" + dependency + configuration)
            smoke = dart_smoke(model)
            (consumer / 'bin/main.dart').write_text(smoke + "\nFuture<void> main() async { print(jsonEncode(await smoke())); }\n")
            invoke(['dart', 'pub', 'get'], consumer, environment, log)
            config = json.loads((consumer / '.dart_tool/package_config.json').read_text())
            lib_entry = next(item for item in config['packages'] if item['name'] == 'lib_onnx')
            if 'pub-cache' not in lib_entry['rootUri']:
                raise ValueError(f'Consumer resolved a workspace dependency: {lib_entry}')
            receipt['resolved_package'] = lib_entry['rootUri']
            receipt['dart_jit_log'] = invoke(['dart', 'run', 'bin/main.dart'], consumer, environment, log)
            bundle_output = work / 'cli-release'
            invoke(['dart', 'build', 'cli', '--target=bin/main.dart', f'--output={bundle_output}'], consumer, environment, log)
            bundle = bundle_output / 'bundle'
            unrelated = work / 'unrelated-cwd'
            unrelated.mkdir()
            binary = next((bundle / 'bin').iterdir())
            receipt['dart_release_log'] = invoke([binary], unrelated, environment, log)
            receipt['dart_release'] = 'passed'
            shutil.copytree(bundle, output / 'dart-bundle', dirs_exist_ok=True)
            receipt['bundled_libraries'] = [
                {'name': file.name, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
                for file in (bundle / 'lib').iterdir() if file.is_file()]
            if os.uname().sysname == 'Darwin':
                for file in (bundle / 'lib').iterdir():
                    if file.is_file():
                        invoke(['otool', '-L', file], unrelated, environment, log)
                        invoke(['codesign', '--verify', '--verbose=2', file], unrelated, environment, log)
            if args.flutter:
                if os.uname().sysname != 'Darwin':
                    raise ValueError('--flutter currently verifies the macOS release consumer only')
                app = work / 'flutter-consumer'
                invoke(['flutter', 'create', '--platforms=macos', '--project-name=archive_flutter_consumer', '--empty', app], work, environment, log)
                (app / 'pubspec.yaml').write_text(
                    "name: archive_flutter_consumer\npublish_to: none\nenvironment:\n  sdk: '>=3.10.0 <4.0.0'\n"
                    "dependencies:\n  flutter:\n    sdk: flutter\n" + dependency + configuration + '\nflutter:\n  uses-material-design: true\n')
                (app / 'lib/main.dart').write_text(smoke + '''
import 'dart:io';
import 'package:flutter/material.dart';
void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MaterialApp(home: Scaffold(body: Text('Real ORT consumer smoke'))));
  WidgetsBinding.instance.addPostFrameCallback((_) async {
    try {
      final result = await smoke();
      stdout.writeln(jsonEncode(result));
      await stdout.flush();
      exit(0);
    } catch (error, stack) {
      stdout.writeln(jsonEncode({'status':'failed','error':'$error','stack':'$stack'}));
      await stdout.flush();
      exit(1);
    }
  });
}
'''.replace("import 'dart:io';\nimport 'package:flutter/material.dart';\n", ''))
                # Imports must precede declarations.
                source = (app / 'lib/main.dart').read_text()
                (app / 'lib/main.dart').write_text("import 'dart:io';\nimport 'package:flutter/material.dart';\n" + source)
                for project in (app / 'macos').rglob('project.pbxproj'):
                    project.write_text(re.sub(r'MACOSX_DEPLOYMENT_TARGET = [0-9.]+;', 'MACOSX_DEPLOYMENT_TARGET = 14.0;', project.read_text()))
                invoke(['flutter', 'pub', 'get'], app, environment, log)
                flutter_environment = environment | {'FLUTTER_MACOS_ARM64_ONLY': 'true'}
                log.write('\n[Scoped build input: FLUTTER_MACOS_ARM64_ONLY=true; no global Flutter config changed]\n')
                invoke(['flutter', 'build', 'macos', '--release'], app, flutter_environment, log, timeout=1200)
                app_bundle = next((app / 'build/macos/Build/Products/Release').glob('*.app'))
                invoke(['codesign', '--verify', '--deep', '--strict', '--verbose=2', app_bundle], unrelated, environment, log)
                app_receipt = output / 'flutter-launch.json'
                executable = app_bundle / 'Contents/MacOS/archive_flutter_consumer'
                launch_output = invoke([executable], unrelated, environment, log, timeout=120)
                emitted = next(line for line in launch_output.splitlines() if line.startswith('{') and '"status"' in line)
                receipt['flutter_release'] = json.loads(emitted)
                if receipt['flutter_release'].get('status') != 'passed':
                    raise ValueError('Flutter consumer reported inference failure')
                app_receipt.write_text(json.dumps(receipt['flutter_release'], indent=2) + '\n')
                shutil.copytree(app_bundle, output / app_bundle.name, dirs_exist_ok=True, symlinks=True)
            if args.android:
                receipt['android_release'] = mobile_consumer(
                    'android', args.android, None, work, output, dependency, configuration, smoke,
                    environment, unrelated, log)
            if args.ios:
                receipt['ios_release'] = mobile_consumer(
                    'ios', args.ios, args.ios_team, work, output, dependency, configuration, smoke,
                    environment, unrelated, log)
            receipt['status'] = 'passed'
    except Exception as error:
        receipt.update({'status': 'failed', 'error': str(error)})
        raise
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if thread:
            thread.join(timeout=5)
        receipt['completed_at'] = datetime.now(timezone.utc).isoformat()
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if not args.keep:
            shutil.rmtree(work)
    print(f'Clean archive consumer passed; evidence: {output}')


if __name__ == '__main__':
    main()
