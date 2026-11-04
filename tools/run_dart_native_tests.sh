#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
onnx_test_ort="${LIB_ONNX_TEST_ORT:-$PWD/native/third_party/onnxruntime/lib/macos_arm64/libonnxruntime.dylib}"
mkdir -p native/build
if [[ "$(uname -s)" == Darwin ]]; then
  onnx_test_bridge="$PWD/native/build/liblib_onnx_test_bridge.dylib"
  platform_flags=(-Wl,-rpath,"$(dirname "$onnx_test_ort")")
else
  onnx_test_bridge="$PWD/native/build/liblib_onnx_test_bridge.so"
  platform_flags=(-Wl,-rpath,"$(dirname "$onnx_test_ort")" -ldl)
fi
clang -shared -fPIC -std=c11 -DLIB_ONNX_TESTING -Wall -Wextra -Werror \
  -I native/include -I native/third_party/onnxruntime/include \
  native/src/lib_onnx_bridge.c native/src/library_locator.c "$onnx_test_ort" \
  "${platform_flags[@]}" -lpthread -o "$onnx_test_bridge"
# Explicit bridge development path; the normal package/consumer route uses hooks.
export LIB_ONNX_TEST_BRIDGE="$onnx_test_bridge"
onnx_test_runner="$(python3 - <<'PY'
import json
from pathlib import Path
from urllib.parse import urlparse,unquote
config=json.loads(Path('.dart_tool/package_config.json').read_text())
print(unquote(urlparse(next(p['rootUri'] for p in config['packages'] if p['name']=='test')).path)+'/bin/test.dart')
PY
)"
dart --packages=.dart_tool/package_config.json "$onnx_test_runner" native/tests/lifecycle_test.dart -r expanded
