#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-address,undefined}"
onnx_test_ort="${LIB_ONNX_TEST_ORT:-$PWD/native/third_party/onnxruntime/lib/macos_arm64/libonnxruntime.dylib}"
if [[ ! -f "$onnx_test_ort" ]]; then
  echo "Set LIB_ONNX_TEST_ORT to the exact pinned ONNX Runtime library" >&2
  exit 1
fi
out="$PWD/.cache/native-tests/${mode//,/+}"
mkdir -p "$out"
python3 tools/make_test_models.py
args=(-std=c11 -g -O1 -Wall -Wextra -Werror -DLIB_ONNX_TESTING -I native/include -I native/third_party/onnxruntime/include)
if [[ "$mode" != none ]]; then args+=("-fsanitize=$mode" -fno-omit-frame-pointer); fi
clang "${args[@]}" native/src/lib_onnx_bridge.c native/tests/runtime_test.c "$onnx_test_ort" -Wl,-rpath,"$(dirname "$onnx_test_ort")" -lpthread -o "$out/runtime_test"
"$out/runtime_test" "$PWD/test/fixtures"
