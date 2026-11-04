#include "lib_onnx.h"
#include <assert.h>
#include <stdio.h>

int main(int argc, char **argv) {
  assert(argc == 2);
  LibOnnxRuntimeHandle rt = lib_onnx_runtime_create(1);
  assert(rt);
  const char *providers[] = {"CPU"};
  LibOnnxSessionHandle s = lib_onnx_session_create(
      rt, argv[1], NULL, 0, providers, 1, NULL, NULL, 0, 1, 0, 0);
  assert(s);
  int64_t shape[] = {1, 4};
  unsigned char data[] = {0, 1, 2, 3};
  LibOnnxTensorHandle input = lib_onnx_tensor_create(6, shape, 2, data, 4, 1);
  LibOnnxTensorHandle output = NULL;
  const char *in[] = {"X"}, *out[] = {"Y"};
  assert(lib_onnx_session_run_sync(s, 1, in, &input, 1, out, &output, 1,
                                   NULL) == 0);
  size_t bytes = lib_onnx_tensor_get_byte_length(output);
  fprintf(stderr, "uint8 output bytes: %zu; expected 4\n", bytes);
  assert(bytes == 4);
  lib_onnx_tensor_destroy(output);
  lib_onnx_tensor_destroy(input);
  lib_onnx_session_destroy(s);
  lib_onnx_runtime_destroy(rt);
}
