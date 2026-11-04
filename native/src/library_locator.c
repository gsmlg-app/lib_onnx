/* The native-assets resolver finds this symbol before DynamicLibrary lookup.
 * dladdr identifies the loaded asset, independent of CWD and app layout. */
#if !defined(_WIN32)
#define _GNU_SOURCE
#include <dlfcn.h>
#include "lib_onnx.h"
LIB_ONNX_EXPORT const char* lib_onnx_library_path(void) {
  Dl_info info;
  if (!dladdr((const void*)&lib_onnx_library_path, &info)) return NULL;
  return info.dli_fname;
}
#endif
