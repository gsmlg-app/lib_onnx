#define _POSIX_C_SOURCE 200809L
#include "lib_onnx.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern void lib_onnx_test_barrier(int32_t);
extern int32_t lib_onnx_test_task_at_barrier(LibOnnxTaskHandle);
extern int32_t lib_onnx_test_cancel_code(LibOnnxTaskHandle);
extern int32_t lib_onnx_test_task_in_run(LibOnnxTaskHandle);
static const char *fixture_root;
static void pause_tick(void) {
  struct timespec p = {0, 1000000};
  nanosleep(&p, NULL);
}
static void terminal(LibOnnxTaskHandle t) {
  for (int i = 0; i < 10000 && lib_onnx_task_state(t) != 2; i++)
    pause_tick();
  assert(lib_onnx_task_state(t) == 2);
}
static void barrier(LibOnnxTaskHandle t) {
  for (int i = 0; i < 10000 && !lib_onnx_test_task_at_barrier(t); i++)
    pause_tick();
  assert(lib_onnx_test_task_at_barrier(t));
}
static LibOnnxSessionHandle load(LibOnnxRuntimeHandle rt, const char *file,
                                 int queue) {
  char path[4096], q[32];
  snprintf(path, sizeof(path), "%s/%s.onnx", fixture_root, file);
  snprintf(q, sizeof(q), "%d", queue);
  const char *providers[] = {"CPU"};
  const char *keys[] = {"bridge.max_queue"}, *vals[] = {q};
  LibOnnxSessionHandle s = lib_onnx_session_create(rt, path, NULL, 0, providers,
                                                   1, keys, vals, 1, 1, 0, 0);
  if (!s)
    fprintf(stderr, "load %s: %s\n", path, lib_onnx_get_last_error());
  assert(s);
  return s;
}
static LibOnnxTaskHandle submit(LibOnnxSessionHandle s, LibOnnxTensorHandle t,
                                int64_t timeout) {
  const char *ins[] = {"X"}, *outs[] = {"Y"};
  return lib_onnx_session_submit(s, 42, ins, &t, 1, outs, 1, timeout);
}
static LibOnnxTensorHandle input(void) {
  float f[] = {1, 2, 3, 4};
  int64_t dims[] = {1, 4};
  LibOnnxTensorHandle t = lib_onnx_tensor_create(1, dims, 2, f, sizeof(f), 1);
  assert(t);
  return t;
}
static void tensor_cases(void) {
  const char *types[] = {"float32", "float16", "int64", "int32",
                         "int8",    "uint8",   "bool"};
  int widths[] = {4, 2, 8, 4, 1, 1, 1};
  const char *suffixes[] = {"", "_scalar", "_empty", "_rank20"};
  LibOnnxRuntimeHandle rt = lib_onnx_runtime_create(2);
  assert(rt);
  for (int dtype = 1; dtype <= 7; dtype++)
    for (int mode = 0; mode < 4; mode++) {
      char model[128];
      snprintf(model, sizeof(model), "identity_%s%s", types[dtype - 1],
               suffixes[mode]);
      LibOnnxSessionHandle s = load(rt, model, 2);
      int64_t dims[20];
      for (int i = 0; i < 20; i++)
        dims[i] = 1;
      int rank = mode == 1 ? 0 : mode == 3 ? 20 : 2;
      if (mode == 0) {
        dims[0] = 1;
        dims[1] = 4;
      }
      if (mode == 2) {
        dims[0] = 0;
        dims[1] = 4;
      }
      size_t n = mode == 0   ? 4
                 : mode == 2 ? 0
                             : 1,
             bytes = n * (size_t)widths[dtype - 1];
      unsigned char data[32] = {0};
      if (dtype == 7)
        for (size_t i = 0; i < bytes; i++)
          data[i] = (unsigned char)(i % 2);
      LibOnnxTensorHandle in =
          lib_onnx_tensor_create(dtype, dims, rank, data, bytes, 1);
      assert(in);
      LibOnnxTaskHandle task = submit(s, in, 0);
      assert(task);
      lib_onnx_tensor_destroy(in);
      terminal(task);
      if (lib_onnx_task_status(task))
        fprintf(stderr, "task: %s\n", lib_onnx_task_error(task));
      assert(lib_onnx_task_status(task) == 0);
      LibOnnxTensorHandle out = lib_onnx_task_output(task, 0);
      assert(out);
      assert(lib_onnx_tensor_get_byte_length(out) == bytes);
      assert(lib_onnx_tensor_get_element_count(out) == (int64_t)n);
      assert(lib_onnx_tensor_get_rank(out) == rank);
      assert(!bytes || !memcmp(data, lib_onnx_tensor_get_data(out), bytes));
      LibOnnxDiagnosticReportHandle d = lib_onnx_task_diagnostic(task);
      assert(d);
      assert(lib_onnx_diagnostic_get_fallback_occurred(d) == 0);
      lib_onnx_session_destroy(
          s); /* Completed result retains its session+environment. */
      assert(lib_onnx_tensor_get_byte_length(out) == bytes);
      lib_onnx_task_release(task);
    }
  int64_t bad[] = {-1, 4}, huge[] = {INT64_MAX, 2}, zero[] = {0, 4};
  unsigned char buf[8] = {0};
  assert(!lib_onnx_tensor_create(6, bad, 2, buf, 4, 1));
  assert(lib_onnx_get_last_error_code() == 2);
  assert(!lib_onnx_tensor_create(3, huge, 2, buf, 8, 1));
  assert(!lib_onnx_tensor_create(6, zero, 2, buf, 1, 1));
  assert(!lib_onnx_tensor_create(99, zero, 2, buf, 0, 1));
  lib_onnx_runtime_destroy(rt);
}
static void unsupported_outputs(void) {
  LibOnnxRuntimeHandle rt = lib_onnx_runtime_create(1);
  assert(rt);
  const char *models[] = {"partial_sequence", "unsupported_double"};
  for (int i = 0; i < 2; i++) {
    LibOnnxSessionHandle s = load(rt, models[i], 1);
    LibOnnxTensorHandle in = input();
    const char *names[] = {"X"}, *outputs[] = {"Y", "sequence"};
    LibOnnxTaskHandle t = lib_onnx_session_submit(s, 1, names, &in, 1, outputs,
                                                  i == 0 ? 2 : 1, 0);
    assert(t);
    lib_onnx_tensor_destroy(in);
    terminal(t);
    assert(lib_onnx_task_status(t) == 11);
    assert(!lib_onnx_task_output(t, 0));
    assert(strlen(lib_onnx_task_error(t)) > 0);
    lib_onnx_task_release(t);
    lib_onnx_session_destroy(s);
  }
  lib_onnx_runtime_destroy(rt);
}
static void lifecycle(void) {
  LibOnnxRuntimeHandle rt = lib_onnx_runtime_create(2);
  assert(rt);
  LibOnnxSessionHandle a = load(rt, "identity_float32", 1),
                       b = load(rt, "identity_float32", 1);
  LibOnnxTensorHandle in = input();
  lib_onnx_test_barrier(1);
  LibOnnxTaskHandle t1 = submit(a, in, 0);
  assert(t1);
  barrier(t1);
  LibOnnxTaskHandle t2 = submit(a, in, 0);
  assert(t2);
  assert(lib_onnx_task_state(t2) == 0);
  assert(!submit(a, in, 0));
  assert(lib_onnx_get_last_error_code() == 9);
  LibOnnxTaskHandle t3 = submit(b, in, 0);
  assert(t3);
  barrier(t3);
  assert(lib_onnx_task_id(t1) != lib_onnx_task_id(t2) &&
         lib_onnx_task_id(t2) != lib_onnx_task_id(t3));
  lib_onnx_task_cancel(t1, 7);
  lib_onnx_task_cancel(t2, 7);
  lib_onnx_tensor_destroy(in);
  lib_onnx_test_barrier(0);
  terminal(t1);
  terminal(t2);
  terminal(t3);
  assert(lib_onnx_task_status(t1) == 7 && lib_onnx_task_status(t2) == 7 &&
         lib_onnx_task_status(t3) == 0);
  lib_onnx_task_cancel(t3, 7);
  assert(lib_onnx_task_status(t3) == 0);
  lib_onnx_session_destroy(a);
  lib_onnx_session_destroy(b);
  lib_onnx_runtime_destroy(rt);
  /* Delayed acknowledgement after all owners close: result and errors still
   * valid. */
  assert(lib_onnx_task_output(t3, 0));
  assert(strstr(lib_onnx_task_error(t1), "Cancellation"));
  lib_onnx_task_release(t1);
  lib_onnx_task_release(t2);
  lib_onnx_task_release(t3);
}
static void timeout_close(void) {
  LibOnnxRuntimeHandle rt = lib_onnx_runtime_create(1);
  assert(rt);
  LibOnnxSessionHandle s = load(rt, "identity_float32", 2);
  LibOnnxTensorHandle in = input();
  lib_onnx_test_barrier(1);
  LibOnnxTaskHandle active = submit(s, in, 0);
  barrier(active);
  LibOnnxTaskHandle timeout = submit(s, in, 1000);
  assert(timeout);
  /* A control-thread deadline is not dependent on ORT Run/event-loop progress.
   */
  for (int i = 0; i < 10000 && !lib_onnx_test_cancel_code(timeout); i++)
    pause_tick();
  assert(lib_onnx_test_cancel_code(timeout) == 8);
  lib_onnx_test_barrier(0);
  terminal(timeout);
  terminal(active);
  assert(lib_onnx_task_status(timeout) == 8);
  assert(lib_onnx_task_status(active) == 0);
  lib_onnx_task_release(timeout);
  lib_onnx_task_release(active);
  lib_onnx_test_barrier(1);
  active = submit(s, in, 0);
  barrier(active);
  lib_onnx_runtime_begin_close(rt);
  assert(!submit(s, in, 0));
  assert(lib_onnx_get_last_error_code() == 10);
  lib_onnx_test_barrier(0);
  terminal(active);
  assert(lib_onnx_task_status(active) == 10);
  lib_onnx_tensor_destroy(in);
  lib_onnx_runtime_destroy(rt);
  lib_onnx_session_destroy(s);
  lib_onnx_task_release(active);
}
static void cancel_real_run(void) {
  LibOnnxRuntimeHandle rt = lib_onnx_runtime_create(1);
  assert(rt);
  LibOnnxSessionHandle s = load(rt, "long_loop", 2);
  LibOnnxTensorHandle in = input();
  LibOnnxTaskHandle t = submit(s, in, 0);
  assert(t);
  for (int i = 0; i < 10000 && !lib_onnx_test_task_in_run(t); i++)
    pause_tick();
  assert(lib_onnx_test_task_in_run(t));
  lib_onnx_task_cancel(t, 7);
  terminal(t);
  assert(lib_onnx_task_status(t) == 7);
  lib_onnx_task_release(t);
  t = submit(s, in, 1000);
  assert(t);
  terminal(t);
  assert(lib_onnx_task_status(t) == 8);
  lib_onnx_task_release(t);
  lib_onnx_tensor_destroy(in);
  lib_onnx_session_destroy(s);
  lib_onnx_runtime_destroy(rt);
}
static void runtime_budget(void) {
  LibOnnxRuntimeHandle rt = lib_onnx_runtime_create(1);
  assert(rt);
  LibOnnxSessionHandle s = load(rt, "identity_float32", 1000);
  LibOnnxTensorHandle in = input();
  LibOnnxTaskHandle tasks[65];
  lib_onnx_test_barrier(1);
  for (int i = 0; i < 65; i++) {
    tasks[i] = submit(s, in, 0);
    assert(tasks[i]);
  }
  assert(!submit(s, in, 0));
  assert(lib_onnx_get_last_error_code() == 9);
  lib_onnx_runtime_begin_close(rt);
  lib_onnx_test_barrier(0);
  for (int i = 0; i < 65; i++) {
    terminal(tasks[i]);
    assert(lib_onnx_task_status(tasks[i]) == 10);
    lib_onnx_task_release(tasks[i]);
  }
  lib_onnx_tensor_destroy(in);
  lib_onnx_session_destroy(s);
  lib_onnx_runtime_destroy(rt);
}
static void early_ack_and_concurrency(void) {
  LibOnnxRuntimeHandle rt = lib_onnx_runtime_create(2);
  assert(rt);
  char path[4096];
  snprintf(path, sizeof(path), "%s/identity_float32.onnx", fixture_root);
  const char *prov[] = {"CPU"},
             *keys[] = {"bridge.max_active", "bridge.max_queue"},
             *vals[] = {"2", "0"};
  LibOnnxSessionHandle s = lib_onnx_session_create(rt, path, NULL, 0, prov, 1,
                                                   keys, vals, 2, 1, 0, 0);
  assert(s);
  LibOnnxTensorHandle in = input();
  lib_onnx_test_barrier(1);
  LibOnnxTaskHandle a = submit(s, in, 0), b = submit(s, in, 0);
  assert(a && b);
  barrier(a);
  barrier(b);
  assert(!submit(s, in, 0));
  assert(lib_onnx_get_last_error_code() == 9);
  /* Releasing the client reference while computation is latched must not free
   * input/session/RunOptions underneath the scheduler reference. */
  lib_onnx_task_release(a);
  lib_onnx_task_release(b);
  lib_onnx_runtime_begin_close(rt);
  lib_onnx_test_barrier(0);
  lib_onnx_runtime_destroy(rt);
  lib_onnx_tensor_destroy(in);
  lib_onnx_session_destroy(s);
}
int main(int argc, char **argv) {
  assert(argc == 2);
  fixture_root = argv[1];
  tensor_cases();
  unsupported_outputs();
  lifecycle();
  timeout_close();
  cancel_real_run();
  runtime_budget();
  early_ack_and_concurrency();
  puts("native safety/lifecycle: PASS");
}
