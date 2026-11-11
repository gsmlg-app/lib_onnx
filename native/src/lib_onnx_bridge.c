#define _POSIX_C_SOURCE 200809L
#include "lib_onnx.h"
#include "onnxruntime_c_api.h"
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

/* All mutable scheduler state is protected by Runtime.mu. Terminal task data is
 * published under that lock and remains owned until the last task reference.
 * No callback crosses a native/Dart thread boundary. */
static const OrtApi *ort;
static pthread_once_t ort_once = PTHREAD_ONCE_INIT;
static atomic_int_fast64_t next_task_id = 1;
typedef struct {
  int code, ort_code;
  char stage[64], message[2048];
} Error;
static _Thread_local Error last_error;
static void init_ort(void) {
  const OrtApiBase *b = OrtGetApiBase();
  if (b)
    ort = b->GetApi(ORT_API_VERSION);
}
static int error(int code, const char *stage, const char *msg) {
  last_error = (Error){.code = code};
  snprintf(last_error.stage, sizeof(last_error.stage), "%s", stage);
  snprintf(last_error.message, sizeof(last_error.message), "%s", msg);
  return code;
}
static int status_error(OrtStatus *status, const char *stage) {
  if (!status)
    return 0;
  int oc = ort->GetErrorCode(status);
  error(LIB_ONNX_ERR_GENERIC, stage, ort->GetErrorMessage(status));
  last_error.ort_code = oc;
  ort->ReleaseStatus(status);
  return last_error.code;
}
static int oom(void) {
  return error(LIB_ONNX_ERR_OUT_OF_MEMORY, "allocation",
               "Native allocation failed");
}
static int64_t now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (int64_t)t.tv_sec * 1000000 + t.tv_nsec / 1000;
}
int64_t lib_onnx_monotonic_us(void) { return now_us(); }
uint32_t lib_onnx_abi_version(void) { return LIB_ONNX_BRIDGE_ABI_VERSION; }
int32_t lib_onnx_version_major(void) { return 0; }
int32_t lib_onnx_version_minor(void) { return 3; }
int32_t lib_onnx_version_patch(void) { return 1; }
const char *lib_onnx_version_string(void) { return "0.3.1"; }
const char *lib_onnx_ort_version_string(void) {
  return OrtGetApiBase()->GetVersionString();
}
const char *lib_onnx_get_last_error(void) { return last_error.message; }
const char *lib_onnx_get_last_error_stage(void) { return last_error.stage; }
int32_t lib_onnx_get_last_error_code(void) { return last_error.code; }
int32_t lib_onnx_get_last_ort_code(void) { return last_error.ort_code; }
void lib_onnx_clear_last_error(void) { last_error = (Error){0}; }

/* The single native dtype table supplies BOTH conversions and byte widths. */
typedef struct {
  ONNXTensorElementDataType ort_type;
  size_t width;
} Dtype;
static const Dtype dtypes[] = {{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, 0},
                               {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4},
                               {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, 2},
                               {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8},
                               {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, 4},
                               {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, 1},
                               {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, 1},
                               {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, 1}};
static int dtype_from_ort(ONNXTensorElementDataType t) {
  for (int i = 1; i < 8; i++)
    if (dtypes[i].ort_type == t)
      return i;
  return 0;
}
static int tensor_size(int dtype, const int64_t *dims, int rank, int64_t *count,
                       size_t *bytes) {
  if (dtype < 1 || dtype > 7)
    return error(11, "tensor.type", "Unsupported tensor element type");
  if (rank < 0 || (rank && !dims) || (size_t)rank > SIZE_MAX / sizeof(int64_t))
    return error(2, "tensor.shape", "Invalid rank/shape pointer");
  /* Check every dimension even if an earlier zero makes the tensor empty. */
  int empty = 0;
  for (int i = 0; i < rank; i++) {
    if (dims[i] < 0)
      return error(2, "tensor.shape",
                   "Actual tensor dimensions must be nonnegative");
    if (!dims[i])
      empty = 1;
  }
  uint64_t n = empty ? 0 : 1;
  if (!empty)
    for (int i = 0; i < rank; i++) {
      if ((uint64_t)dims[i] > (uint64_t)INT64_MAX / n)
        return error(2, "tensor.shape", "Element count overflow");
      n *= (uint64_t)dims[i];
    }
  if (n > SIZE_MAX / dtypes[dtype].width ||
      n > (uint64_t)INT64_MAX / dtypes[dtype].width)
    return error(2, "tensor.shape", "Tensor byte size overflow");
  *count = (int64_t)n;
  *bytes = (size_t)n * dtypes[dtype].width;
  return 0;
}
struct LibOnnxTensor {
  int dtype, rank;
  int64_t *shape;
  int64_t count;
  void *data;
  size_t bytes;
  int owns_data;
  OrtValue *value;
};
void lib_onnx_tensor_destroy(LibOnnxTensorHandle t) {
  if (!t)
    return;
  if (t->value)
    ort->ReleaseValue(t->value);
  if (t->owns_data)
    free(t->data);
  free(t->shape);
  free(t);
}
LibOnnxTensorHandle lib_onnx_tensor_create(int32_t dtype, const int64_t *shape,
                                           int32_t rank, const void *data,
                                           size_t length, int32_t copy_data) {
  pthread_once(&ort_once, init_ort);
  if (!ort) {
    error(1, "runtime.init", "ORT API version unavailable");
    return NULL;
  }
  int64_t count;
  size_t bytes;
  if (tensor_size(dtype, shape, rank, &count, &bytes))
    return NULL;
  if (length != bytes || (bytes && !data)) {
    error(2, "tensor.buffer",
          "Buffer length must exactly match tensor shape/type");
    return NULL;
  }
  struct LibOnnxTensor *t = calloc(1, sizeof(*t));
  if (!t) {
    oom();
    return NULL;
  }
  t->dtype = dtype;
  t->rank = rank;
  t->count = count;
  t->bytes = bytes;
  if (rank) {
    t->shape = malloc(sizeof(int64_t) * (size_t)rank);
    if (!t->shape)
      goto fail;
    memcpy(t->shape, shape, sizeof(int64_t) * (size_t)rank);
  }
  /* ORT expects a non-null backing pointer even for some zero-byte tensors. */
  t->owns_data = copy_data || !bytes;
  t->data = t->owns_data ? malloc(bytes ? bytes : 1) : (void *)data;
  if (!t->data)
    goto fail;
  if (bytes && copy_data)
    memcpy(t->data, data, bytes);
  OrtMemoryInfo *mem = NULL;
  if (status_error(
          ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem),
          "tensor.memory"))
    goto fail_status;
  OrtStatus *st = ort->CreateTensorWithDataAsOrtValue(
      mem, t->data, bytes, t->shape, (size_t)rank, dtypes[dtype].ort_type,
      &t->value);
  ort->ReleaseMemoryInfo(mem);
  if (status_error(st, "tensor.create"))
    goto fail_status;
  return t;
fail:
  oom();
fail_status:
  lib_onnx_tensor_destroy(t);
  return NULL;
}
int32_t lib_onnx_tensor_get_dtype(LibOnnxTensorHandle t) {
  return t ? t->dtype : 0;
}
int32_t lib_onnx_tensor_get_rank(LibOnnxTensorHandle t) {
  return t ? t->rank : -1;
}
int32_t lib_onnx_tensor_get_shape(LibOnnxTensorHandle t, int64_t *dst,
                                  int32_t capacity) {
  if (!t)
    return -1;
  if (!dst)
    return t->rank;
  if (capacity < t->rank) {
    error(2, "tensor.shape", "Shape buffer too short");
    return -1;
  }
  if (t->rank)
    memcpy(dst, t->shape, (size_t)t->rank * sizeof(int64_t));
  return t->rank;
}
const void *lib_onnx_tensor_get_data(LibOnnxTensorHandle t) {
  return t ? t->data : NULL;
}
size_t lib_onnx_tensor_get_byte_length(LibOnnxTensorHandle t) {
  return t ? t->bytes : 0;
}
int64_t lib_onnx_tensor_get_element_count(LibOnnxTensorHandle t) {
  return t ? t->count : -1;
}

struct LibOnnxDiagnosticReport {
  char provider[16], registered[128], details[2304];
  int evidence, fallback;
  int64_t duration;
};
const char *lib_onnx_diagnostic_get_provider(LibOnnxDiagnosticReportHandle d) {
  return d ? d->provider : "";
}
const char *
lib_onnx_diagnostic_get_registered_providers(LibOnnxDiagnosticReportHandle d) {
  return d ? d->registered : "";
}
int32_t
lib_onnx_diagnostic_get_evidence_level(LibOnnxDiagnosticReportHandle d) {
  return d ? d->evidence : 0;
}
int64_t lib_onnx_diagnostic_get_duration_us(LibOnnxDiagnosticReportHandle d) {
  return d ? d->duration : 0;
}
const char *
lib_onnx_diagnostic_get_hardware_details(LibOnnxDiagnosticReportHandle d) {
  return d ? d->details : "";
}
const char *
lib_onnx_diagnostic_get_model_hash(LibOnnxDiagnosticReportHandle d) {
  (void)d;
  return "";
}
int32_t
lib_onnx_diagnostic_get_fallback_occurred(LibOnnxDiagnosticReportHandle d) {
  return d ? d->fallback : -1;
}
void lib_onnx_diagnostic_destroy(LibOnnxDiagnosticReportHandle d) { free(d); }

typedef struct {
  int count;
  char **names;
  int *types;
  int *ranks;
  int64_t **shapes;
} Metadata;
struct LibOnnxRuntime {
  atomic_int refs;
  OrtEnv *env;
  pthread_mutex_t mu;
  pthread_cond_t cv;
  pthread_t *workers;
  pthread_t monitor;
  int nworkers, monitor_started, closing, pending;
  struct LibOnnxTask *head, *tail, *live;
};
struct LibOnnxSession {
  atomic_int refs;
  struct LibOnnxRuntime *rt;
  OrtSession *value;
  Metadata inputs, outputs;
  int closing, queued, active, pending, max_queue, max_active;
  int cpu_only, accelerators_requested, fallback;
  char registered[128], initialization[2048];
};
struct LibOnnxTask {
  atomic_int refs;
  struct LibOnnxSession *s;
  struct LibOnnxTask *next, *live_next;
  int64_t id, request_id, deadline, started;
  int state, cancel_code, cancel_ort_code, status;
  Error failure;
  OrtRunOptions *options;
  int ni, no;
  char **in_names, **out_names;
  LibOnnxTensorHandle *inputs, *outputs;
  LibOnnxDiagnosticReportHandle diagnostic;
#ifdef LIB_ONNX_TESTING
  int at_barrier, in_run;
#endif
};
static void runtime_release(struct LibOnnxRuntime *rt) {
  if (atomic_fetch_sub(&rt->refs, 1) != 1)
    return;
  ort->ReleaseEnv(rt->env);
  free(rt->workers);
  pthread_mutex_destroy(&rt->mu);
  pthread_cond_destroy(&rt->cv);
  free(rt);
}
static void free_metadata(Metadata *m) {
  for (int i = 0; i < m->count; i++) {
    if (m->names)
      free(m->names[i]);
    if (m->shapes)
      free(m->shapes[i]);
  }
  free(m->names);
  free(m->types);
  free(m->ranks);
  free(m->shapes);
}
static void session_release(struct LibOnnxSession *s) {
  if (atomic_fetch_sub(&s->refs, 1) != 1)
    return;
  free_metadata(&s->inputs);
  free_metadata(&s->outputs);
  if (s->value)
    ort->ReleaseSession(s->value);
  runtime_release(s->rt);
  free(s);
}
void lib_onnx_task_release(LibOnnxTaskHandle t) {
  if (!t || atomic_fetch_sub(&t->refs, 1) != 1)
    return;
  for (int i = 0; i < t->ni; i++) {
    if (t->in_names)
      free(t->in_names[i]);
    if (t->inputs)
      lib_onnx_tensor_destroy(t->inputs[i]);
  }
  for (int i = 0; i < t->no; i++) {
    if (t->out_names)
      free(t->out_names[i]);
    if (t->outputs)
      lib_onnx_tensor_destroy(t->outputs[i]);
  }
  free(t->in_names);
  free(t->out_names);
  free(t->inputs);
  free(t->outputs);
  if (t->options)
    ort->ReleaseRunOptions(t->options);
  free(t->diagnostic);
  session_release(t->s);
  free(t);
}
static void cancel_locked(struct LibOnnxTask *t, int code) {
  if (t->state == 2 || t->cancel_code)
    return;
  t->cancel_code = code;
  /* Every accepted task has its own RunOptions, including the dequeue gap. */
  OrtStatus *st = ort->RunOptionsSetTerminate(t->options);
  if (st) { /* Cancellation remains requested, never pretend hardware stopped.
             */
    t->cancel_ort_code = ort->GetErrorCode(st);
    ort->ReleaseStatus(st);
  }
}
void lib_onnx_runtime_begin_close(LibOnnxRuntimeHandle rt) {
  if (!rt)
    return;
  pthread_mutex_lock(&rt->mu);
  rt->closing = 1;
  for (struct LibOnnxTask *t = rt->live; t; t = t->live_next)
    cancel_locked(t, LIB_ONNX_ERR_CLOSED);
  pthread_cond_broadcast(&rt->cv);
  pthread_mutex_unlock(&rt->mu);
}
void lib_onnx_session_begin_close(LibOnnxSessionHandle s) {
  if (!s)
    return;
  pthread_mutex_lock(&s->rt->mu);
  s->closing = 1;
  for (struct LibOnnxTask *t = s->rt->live; t; t = t->live_next)
    if (t->s == s)
      cancel_locked(t, LIB_ONNX_ERR_CLOSED);
  pthread_cond_broadcast(&s->rt->cv);
  pthread_mutex_unlock(&s->rt->mu);
}
void lib_onnx_session_destroy(LibOnnxSessionHandle s) {
  if (!s)
    return;
  lib_onnx_session_begin_close(s);
  pthread_mutex_lock(&s->rt->mu);
  while (s->pending)
    pthread_cond_wait(&s->rt->cv, &s->rt->mu);
  pthread_mutex_unlock(&s->rt->mu);
  session_release(s);
}
static void *worker_loop(void *arg);
static void *monitor_loop(void *arg);
LibOnnxRuntimeHandle lib_onnx_runtime_create(int32_t concurrency) {
  pthread_once(&ort_once, init_ort);
  if (!ort) {
    error(1, "runtime.init", "ORT API unavailable");
    return NULL;
  }
  if (concurrency < 1 || concurrency > 64) {
    error(2, "runtime.init", "maxConcurrency must be 1..64");
    return NULL;
  }
  struct LibOnnxRuntime *rt = calloc(1, sizeof(*rt));
  if (!rt) {
    oom();
    return NULL;
  }
  atomic_init(&rt->refs, 1);
  if (pthread_mutex_init(&rt->mu, NULL)) {
    free(rt);
    error(1, "runtime.init", "Mutex initialization failed");
    return NULL;
  }
  if (pthread_cond_init(&rt->cv, NULL)) {
    pthread_mutex_destroy(&rt->mu);
    free(rt);
    error(1, "runtime.init", "Condition initialization failed");
    return NULL;
  }
  if (status_error(
          ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "lib_onnx", &rt->env),
          "runtime.init"))
    goto fail;
  rt->workers = calloc((size_t)concurrency, sizeof(pthread_t));
  if (!rt->workers) {
    oom();
    goto fail;
  }
  for (int i = 0; i < concurrency; i++) {
    if (pthread_create(&rt->workers[i], NULL, worker_loop, rt)) {
      error(1, "runtime.thread", "Worker creation failed");
      goto fail;
    }
    rt->nworkers++;
  }
  if (pthread_create(&rt->monitor, NULL, monitor_loop, rt)) {
    error(1, "runtime.thread", "Deadline monitor creation failed");
    goto fail;
  }
  rt->monitor_started = 1;
  return rt;
fail:
  lib_onnx_runtime_begin_close(rt);
  for (int i = 0; i < rt->nworkers; i++)
    pthread_join(rt->workers[i], NULL);
  if (rt->env)
    ort->ReleaseEnv(rt->env);
  free(rt->workers);
  pthread_cond_destroy(&rt->cv);
  pthread_mutex_destroy(&rt->mu);
  free(rt);
  return NULL;
}
void lib_onnx_runtime_destroy(LibOnnxRuntimeHandle rt) {
  if (!rt)
    return;
  lib_onnx_runtime_begin_close(rt);
  for (int i = 0; i < rt->nworkers; i++)
    pthread_join(rt->workers[i], NULL);
  if (rt->monitor_started)
    pthread_join(rt->monitor, NULL);
  runtime_release(rt);
}
int32_t lib_onnx_runtime_is_provider_supported(LibOnnxRuntimeHandle rt,
                                               const char *name) {
  (void)rt;
  pthread_once(&ort_once, init_ort);
  if (!ort || !name)
    return 0;
  char **providers = NULL;
  int n = 0;
  if (status_error(ort->GetAvailableProviders(&providers, &n),
                   "provider.available"))
    return -1;
  char full[96];
  snprintf(full, sizeof(full), "%sExecutionProvider", name);
  int found = 0;
  for (int i = 0; i < n; i++)
    if (!strcasecmp(name, providers[i]) || !strcasecmp(full, providers[i]))
      found = 1;
  if (status_error(ort->ReleaseAvailableProviders(providers, n),
                   "provider.release"))
    return -1;
  return found;
}
int32_t lib_onnx_runtime_get_available_providers(LibOnnxRuntimeHandle rt,
                                                 char *buf, size_t cap) {
  (void)rt;
  if (!buf || !cap)
    return -1;
  buf[0] = 0;
  char **names = NULL;
  int n = 0;
  if (status_error(ort->GetAvailableProviders(&names, &n),
                   "provider.available"))
    return -1;
  size_t needed = 1;
  for (int i = 0; i < n; i++)
    needed += strlen(names[i]) + (i ? 1 : 0);
  if (needed > cap) {
    status_error(ort->ReleaseAvailableProviders(names, n), "provider.release");
    error(2, "provider.available", "Provider buffer too short");
    return -1;
  }
  for (int i = 0; i < n; i++) {
    if (i)
      strcat(buf, ",");
    strcat(buf, names[i]);
  }
  if (status_error(ort->ReleaseAvailableProviders(names, n),
                   "provider.release"))
    return -1;
  return n;
}
static int metadata(OrtSession *session, int output, Metadata *m) {
  size_t count = 0;
  OrtAllocator *a = NULL;
  if (status_error(output ? ort->SessionGetOutputCount(session, &count)
                          : ort->SessionGetInputCount(session, &count),
                   "metadata.count"))
    return 1;
  if (count > INT32_MAX) {
    error(2, "metadata.count", "Too many model inputs/outputs");
    return 1;
  }
  m->count = (int)count;
  m->names = calloc(count ? count : 1, sizeof(char *));
  m->types = calloc(count ? count : 1, sizeof(int));
  m->ranks = calloc(count ? count : 1, sizeof(int));
  m->shapes = calloc(count ? count : 1, sizeof(int64_t *));
  if (!m->names || !m->types || !m->ranks || !m->shapes)
    return oom();
  if (status_error(ort->GetAllocatorWithDefaultOptions(&a),
                   "metadata.allocator"))
    return 1;
  for (int i = 0; i < m->count; i++) {
    OrtTypeInfo *info = NULL;
    char *name = NULL;
    if (status_error(output ? ort->SessionGetOutputName(session, i, a, &name)
                            : ort->SessionGetInputName(session, i, a, &name),
                     "metadata.name"))
      return 1;
    m->names[i] = strdup(name);
    a->Free(a, name);
    if (!m->names[i])
      return oom();
    if (status_error(output ? ort->SessionGetOutputTypeInfo(session, i, &info)
                            : ort->SessionGetInputTypeInfo(session, i, &info),
                     "metadata.type"))
      return 1;
    ONNXType kind = ONNX_TYPE_UNKNOWN;
    const OrtTensorTypeAndShapeInfo *shape = NULL;
    if (status_error(ort->GetOnnxTypeFromTypeInfo(info, &kind),
                     "metadata.type"))
      goto item_fail;
    if (kind != ONNX_TYPE_TENSOR) {
      m->ranks[i] = -1;
      ort->ReleaseTypeInfo(info);
      continue;
    }
    if (status_error(ort->CastTypeInfoToTensorInfo(info, &shape),
                     "metadata.tensor"))
      goto item_fail;
    ONNXTensorElementDataType dt;
    size_t rank = 0;
    if (status_error(ort->GetTensorElementType(shape, &dt), "metadata.dtype"))
      goto item_fail;
    m->types[i] = dtype_from_ort(dt);
    if (status_error(ort->GetDimensionsCount(shape, &rank), "metadata.rank"))
      goto item_fail;
    if (rank > INT32_MAX || rank > SIZE_MAX / sizeof(int64_t)) {
      error(2, "metadata.rank", "Rank overflow");
      goto item_fail;
    }
    m->ranks[i] = (int)rank;
    if (rank) {
      m->shapes[i] = malloc(rank * sizeof(int64_t));
      if (!m->shapes[i]) {
        oom();
        goto item_fail;
      }
      if (status_error(ort->GetDimensions(shape, m->shapes[i], rank),
                       "metadata.shape"))
        goto item_fail;
    }
    ort->ReleaseTypeInfo(info);
    continue;
  item_fail:
    ort->ReleaseTypeInfo(info);
    return 1;
  }
  return 0;
}
static const char *option(const char **keys, const char **vals, int n,
                          const char *key) {
  for (int i = 0; i < n; i++)
    if (!strcmp(keys[i], key))
      return vals[i];
  return NULL;
}
static int int_option(const char **keys, const char **vals, int n,
                      const char *key, int def, int min, int max, int *dest) {
  const char *v = option(keys, vals, n, key);
  if (!v) {
    *dest = def;
    return 0;
  }
  char *end;
  long x = strtol(v, &end, 10);
  if (!*v || *end || x < min || x > max)
    return error(2, "session.options", "Invalid integer option");
  *dest = (int)x;
  return 0;
}
static void append_csv(char *buf, size_t cap, const char *value) {
  size_t used = strlen(buf);
  snprintf(buf + used, cap - used, "%s%s", used ? "," : "", value);
}
LibOnnxSessionHandle lib_onnx_session_create(
    LibOnnxRuntimeHandle rt, const char *path, const void *bytes, size_t length,
    const char **providers, int32_t np, const char **keys, const char **vals,
    int32_t no, int32_t threads, int32_t fallback, int32_t hardware) {
  lib_onnx_clear_last_error();
  if (!rt || (!path && (!bytes || !length)) || np < 0 || no < 0 ||
      (np && !providers) || (no && (!keys || !vals)) || threads < 1) {
    error(2, "session.arguments", "Invalid session arguments");
    return NULL;
  }
  for (int i = 0; i < no; i++)
    if (!keys[i] || !vals[i]) {
      error(2, "session.options", "Null option");
      return NULL;
    }
  for (int i = 0; i < no; i++)
    for (int j = 0; j < i; j++)
      if (!strcmp(keys[i], keys[j])) {
        error(2, "session.options", "Duplicate provider/session option key");
        return NULL;
      }
  if (fallback < 0 || fallback > 2 || hardware < 0 || hardware > 2) {
    error(2, "policy", "Invalid policy enum");
    return NULL;
  }
  if (fallback == 2) {
    error(12, "policy",
          "UnsupportedPolicy: reliable whitelist validation of fused "
          "nodes/subgraphs is unavailable");
    return NULL;
  }
  if (hardware) {
    error(
        6, "policy.hardware",
        "HardwareVerificationUnavailable: no per-run physical GPU/NPU "
        "telemetry is available; use hardwareRequirement.any for a preference");
    return NULL;
  }
  pthread_mutex_lock(&rt->mu);
  if (rt->closing) {
    pthread_mutex_unlock(&rt->mu);
    error(10, "session.create", "Runtime closed");
    return NULL;
  }
  atomic_fetch_add(&rt->refs, 1);
  pthread_mutex_unlock(&rt->mu);
  struct LibOnnxSession *s = calloc(1, sizeof(*s));
  OrtSessionOptions *opts = NULL;
  if (!s) {
    runtime_release(rt);
    oom();
    return NULL;
  }
  s->rt = rt;
  atomic_init(&s->refs, 1);
  s->fallback = fallback;
  s->cpu_only = 1;
  if (int_option(keys, vals, no, "bridge.max_queue", 16, 0, 65536,
                 &s->max_queue) ||
      int_option(keys, vals, no, "bridge.max_active", 1, 1, 64, &s->max_active))
    goto fail;
  if (status_error(ort->CreateSessionOptions(&opts), "session.options"))
    goto fail;
  if (status_error(ort->SetIntraOpNumThreads(opts, threads), "session.threads"))
    goto fail;
  int inter, level, pattern;
  if (int_option(keys, vals, no, "session.inter_threads", 1, 1, 65536,
                 &inter) ||
      int_option(keys, vals, no, "session.optimization", 99, 0, 99, &level) ||
      int_option(keys, vals, no, "session.memory_pattern", 1, 0, 1, &pattern))
    goto fail;
  if (level != 0 && level != 1 && level != 2 && level != 99) {
    error(2, "session.options", "Invalid graph optimization level");
    goto fail;
  }
  if (status_error(ort->SetInterOpNumThreads(opts, inter), "session.threads") ||
      status_error(ort->SetSessionGraphOptimizationLevel(
                       opts, (GraphOptimizationLevel)level),
                   "session.optimization") ||
      status_error(pattern ? ort->EnableMemPattern(opts)
                           : ort->DisableMemPattern(opts),
                   "session.memory"))
    goto fail;
  const char *log = option(keys, vals, no, "session.log_id");
  if (log && status_error(ort->SetSessionLogId(opts, log), "session.logging"))
    goto fail;
  int configured = 0;
  for (int p = 0; p < np; p++) {
    const char *name = providers[p];
    if (!name) {
      error(2, "provider", "Null provider name");
      goto fail;
    }
    for (int q = 0; q < p; q++)
      if (!strcmp(name, providers[q])) {
        error(2, "provider", "Duplicate provider");
        goto fail;
      }
    if (!strcmp(name, "CPU")) {
      if (p != np - 1) {
        error(2, "policy.priority",
              "CPU EP must be last in graph allocation priority");
        goto fail;
      }
      if (fallback == 1 && np > 1) {
        error(5, "policy",
              "Strict accelerator policy conflicts with explicit CPU EP");
        goto fail;
      }
      continue;
    }
    if (strcmp(name, "CoreML") && strcmp(name, "QNN")) {
      error(4, "provider",
            "Only CPU, CoreML and QNN are supported by this bridge");
      goto fail;
    }
    s->accelerators_requested = 1;
    const char *prefix = !strcmp(name, "CoreML") ? "coreml." : "qnn.";
    size_t plen = strlen(prefix);
    const char **pk = calloc((size_t)no + 1, sizeof(char *));
    const char **pv = calloc((size_t)no + 1, sizeof(char *));
    if (!pk || !pv) {
      free(pk);
      free(pv);
      oom();
      goto fail;
    }
    size_t n = 0;
    for (int i = 0; i < no; i++)
      if (!strncmp(keys[i], prefix, plen)) {
        pk[n] = keys[i] + plen;
        pv[n++] = vals[i];
      }
    if (!strcmp(name, "QNN")) {
      const char *backend = option(keys, vals, no, "qnn.backend_path");
      const char *base = backend ? strrchr(backend, '/') : NULL;
      base = base ? base + 1 : backend;
      if (!base || strcmp(base, "libQnnHtp.so")) {
        free(pk);
        free(pv);
        error(2, "qnn.backend",
              "QNN HTP requires backend_path ending in libQnnHtp.so, not a "
              "CPU/GPU backend");
        goto fail;
      }
    }
    OrtStatus *st =
        ort->SessionOptionsAppendExecutionProvider(opts, name, pk, pv, n);
    free(pk);
    free(pv);
    if (st) {
      status_error(st,
                   !strcmp(name, "QNN") ? "qnn.initialize" : "coreml.register");
      size_t used = strlen(s->initialization);
      snprintf(s->initialization + used, sizeof(s->initialization) - used,
               "%s%s initialization failed (ORT %d): %.800s", used ? "; " : "",
               name, last_error.ort_code, last_error.message);
      if (fallback == 1) {
        last_error.code = 4;
        goto fail;
      }
      /* Explicit allowed policy permits initialization fallback; diagnostics
       * retain that fact rather than claiming EP assignment or physical
       * utilization. */
      continue;
    }
    configured++;
    s->cpu_only = 0;
    append_csv(s->registered, sizeof(s->registered), name);
  }
  if (!configured)
    strcpy(s->registered, "CPU");
  else if (fallback == 0)
    append_csv(s->registered, sizeof(s->registered), "CPU");
  if (configured && fallback == 1 &&
      status_error(ort->AddSessionConfigEntry(
                       opts, "session.disable_cpu_ep_fallback", "1"),
                   "policy.cpu_fallback"))
    goto fail;
  OrtStatus *st = path ? ort->CreateSession(rt->env, path, opts, &s->value)
                       : ort->CreateSessionFromArray(rt->env, bytes, length,
                                                     opts, &s->value);
  if (status_error(st, "session.load")) {
    if (configured && fallback == 1)
      last_error.code = 5;
    else
      last_error.code = 3;
    goto fail;
  }
  ort->ReleaseSessionOptions(opts);
  opts = NULL;
  if (metadata(s->value, 0, &s->inputs) || metadata(s->value, 1, &s->outputs))
    goto fail;
  pthread_mutex_lock(&rt->mu);
  int closing = rt->closing;
  pthread_mutex_unlock(&rt->mu);
  if (closing) {
    error(10, "session.load", "Runtime closed during load");
    goto fail;
  }
  lib_onnx_clear_last_error();
  return s;
fail:
  if (opts)
    ort->ReleaseSessionOptions(opts);
  session_release(s);
  return NULL;
}
#define META_ACCESSORS(kind, member)                                           \
  int32_t lib_onnx_session_get_##kind##_count(LibOnnxSessionHandle s) {        \
    return s ? s->member.count : 0;                                            \
  }                                                                            \
  const char *lib_onnx_session_get_##kind##_name(LibOnnxSessionHandle s,       \
                                                 int32_t i) {                  \
    return s && i >= 0 && i < s->member.count ? s->member.names[i] : NULL;     \
  }                                                                            \
  int32_t lib_onnx_session_get_##kind##_type(LibOnnxSessionHandle s,           \
                                             int32_t i) {                      \
    return s && i >= 0 && i < s->member.count ? s->member.types[i] : 0;        \
  }                                                                            \
  int32_t lib_onnx_session_get_##kind##_shape(                                 \
      LibOnnxSessionHandle s, int32_t i, int64_t *dst, int32_t cap) {          \
    if (!s || i < 0 || i >= s->member.count)                                   \
      return -1;                                                               \
    int rank = s->member.ranks[i];                                             \
    if (!dst)                                                                  \
      return rank;                                                             \
    if (cap < rank)                                                            \
      return -1;                                                               \
    if (rank > 0)                                                              \
      memcpy(dst, s->member.shapes[i], (size_t)rank * sizeof(int64_t));        \
    return rank;                                                               \
  }
META_ACCESSORS(input, inputs)
META_ACCESSORS(output, outputs)

LibOnnxTaskHandle lib_onnx_session_submit(LibOnnxSessionHandle s,
                                          int64_t request, const char **names,
                                          LibOnnxTensorHandle *tensors,
                                          int32_t ni, const char **out,
                                          int32_t no, int64_t timeout) {
  lib_onnx_clear_last_error();
  if (!s || ni < 0 || no < 1 || (ni && (!names || !tensors)) || !out ||
      timeout < 0) {
    error(2, "submit", "Invalid task arguments");
    return NULL;
  }
  struct LibOnnxRuntime *rt = s->rt;
  pthread_mutex_lock(&rt->mu);
  if (rt->closing || s->closing) {
    error(10, "submit", "Runtime or session is closed");
    pthread_mutex_unlock(&rt->mu);
    return NULL;
  }
  if (rt->pending >= rt->nworkers + 64 ||
      s->pending >= s->max_active + s->max_queue) {
    error(9, "submit", "Session bounded queue is full");
    pthread_mutex_unlock(&rt->mu);
    return NULL;
  }
  /* Reserve bounded admission and a Session reference, then copy outside the
   * scheduler mutex so control/close never waits behind large input copies. */
  rt->pending++;
  s->pending++;
  atomic_fetch_add(&s->refs, 1);
  pthread_mutex_unlock(&rt->mu);
  struct LibOnnxTask *t = calloc(1, sizeof(*t));
  if (!t) {
    oom();
    goto failed_reservation;
  }
  atomic_init(&t->refs, 1);
  t->s = s;
  t->id = atomic_fetch_add(&next_task_id, 1);
  t->request_id = request;
  t->ni = ni;
  t->no = no;
  int64_t accepted = now_us();
  if (timeout > INT64_MAX - accepted) {
    error(2, "submit.timeout", "Timeout overflow");
    goto failed;
  }
  t->deadline = timeout ? accepted + timeout : 0;
  t->in_names = calloc(ni ? ni : 1, sizeof(char *));
  t->inputs = calloc(ni ? ni : 1, sizeof(LibOnnxTensorHandle));
  t->out_names = calloc((size_t)no, sizeof(char *));
  t->outputs = calloc((size_t)no, sizeof(LibOnnxTensorHandle));
  if (!t->in_names || !t->inputs || !t->out_names || !t->outputs) {
    oom();
    goto failed;
  }
  if (status_error(ort->CreateRunOptions(&t->options), "submit.run_options"))
    goto failed;
  for (int i = 0; i < ni; i++) {
    if (!names[i] || !tensors[i]) {
      error(2, "submit.inputs", "Null input name/tensor");
      goto failed;
    }
    t->in_names[i] = strdup(names[i]);
    if (!t->in_names[i]) {
      oom();
      goto failed;
    }
    LibOnnxTensorHandle in = tensors[i];
    t->inputs[i] = lib_onnx_tensor_create(in->dtype, in->shape, in->rank,
                                          in->data, in->bytes, 1);
    if (!t->inputs[i])
      goto failed;
  }
  for (int i = 0; i < no; i++) {
    if (!out[i]) {
      error(2, "submit.outputs", "Null output name");
      goto failed;
    }
    t->out_names[i] = strdup(out[i]);
    if (!t->out_names[i]) {
      oom();
      goto failed;
    }
  }
  pthread_mutex_lock(&rt->mu);
  if (rt->closing || s->closing) {
    pthread_mutex_unlock(&rt->mu);
    error(10, "submit", "Closed during input preparation");
    goto failed;
  }
  atomic_fetch_add(&t->refs, 1);
  if (rt->tail)
    rt->tail->next = t;
  else
    rt->head = t;
  rt->tail = t;
  t->live_next = rt->live;
  rt->live = t;
  s->queued++;
  pthread_cond_broadcast(&rt->cv);
  pthread_mutex_unlock(&rt->mu);
  return t;
failed:
  /* Release pending reservation before its last Session reference. */
  pthread_mutex_lock(&rt->mu);
  rt->pending--;
  s->pending--;
  pthread_cond_broadcast(&rt->cv);
  pthread_mutex_unlock(&rt->mu);
  lib_onnx_task_release(t);
  return NULL;
failed_reservation:
  pthread_mutex_lock(&rt->mu);
  rt->pending--;
  s->pending--;
  pthread_cond_broadcast(&rt->cv);
  pthread_mutex_unlock(&rt->mu);
  session_release(s);
  return NULL;
}
static LibOnnxTensorHandle wrap_output(OrtValue *value) {
  int is_tensor = 0;
  OrtTensorTypeAndShapeInfo *info = NULL;
  struct LibOnnxTensor *t = NULL;
  if (status_error(ort->IsTensor(value, &is_tensor), "output.kind"))
    return NULL;
  if (!is_tensor) {
    error(11, "output.kind", "Unsupported non-Tensor OrtValue");
    return NULL;
  }
  if (status_error(ort->GetTensorTypeAndShape(value, &info), "output.type"))
    return NULL;
  t = calloc(1, sizeof(*t));
  if (!t) {
    oom();
    goto fail;
  }
  ONNXTensorElementDataType dt;
  size_t rank = 0, ort_count = 0, ort_bytes = 0;
  if (status_error(ort->GetTensorElementType(info, &dt), "output.dtype") ||
      status_error(ort->GetDimensionsCount(info, &rank), "output.rank"))
    goto fail;
  t->dtype = dtype_from_ort(dt);
  if (rank > INT32_MAX || rank > SIZE_MAX / sizeof(int64_t)) {
    error(2, "output.rank", "Output rank overflow");
    goto fail;
  }
  t->rank = (int)rank;
  if (rank) {
    t->shape = malloc(rank * sizeof(int64_t));
    if (!t->shape) {
      oom();
      goto fail;
    }
    if (status_error(ort->GetDimensions(info, t->shape, rank), "output.shape"))
      goto fail;
  }
  if (tensor_size(t->dtype, t->shape, t->rank, &t->count, &t->bytes) ||
      status_error(ort->GetTensorShapeElementCount(info, &ort_count),
                   "output.elements") ||
      status_error(ort->GetTensorSizeInBytes(value, &ort_bytes),
                   "output.bytes"))
    goto fail;
  if (ort_count != (size_t)t->count || ort_bytes != t->bytes) {
    error(2, "output.buffer", "ORT output buffer does not match shape/type");
    goto fail;
  }
  if (status_error(ort->GetTensorMutableData(value, &t->data), "output.data"))
    goto fail;
  if (t->bytes && !t->data) {
    error(2, "output.data", "Null output buffer");
    goto fail;
  }
  ort->ReleaseTensorTypeAndShapeInfo(info);
  t->value = value;
  return t;
fail:
  if (info)
    ort->ReleaseTensorTypeAndShapeInfo(info);
  lib_onnx_tensor_destroy(t);
  return NULL;
}
#ifdef LIB_ONNX_TESTING
/* Test-only native latch. Never included in distributed hook builds. */
static atomic_int test_barrier = 0;
static atomic_int test_waiting = 0;
int32_t lib_onnx_test_barrier_waiting(void) {
  return atomic_load(&test_waiting);
}
int32_t lib_onnx_test_cancel_code(LibOnnxTaskHandle t) {
  pthread_mutex_lock(&t->s->rt->mu);
  int v = t->cancel_code;
  pthread_mutex_unlock(&t->s->rt->mu);
  return v;
}
void lib_onnx_test_barrier(int32_t hold) { atomic_store(&test_barrier, hold); }
int32_t lib_onnx_test_task_in_run(LibOnnxTaskHandle t) {
  pthread_mutex_lock(&t->s->rt->mu);
  int v = t->in_run;
  pthread_mutex_unlock(&t->s->rt->mu);
  return v;
}
int32_t lib_onnx_test_task_at_barrier(LibOnnxTaskHandle t) {
  pthread_mutex_lock(&t->s->rt->mu);
  int v = t->at_barrier;
  pthread_mutex_unlock(&t->s->rt->mu);
  return v;
}
#endif
static void execute(struct LibOnnxTask *t) {
  const OrtValue **inputs = calloc(t->ni ? t->ni : 1, sizeof(OrtValue *));
  OrtValue **outputs = calloc((size_t)t->no, sizeof(OrtValue *));
  if (!inputs || !outputs) {
    oom();
    goto done;
  }
  for (int i = 0; i < t->ni; i++)
    inputs[i] = t->inputs[i]->value;
#ifdef LIB_ONNX_TESTING
  pthread_mutex_lock(&t->s->rt->mu);
  t->in_run = 1;
  pthread_mutex_unlock(&t->s->rt->mu);
#endif
  if (status_error(ort->Run(t->s->value, t->options,
                            (const char *const *)t->in_names, inputs,
                            (size_t)t->ni, (const char *const *)t->out_names,
                            (size_t)t->no, outputs),
                   "run"))
    goto done;
  for (int i = 0; i < t->no; i++) {
    t->outputs[i] = wrap_output(outputs[i]);
    if (!t->outputs[i])
      goto done;
    outputs[i] = NULL;
  }
  t->diagnostic = calloc(1, sizeof(*t->diagnostic));
  if (!t->diagnostic) {
    oom();
    goto done;
  }
  snprintf(t->diagnostic->registered, sizeof(t->diagnostic->registered), "%s",
           t->s->registered);
  snprintf(t->diagnostic->details, sizeof(t->diagnostic->details),
           "Physical execution unobserved. Registration does not prove graph "
           "assignment or hardware telemetry.%s%s",
           t->s->initialization[0] ? " Initialization diagnostics: " : "",
           t->s->initialization);
  if (t->s->cpu_only) {
    strcpy(t->diagnostic->provider, "CPU");
    t->diagnostic->evidence = 1;
    t->diagnostic->fallback = t->s->accelerators_requested ? 1 : 0;
  } else {
    t->diagnostic->evidence = 2;
    t->diagnostic->fallback = t->s->fallback == 1 ? 0 : -1;
  }
  t->diagnostic->duration = now_us() - t->started;
done:
  if (outputs)
    for (int i = 0; i < t->no; i++)
      if (outputs[i])
        ort->ReleaseValue(outputs[i]);
  free(outputs);
  free(inputs);
  t->failure = last_error;
  t->status = last_error.code;
}
static void *worker_loop(void *arg) {
  struct LibOnnxRuntime *rt = arg;
  for (;;) {
    pthread_mutex_lock(&rt->mu);
    struct LibOnnxTask *t = NULL, *prev = NULL;
    for (;;) {
      for (t = rt->head, prev = NULL; t; prev = t, t = t->next)
        if (t->cancel_code || t->s->active < t->s->max_active)
          break;
      if (t)
        break;
      if (rt->closing && !rt->head) {
        pthread_mutex_unlock(&rt->mu);
        return NULL;
      }
      pthread_cond_wait(&rt->cv, &rt->mu);
    }
    if (prev)
      prev->next = t->next;
    else
      rt->head = t->next;
    if (rt->tail == t)
      rt->tail = prev;
    t->next = NULL;
    t->state = 1;
    t->s->queued--;
    t->s->active++;
    t->started = now_us();
    if (t->deadline && now_us() >= t->deadline)
      cancel_locked(t, 8);
#ifdef LIB_ONNX_TESTING
    t->at_barrier = 1;
    atomic_fetch_add(&test_waiting, 1);
    pthread_mutex_unlock(&rt->mu);
    while (atomic_load(&test_barrier)) {
      struct timespec pause = {0, 1000000};
      nanosleep(&pause, NULL);
    }
    pthread_mutex_lock(&rt->mu);
    t->at_barrier = 0;
    atomic_fetch_sub(&test_waiting, 1);
#endif
    int cancelled = t->cancel_code;
    pthread_mutex_unlock(&rt->mu);
    lib_onnx_clear_last_error();
    if (!cancelled)
      execute(t);
    pthread_mutex_lock(&rt->mu);
    if (t->deadline && now_us() >= t->deadline && !t->cancel_code)
      cancel_locked(t, 8);
    if (t->cancel_code) {
      if (t->cancel_ort_code)
        t->failure.ort_code = t->cancel_ort_code;
      t->status = t->cancel_code;
      t->failure.code = t->cancel_code;
      snprintf(t->failure.stage, sizeof(t->failure.stage), "run.cancel");
      snprintf(t->failure.message, sizeof(t->failure.message),
               "%s requested; native execution has now returned (EP "
               "interruption is cooperative)",
               t->cancel_code == 8    ? "Timeout"
               : t->cancel_code == 10 ? "Close"
                                      : "Cancellation");
    }
    if (t->status) {
      for (int i = 0; i < t->no; i++) {
        lib_onnx_tensor_destroy(t->outputs[i]);
        t->outputs[i] = NULL;
      }
      free(t->diagnostic);
      t->diagnostic = NULL;
    }
    struct LibOnnxTask **link = &rt->live;
    while (*link && *link != t)
      link = &(*link)->live_next;
    if (*link)
      *link = t->live_next;
    t->s->active--;
    t->s->pending--;
    rt->pending--;
    t->state = 2;
    pthread_cond_broadcast(&rt->cv);
    pthread_mutex_unlock(&rt->mu);
    lib_onnx_task_release(t);
  }
}
static void *monitor_loop(void *arg) {
  struct LibOnnxRuntime *rt = arg;
  pthread_mutex_lock(&rt->mu);
  while (!rt->closing || rt->live) {
    int64_t now = now_us();
    for (struct LibOnnxTask *t = rt->live; t; t = t->live_next)
      if (t->deadline && now >= t->deadline)
        cancel_locked(t, 8);
    pthread_cond_broadcast(&rt->cv);
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_nsec += 2000000;
    if (deadline.tv_nsec >= 1000000000) {
      deadline.tv_sec++;
      deadline.tv_nsec -= 1000000000;
    }
    /* Wait scheduling may use realtime, but all timeout decisions use
     * monotonic. */
    pthread_cond_timedwait(&rt->cv, &rt->mu, &deadline);
  }
  pthread_mutex_unlock(&rt->mu);
  return NULL;
}
int64_t lib_onnx_task_id(LibOnnxTaskHandle t) { return t ? t->id : 0; }
int32_t lib_onnx_task_state(LibOnnxTaskHandle t) {
  if (!t)
    return -1;
  pthread_mutex_lock(&t->s->rt->mu);
  int state = t->state;
  pthread_mutex_unlock(&t->s->rt->mu);
  return state;
}
int32_t lib_onnx_task_status(LibOnnxTaskHandle t) {
  return lib_onnx_task_state(t) == 2 ? t->status : -1;
}
int32_t lib_onnx_task_ort_code(LibOnnxTaskHandle t) {
  return lib_onnx_task_state(t) == 2 ? t->failure.ort_code : 0;
}
const char *lib_onnx_task_error(LibOnnxTaskHandle t) {
  return lib_onnx_task_state(t) == 2 ? t->failure.message : "";
}
const char *lib_onnx_task_stage(LibOnnxTaskHandle t) {
  return lib_onnx_task_state(t) == 2 ? t->failure.stage : "";
}
int32_t lib_onnx_task_cancel(LibOnnxTaskHandle t, int32_t reason) {
  if (!t || (reason != 7 && reason != 8))
    return 2;
  pthread_mutex_lock(&t->s->rt->mu);
  cancel_locked(t, reason);
  pthread_cond_broadcast(&t->s->rt->cv);
  pthread_mutex_unlock(&t->s->rt->mu);
  return 0;
}
LibOnnxTensorHandle lib_onnx_task_output(LibOnnxTaskHandle t, int32_t i) {
  return lib_onnx_task_state(t) == 2 && i >= 0 && i < t->no ? t->outputs[i]
                                                            : NULL;
}
LibOnnxDiagnosticReportHandle lib_onnx_task_diagnostic(LibOnnxTaskHandle t) {
  return lib_onnx_task_state(t) == 2 ? t->diagnostic : NULL;
}
int32_t lib_onnx_session_run_sync(LibOnnxSessionHandle s, int64_t req,
                                  const char **names,
                                  LibOnnxTensorHandle *tensors, int32_t ni,
                                  const char **out,
                                  LibOnnxTensorHandle *outputs, int32_t no,
                                  LibOnnxDiagnosticReportHandle *diag) {
  if (!outputs)
    return error(2, "run.sync", "Null output buffer");
  for (int i = 0; i < no; i++)
    outputs[i] = NULL;
  if (diag)
    *diag = NULL;
  struct LibOnnxTask *t =
      lib_onnx_session_submit(s, req, names, tensors, ni, out, no, 0);
  if (!t)
    return last_error.code;
  pthread_mutex_lock(&s->rt->mu);
  while (t->state != 2)
    pthread_cond_wait(&s->rt->cv, &s->rt->mu);
  int status = t->status;
  last_error = t->failure;
  if (!status) {
    for (int i = 0; i < no; i++) {
      outputs[i] = t->outputs[i];
      t->outputs[i] = NULL;
    }
    if (diag) {
      *diag = t->diagnostic;
      t->diagnostic = NULL;
    }
  }
  pthread_mutex_unlock(&s->rt->mu);
  lib_onnx_task_release(t);
  return status;
}
