#include "lib_onnx.h"
#include "onnxruntime_c_api.h"
#include "coreml_provider_factory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#define BRIDGE_VERSION_MAJOR 0
#define BRIDGE_VERSION_MINOR 1
#define BRIDGE_VERSION_PATCH 0
#define BRIDGE_VERSION_STRING "0.1.0"

static const OrtApi* g_ort = NULL;
static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;
static char g_last_error[1024] = {0};
static pthread_mutex_t g_error_mutex = PTHREAD_MUTEX_INITIALIZER;

static void set_last_error(const char* msg) {
  pthread_mutex_lock(&g_error_mutex);
  strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
  g_last_error[sizeof(g_last_error) - 1] = '\0';
  pthread_mutex_unlock(&g_error_mutex);
}

static int check_ort_status(OrtStatus* status) {
  if (status == NULL) return LIB_ONNX_OK;
  const char* msg = g_ort->GetErrorMessage(status);
  set_last_error(msg ? msg : "Unknown ONNX Runtime error");
  g_ort->ReleaseStatus(status);
  return LIB_ONNX_ERR_GENERIC;
}

static void ensure_ort_initialized(void) {
  pthread_mutex_lock(&g_init_mutex);
  if (g_ort == NULL) {
    const OrtApiBase* base = OrtGetApiBase();
    if (base != NULL) {
      g_ort = base->GetApi(ORT_API_VERSION);
    }
  }
  pthread_mutex_unlock(&g_init_mutex);
}

/* Version API */
LIB_ONNX_EXPORT int32_t lib_onnx_version_major(void) { return BRIDGE_VERSION_MAJOR; }
LIB_ONNX_EXPORT int32_t lib_onnx_version_minor(void) { return BRIDGE_VERSION_MINOR; }
LIB_ONNX_EXPORT int32_t lib_onnx_version_patch(void) { return BRIDGE_VERSION_PATCH; }
LIB_ONNX_EXPORT const char* lib_onnx_version_string(void) { return BRIDGE_VERSION_STRING; }

LIB_ONNX_EXPORT const char* lib_onnx_ort_version_string(void) {
  ensure_ort_initialized();
  if (g_ort && OrtGetApiBase()) {
    return OrtGetApiBase()->GetVersionString();
  }
  return "Unknown";
}

LIB_ONNX_EXPORT const char* lib_onnx_get_last_error(void) {
  pthread_mutex_lock(&g_error_mutex);
  const char* res = g_last_error;
  pthread_mutex_unlock(&g_error_mutex);
  return res;
}

LIB_ONNX_EXPORT void lib_onnx_clear_last_error(void) {
  pthread_mutex_lock(&g_error_mutex);
  g_last_error[0] = '\0';
  pthread_mutex_unlock(&g_error_mutex);
}

/* Worker Task for Async Runner */
typedef struct AsyncTask {
  int64_t run_id;
  struct LibOnnxSession* session;
  char** input_names;
  struct LibOnnxTensor** input_tensors;
  int32_t num_inputs;
  char** output_names;
  int32_t num_outputs;
  int64_t timeout_ms;
  LibOnnxCompletionCallback callback;
  void* user_data;
  int is_cancelled;
  struct AsyncTask* next;
} AsyncTask;

/* Runtime */
struct LibOnnxRuntime {
  OrtEnv* env;
  int32_t max_concurrency;
  pthread_t* worker_threads;
  int32_t num_workers;
  AsyncTask* task_queue_head;
  AsyncTask* task_queue_tail;
  int32_t task_queue_size;
  int32_t is_shutting_down;
  pthread_mutex_t queue_mutex;
  pthread_cond_t queue_cond;
};

static void* worker_thread_loop(void* arg);

LIB_ONNX_EXPORT LibOnnxRuntimeHandle lib_onnx_runtime_create(int32_t max_concurrency) {
  ensure_ort_initialized();
  if (g_ort == NULL) {
    set_last_error("Failed to acquire ONNX Runtime C API pointer");
    return NULL;
  }

  struct LibOnnxRuntime* rt = (struct LibOnnxRuntime*)calloc(1, sizeof(struct LibOnnxRuntime));
  if (!rt) return NULL;

  rt->max_concurrency = max_concurrency > 0 ? max_concurrency : 4;
  pthread_mutex_init(&rt->queue_mutex, NULL);
  pthread_cond_init(&rt->queue_cond, NULL);

  OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "lib_onnx", &rt->env);
  if (status != NULL) {
    check_ort_status(status);
    free(rt);
    return NULL;
  }

  // Start background worker pool
  rt->num_workers = rt->max_concurrency;
  rt->worker_threads = (pthread_t*)malloc(sizeof(pthread_t) * rt->num_workers);
  for (int i = 0; i < rt->num_workers; i++) {
    pthread_create(&rt->worker_threads[i], NULL, worker_thread_loop, rt);
  }

  return rt;
}

LIB_ONNX_EXPORT void lib_onnx_runtime_destroy(LibOnnxRuntimeHandle runtime) {
  if (!runtime) return;

  pthread_mutex_lock(&runtime->queue_mutex);
  runtime->is_shutting_down = 1;
  pthread_cond_broadcast(&runtime->queue_cond);
  pthread_mutex_unlock(&runtime->queue_mutex);

  for (int i = 0; i < runtime->num_workers; i++) {
    pthread_join(runtime->worker_threads[i], NULL);
  }
  free(runtime->worker_threads);

  // Drain remaining tasks
  AsyncTask* curr = runtime->task_queue_head;
  while (curr != NULL) {
    AsyncTask* next = curr->next;
    if (curr->callback) {
      curr->callback(curr->run_id, LIB_ONNX_ERR_CLOSED, "Runtime shutting down", NULL, 0, NULL, curr->user_data);
    }
    for (int i = 0; i < curr->num_inputs; i++) free(curr->input_names[i]);
    free(curr->input_names);
    free(curr->input_tensors);
    for (int i = 0; i < curr->num_outputs; i++) free(curr->output_names[i]);
    free(curr->output_names);
    free(curr);
    curr = next;
  }

  if (runtime->env) {
    g_ort->ReleaseEnv(runtime->env);
  }

  pthread_mutex_destroy(&runtime->queue_mutex);
  pthread_cond_destroy(&runtime->queue_cond);
  free(runtime);
}

LIB_ONNX_EXPORT int32_t lib_onnx_runtime_is_provider_supported(LibOnnxRuntimeHandle runtime, const char* provider_name) {
  (void)runtime;
  ensure_ort_initialized();
  if (!g_ort || !provider_name) return 0;

  char** providers = NULL;
  int num_providers = 0;
  OrtStatus* status = g_ort->GetAvailableProviders(&providers, &num_providers);
  if (status != NULL) {
    g_ort->ReleaseStatus(status);
    return 0;
  }

  int supported = 0;
  for (int i = 0; i < num_providers; i++) {
    if (strcasecmp(providers[i], provider_name) == 0 ||
        (strncasecmp(providers[i], provider_name, strlen(provider_name)) == 0)) {
      supported = 1;
      break;
    }
  }

  g_ort->ReleaseAvailableProviders(providers, num_providers);
  return supported;
}

LIB_ONNX_EXPORT int32_t lib_onnx_runtime_get_available_providers(LibOnnxRuntimeHandle runtime, char* buffer, size_t buffer_len) {
  (void)runtime;
  ensure_ort_initialized();
  if (!g_ort || !buffer || buffer_len == 0) return 0;

  char** providers = NULL;
  int num_providers = 0;
  OrtStatus* status = g_ort->GetAvailableProviders(&providers, &num_providers);
  if (status != NULL) {
    g_ort->ReleaseStatus(status);
    return 0;
  }

  buffer[0] = '\0';
  for (int i = 0; i < num_providers; i++) {
    if (i > 0) strncat(buffer, ",", buffer_len - strlen(buffer) - 1);
    strncat(buffer, providers[i], buffer_len - strlen(buffer) - 1);
  }

  g_ort->ReleaseAvailableProviders(providers, num_providers);
  return num_providers;
}

/* Tensor */
struct LibOnnxTensor {
  int32_t dtype;
  int64_t* shape;
  int32_t rank;
  void* data;
  size_t byte_length;
  int owns_data;
  OrtValue* ort_value;
};

static ONNXTensorElementDataType map_lib_dtype_to_ort(int32_t dtype) {
  switch (dtype) {
    case LIB_ONNX_DTYPE_FLOAT32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case LIB_ONNX_DTYPE_FLOAT16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case LIB_ONNX_DTYPE_INT64:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case LIB_ONNX_DTYPE_INT32:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case LIB_ONNX_DTYPE_INT8:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case LIB_ONNX_DTYPE_UINT8:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case LIB_ONNX_DTYPE_BOOL:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    default:                     return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

static int32_t map_ort_dtype_to_lib(ONNXTensorElementDataType ort_type) {
  switch (ort_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return LIB_ONNX_DTYPE_FLOAT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return LIB_ONNX_DTYPE_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return LIB_ONNX_DTYPE_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return LIB_ONNX_DTYPE_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return LIB_ONNX_DTYPE_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return LIB_ONNX_DTYPE_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return LIB_ONNX_DTYPE_BOOL;
    default:                                    return LIB_ONNX_DTYPE_UNKNOWN;
  }
}

LIB_ONNX_EXPORT LibOnnxTensorHandle lib_onnx_tensor_create(
    int32_t dtype,
    const int64_t* shape,
    int32_t rank,
    const void* data,
    size_t byte_length,
    int32_t copy_data
) {
  ensure_ort_initialized();
  if (!g_ort || !shape || rank <= 0 || !data || byte_length == 0) {
    set_last_error("Invalid arguments to lib_onnx_tensor_create");
    return NULL;
  }

  ONNXTensorElementDataType ort_type = map_lib_dtype_to_ort(dtype);
  if (ort_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
    set_last_error("Unsupported data type for tensor creation");
    return NULL;
  }

  struct LibOnnxTensor* t = (struct LibOnnxTensor*)calloc(1, sizeof(struct LibOnnxTensor));
  if (!t) return NULL;

  t->dtype = dtype;
  t->rank = rank;
  t->shape = (int64_t*)malloc(sizeof(int64_t) * rank);
  memcpy(t->shape, shape, sizeof(int64_t) * rank);
  t->byte_length = byte_length;

  if (copy_data) {
    t->data = malloc(byte_length);
    memcpy(t->data, data, byte_length);
    t->owns_data = 1;
  } else {
    t->data = (void*)data;
    t->owns_data = 0;
  }

  OrtMemoryInfo* mem_info = NULL;
  OrtStatus* status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
  if (status != NULL) {
    check_ort_status(status);
    if (t->owns_data) free(t->data);
    free(t->shape);
    free(t);
    return NULL;
  }

  status = g_ort->CreateTensorWithDataAsOrtValue(
      mem_info,
      t->data,
      t->byte_length,
      t->shape,
      t->rank,
      ort_type,
      &t->ort_value
  );
  g_ort->ReleaseMemoryInfo(mem_info);

  if (status != NULL) {
    check_ort_status(status);
    if (t->owns_data) free(t->data);
    free(t->shape);
    free(t);
    return NULL;
  }

  return t;
}

LIB_ONNX_EXPORT void lib_onnx_tensor_destroy(LibOnnxTensorHandle tensor) {
  if (!tensor) return;
  if (tensor->ort_value) {
    g_ort->ReleaseValue(tensor->ort_value);
  }
  if (tensor->owns_data && tensor->data) {
    free(tensor->data);
  }
  if (tensor->shape) {
    free(tensor->shape);
  }
  free(tensor);
}

LIB_ONNX_EXPORT int32_t lib_onnx_tensor_get_dtype(LibOnnxTensorHandle tensor) {
  return tensor ? tensor->dtype : LIB_ONNX_DTYPE_UNKNOWN;
}

LIB_ONNX_EXPORT int32_t lib_onnx_tensor_get_rank(LibOnnxTensorHandle tensor) {
  return tensor ? tensor->rank : 0;
}

LIB_ONNX_EXPORT int32_t lib_onnx_tensor_get_shape(LibOnnxTensorHandle tensor, int64_t* out_shape, int32_t max_rank) {
  if (!tensor || !out_shape) return 0;
  int32_t count = tensor->rank < max_rank ? tensor->rank : max_rank;
  for (int32_t i = 0; i < count; i++) {
    out_shape[i] = tensor->shape[i];
  }
  return count;
}

LIB_ONNX_EXPORT const void* lib_onnx_tensor_get_data(LibOnnxTensorHandle tensor) {
  return tensor ? tensor->data : NULL;
}

LIB_ONNX_EXPORT size_t lib_onnx_tensor_get_byte_length(LibOnnxTensorHandle tensor) {
  return tensor ? tensor->byte_length : 0;
}

LIB_ONNX_EXPORT int64_t lib_onnx_tensor_get_element_count(LibOnnxTensorHandle tensor) {
  if (!tensor || tensor->rank == 0) return 0;
  int64_t count = 1;
  for (int32_t i = 0; i < tensor->rank; i++) {
    count *= tensor->shape[i];
  }
  return count;
}

/* Diagnostic Report */
struct LibOnnxDiagnosticReport {
  char provider[64];
  int32_t evidence_level;
  int64_t duration_us;
  char hardware_details[256];
  char model_hash[65];
  int32_t fallback_occurred;
};

LIB_ONNX_EXPORT const char* lib_onnx_diagnostic_get_provider(LibOnnxDiagnosticReportHandle report) {
  return report ? report->provider : "Unknown";
}

LIB_ONNX_EXPORT int32_t lib_onnx_diagnostic_get_evidence_level(LibOnnxDiagnosticReportHandle report) {
  return report ? report->evidence_level : LIB_ONNX_EVIDENCE_UNKNOWN;
}

LIB_ONNX_EXPORT int64_t lib_onnx_diagnostic_get_duration_us(LibOnnxDiagnosticReportHandle report) {
  return report ? report->duration_us : 0;
}

LIB_ONNX_EXPORT const char* lib_onnx_diagnostic_get_hardware_details(LibOnnxDiagnosticReportHandle report) {
  return report ? report->hardware_details : "";
}

LIB_ONNX_EXPORT const char* lib_onnx_diagnostic_get_model_hash(LibOnnxDiagnosticReportHandle report) {
  return report ? report->model_hash : "";
}

LIB_ONNX_EXPORT int32_t lib_onnx_diagnostic_get_fallback_occurred(LibOnnxDiagnosticReportHandle report) {
  return report ? report->fallback_occurred : 0;
}

LIB_ONNX_EXPORT void lib_onnx_diagnostic_destroy(LibOnnxDiagnosticReportHandle report) {
  if (report) free(report);
}

/* Session */
struct LibOnnxSession {
  struct LibOnnxRuntime* runtime;
  OrtSession* session;
  char chosen_provider[64];
  int32_t fallback_policy;
  int32_t hardware_requirement;
  int32_t evidence_level;
  char hardware_details[256];
  char model_hash[65];

  // Cached input metadata
  int32_t input_count;
  char** input_names;
  int32_t* input_types;
  int64_t** input_shapes;
  int32_t* input_ranks;

  // Cached output metadata
  int32_t output_count;
  char** output_names;
  int32_t* output_types;
  int64_t** output_shapes;
  int32_t* output_ranks;

  // Active in-flight cancellation tracking
  pthread_mutex_t run_mutex;
  int64_t active_run_id;
  int active_cancelled;
  int is_closed;
};

/* Hardware check on Apple */
static void probe_apple_hardware(char* details, size_t len, int* has_ane, int* has_gpu) {
  *has_ane = 0;
  *has_gpu = 0;
#if defined(__APPLE__)
  char cpu_brand[128] = {0};
  size_t size = sizeof(cpu_brand);
  if (sysctlbyname("machdep.cpu.brand_string", cpu_brand, &size, NULL, 0) == 0) {
    snprintf(details, len, "%s", cpu_brand);
    if (strstr(cpu_brand, "Apple M") || strstr(cpu_brand, "Apple A")) {
      *has_ane = 1; // All Apple Silicon M/A chips have ANE
      *has_gpu = 1;
    }
  } else {
    snprintf(details, len, "Apple Hardware");
    *has_gpu = 1;
  }
#else
  snprintf(details, len, "Non-Apple Target");
#endif
}

LIB_ONNX_EXPORT LibOnnxSessionHandle lib_onnx_session_create(
    LibOnnxRuntimeHandle runtime,
    const char* model_path,
    const void* model_bytes,
    size_t model_bytes_len,
    const char** providers,
    int32_t num_providers,
    const char** option_keys,
    const char** option_values,
    int32_t num_options,
    int32_t thread_count,
    int32_t fallback_policy,
    int32_t hardware_requirement
) {
  ensure_ort_initialized();
  if (!runtime || (!model_path && !model_bytes)) {
    set_last_error("Invalid session creation parameters");
    return NULL;
  }

  OrtSessionOptions* session_options = NULL;
  OrtStatus* status = g_ort->CreateSessionOptions(&session_options);
  if (status != NULL) {
    check_ort_status(status);
    return NULL;
  }

  if (thread_count > 0) {
    g_ort->SetIntraOpNumThreads(session_options, thread_count);
  }

  struct LibOnnxSession* s = (struct LibOnnxSession*)calloc(1, sizeof(struct LibOnnxSession));
  if (!s) {
    g_ort->ReleaseSessionOptions(session_options);
    return NULL;
  }

  s->runtime = runtime;
  s->fallback_policy = fallback_policy;
  s->hardware_requirement = hardware_requirement;
  pthread_mutex_init(&s->run_mutex, NULL);

  int has_ane = 0, has_gpu = 0;
  probe_apple_hardware(s->hardware_details, sizeof(s->hardware_details), &has_ane, &has_gpu);

  // Configure Providers
  strcpy(s->chosen_provider, "CPU");
  s->evidence_level = LIB_ONNX_EVIDENCE_NONE;

  int accelerator_requested = 0;
  int accelerator_configured = 0;

  for (int p = 0; p < num_providers; p++) {
    const char* prov = providers[p];
    if (strcasecmp(prov, "CPU") != 0) {
      accelerator_requested = 1;
    }
    if (strcasecmp(prov, "CoreML") == 0) {
      uint32_t coreml_flags = COREML_FLAG_ENABLE_ON_SUBGRAPH | COREML_FLAG_CREATE_MLPROGRAM;

      // Parse custom CoreML options
      for (int o = 0; o < num_options; o++) {
        if (strcmp(option_keys[o], "coreml.MLComputeUnits") == 0) {
          if (strcasecmp(option_values[o], "CPUAndGPU") == 0) {
            coreml_flags |= COREML_FLAG_USE_CPU_AND_GPU;
          } else if (strcasecmp(option_values[o], "CPUOnly") == 0) {
            coreml_flags |= COREML_FLAG_USE_CPU_ONLY;
          }
        } else if (strcmp(option_keys[o], "coreml.RequireANE") == 0) {
          if (strcmp(option_values[o], "1") == 0) {
            coreml_flags |= COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE;
          }
        }
      }

      OrtStatus* c_status = OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml_flags);
      if (c_status == NULL) {
        strcpy(s->chosen_provider, "CoreML");
        s->evidence_level = LIB_ONNX_EVIDENCE_EP_COMPILED;
        accelerator_configured = 1;
        break; // Provider configured
      } else {
        g_ort->ReleaseStatus(c_status);
        if (fallback_policy == LIB_ONNX_FALLBACK_STRICT) {
          set_last_error("CoreML execution provider failed to initialize under strict fallback policy");
          g_ort->ReleaseSessionOptions(session_options);
          pthread_mutex_destroy(&s->run_mutex);
          free(s);
          return NULL;
        }
      }
    } else if (strcasecmp(prov, "QNN") == 0) {
      accelerator_requested = 1;
      // QNN provider option keys/values
      const char* qnn_keys[8];
      const char* qnn_vals[8];
      int qnn_opt_count = 0;
      for (int o = 0; o < num_options; o++) {
        if (strncmp(option_keys[o], "qnn.", 4) == 0 && qnn_opt_count < 8) {
          qnn_keys[qnn_opt_count] = option_keys[o] + 4;
          qnn_vals[qnn_opt_count] = option_values[o];
          qnn_opt_count++;
        }
      }
      OrtStatus* q_status = g_ort->SessionOptionsAppendExecutionProvider(
          session_options, "QNN", qnn_keys, qnn_vals, qnn_opt_count
      );
      if (q_status == NULL) {
        strcpy(s->chosen_provider, "QNN");
        s->evidence_level = LIB_ONNX_EVIDENCE_EP_COMPILED;
        accelerator_configured = 1;
        break;
      } else {
        g_ort->ReleaseStatus(q_status);
        if (fallback_policy == LIB_ONNX_FALLBACK_STRICT) {
          set_last_error("QNN execution provider failed to initialize under strict fallback policy");
          g_ort->ReleaseSessionOptions(session_options);
          pthread_mutex_destroy(&s->run_mutex);
          free(s);
          return NULL;
        }
      }
    }
  }

  if (accelerator_requested && !accelerator_configured && fallback_policy == LIB_ONNX_FALLBACK_STRICT) {
    set_last_error("Requested accelerator could not be configured under strict fallback policy");
    g_ort->ReleaseSessionOptions(session_options);
    pthread_mutex_destroy(&s->run_mutex);
    free(s);
    return NULL;
  }

  // Check hardware requirement
  if (hardware_requirement == LIB_ONNX_HARDWARE_NPU && !has_ane && accelerator_configured) {
    if (fallback_policy == LIB_ONNX_FALLBACK_STRICT) {
      set_last_error("Strict NPU/ANE hardware requirement requested but device has no verified NPU");
      g_ort->ReleaseSessionOptions(session_options);
      pthread_mutex_destroy(&s->run_mutex);
      free(s);
      return NULL;
    }
    s->evidence_level = LIB_ONNX_EVIDENCE_UNVERIFIABLE;
  }

  // Create OrtSession
  if (model_path != NULL) {
    status = g_ort->CreateSession(runtime->env, model_path, session_options, &s->session);
  } else {
    status = g_ort->CreateSessionFromArray(runtime->env, model_bytes, model_bytes_len, session_options, &s->session);
  }
  g_ort->ReleaseSessionOptions(session_options);

  if (status != NULL) {
    check_ort_status(status);
    pthread_mutex_destroy(&s->run_mutex);
    free(s);
    return NULL;
  }

  // Model graph compilation succeeded on EP
  if (accelerator_configured) {
    s->evidence_level = has_ane ? LIB_ONNX_EVIDENCE_HARDWARE_OBSERVED : LIB_ONNX_EVIDENCE_GRAPH_PARTITIONED;
  }

  // Query metadata: inputs
  OrtAllocator* allocator = NULL;
  g_ort->GetAllocatorWithDefaultOptions(&allocator);

  size_t in_count = 0;
  g_ort->SessionGetInputCount(s->session, &in_count);
  s->input_count = (int32_t)in_count;
  s->input_names = (char**)calloc(s->input_count, sizeof(char*));
  s->input_types = (int32_t*)calloc(s->input_count, sizeof(int32_t));
  s->input_shapes = (int64_t**)calloc(s->input_count, sizeof(int64_t*));
  s->input_ranks = (int32_t*)calloc(s->input_count, sizeof(int32_t));

  for (int32_t i = 0; i < s->input_count; i++) {
    char* name = NULL;
    g_ort->SessionGetInputName(s->session, i, allocator, &name);
    s->input_names[i] = strdup(name);
    allocator->Free(allocator, name);

    OrtTypeInfo* type_info = NULL;
    g_ort->SessionGetInputTypeInfo(s->session, i, &type_info);
    const OrtTensorTypeAndShapeInfo* shape_info = NULL;
    g_ort->CastTypeInfoToTensorInfo(type_info, &shape_info);

    ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    g_ort->GetTensorElementType(shape_info, &dt);
    s->input_types[i] = map_ort_dtype_to_lib(dt);

    size_t dim_count = 0;
    g_ort->GetDimensionsCount(shape_info, &dim_count);
    s->input_ranks[i] = (int32_t)dim_count;
    s->input_shapes[i] = (int64_t*)malloc(sizeof(int64_t) * dim_count);
    g_ort->GetDimensions(shape_info, s->input_shapes[i], dim_count);

    g_ort->ReleaseTypeInfo(type_info);
  }

  // Query metadata: outputs
  size_t out_count = 0;
  g_ort->SessionGetOutputCount(s->session, &out_count);
  s->output_count = (int32_t)out_count;
  s->output_names = (char**)calloc(s->output_count, sizeof(char*));
  s->output_types = (int32_t*)calloc(s->output_count, sizeof(int32_t));
  s->output_shapes = (int64_t**)calloc(s->output_count, sizeof(int64_t*));
  s->output_ranks = (int32_t*)calloc(s->output_count, sizeof(int32_t));

  for (int32_t i = 0; i < s->output_count; i++) {
    char* name = NULL;
    g_ort->SessionGetOutputName(s->session, i, allocator, &name);
    s->output_names[i] = strdup(name);
    allocator->Free(allocator, name);

    OrtTypeInfo* type_info = NULL;
    g_ort->SessionGetOutputTypeInfo(s->session, i, &type_info);
    const OrtTensorTypeAndShapeInfo* shape_info = NULL;
    g_ort->CastTypeInfoToTensorInfo(type_info, &shape_info);

    ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    g_ort->GetTensorElementType(shape_info, &dt);
    s->output_types[i] = map_ort_dtype_to_lib(dt);

    size_t dim_count = 0;
    g_ort->GetDimensionsCount(shape_info, &dim_count);
    s->output_ranks[i] = (int32_t)dim_count;
    s->output_shapes[i] = (int64_t*)malloc(sizeof(int64_t) * dim_count);
    g_ort->GetDimensions(shape_info, s->output_shapes[i], dim_count);

    g_ort->ReleaseTypeInfo(type_info);
  }

  return s;
}

LIB_ONNX_EXPORT void lib_onnx_session_destroy(LibOnnxSessionHandle session) {
  if (!session) return;

  pthread_mutex_lock(&session->run_mutex);
  session->is_closed = 1;
  session->active_cancelled = 1;
  pthread_mutex_unlock(&session->run_mutex);

  // Free inputs
  for (int32_t i = 0; i < session->input_count; i++) {
    free(session->input_names[i]);
    free(session->input_shapes[i]);
  }
  free(session->input_names);
  free(session->input_types);
  free(session->input_shapes);
  free(session->input_ranks);

  // Free outputs
  for (int32_t i = 0; i < session->output_count; i++) {
    free(session->output_names[i]);
    free(session->output_shapes[i]);
  }
  free(session->output_names);
  free(session->output_types);
  free(session->output_shapes);
  free(session->output_ranks);

  if (session->session) {
    g_ort->ReleaseSession(session->session);
  }

  pthread_mutex_destroy(&session->run_mutex);
  free(session);
}

LIB_ONNX_EXPORT int32_t lib_onnx_session_get_input_count(LibOnnxSessionHandle session) {
  return session ? session->input_count : 0;
}

LIB_ONNX_EXPORT const char* lib_onnx_session_get_input_name(LibOnnxSessionHandle session, int32_t index) {
  if (!session || index < 0 || index >= session->input_count) return NULL;
  return session->input_names[index];
}

LIB_ONNX_EXPORT int32_t lib_onnx_session_get_input_type(LibOnnxSessionHandle session, int32_t index) {
  if (!session || index < 0 || index >= session->input_count) return LIB_ONNX_DTYPE_UNKNOWN;
  return session->input_types[index];
}

LIB_ONNX_EXPORT int32_t lib_onnx_session_get_input_shape(LibOnnxSessionHandle session, int32_t index, int64_t* out_shape, int32_t max_rank) {
  if (!session || index < 0 || index >= session->input_count || !out_shape) return 0;
  int32_t count = session->input_ranks[index] < max_rank ? session->input_ranks[index] : max_rank;
  for (int32_t i = 0; i < count; i++) {
    out_shape[i] = session->input_shapes[index][i];
  }
  return count;
}

LIB_ONNX_EXPORT int32_t lib_onnx_session_get_output_count(LibOnnxSessionHandle session) {
  return session ? session->output_count : 0;
}

LIB_ONNX_EXPORT const char* lib_onnx_session_get_output_name(LibOnnxSessionHandle session, int32_t index) {
  if (!session || index < 0 || index >= session->output_count) return NULL;
  return session->output_names[index];
}

LIB_ONNX_EXPORT int32_t lib_onnx_session_get_output_type(LibOnnxSessionHandle session, int32_t index) {
  if (!session || index < 0 || index >= session->output_count) return LIB_ONNX_DTYPE_UNKNOWN;
  return session->output_types[index];
}

LIB_ONNX_EXPORT int32_t lib_onnx_session_get_output_shape(LibOnnxSessionHandle session, int32_t index, int64_t* out_shape, int32_t max_rank) {
  if (!session || index < 0 || index >= session->output_count || !out_shape) return 0;
  int32_t count = session->output_ranks[index] < max_rank ? session->output_ranks[index] : max_rank;
  for (int32_t i = 0; i < count; i++) {
    out_shape[i] = session->output_shapes[index][i];
  }
  return count;
}

/* Run execution */
static int64_t get_current_time_us(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return ((int64_t)tv.tv_sec * 1000000LL) + tv.tv_usec;
}

LIB_ONNX_EXPORT int32_t lib_onnx_session_run_sync(
    LibOnnxSessionHandle session,
    int64_t run_id,
    const char** input_names,
    LibOnnxTensorHandle* input_tensors,
    int32_t num_inputs,
    const char** output_names,
    LibOnnxTensorHandle* out_output_tensors,
    int32_t num_outputs,
    LibOnnxDiagnosticReportHandle* out_diagnostic
) {
  if (!session || session->is_closed) {
    set_last_error("Session is null or closed");
    return LIB_ONNX_ERR_CLOSED;
  }

  pthread_mutex_lock(&session->run_mutex);
  session->active_run_id = run_id;
  session->active_cancelled = 0;
  pthread_mutex_unlock(&session->run_mutex);

  // Prepare input OrtValues
  const OrtValue** in_values = (const OrtValue**)malloc(sizeof(OrtValue*) * num_inputs);
  for (int32_t i = 0; i < num_inputs; i++) {
    in_values[i] = input_tensors[i]->ort_value;
  }

  OrtValue** out_values = (OrtValue**)calloc(num_outputs, sizeof(OrtValue*));

  OrtRunOptions* run_opts = NULL;
  g_ort->CreateRunOptions(&run_opts);

  int64_t start_time = get_current_time_us();

  OrtStatus* status = g_ort->Run(
      session->session,
      run_opts,
      input_names,
      in_values,
      num_inputs,
      output_names,
      num_outputs,
      out_values
  );

  int64_t duration = get_current_time_us() - start_time;
  g_ort->ReleaseRunOptions(run_opts);
  free(in_values);

  pthread_mutex_lock(&session->run_mutex);
  int was_cancelled = session->active_cancelled;
  session->active_run_id = 0;
  pthread_mutex_unlock(&session->run_mutex);

  if (was_cancelled) {
    for (int32_t i = 0; i < num_outputs; i++) {
      if (out_values[i]) g_ort->ReleaseValue(out_values[i]);
    }
    free(out_values);
    if (status) g_ort->ReleaseStatus(status);
    set_last_error("Execution was cancelled");
    return LIB_ONNX_ERR_CANCELLED;
  }

  if (status != NULL) {
    for (int32_t i = 0; i < num_outputs; i++) {
      if (out_values[i]) g_ort->ReleaseValue(out_values[i]);
    }
    free(out_values);
    check_ort_status(status);
    return LIB_ONNX_ERR_GENERIC;
  }

  // Wrap output values into LibOnnxTensor
  for (int32_t i = 0; i < num_outputs; i++) {
    struct LibOnnxTensor* t = (struct LibOnnxTensor*)calloc(1, sizeof(struct LibOnnxTensor));
    t->ort_value = out_values[i];

    OrtTensorTypeAndShapeInfo* shape_info = NULL;
    g_ort->GetTensorTypeAndShape(out_values[i], &shape_info);

    ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    g_ort->GetTensorElementType(shape_info, &dt);
    t->dtype = map_ort_dtype_to_lib(dt);

    size_t dim_count = 0;
    g_ort->GetDimensionsCount(shape_info, &dim_count);
    t->rank = (int32_t)dim_count;
    t->shape = (int64_t*)malloc(sizeof(int64_t) * dim_count);
    g_ort->GetDimensions(shape_info, t->shape, dim_count);

    size_t elem_count = 0;
    g_ort->GetTensorShapeElementCount(shape_info, &elem_count);

    g_ort->GetTensorMutableData(out_values[i], &t->data);
    t->owns_data = 0; // Managed by OrtValue
    t->byte_length = elem_count * (t->dtype == LIB_ONNX_DTYPE_INT64 ? 8 : (t->dtype == LIB_ONNX_DTYPE_FLOAT16 ? 2 : 4));

    g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);
    out_output_tensors[i] = t;
  }
  free(out_values);

  // Diagnostic Report
  if (out_diagnostic != NULL) {
    struct LibOnnxDiagnosticReport* report = (struct LibOnnxDiagnosticReport*)calloc(1, sizeof(struct LibOnnxDiagnosticReport));
    strncpy(report->provider, session->chosen_provider, sizeof(report->provider) - 1);
    report->evidence_level = session->evidence_level;
    report->duration_us = duration;
    strncpy(report->hardware_details, session->hardware_details, sizeof(report->hardware_details) - 1);
    strncpy(report->model_hash, session->model_hash, sizeof(report->model_hash) - 1);
    report->fallback_occurred = (session->evidence_level == LIB_ONNX_EVIDENCE_NONE) ? 1 : 0;
    *out_diagnostic = report;
  }

  return LIB_ONNX_OK;
}

LIB_ONNX_EXPORT int32_t lib_onnx_session_cancel_run(LibOnnxSessionHandle session, int64_t run_id) {
  if (!session) return LIB_ONNX_ERR_INVALID_ARGUMENT;

  pthread_mutex_lock(&session->run_mutex);
  if (session->active_run_id == run_id) {
    session->active_cancelled = 1;
  }
  pthread_mutex_unlock(&session->run_mutex);

  // Also check queued tasks in runtime
  struct LibOnnxRuntime* rt = session->runtime;
  if (rt) {
    pthread_mutex_lock(&rt->queue_mutex);
    AsyncTask* curr = rt->task_queue_head;
    while (curr) {
      if (curr->run_id == run_id) {
        curr->is_cancelled = 1;
        break;
      }
      curr = curr->next;
    }
    pthread_mutex_unlock(&rt->queue_mutex);
  }

  return LIB_ONNX_OK;
}

/* Worker loop executing async tasks */
static void* worker_thread_loop(void* arg) {
  struct LibOnnxRuntime* rt = (struct LibOnnxRuntime*)arg;

  while (1) {
    pthread_mutex_lock(&rt->queue_mutex);
    while (rt->task_queue_head == NULL && !rt->is_shutting_down) {
      pthread_cond_wait(&rt->queue_cond, &rt->queue_mutex);
    }

    if (rt->is_shutting_down && rt->task_queue_head == NULL) {
      pthread_mutex_unlock(&rt->queue_mutex);
      break;
    }

    AsyncTask* task = rt->task_queue_head;
    if (task) {
      rt->task_queue_head = task->next;
      if (rt->task_queue_head == NULL) rt->task_queue_tail = NULL;
      rt->task_queue_size--;
    }
    pthread_mutex_unlock(&rt->queue_mutex);

    if (!task) continue;

    if (task->is_cancelled) {
      if (task->callback) {
        task->callback(task->run_id, LIB_ONNX_ERR_CANCELLED, "Task cancelled in queue", NULL, 0, NULL, task->user_data);
      }
    } else {
      LibOnnxTensorHandle* outputs = (LibOnnxTensorHandle*)calloc(task->num_outputs, sizeof(LibOnnxTensorHandle));
      LibOnnxDiagnosticReportHandle diag = NULL;

      int32_t res = lib_onnx_session_run_sync(
          task->session,
          task->run_id,
          (const char**)task->input_names,
          task->input_tensors,
          task->num_inputs,
          (const char**)task->output_names,
          outputs,
          task->num_outputs,
          &diag
      );

      if (task->callback) {
        task->callback(
            task->run_id,
            res,
            res == LIB_ONNX_OK ? NULL : lib_onnx_get_last_error(),
            outputs,
            task->num_outputs,
            diag,
            task->user_data
        );
      } else {
        for (int i = 0; i < task->num_outputs; i++) {
          if (outputs[i]) lib_onnx_tensor_destroy(outputs[i]);
        }
        if (diag) lib_onnx_diagnostic_destroy(diag);
      }
      free(outputs);
    }

    for (int i = 0; i < task->num_inputs; i++) free(task->input_names[i]);
    free(task->input_names);
    free(task->input_tensors);
    for (int i = 0; i < task->num_outputs; i++) free(task->output_names[i]);
    free(task->output_names);
    free(task);
  }

  return NULL;
}

LIB_ONNX_EXPORT int32_t lib_onnx_session_run_async(
    LibOnnxSessionHandle session,
    int64_t run_id,
    const char** input_names,
    LibOnnxTensorHandle* input_tensors,
    int32_t num_inputs,
    const char** output_names,
    int32_t num_outputs,
    int64_t timeout_ms,
    LibOnnxCompletionCallback callback,
    void* user_data
) {
  if (!session || session->is_closed || !session->runtime) {
    set_last_error("Session is closed or invalid");
    return LIB_ONNX_ERR_CLOSED;
  }

  struct LibOnnxRuntime* rt = session->runtime;
  pthread_mutex_lock(&rt->queue_mutex);

  if (rt->is_shutting_down) {
    pthread_mutex_unlock(&rt->queue_mutex);
    set_last_error("Runtime is shutting down");
    return LIB_ONNX_ERR_CLOSED;
  }

  // Bounded queue limit (e.g. 64)
  if (rt->task_queue_size >= 64) {
    pthread_mutex_unlock(&rt->queue_mutex);
    set_last_error("Session task queue is full (queue overflow)");
    return LIB_ONNX_ERR_QUEUE_OVERFLOW;
  }

  AsyncTask* task = (AsyncTask*)calloc(1, sizeof(AsyncTask));
  task->run_id = run_id;
  task->session = session;
  task->num_inputs = num_inputs;
  task->input_names = (char**)malloc(sizeof(char*) * num_inputs);
  task->input_tensors = (LibOnnxTensorHandle*)malloc(sizeof(LibOnnxTensorHandle) * num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    task->input_names[i] = strdup(input_names[i]);
    task->input_tensors[i] = input_tensors[i];
  }

  task->num_outputs = num_outputs;
  task->output_names = (char**)malloc(sizeof(char*) * num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    task->output_names[i] = strdup(output_names[i]);
  }

  task->timeout_ms = timeout_ms;
  task->callback = callback;
  task->user_data = user_data;

  if (rt->task_queue_tail == NULL) {
    rt->task_queue_head = task;
    rt->task_queue_tail = task;
  } else {
    rt->task_queue_tail->next = task;
    rt->task_queue_tail = task;
  }
  rt->task_queue_size++;

  pthread_cond_signal(&rt->queue_cond);
  pthread_mutex_unlock(&rt->queue_mutex);

  return LIB_ONNX_OK;
}
