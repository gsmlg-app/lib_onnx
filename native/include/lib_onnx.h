#ifndef LIB_ONNX_H
#define LIB_ONNX_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #define LIB_ONNX_EXPORT __declspec(dllexport)
#else
  #define LIB_ONNX_EXPORT __attribute__((visibility("default")))
#endif

/* Status Codes */
#define LIB_ONNX_OK 0
#define LIB_ONNX_ERR_GENERIC 1
#define LIB_ONNX_ERR_INVALID_ARGUMENT 2
#define LIB_ONNX_ERR_MODEL_LOAD 3
#define LIB_ONNX_ERR_PROVIDER_UNAVAILABLE 4
#define LIB_ONNX_ERR_FALLBACK_VIOLATION 5
#define LIB_ONNX_ERR_HARDWARE_UNMET 6
#define LIB_ONNX_ERR_CANCELLED 7
#define LIB_ONNX_ERR_TIMEOUT 8
#define LIB_ONNX_ERR_QUEUE_OVERFLOW 9
#define LIB_ONNX_ERR_CLOSED 10
#define LIB_ONNX_ERR_UNSUPPORTED_TYPE 11

/* Data Types */
#define LIB_ONNX_DTYPE_UNKNOWN 0
#define LIB_ONNX_DTYPE_FLOAT32 1
#define LIB_ONNX_DTYPE_FLOAT16 2
#define LIB_ONNX_DTYPE_INT64 3
#define LIB_ONNX_DTYPE_INT32 4
#define LIB_ONNX_DTYPE_INT8 5
#define LIB_ONNX_DTYPE_UINT8 6
#define LIB_ONNX_DTYPE_BOOL 7

/* Fallback Policy */
#define LIB_ONNX_FALLBACK_ALLOWED 0
#define LIB_ONNX_FALLBACK_STRICT 1
#define LIB_ONNX_FALLBACK_WHITELISTED_OPS 2

/* Hardware Requirement */
#define LIB_ONNX_HARDWARE_ANY 0
#define LIB_ONNX_HARDWARE_GPU 1
#define LIB_ONNX_HARDWARE_NPU 2

/* Evidence Level */
#define LIB_ONNX_EVIDENCE_UNKNOWN 0
#define LIB_ONNX_EVIDENCE_NONE 1
#define LIB_ONNX_EVIDENCE_EP_COMPILED 2
#define LIB_ONNX_EVIDENCE_GRAPH_PARTITIONED 3
#define LIB_ONNX_EVIDENCE_HARDWARE_OBSERVED 4
#define LIB_ONNX_EVIDENCE_UNVERIFIABLE 5

/* Opaque handles */
typedef struct LibOnnxRuntime* LibOnnxRuntimeHandle;
typedef struct LibOnnxSession* LibOnnxSessionHandle;
typedef struct LibOnnxTensor* LibOnnxTensorHandle;
typedef struct LibOnnxDiagnosticReport* LibOnnxDiagnosticReportHandle;

/* Version API */
LIB_ONNX_EXPORT int32_t lib_onnx_version_major(void);
LIB_ONNX_EXPORT int32_t lib_onnx_version_minor(void);
LIB_ONNX_EXPORT int32_t lib_onnx_version_patch(void);
LIB_ONNX_EXPORT const char* lib_onnx_version_string(void);
LIB_ONNX_EXPORT const char* lib_onnx_ort_version_string(void);

/* Error API */
LIB_ONNX_EXPORT const char* lib_onnx_get_last_error(void);
LIB_ONNX_EXPORT void lib_onnx_clear_last_error(void);

/* Runtime Management */
LIB_ONNX_EXPORT LibOnnxRuntimeHandle lib_onnx_runtime_create(int32_t max_concurrency);
LIB_ONNX_EXPORT void lib_onnx_runtime_destroy(LibOnnxRuntimeHandle runtime);
LIB_ONNX_EXPORT int32_t lib_onnx_runtime_is_provider_supported(LibOnnxRuntimeHandle runtime, const char* provider_name);
LIB_ONNX_EXPORT int32_t lib_onnx_runtime_get_available_providers(LibOnnxRuntimeHandle runtime, char* buffer, size_t buffer_len);

/* Tensor Management */
LIB_ONNX_EXPORT LibOnnxTensorHandle lib_onnx_tensor_create(
    int32_t dtype,
    const int64_t* shape,
    int32_t rank,
    const void* data,
    size_t byte_length,
    int32_t copy_data
);
LIB_ONNX_EXPORT void lib_onnx_tensor_destroy(LibOnnxTensorHandle tensor);
LIB_ONNX_EXPORT int32_t lib_onnx_tensor_get_dtype(LibOnnxTensorHandle tensor);
LIB_ONNX_EXPORT int32_t lib_onnx_tensor_get_rank(LibOnnxTensorHandle tensor);
LIB_ONNX_EXPORT int32_t lib_onnx_tensor_get_shape(LibOnnxTensorHandle tensor, int64_t* out_shape, int32_t max_rank);
LIB_ONNX_EXPORT const void* lib_onnx_tensor_get_data(LibOnnxTensorHandle tensor);
LIB_ONNX_EXPORT size_t lib_onnx_tensor_get_byte_length(LibOnnxTensorHandle tensor);
LIB_ONNX_EXPORT int64_t lib_onnx_tensor_get_element_count(LibOnnxTensorHandle tensor);

/* Session Management */
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
);
LIB_ONNX_EXPORT void lib_onnx_session_destroy(LibOnnxSessionHandle session);

/* Session Metadata */
LIB_ONNX_EXPORT int32_t lib_onnx_session_get_input_count(LibOnnxSessionHandle session);
LIB_ONNX_EXPORT const char* lib_onnx_session_get_input_name(LibOnnxSessionHandle session, int32_t index);
LIB_ONNX_EXPORT int32_t lib_onnx_session_get_input_type(LibOnnxSessionHandle session, int32_t index);
LIB_ONNX_EXPORT int32_t lib_onnx_session_get_input_shape(LibOnnxSessionHandle session, int32_t index, int64_t* out_shape, int32_t max_rank);

LIB_ONNX_EXPORT int32_t lib_onnx_session_get_output_count(LibOnnxSessionHandle session);
LIB_ONNX_EXPORT const char* lib_onnx_session_get_output_name(LibOnnxSessionHandle session, int32_t index);
LIB_ONNX_EXPORT int32_t lib_onnx_session_get_output_type(LibOnnxSessionHandle session, int32_t index);
LIB_ONNX_EXPORT int32_t lib_onnx_session_get_output_shape(LibOnnxSessionHandle session, int32_t index, int64_t* out_shape, int32_t max_rank);

/* Execution and Cancellation */
typedef void (*LibOnnxCompletionCallback)(
    int64_t run_id,
    int32_t status_code,
    const char* error_msg,
    LibOnnxTensorHandle* outputs,
    int32_t num_outputs,
    LibOnnxDiagnosticReportHandle diagnostic,
    void* user_data
);

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
);

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
);

LIB_ONNX_EXPORT int32_t lib_onnx_session_cancel_run(LibOnnxSessionHandle session, int64_t run_id);

/* Diagnostic Report */
LIB_ONNX_EXPORT const char* lib_onnx_diagnostic_get_provider(LibOnnxDiagnosticReportHandle report);
LIB_ONNX_EXPORT int32_t lib_onnx_diagnostic_get_evidence_level(LibOnnxDiagnosticReportHandle report);
LIB_ONNX_EXPORT int64_t lib_onnx_diagnostic_get_duration_us(LibOnnxDiagnosticReportHandle report);
LIB_ONNX_EXPORT const char* lib_onnx_diagnostic_get_hardware_details(LibOnnxDiagnosticReportHandle report);
LIB_ONNX_EXPORT const char* lib_onnx_diagnostic_get_model_hash(LibOnnxDiagnosticReportHandle report);
LIB_ONNX_EXPORT int32_t lib_onnx_diagnostic_get_fallback_occurred(LibOnnxDiagnosticReportHandle report);
LIB_ONNX_EXPORT void lib_onnx_diagnostic_destroy(LibOnnxDiagnosticReportHandle report);

#ifdef __cplusplus
}
#endif

#endif /* LIB_ONNX_H */
