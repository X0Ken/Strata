// Source-level BLAS names for the shared engine; all calls execute DTK's native rocBLAS.
#pragma once
#include "strata_hip.h"
#include <rocblas/rocblas.h>

using cublasHandle_t = rocblas_handle;
using cublasStatus_t = rocblas_status;
using cublasGemmAlgo_t = rocblas_gemm_algo;
#define CUBLAS_STATUS_SUCCESS rocblas_status_success
#define CUBLAS_OP_N rocblas_operation_none
#define CUBLAS_OP_T rocblas_operation_transpose
#define CUBLAS_GEMM_DEFAULT rocblas_gemm_algo_standard
#define CUBLAS_COMPUTE_32F rocblas_datatype_f32_r
#define CUDA_R_32F rocblas_datatype_f32_r
#define CUDA_R_16F rocblas_datatype_f16_r
#define CUDA_R_16BF rocblas_datatype_bf16_r
#define CUBLAS_DEFAULT_MATH 0
#define cublasCreate rocblas_create_handle
#define cublasDestroy rocblas_destroy_handle
#define cublasSetStream rocblas_set_stream
#define cublasSetWorkspace rocblas_set_workspace
#define cublasSgemm rocblas_sgemm

inline rocblas_status cublasSetMathMode(rocblas_handle, int mode) {
    return mode == CUBLAS_DEFAULT_MATH ? rocblas_status_success : rocblas_status_invalid_value;
}

inline rocblas_status cublasGemmEx(rocblas_handle handle, rocblas_operation trans_a, rocblas_operation trans_b,
                                 int m, int n, int k, const void* alpha, const void* a, rocblas_datatype a_type,
                                 int lda, const void* b, rocblas_datatype b_type, int ldb, const void* beta,
                                 void* c, rocblas_datatype c_type, int ldc, rocblas_datatype compute_type,
                                 rocblas_gemm_algo algo) {
    return rocblas_gemm_ex(handle, trans_a, trans_b, m, n, k, alpha, a, a_type, lda, b, b_type, ldb,
                          beta, c, c_type, ldc, c, c_type, ldc, compute_type, algo, 0, 0);
}
