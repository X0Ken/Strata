#include <cublas_v2.h>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void hip_check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

static void blas_check(rocblas_status status) {
    if (status != rocblas_status_success) throw std::runtime_error("rocBLAS status " + std::to_string(status));
}

int main() {
    try {
        constexpr int m = 4, n = 3, k = 8;
        cublasHandle_t handle;
        blas_check(cublasCreate(&handle));
        hipStream_t stream;
        hip_check(hipStreamCreate(&stream));
        blas_check(cublasSetStream(handle, stream));
        void* workspace;
        hip_check(hipMalloc(&workspace, 32 << 20));
        blas_check(cublasSetWorkspace(handle, workspace, 32 << 20));
        for (auto type : {CUDA_R_16F, CUDA_R_16BF}) {
            const uint16_t one = type == CUDA_R_16F ? 0x3c00 : 0x3f80;
            auto encode = [one](int x) -> uint16_t { return x == 0 ? 0 : x == 1 ? one : 0x4000; };
            std::vector<uint16_t> a(k * m), b(k * n);
            for (int row = 0; row < m; ++row)
                for (int inner = 0; inner < k; ++inner) a[inner + k * row] = encode((inner + 2 * row) % 3);
            for (int col = 0; col < n; ++col)
                for (int inner = 0; inner < k; ++inner) b[inner + k * col] = encode((2 * inner + col) % 3);
            uint16_t *da, *db;
            float* dc;
            hip_check(hipMalloc(&da, a.size() * 2));
            hip_check(hipMalloc(&db, b.size() * 2));
            hip_check(hipMalloc(&dc, m * n * sizeof(float)));
            hip_check(hipMemcpy(da, a.data(), a.size() * 2, hipMemcpyHostToDevice));
            hip_check(hipMemcpy(db, b.data(), b.size() * 2, hipMemcpyHostToDevice));
            for (float beta : {0.0f, 1.0f}) {
                std::vector<float> result(m * n, 2.0f);
                hip_check(hipMemcpy(dc, result.data(), result.size() * sizeof(float), hipMemcpyHostToDevice));
                const float alpha = 1.0f;
                blas_check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                                       da, type, k, db, type, k, &beta, dc, CUDA_R_32F, m,
                                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                hip_check(hipStreamSynchronize(stream));
                hip_check(hipMemcpy(result.data(), dc, result.size() * sizeof(float), hipMemcpyDeviceToHost));
                for (int col = 0; col < n; ++col) {
                    for (int row = 0; row < m; ++row) {
                        float expected = 2.0f * beta;
                        for (int inner = 0; inner < k; ++inner) expected += ((inner + 2 * row) % 3) * ((2 * inner + col) % 3);
                        if (result[row + m * col] != expected) throw std::runtime_error("GEMM differs from CPU reference");
                    }
                }
            }
            hip_check(hipFree(da)); hip_check(hipFree(db)); hip_check(hipFree(dc));
            std::printf("PASS DTK rocBLAS %s -> F32, transpose and beta=0/1\n", type == CUDA_R_16F ? "F16" : "BF16");
        }
        blas_check(cublasDestroy(handle));
        hip_check(hipFree(workspace));
        hip_check(hipStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL DTK BLAS: %s\n", error.what());
        return 1;
    }
}
