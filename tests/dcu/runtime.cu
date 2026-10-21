#include <cuda_runtime.h>
#include <cstdio>

__global__ void thread_limit_probe(int* result) {
    if (threadIdx.x == 1023) *result = 1024;
}

int main() {
    int limit = 0;
    if (cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0) != cudaSuccess || limit < 65536) {
        std::fprintf(stderr, "FAIL DTK LDS limit: %d\n", limit);
        return 1;
    }
    int* device = nullptr;
    if (cudaMalloc(&device, sizeof(int)) != cudaSuccess || cudaMemset(device, 0, sizeof(int)) != cudaSuccess) return 1;
    thread_limit_probe<<<1, 1024>>>(device);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 1;
    int result = 0;
    if (cudaMemcpy(&result, device, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    cudaFree(device);
    if (result != 1024) return 1;
    std::printf("PASS DTK 1024-thread launch and LDS limit %d bytes\n", limit);
    return 0;
}
