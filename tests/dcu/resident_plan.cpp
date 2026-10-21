#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(cudaError_t status) {
    if (status != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); std::exit(1); }
}

int main(int argc, char** argv) {
    constexpr int cap = 80, experts = 32, stride = 512, words = 1024;
    int32_t *ids_d = nullptr, *res_d = nullptr, *plan_d = nullptr;
    unsigned long long* offsets_d = nullptr;
    uint8_t* cache = nullptr;
    check(cudaMalloc((void**) &ids_d, cap * sizeof(int32_t)));
    check(cudaMalloc((void**) &res_d, experts * sizeof(int32_t)));
    check(cudaMalloc((void**) &plan_d, words * sizeof(int32_t)));
    check(cudaMalloc((void**) &offsets_d, experts * sizeof(unsigned long long)));
    check(cudaMalloc((void**) &cache, experts * stride + 128));
    std::vector<unsigned long long> offsets(experts);
    for (int i = 0; i < experts; ++i) offsets[i] = i * stride + (i % 2) * 64;
    check(cudaMemcpy(offsets_d, offsets.data(), offsets.size() * sizeof(offsets[0]), cudaMemcpyHostToDevice));
    std::mt19937 rng(7391);
    for (int trial = 0; trial < 128; ++trial) {
        const int n = (trial % 8 + 1) * 10;
        const bool variable_stride = trial % 2;
        std::vector<int32_t> ids(n), res(experts), got(words), expected(words, -777);
        for (int i = 0; i < experts; ++i) res[i] = trial < 8 ? -1 : trial < 16 ? i : rng() % 3 ? i : -1;
        for (int& id : ids) id = (int) (rng() % (experts + 2)) - 1;
        const int po = ((4 + (cap + 1) + 2 * cap) + 1) & ~1;
        int* starts = expected.data() + 4;
        int* dst = starts + cap + 1;
        int* tok = dst + cap;
        int groups = 0, entries = 0;
        std::vector<uint64_t> ptrs;
        for (int i = 0; i < n; ++i) {
            const int e = ids[i];
            if (e < 0 || e >= experts || res[e] < 0 || std::find(ids.begin(), ids.begin() + i, e) != ids.begin() + i) continue;
            starts[groups++] = entries;
            ptrs.push_back((uint64_t) cache + (variable_stride ? offsets[res[e]] : (uint64_t) res[e] * stride));
            for (int j = i; j < n; ++j) if (ids[j] == e) { dst[entries] = j; tok[entries++] = j / 10; }
        }
        starts[groups] = entries;
        expected[0] = groups; expected[1] = entries; expected[2] = 0;
        expected[po + 4 * cap] = entries;
        check(cudaMemcpy(ids_d, ids.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
        check(cudaMemcpy(res_d, res.data(), experts * sizeof(int32_t), cudaMemcpyHostToDevice));
        std::vector<int32_t> initial(words, -777);
        check(cudaMemcpy(plan_d, initial.data(), words * sizeof(int32_t), cudaMemcpyHostToDevice));
        strata::kernels::resident_hits_plan(ids_d, n, 10, res_d, experts, cache,
            variable_stride ? offsets_d : nullptr, stride, plan_d, cap, nullptr);
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(got.data(), plan_d, words * sizeof(int32_t), cudaMemcpyDeviceToHost));
        for (int i = 0; i < words; ++i) {
            if (i >= po && i < po + 2 * groups) continue;
            if (got[i] != expected[i]) { std::fprintf(stderr, "plan mismatch trial %d word %d: %d != %d\n", trial, i, got[i], expected[i]); return 1; }
        }
        for (int i = 0; i < groups; ++i) {
            uint64_t pointer = 0;
            std::memcpy(&pointer, got.data() + po + 2 * i, sizeof pointer);
            if (pointer != ptrs[i]) { std::fprintf(stderr, "plan pointer mismatch\n"); return 1; }
        }
    }
    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
        cudaEvent_t begin, end;
        check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        std::vector<int32_t> ids(cap), res(experts);
        for (int i = 0; i < experts; ++i) res[i] = i % 5 ? i : -1;
        for (int i = 0; i < cap; ++i) ids[i] = i % experts;
        check(cudaMemcpy(ids_d, ids.data(), cap * sizeof(int32_t), cudaMemcpyHostToDevice));
        check(cudaMemcpy(res_d, res.data(), experts * sizeof(int32_t), cudaMemcpyHostToDevice));
        for (int n : {10, 40, 80}) {
            auto launch = [&] { strata::kernels::resident_hits_plan(ids_d, n, 10, res_d, experts,
                cache, offsets_d, stride, plan_d, cap, nullptr); };
            for (int i = 0; i < 32; ++i) launch();
            check(cudaEventRecord(begin, nullptr));
            for (int i = 0; i < 512; ++i) launch();
            check(cudaEventRecord(end, nullptr)); check(cudaEventSynchronize(end));
            float ms = 0; check(cudaEventElapsedTime(&ms, begin, end));
            const char* par = std::getenv("STRATA_DCU_PAR_PLAN");
            std::printf("{\"entries\":%d,\"parallel\":%d,\"iterations\":512,\"us\":%.6f}\n",
                n, par ? std::atoi(par) != 0 : 0, ms * 1000 / 512);
        }
        check(cudaEventDestroy(end)); check(cudaEventDestroy(begin));
    }
    check(cudaFree(cache)); check(cudaFree(offsets_d)); check(cudaFree(plan_d)); check(cudaFree(res_d)); check(cudaFree(ids_d));
    std::puts("PASS 128 resident/missing/duplicate/invalid-id plans, 1..8 rows, fixed/variable slot offsets");
}
