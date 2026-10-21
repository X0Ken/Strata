#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include <cstdio>

int main() {
    using namespace strata::prefill;
    set_act_f16(false);
    set_act_f16(true);
    set_act_f16(false);
    set_f16_range(true);
    gemm_set_f16_range(true);
    float activation_max = -1;
    unsigned long long activation_over = 1;
    float gemm_max[2] = {-1, -1};
    unsigned long long gemm_over[2] = {1, 1};
    if (!f16_range_read(activation_max, activation_over, true) ||
        !gemm_f16_range_read(gemm_max, gemm_over, true) ||
        activation_max != 0 || activation_over != 0 ||
        gemm_max[0] != 0 || gemm_max[1] != 0 || gemm_over[0] != 0 || gemm_over[1] != 0) {
        std::fprintf(stderr, "FAIL DTK prefill device-symbol state\n");
        return 1;
    }
    set_f16_range(false);
    gemm_set_f16_range(false);
    std::printf("PASS DTK prefill device-symbol setters and counter reads/resets\n");
    return 0;
}
