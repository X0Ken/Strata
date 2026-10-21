#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
static void ck(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(1); } }
int main(int argc, char**) {
    constexpr int N=3072, K=10, T=8, R=K*T;
    float *host=nullptr,*mapped=nullptr,*hits=nullptr,*parts=nullptr,*weights=nullptr,*shared=nullptr,*a=nullptr,*b=nullptr;
    int32_t *dst=nullptr,*count=nullptr;
    cudaStream_t stream; ck(cudaStreamCreate(&stream));
    ck(cudaHostAlloc((void**)&host,R*N*4,cudaHostAllocMapped)); ck(cudaHostGetDevicePointer((void**)&mapped,host,0));
    auto alloc=[](auto** p,size_t bytes){ck(cudaMalloc((void**)p,bytes));};
    alloc(&hits,R*N*4);alloc(&parts,R*N*4);alloc(&weights,R*4);alloc(&shared,T*N*4);alloc(&a,T*N*4);alloc(&b,T*N*4);alloc(&dst,R*4);alloc(&count,4);
    std::mt19937 rng(7391); std::vector<float> h(R*N),w(R),sh(T*N),got(T*N),ref(T*N);
    auto fill=[&](auto& x){for(auto& v:x)v=(int(rng()%20001)-10000)*0.00031f;};
    fill(h);fill(w);fill(sh);
    ck(cudaMemcpy(hits,h.data(),h.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(weights,w.data(),w.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(shared,sh.data(),sh.size()*4,cudaMemcpyHostToDevice));
    for(int trial=0;trial<128;++trial) {
        int tokens=trial%8+1; std::vector<int32_t> indices;
        for(int r=0;r<tokens*K;++r) {
            bool hit=trial<8 ? false : trial<16 ? true : rng()%2;
            if(hit)indices.push_back(r);
            // Poison GPU hit rows in mapped memory: the fused path must never read them.
            for(int c=0;c<N;++c)host[r*N+c]=hit ? __builtin_nanf("") : (int(rng()%20001)-10000)*0.00027f;
        }
        int n=indices.size(); if(n)ck(cudaMemcpy(dst,indices.data(),n*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(count,&n,4,cudaMemcpyHostToDevice));
        strata::kernels::copy_rows_from_mapped(parts,mapped,tokens*K,N,dst,count,stream);
        strata::kernels::moe_hit_add(parts,hits,dst,count,R,N,stream);
        strata::kernels::native_moe_combine_multi(parts,weights,shared,a,N,K,tokens,stream);
        strata::kernels::native_moe_combine_mapped(mapped,hits,dst,count,weights,shared,b,N,K,tokens,stream);
        ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(ref.data(),a,tokens*N*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(got.data(),b,tokens*N*4,cudaMemcpyDeviceToHost));
        if(std::memcmp(ref.data(),got.data(),tokens*N*4)) {std::fprintf(stderr,"mismatch trial %d\n",trial);return 1;}
    }
    if (argc > 1) {
        cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));
        for(int nh : {0,2,5,8,10}) {
            std::vector<int32_t> indices;for(int i=0;i<nh;++i)indices.push_back(i);
            if(nh)ck(cudaMemcpy(dst,indices.data(),nh*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(count,&nh,4,cudaMemcpyHostToDevice));
            for(int i=0;i<K*N;++i)host[i]=0.125f;
            for(int mode=0;mode<2;++mode) {
                auto launch=[&] {
                    if(mode) strata::kernels::native_moe_combine_mapped(mapped,hits,dst,count,weights,shared,b,N,K,1,stream);
                    else {
                        strata::kernels::copy_rows_from_mapped(parts,mapped,K,N,dst,count,stream);
                        strata::kernels::moe_hit_add(parts,hits,dst,count,R,N,stream);
                        strata::kernels::native_moe_combine_multi(parts,weights,shared,a,N,K,1,stream);
                    }
                };
                for(int i=0;i<16;++i)launch();ck(cudaStreamSynchronize(stream));
                ck(cudaEventRecord(start,stream));for(int i=0;i<256;++i)launch();ck(cudaEventRecord(stop,stream));ck(cudaEventSynchronize(stop));
                float ms;ck(cudaEventElapsedTime(&ms,start,stop));
                std::printf("{\"hits\":%d,\"fused\":%d,\"width\":%d,\"iterations\":256,\"us\":%.6f}\n",nh,mode,N,ms*1000/256);
            }
        }
        ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));
    }
    ck(cudaFreeHost(host));for(void* p:{(void*)hits,(void*)parts,(void*)weights,(void*)shared,(void*)a,(void*)b,(void*)dst,(void*)count})ck(cudaFree(p));ck(cudaStreamDestroy(stream));
    std::puts("PASS 128 bitwise mapped-combine cases, 1..8 tokens, all/mixed/no hits, poisoned host hit rows");
}
