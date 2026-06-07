#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WM = 64,  WN = 64;
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;

constexpr int WARPS_M = BM / WM;                  // 2
constexpr int WARPS_N = BN / WN;                  // 2
constexpr int NUM_WARPS = WARPS_M * WARPS_N;      // 4
constexpr int THREADS = NUM_WARPS * 32;           // 128

constexpr int MMA_PER_WARP_M = WM / MMA_M;        // 4
constexpr int MMA_PER_WARP_N = WN / MMA_N;        // 8
constexpr int K_ITER_PER_TILE = BK / MMA_K;       // 2

constexpr int STAGES=3;

__device__ __forceinline__ half &A_at(half *A, int i, int j, int M=10240) {return A[i+j*M];}
__device__ __forceinline__ half &B_at(half *B, int i, int j, int K=4096) {return B[i+j*K];}
__device__ __forceinline__ float &C_at(float *C, int i, int j, int M=10240) {return C[i+j*M];}

__device__ __forceinline__ uint32_t smem_ptr_to_uint(const void *ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_16(void *dst, const void *src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_ptr_to_uint(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}


__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
    const void *smem_ptr)
{
    uint32_t addr = smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
    const void *smem_ptr)
{
    uint32_t addr = smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t &r0, uint32_t &r1,
    const void *smem_ptr)
{
    uint32_t addr = smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(
    uint32_t &r0, uint32_t &r1,
    const void *smem_ptr)
{
    uint32_t addr = smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__global__ void kernel(int M, int N, int K, half *A, half *B, float *C)
{
    int block_m = BM * blockIdx.x;
    int block_n = BN * blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;                  // 0..3
    int lane_id = tid % 32;                  // 0..31
    int warp_m = warp_id / WARPS_N;          // 0..1
    int warp_n = warp_id % WARPS_N;          // 0..1
    int warp_m_offset = warp_m * WM;
    int warp_n_offset = warp_n * WN;

    __shared__ half block_A[BK][BM];         // 128 × 32 = 8 KB
    __shared__ half block_B[BN][BK];         // 32 × 128 = 8 KB
    float res[MMA_PER_WARP_M][MMA_PER_WARP_N][4];
    #pragma unroll
    for (int i=0; i<MMA_PER_WARP_M; i++)
        #pragma unroll
        for (int j=0; j<MMA_PER_WARP_N; j++)
            #pragma unroll
            for (int k=0; k<4; k++)
                res[i][j][k] = 0.0f;

    for (int k_tile=0; k_tile<K; k_tile+=BK)
    {
        __syncthreads();
        #pragma unroll
        for (int k=0; k<BK; k++)
        {   
            block_A[k][tid] = A_at(A, block_m+tid, k_tile+k);
        }
        #pragma unroll
        for (int k=0; k<BK; k++)
        {
            block_B[tid][k] = B_at(B, k_tile+k, block_n+tid);
        }
        __syncthreads();

        #pragma unroll
        for (int k=0; k<K_ITER_PER_TILE; k++)
        {
            int mma_k = k * MMA_K;
            uint32_t frag_a[MMA_PER_WARP_M][4];
            #pragma unroll
            for (int i=0; i<MMA_PER_WARP_M; i++)
            {
                int mma_m = i * MMA_M;
                const half *src = &block_A[mma_k+lane_id%16][warp_m_offset+mma_m+lane_id/16*8];
                ldmatrix_x4_trans(frag_a[i][0], frag_a[i][2], frag_a[i][1], frag_a[i][3], src); //^T
            }
            uint32_t frag_b[MMA_PER_WARP_N][2];
            #pragma unroll
            for (int j=0; j<MMA_PER_WARP_N/2; j++)
            {
                int mma_n = j * MMA_N * 2;
                int row = (lane_id % 8) + (lane_id / 16) * 8;
                int col = ((lane_id / 8) % 2) * 8;
                const half *src =&block_B[warp_n_offset+mma_n+row][mma_k+col];
                ldmatrix_x4(frag_b[j<<1][0], frag_b[j<<1][1], frag_b[j<<1|1][0], frag_b[j<<1|1][1], src);
            }
            #pragma unroll
            for (int i=0; i<MMA_PER_WARP_M; i++)
            {
                #pragma unroll
                for (int j=0; j<MMA_PER_WARP_N; j++)
                {
                    mma_m16n8k16(
                        res[i][j][0], res[i][j][1], res[i][j][2], res[i][j][3],
                        frag_a[i][0], frag_a[i][1], frag_a[i][2], frag_a[i][3],
                        frag_b[j][0], frag_b[j][1],
                        res[i][j][0], res[i][j][1], res[i][j][2], res[i][j][3]);
                }
            }
        }
    }
    int group_row = lane_id / 4;
    int group_col = lane_id % 4;
    #pragma unroll
    for (int i=0; i<MMA_PER_WARP_M; i++)
    {
        #pragma unroll
        for (int j=0; j<MMA_PER_WARP_N; j++)
        {
            int base_m = block_m + warp_m_offset + i * MMA_M;
            int base_n = block_n + warp_n_offset + j * MMA_N;
            int row_0 = base_m + group_row;
            int row_1 = base_m + group_row + 8;
            int col_0 = base_n + group_col * 2;
            int col_1 = base_n + group_col * 2 + 1;
            C_at(C, row_0, col_0) = res[i][j][0];
            C_at(C, row_0, col_1) = res[i][j][1];
            C_at(C, row_1, col_0) = res[i][j][2];
            C_at(C, row_1, col_1) = res[i][j][3];
        }
    }
}

__global__ void float_to_half_vec4(const float *src, half *dst, int64_t n)
{
    int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 >= n) return;
    float4 f = *reinterpret_cast<const float4*>(src + i);
    half2 h01 = __floats2half2_rn(f.x, f.y);
    half2 h23 = __floats2half2_rn(f.z, f.w);
    reinterpret_cast<half2*>(dst + i)[0] = h01;
    reinterpret_cast<half2*>(dst + i)[1] = h23;
}

int main(int argc, const char **argv) {
    int m = 10240;
    int k = 4096;
    int n = 8192;
    float alpha = 1.0;
    float beta = 0.0;
    int Nt = 10;
    float *A, *B, *C, *C2;
    half *Ah, *Bh;
    cudaMallocManaged(&A, m * k * sizeof(float));
    cudaMallocManaged(&B, k * n * sizeof(float));
    cudaMallocManaged(&C, m * n * sizeof(float));
    cudaMallocManaged(&C2, m * n * sizeof(float));
    cudaMallocManaged(&Ah, m * k * sizeof(half));
    cudaMallocManaged(&Bh, k * n * sizeof(half));
    for (int i=0; i<m; i++)
        for (int j=0; j<k; j++)
            A[k*i+j] = drand48();
    for (int i=0; i<k; i++)
        for (int j=0; j<n; j++)
            B[n*i+j] = drand48();
    for (int i=0; i<n; i++)
        for (int j=0; j<m; j++)
            C[m*i+j] = C2[m*i+j] = 0;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    auto tic = chrono::steady_clock::now();
    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        cublasGemmEx(cublas_handle,
         CUBLAS_OP_N,
         CUBLAS_OP_N,
         m,
         n,
         k,
         &alpha,
         A, CUDA_R_32F, m,
         B, CUDA_R_32F, k,
         &beta,
         C, CUDA_R_32F, m,
         CUBLAS_COMPUTE_32F_FAST_16F,
         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
    }
    auto toc = chrono::steady_clock::now();
    int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
    double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
    double cublas_flops = double(num_flops) / tcublas / 1.0e9;

    int tile = 128;
    dim3 block = dim3(tile);
    dim3 grid = dim3((m+tile-1)/tile, (n+tile-1)/tile);
    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        float_to_half_vec4<<< (m*k/4+255)/256, 256>>>(A, Ah, m*k);
        float_to_half_vec4<<< (k*n/4+255)/256, 256>>>(B, Bh, k*n);
        kernel<<< grid, block>>>(m, n, k, Ah, Bh, C2);
        cudaDeviceSynchronize();
    }

    toc = chrono::steady_clock::now();
    double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
    double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
    printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
    
    double err = 0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            err += fabs(C[m*i+j] - C2[m*i+j]);
        }
    }
    printf("error: %lf\n", err/n/m);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(C2);
    cublasDestroy(cublas_handle);
}
