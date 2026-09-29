#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 分块配置：根据硬件调优参数静态指定
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;

constexpr int THREADS_X = BN / TN; // 16
constexpr int THREADS_Y = BM / TM; // 16

// =====================================================================
// Fused GEMM + Bias + ReLU Kernel
// 计算公式: C = ReLU(A * B + Bias)
// 其中: A 是 M x K, B 是 K x N, C 是 M x N
// Bias 是 1 x N 的行向量 (在 N 维度做广播操作)
// =====================================================================
__global__ void __launch_bounds__(THREADS_X * THREADS_Y)//指导编译器分配物理寄存器。
fused_gemm_bias_relu_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ Bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    // 1. 结果累加寄存器堆
    float r_c[TM][TN] = {0.0f};
    float r_a[TM];
    float r_b[TN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int tid = ty * THREADS_X + tx;
    int num_threads = THREADS_X * THREADS_Y; // 256 线程

    // 2. 主循环：GEMM 内积累加
    for (int bk = 0; bk < K; bk += BK) {
        // 协同搬运 A 到 SMEM
        for (int load_idx = tid; load_idx < BM * BK; load_idx += num_threads) {
            int r = load_idx / BK;
            int c = load_idx % BK;
            int g_row = block_row + r;
            int g_col = bk + c;
            s_a[r][c] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }

        // 协同搬运 B 到 SMEM
        for (int load_idx = tid; load_idx < BK * BN; load_idx += num_threads) {
            int r = load_idx / BN;
            int c = load_idx % BN;
            int g_row = bk + r;
            int g_col = block_col + c;
            s_b[r][c] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }

        __syncthreads();

        // 寄存器点积展开
        #pragma unroll
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                r_a[i] = s_a[ty * TM + i][dot_idx];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                r_b[j] = s_b[dot_idx][tx * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    r_c[i][j] += r_a[i] * r_b[j];
                }
            }
        }

        __syncthreads();
    }

    // =================================================================
    // 3. Epilogue 阶段 (Fusion: 加 Bias 并执行 ReLU 激活)
    // 关键优化：先预加载当前线程负责的列对应的 Bias 到寄存器中复用
    // =================================================================
    float r_bias[TN];
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        int g_col = block_col + tx * TN + j;
        r_bias[j] = (g_col < N) ? Bias[g_col] : 0.0f;
    }

    // 寄存器级别完成加法与截断，并直接写回 Global Memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int g_row = block_row + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int g_col = block_col + tx * TN + j;
            if (g_row < M && g_col < N) {
                // 原地寄存器运算: + Bias
                float val = r_c[i][j] + r_bias[j];
                // 原地寄存器运算: ReLU (映射为 GPU 的单周期 FMNMX / PRMT 汇编指令)
                val = fmaxf(0.0f, val);
                // 仅发生一次真实的 VRAM 物理写操作
                C[g_row * N + g_col] = val;
            }
        }
    }
}




int main() {
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_Bias = N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 0.5f);
    std::vector<float> h_Bias(N, -1000.0f); // 设为负数，验证 ReLU 是否正确截断为 0
    std::vector<float> h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_Bias, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_Bias, size_Bias));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bias, h_Bias.data(), size_Bias, cudaMemcpyHostToDevice));

    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    // 运行 Fused Kernel
    fused_gemm_bias_relu_kernel<<<grid, block>>>(d_A, d_B, d_Bias, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // 逻辑验证:
    // A * B 的点积应为: 2048 * (1.0 * 0.5) = 1024.0f
    // 加 Bias: 1024.0f + (-1000.0f) = 24.0f > 0，ReLU 保留 24.0f
    std::cout << "验证结果 C[0]: " << h_C[0] << " (预期: 24.0)" << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_Bias));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}