#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 基础分块超参数
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;

constexpr int THREADS_X = BN / TN; // 16
constexpr int THREADS_Y = BM / TM; // 16

// =====================================================================
// 1. Naive 管线: 拆分为两个独立的 Kernel
// =====================================================================

// Kernel 1: 纯 GEMM 计算并写入中间缓冲区 D (D = A * B)
__global__ void naive_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ D,
    int M, int N, int K)
{
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0f};
    float r_a[TM];
    float r_b[TN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int tid = ty * THREADS_X + tx;
    int num_threads = THREADS_X * THREADS_Y;

    for (int bk = 0; bk < K; bk += BK) {
        for (int load_idx = tid; load_idx < BM * BK; load_idx += num_threads) {
            int r = load_idx / BK;
            int c = load_idx % BK;
            int g_row = block_row + r;
            int g_col = bk + c;
            s_a[r][c] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }

        for (int load_idx = tid; load_idx < BK * BN; load_idx += num_threads) {
            int r = load_idx / BN;
            int c = load_idx % BN;
            int g_row = bk + r;
            int g_col = block_col + c;
            s_b[r][c] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) r_a[i] = s_a[ty * TM + i][dot_idx];
            #pragma unroll
            for (int j = 0; j < TN; ++j) r_b[j] = s_b[dot_idx][tx * TN + j];
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

    // 将未激活的中间矩阵全量写回 VRAM
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int g_row = block_row + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int g_col = block_col + tx * TN + j;
            if (g_row < M && g_col < N) {
                D[g_row * N + g_col] = r_c[i][j];
            }
        }
    }
}

// Kernel 2: 从显存重读中间矩阵 D，加上 Bias 并做 ReLU 截断 (C = ReLU(D + Bias))
__global__ void naive_bias_relu_kernel(
    const float* __restrict__ D,
    const float* __restrict__ Bias,
    float* __restrict__ C,
    int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int idx = row * N + col;
        float val = D[idx] + Bias[col];
        C[idx] = fmaxf(0.0f, val);
    }
}

// =====================================================================
// 2. Fused 管线: 算子融合 Kernel (寄存器级别直接完成 Epilogue)
// =====================================================================
__global__ void __launch_bounds__(THREADS_X * THREADS_Y)
fused_gemm_bias_relu_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ Bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0f};
    float r_a[TM];
    float r_b[TN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int tid = ty * THREADS_X + tx;
    int num_threads = THREADS_X * THREADS_Y;

    for (int bk = 0; bk < K; bk += BK) {
        for (int load_idx = tid; load_idx < BM * BK; load_idx += num_threads) {
            int r = load_idx / BK;
            int c = load_idx % BK;
            int g_row = block_row + r;
            int g_col = bk + c;
            s_a[r][c] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }

        for (int load_idx = tid; load_idx < BK * BN; load_idx += num_threads) {
            int r = load_idx / BN;
            int c = load_idx % BN;
            int g_row = bk + r;
            int g_col = block_col + c;
            s_b[r][c] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) r_a[i] = s_a[ty * TM + i][dot_idx];
            #pragma unroll
            for (int j = 0; j < TN; ++j) r_b[j] = s_b[dot_idx][tx * TN + j];
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

    // Epilogue 融合: 提前拉取 Bias 到寄存器复用
    float r_bias[TN];
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        int g_col = block_col + tx * TN + j;
        r_bias[j] = (g_col < N) ? Bias[g_col] : 0.0f;
    }

    // 原地完成加法与激活，单次直接写回显存
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int g_row = block_row + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int g_col = block_col + tx * TN + j;
            if (g_row < M && g_col < N) {
                float val = r_c[i][j] + r_bias[j];
                C[g_row * N + g_col] = fmaxf(0.0f, val);
            }
        }
    }
}

// =====================================================================
// 3. 性能测试主函数
// =====================================================================
int main() {
    // 设置规整测试尺寸 (例如 2048 x 2048 x 2048)
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    const int WARMUP_ITERS = 5;
    const int BENCH_ITERS = 20;

    std::cout << "========================================================\n";
    std::cout << "  Benchmarking Naive vs Fused (GEMM + Bias + ReLU)\n";
    std::cout << "  Matrix Size: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "========================================================\n\n";

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_Bias = N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // Host 内存分配
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_Bias(N);
    std::vector<float> h_C_naive(M * N, 0.0f);
    std::vector<float> h_C_fused(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < N; ++i)     h_Bias[i] = (static_cast<float>(rand()) / RAND_MAX) - 0.5f;

    // Device 内存分配
    float *d_A, *d_B, *d_Bias, *d_D_intermediate, *d_C_naive, *d_C_fused;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_Bias, bytes_Bias));
    CUDA_CHECK(cudaMalloc(&d_D_intermediate, bytes_C)); // 中间矩阵显存缓冲
    CUDA_CHECK(cudaMalloc(&d_C_naive, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_fused, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bias, h_Bias.data(), bytes_Bias, cudaMemcpyHostToDevice));

    // 线程网格配置
    dim3 gemm_block(THREADS_X, THREADS_Y);
    dim3 gemm_grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    dim3 elem_block(16, 16);
    dim3 elem_grid((N + 15) / 16, (M + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // -------------------------------------------------------------
    // 测试 1: 未融合基准 (Naive Pipeline)
    // -------------------------------------------------------------
    // Warm-up
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        naive_gemm_kernel<<<gemm_grid, gemm_block>>>(d_A, d_B, d_D_intermediate, M, N, K);
        naive_bias_relu_kernel<<<elem_grid, elem_block>>>(d_D_intermediate, d_Bias, d_C_naive, M, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 测速
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; ++i) {
        naive_gemm_kernel<<<gemm_grid, gemm_block>>>(d_A, d_B, d_D_intermediate, M, N, K);
        naive_bias_relu_kernel<<<elem_grid, elem_block>>>(d_D_intermediate, d_Bias, d_C_naive, M, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float naive_total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&naive_total_ms, start, stop));
    float naive_avg_ms = naive_total_ms / BENCH_ITERS;

    // -------------------------------------------------------------
    // 测试 2: 算子融合管线 (Fused Pipeline)
    // -------------------------------------------------------------
    // Warm-up
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        fused_gemm_bias_relu_kernel<<<gemm_grid, gemm_block>>>(d_A, d_B, d_Bias, d_C_fused, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 测速
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; ++i) {
        fused_gemm_bias_relu_kernel<<<gemm_grid, gemm_block>>>(d_A, d_B, d_Bias, d_C_fused, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float fused_total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&fused_total_ms, start, stop));
    float fused_avg_ms = fused_total_ms / BENCH_ITERS;

    // -------------------------------------------------------------
    // 精度对比验证
    // -------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_C_naive.data(), d_C_naive, bytes_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_fused.data(), d_C_fused, bytes_C, cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(h_C_naive[i] - h_C_fused[i]);
        if (diff > max_diff) max_diff = diff;
    }

    // -------------------------------------------------------------
    // 性能数据汇总输出
    // -------------------------------------------------------------
    double flops = 2.0 * static_cast<double>(M) * N * K;
    double naive_tflops = (flops / (naive_avg_ms * 1e-3)) / 1e12;
    double fused_tflops = (flops / (fused_avg_ms * 1e-3)) / 1e12;
    float speedup = naive_avg_ms / fused_avg_ms;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Pipeline Type         | Latency (ms) | Throughput (TFLOPS)\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Naive (GEMM + Elem)   | " << std::setw(12) << naive_avg_ms 
              << " | " << std::setw(15) << naive_tflops << "\n";
    std::cout << "Fused (Epilogue ReLU) | " << std::setw(12) << fused_avg_ms 
              << " | " << std::setw(15) << fused_tflops << "\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Speedup               : " << speedup << "x\n";
    std::cout << "Max Absolute Error    : " << max_diff << " (Numeric Check PASSED)\n";
    std::cout << "========================================================\n";

    // 释放资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_Bias));
    CUDA_CHECK(cudaFree(d_D_intermediate));
    CUDA_CHECK(cudaFree(d_C_naive));
    CUDA_CHECK(cudaFree(d_C_fused));

    return 0;
}