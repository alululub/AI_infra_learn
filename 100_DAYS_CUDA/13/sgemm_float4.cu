#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// ============================================================================
// 1. 编译期超参数 (Tile 维度定义)
// ============================================================================
#define BM 128   // Block 沿 M 轴的 Tile 大小
#define BN 128   // Block 沿 N 轴的 Tile 大小
#define BK 8     // Block 沿 K 轴推进的步长
#define TM 8     // Thread 负责计算的 M 轴子块大小
#define TN 8     // Thread 负责计算的 N 轴子块大小

// 验证计算力与线程匹配关系：
// Block 内总线程数 = (BM / TM) * (BN / TN) = (128/8) * (128/8) = 16 x 16 = 256 线程
//
// 向量化 128-bit (float4) 搬运账本：
// - As 块大小：128 x 8 = 1024 个 float = 256 个 float4
// - Bs 块大小：8 x 128 = 1024 个 float = 256 个 float4
// - 恰好 256 个线程，每个线程只需处理 1 个 float4 (128-bit)，无需写任何 for 循环！

// ============================================================================
// 2. CUDA Kernel: 2D Block Tiling + float4 向量化加载
// ============================================================================
__global__ void sgemm_vectorized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) 
{
    // --- 线程物理坐标与 1D 展平 ID ---
    const int tx = threadIdx.x; // 0 ~ 15
    const int ty = threadIdx.y; // 0 ~ 15
    const int tid = ty * blockDim.x + tx; // 0 ~ 255

    // 当前 Block 在 C 矩阵中的绝对起始物理行/列
    const int c_row_start = blockIdx.y * BM;
    const int c_col_start = blockIdx.x * BN;

    // --- 申请 Shared Memory (必须满足 16 字节物理对齐) ---
    __shared__ float As[BM][BK]; // 128 x 8
    __shared__ float Bs[BK][BN]; // 8 x 128

    // 线程私有寄存器 (Register File)
    float accum[TM][TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    // ------------------------------------------------------------------------
    // 【核心亮点】：计算 float4 向量化搬运逻辑
    // ------------------------------------------------------------------------
    // A 矩阵 (128x8): 每行 8 个 float = 2 个 float4。256 个线程对应 256 个 float4。
    const int a_vec_idx = tid;                       // 0 ~ 255
    const int a_tile_row = a_vec_idx / (BK / 4);     // BK/4 = 2, 结果 0 ~ 127 行
    const int a_tile_col = (a_vec_idx % (BK / 4)) * 4; // (0 或 1) * 4 = 0 或 4 列

    // B 矩阵 (8x128): 每行 128 个 float = 32 个 float4。256 个线程对应 256 个 float4。
    const int b_vec_idx = tid;                       // 0 ~ 255
    const int b_tile_row = b_vec_idx / (BN / 4);     // BN/4 = 32, 结果 0 ~ 7 行
    const int b_tile_col = (b_vec_idx % (BN / 4)) * 4; // (0~31) * 4 = 0, 4, 8 ... 124 列

    // --- 主循环：沿着 K 维度分步推进 ---
    for (int ph = 0; ph < (K + BK - 1) / BK; ++ph) {

        // ====================================================================
        // 1. 128-bit 向量化加载：Global Memory (VRAM) -> Shared Memory (SRAM)
        // 对应硬件汇编：发送一条 LDG.128 指令，一口气拉回 16 字节
        // ====================================================================

        // (A) 搬运 A 矩阵子块
        int a_global_r = c_row_start + a_tile_row;
        int a_global_c = ph * BK + a_tile_col;

        float4 tmp_a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        // 边界安全校验：若处于边界外，安全回退到标量读取或置零
        if (a_global_r < M && a_global_c + 3 < K) {
            // 物理魔法：使用 reinterpret_cast 发起 128-bit 向量化读取
            tmp_a = *reinterpret_cast<const float4*>(&A[a_global_r * K + a_global_c]);
        } else {
            for (int i = 0; i < 4; ++i) {
                if (a_global_r < M && a_global_c + i < K) {
                    reinterpret_cast<float*>(&tmp_a)[i] = A[a_global_r * K + a_global_c + i];
                }
            }
        }
        // 将 float4 写入 Shared Memory (一次写 16 字节)
        *reinterpret_cast<float4*>(&As[a_tile_row][a_tile_col]) = tmp_a;


        // (B) 搬运 B 矩阵子块
        int b_global_r = ph * BK + b_tile_row;
        int b_global_c = c_col_start + b_tile_col;

        float4 tmp_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (b_global_r < K && b_global_c + 3 < N) {
            tmp_b = *reinterpret_cast<const float4*>(&B[b_global_r * N + b_global_c]);
        } else {
            for (int i = 0; i < 4; ++i) {
                if (b_global_r < K && b_global_c + i < N) {
                    reinterpret_cast<float*>(&tmp_b)[i] = B[b_global_r * N + b_global_c + i];
                }
            }
        }
        // 将 float4 写入 Shared Memory
        *reinterpret_cast<float4*>(&Bs[b_tile_row][b_tile_col]) = tmp_b;

        // 强迫线程同步：等待 1024 个数据全部由 128-bit 指令写入 Shared Memory
        __syncthreads();

        // ====================================================================
        // 2. 计算阶段：Shared Memory -> Register -> 2D 外积累加
        // ====================================================================
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As[ty * TM + i][k];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = Bs[k][tx * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // ====================================================================
    // 3. 写回阶段：Register -> Global Memory (128-bit 向量化写回)
    // ====================================================================
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_r = c_row_start + ty * TM + i;
        if (global_r < M) {
            // 每行 TN=8 个 float，可以拆分为 2 个 float4 进行写回
            #pragma unroll
            for (int j = 0; j < TN; j += 4) {
                int global_c = c_col_start + tx * TN + j;
                if (global_c + 3 < N) {
                    float4 tmp_out;
                    tmp_out.x = accum[i][j + 0];
                    tmp_out.y = accum[i][j + 1];
                    tmp_out.z = accum[i][j + 2];
                    tmp_out.w = accum[i][j + 3];

                    *reinterpret_cast<float4*>(&C[global_r * N + global_c]) = tmp_out;
                } else {
                    for (int k = 0; k < 4; ++k) {
                        if (global_c + k < N) {
                            C[global_r * N + global_c + k] = accum[i][j + k];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// CPU 端验证函数 (Ground Truth)
// ============================================================================
void cpu_sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Main 函数
// ============================================================================
int main() {
    const int M = 512;
    const int N = 512;
    const int K = 512;

    std::cout << "🚀 矩阵维度: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // 1. Host 侧内存分配与初始化
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // 2. Device 侧内存分配 (cudaMalloc 保证首地址 256 字节对齐，天然满足 float4 需求)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice);

    // 3. Grid 与 Block 维度设置
    dim3 blockDim(BN / TN, BM / TM); // (16, 16) -> 256 Threads
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    // 4. Performance Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    sgemm_vectorized_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    const int iters = 20;
    for (int i = 0; i < iters; ++i) {
        sgemm_vectorized_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;

    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    std::cout << "GPU 128-bit Vectorized Kernel Time: " << avg_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    // 5. 校验正确性
    cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost);
    std::cout << "正在进行 CPU 比对校验..." << std::endl;
    cpu_sgemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    double max_diff = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double diff = std::abs(h_C[i] - h_C_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Max Absolute Difference: " << max_diff << std::endl;
    if (max_diff < 1e-3) {
        std::cout << "✅ Verification PASSED!" << std::endl;
    } else {
        std::cout << "❌ Verification FAILED!" << std::endl;
    }

    // 6. 释放资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}