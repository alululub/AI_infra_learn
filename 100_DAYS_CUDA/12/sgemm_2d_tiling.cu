#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// ============================================================================
// 超参数定义 (Heuristics Config)
// ============================================================================
#define BM 128   // Block 沿 M 轴的 Tile 大小
#define BN 128   // Block 沿 N 轴的 Tile 大小
#define BK 8     // Block 沿 K 轴推进的步长
#define TM 8     // Thread 负责计算的 M 轴子块大小
#define TN 8     // Thread 负责计算的 N 轴子块大小

// ============================================================================
// 核函数：SGEMM 2D Block Tiling
// ============================================================================
__global__ void sgemm_2d_tiling_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) 
{
    const int tx = threadIdx.x; // 0 ~ 15
    const int ty = threadIdx.y; // 0 ~ 15
    const int tid = ty * blockDim.x + tx; // 线性线程 ID: 0 ~ 255

    const int c_row_start = blockIdx.y * BM;
    const int c_col_start = blockIdx.x * BN;

    __shared__ float As[BM][BK]; // 128 x 8  = 1024 个 float
    __shared__ float Bs[BK][BN]; // 8   x 128 = 1024 个 float

    float accum[TM][TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    for (int ph = 0; ph < (K + BK - 1) / BK; ++ph) 
    {
        // --- 协同搬运 A 矩阵子块 (128 x 8 = 1024 元素，256 线程循环 4 次) ---
        #pragma unroll
        for (int load_idx = 0; load_idx < 4; ++load_idx)
        {
            int tid_cur = tid + load_idx * 256;
            int a_row_cur = tid_cur / BK;
            int a_col_cur = tid_cur % BK;

            int a_row_global = c_row_start + a_row_cur;
            int a_col_global = a_col_cur + BK * ph;
            if (a_row_global<M && a_col_global<K) 
            {
                As[a_row_cur][a_col_cur] = A[a_row_global*K + a_col_global];
            }
            else
            {
                As[a_row_cur][a_col_cur] = 0.0f;
            }
            

        }
        

        // --- 协同搬运 B 矩阵子块 (8 x 128 = 1024 元素，256 线程循环 4 次) ---
        #pragma unroll
        for (int load_idx = 0; load_idx < 4; ++load_idx) {
            int tid_cur = tid + load_idx * 256; // 0 ~ 1023
            int b_row_cur = tid_cur / BN;      // 0 ~ 7
            int b_col_cur = tid_cur % BN;      // 0 ~ 127

            int b_row_global = ph * BK + b_row_cur;
            int b_col_global = c_col_start + b_col_cur;

            if (b_row_global < K && b_col_global < N) {
                Bs[b_row_cur][b_col_cur] = B[b_row_global * N + b_col_global];
            } else {
                Bs[b_row_cur][b_col_cur] = 0.0f;
            }
        }

        __syncthreads();

        // --- 2D Register ！！！！外积！！！！乘加计算 ---
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

    // --- 写回 Global Memory ---
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_r = c_row_start + ty * TM + i;
            int global_c = c_col_start + tx * TN + j;

            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = accum[i][j];
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
    // 设一个适合快速验证的尺寸 (也可替换为 1024, 2048 等大矩阵)
    const int M = 512;
    const int N = 512;
    const int K = 512;

    std::cout << "Matrix Dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

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

    // 2. Device 侧内存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice);

    // 3. Grid 与 Block 维度设置
    dim3 blockDim(BN / TN, BM / TM); // (16, 16) -> 256 Threads
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    // 4. Warmup + Performance Timing (CUDA Event)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    sgemm_2d_tiling_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // 正式计费执行
    cudaEventRecord(start);
    sgemm_2d_tiling_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 计算 TFLOPS: (2 * M * N * K) / (time_in_seconds * 1e12)
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (milliseconds / 1000.0)) / 1e12;

    std::cout << "GPU Kernel Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    // 5. 数据拷贝回 Host 并校验正确性
    cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost);

    std::cout << "Calculating CPU reference for verification..." << std::endl;
    cpu_sgemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    // 误差校验
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

    // 6. 释放显存与资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}