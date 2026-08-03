#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================================
// 1. 编译期超参数定义 (Tile 维度)
// ============================================================================
// 每个 Thread Block 负责计算 C 矩阵中 128 x 128 的大分块
#define BM 128
#define BN 128
#define BK 8

// 每个 Thread 负责计算 C 矩阵中 8 x 8 的小分块（寄存器 Tile）
#define TM 8
#define TN 8

// 检查线程配置逻辑：(128/8) * (128/8) = 16 x 16 = 256 线程/Block
// 256 线程协同搬运 As (128x8=1024 元素) -> 每人搬 4 个
// 256 线程协同搬运 Bs (8x128=1024 元素) -> 每人搬 4 个

// ============================================================================
// 2. CUDA Kernel: 2D Block Tiling SGEMM
// ============================================================================
__global__ void sgemm_2d_block_tiling(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int N, int K) 
{
    // --- 线程与 Block 的坐标推导 ---
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x; // 0 ~ 15
    const int ty = threadIdx.y; // 0 ~ 15

    const int c_row_start = by * BM;
    const int c_col_start = bx * BN;

    const int thread_row_in_block = ty * TM;
    const int thread_col_in_block = tx * TN;

    // --- 片上 Shared Memory 与 线程私有寄存器 (Registers) ---
    __shared__ float As[BM][BK]; // 128 x 8
    __shared__ float Bs[BK][BN]; // 8 x 128

    float accum[TM][TN] = {0.0f}; // 累加器，存储在寄存器中
    float reg_a[TM] = {0.0f};     // 缓存当前 K 步下的 A 向量
    float reg_b[TN] = {0.0f};     // 缓存当前 K 步下的 B 向量

    // --- 协同搬运逻辑：将 2D 线程映射到 1D 展平索引 ---
    const int tid = ty * blockDim.x + tx; // 0 ~ 255

    const int a_tile_row = tid / BK; // 0 ~ 127
    const int a_tile_col = tid % BK; // 0 ~ 7

    const int b_tile_row = tid / BN; // 0 ~ 7
    const int b_tile_col = tid % BN; // 0 ~ 127

    // --- 主循环：沿着 K 维度分块推进 ---
    for (int ph = 0; ph < (K + BK - 1) / BK; ++ph) {

        // 1. 全局显存 -> Shared Memory 搬运 (A 矩阵)
        int a_global_r = c_row_start + a_tile_row;
        int a_global_c = ph * BK + a_tile_col;
        if (a_global_r < M && a_global_c < K) {
            As[a_tile_row][a_tile_col] = A[a_global_r * K + a_global_c];
        } else {
            As[a_tile_row][a_tile_col] = 0.0f;
        }

        // 2. 全局显存 -> Shared Memory 搬运 (B 矩阵，满足合并访存)
        int b_global_r = ph * BK + b_tile_row;
        int b_global_c = c_col_start + b_tile_col;
        if (b_global_r < K && b_global_c < N) {
            Bs[b_tile_row][b_tile_col] = B[b_global_r * N + b_global_c];
        } else {
            Bs[b_tile_row][b_tile_col] = 0.0f;
        }

        __syncthreads(); // 等待 Shared Memory 写入完毕

        // 3. 寄存器级分块计算
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 从 Shared Memory 读取到 Register
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As[thread_row_in_block + i][k];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = Bs[k][thread_col_in_block + j];
            }

            // 在寄存器内点积外积累加 (FFMA)
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads(); // 等待所有线程完成读取，防止写后读冲突
    }

    // --- 4. 写回 Global Memory ---
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_r = c_row_start + thread_row_in_block + i;
            int global_c = c_col_start + thread_col_in_block + j;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = accum[i][j];
            }
        }
    }
}

// ============================================================================
// 3. Host 端 CPU 黄金参考实现 (用于比对结果正确性)
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

// 结果验证
bool verify_result(const float* host_c, const float* device_c, int num_elements, float tol = 1e-3f) {
    for (int i = 0; i < num_elements; ++i) {
        if (std::abs(host_c[i] - device_c[i]) > tol) {
            std::cout << "验证失败! 索引 " << i 
                      << " CPU=" << host_c[i] 
                      << " GPU=" << device_c[i] << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// 4. Main 主函数
// ============================================================================
int main() {
    // 矩阵规模定义 (可任意按需修改)
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    std::cout << "正在初始化矩阵: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // 1. Host 内存分配
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_device(M * N);
    std::vector<float> h_C_cpu(M * N);

    // 2. 随机初始化输入数据
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // 3. Device 显存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    // 4. 拷贝数据到 GPU
    cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice);

    // 5. 配置 Kernel 网格与线程块维度
    dim3 blockDim(BN / TN, BM / TM); // (128/8, 128/8) = (16, 16) = 256 线程
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM); // (8, 8)

    // 6. 预热运行 (Warmup)
    sgemm_2d_block_tiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // 7. 性能测试 (使用 CUDA Event 测毫秒数)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    const int iterations = 10;
    for (int i = 0; i < iterations; ++i) {
        sgemm_2d_block_tiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iterations;

    // 计算 TFLOPS: 2 * M * N * K 次浮点运算
    double flops = 2.0 * static_cast<double>(M) * N * K;
    double tflops = (flops * 1e-12) / (avg_ms * 1e-3);

    std::cout << "GPU 平均耗时: " << avg_ms << " ms" << std::endl;
    std::cout << "GPU 计算性能: " << tflops << " TFLOPS" << std::endl;

    // 8. 结果正确性校验
    std::cout << "正在运行 CPU 黄金参考进行比对校验..." << std::endl;
    cudaMemcpy(h_C_device.data(), d_C, bytes_C, cudaMemcpyDeviceToHost);
    cpu_sgemm(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);

    if (verify_result(h_C_cpu.data(), h_C_device.data(), M * N)) {
        std::cout << ">> 校验通过！计算结果完全正确 (PASSED) <<" << std::endl;
    }

    // 9. 释放资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}