#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

constexpr int TILE_DIM = 16;

// =================================================================
// 1. 单 GEMM Kernel (基础 Tiling + SMEM)
//    每次只负责计算单个矩阵乘法: C_single = A_single * B_single
// =================================================================
__global__ void single_gemm_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, int N, int K) 
{
    __shared__ float s_a[TILE_DIM][TILE_DIM];
    __shared__ float s_b[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // 加载 A 的分块到 SMEM
        if (row < M && (t * TILE_DIM + threadIdx.x) < K) {
            s_a[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_DIM + threadIdx.x];
        } else {
            s_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 加载 B 的分块到 SMEM
        if ((t * TILE_DIM + threadIdx.y) < K && col < N) {
            s_b[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
        } else {
            s_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// =================================================================
// 2. Batched GEMM Kernel (升维 3D Grid + SMEM)
//    blockIdx.z 负责定位具体的 Batch 索引，内部计算与单 GEMM 完全一致
// =================================================================
__global__ void batched_gemm_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, int N, int K,
    int stride_a, int stride_b, int stride_c) 
{
    // 关键核心：利用 blockIdx.z 瞬间完成当前 Batch 的首地址偏移
    int batch_idx = blockIdx.z;
    const float* A_batch = A + batch_idx * stride_a;
    const float* B_batch = B + batch_idx * stride_b;
    float* C_batch       = C + batch_idx * stride_c;

    __shared__ float s_a[TILE_DIM][TILE_DIM];
    __shared__ float s_b[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        if (row < M && (t * TILE_DIM + threadIdx.x) < K) {
            s_a[threadIdx.y][threadIdx.x] = A_batch[row * K + t * TILE_DIM + threadIdx.x];
        } else {
            s_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((t * TILE_DIM + threadIdx.y) < K && col < N) {
            s_b[threadIdx.y][threadIdx.x] = B_batch[(t * TILE_DIM + threadIdx.y) * N + col];
        } else {
            s_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C_batch[row * N + col] = acc;
    }
}

// =================================================================
// 3. 性能基准测试与对比
// =================================================================
int main() {
    // 模拟 Multi-Head Attention 中的批量场景:
    // 例如 Batch * Heads = 64, 每个头矩阵尺寸 128x128
    const int BATCH_SIZE = 10;
    const int M = 128;
    const int N = 128;
    const int K = 128;

    const int STRIDE_A = M * K;
    const int STRIDE_B = K * N;
    const int STRIDE_C = M * N;

    const size_t bytes_A = BATCH_SIZE * STRIDE_A * sizeof(float);
    const size_t bytes_B = BATCH_SIZE * STRIDE_B * sizeof(float);
    const size_t bytes_C = BATCH_SIZE * STRIDE_C * sizeof(float);

    std::vector<float> h_A(BATCH_SIZE * STRIDE_A, 1.0f);
    std::vector<float> h_B(BATCH_SIZE * STRIDE_B, 2.0f);
    std::vector<float> h_C_naive(BATCH_SIZE * STRIDE_C, 0.0f);
    std::vector<float> h_C_batched(BATCH_SIZE * STRIDE_C, 0.0f);

    float *d_A, *d_B, *d_C_naive, *d_C_batched;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C_naive, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_batched, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    dim3 block(TILE_DIM, TILE_DIM); // 16x16 = 256 线程
    dim3 grid_2d((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM); // (8, 8)
    dim3 grid_3d(grid_2d.x, grid_2d.y, BATCH_SIZE); // (8, 8, 64)

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    batched_gemm_kernel<<<grid_3d, block>>>(d_A, d_B, d_C_batched, M, N, K, STRIDE_A, STRIDE_B, STRIDE_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------------------------------------
    // 测试 1: Naive 方案 (Host 循环下发 B 次 Kernel)
    // -------------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start));
    for (int b = 0; b < BATCH_SIZE; ++b) {
        const float* pA = d_A + b * STRIDE_A;
        const float* pB = d_B + b * STRIDE_B;
        float* pC       = d_C_naive + b * STRIDE_C;
        single_gemm_kernel<<<grid_2d, block>>>(pA, pB, pC, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_naive = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));

    // -------------------------------------------------------------
    // 测试 2: Batched GEMM 方案 (一次 Kernel Launch 铺满 GPU)
    // -------------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start));
    batched_gemm_kernel<<<grid_3d, block>>>(
        d_A, d_B, d_C_batched, M, N, K, STRIDE_A, STRIDE_B, STRIDE_C);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_batched = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_batched, start, stop));

    // -------------------------------------------------------------
    // 结果正确性比对
    // -------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_C_naive.data(), d_C_naive, bytes_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_batched.data(), d_C_batched, bytes_C, cudaMemcpyDeviceToHost));

    bool match = true;
    for (size_t i = 0; i < h_C_naive.size(); ++i) {
        if (std::fabs(h_C_naive[i] - h_C_batched[i]) > 1e-4) {
            match = false;
            break;
        }
    }

    std::cout << ">>> 验证结果: " << (match ? "PASS (两者输出完全一致)" : "FAIL") << std::endl;
    std::cout << "期望值验证 (1.0 * 2.0 * 128 = 256.0): " << h_C_batched[0] << std::endl;

    std::cout << "\n>>> 性能对比 (Batch=" << BATCH_SIZE << ", " 
              << M << "x" << N << "x" << K << "):" << std::endl;
    std::cout << "Naive 方案耗时 (Host 循环 64 次启动): " << ms_naive << " ms" << std::endl;
    std::cout << "Batched GEMM 耗时 (一次 3D Grid 启动): " << ms_batched << " ms" << std::endl;
    std::cout << "加速比 (Speedup):                    " << (ms_naive / ms_batched) << "x" << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_naive));
    CUDA_CHECK(cudaFree(d_C_batched));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}