#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h> 

using namespace nvcuda;

// 定义 WMMA Tile 几何尺寸
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// -----------------------------------------------------------------
// WMMA GEMM Kernel: D = A * B + C
// -----------------------------------------------------------------
__global__ void wmma_gemm_kernel(
    const half* __restrict__ A, 
    const half* __restrict__ B, 
    float* __restrict__ C, 
    int M, int N, int K) 
{
    // 每个 Block 包含 4 个 Warp (共 128 线程)
    // Warp 网格按 2x2 划分：每个 Block 计算 (2*16) x (2*16) = 32x32 的结果
    const int warp_id = threadIdx.x / 32;
    const int warp_row = warp_id / 2; // 0 或 1
    const int warp_col = warp_id % 2; // 0 或 1

    // 计算当前 Warp 负责计算的 C 矩阵全局行/列坐标
    const int c_row = blockIdx.y * (2 * WMMA_M) + warp_row * WMMA_M;
    const int c_col = blockIdx.x * (2 * WMMA_N) + warp_col * WMMA_N;

    // 边界检查
    if (c_row >= M || c_col >= N) return;

    // -------------------------------------------------------------
    // Step 1: 声明 Fragment (寄存器容器)
    // -------------------------------------------------------------
    // A 矩阵切片: 16x16x16, half 类型, 行优先 (row_major)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;

    // B 矩阵切片: 16x16x16, half 类型, 行优先 (row_major)
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

    // 累加器切片: 16x16x16, float 类型
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // 初始化累加器为 0
    wmma::fill_fragment(acc_frag, 0.0f);

    // -------------------------------------------------------------
    // Step 2 & 3: 沿 K 轴切片推进并执行 Tensor Core 乘加
    // -------------------------------------------------------------
    for (int k = 0; k < K; k += WMMA_K) {
        // 从 Global Memory 读取数据到 Fragment
        // 参数: fragment, 起始内存地址, 跨度(Stride / Leading Dimension)
        wmma::load_matrix_sync(a_frag, A + c_row * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + c_col, N);

        // 驱动 Tensor Core 执行硬件矩阵乘累加: acc = A * B + acc
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // -------------------------------------------------------------
    // Step 4: 将结果写回内存 (Global Memory)
    // -------------------------------------------------------------
    wmma::store_matrix_sync(C + c_row * N + c_col, acc_frag, N, wmma::mem_row_major);
}

// -----------------------------------------------------------------
// Host 端验证与主函数
// -----------------------------------------------------------------
int main() {
    int M = 256, N = 256, K = 256;
    std::cout << "Running WMMA Tensor Core Demo [M=" << M << ", N=" << N << ", K=" << K << "]..." << std::endl;

    size_t bytes_A = M * K * sizeof(half);
    size_t bytes_B = K * N * sizeof(half);
    size_t bytes_C = M * N * sizeof(float);

    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);

    // 初始化测试数据
    for (int i = 0; i < M * K; ++i) h_A[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);

    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    // 启动配置：每个 Block 128 个线程（4 个 Warp）
    dim3 block(128); 
    // 每个 Block 计算 32x32 的矩阵块
    dim3 grid((N + 31) / 32, (M + 31) / 32);

    wmma_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    // CPU 简易校验前 5 个元素
    std::cout << "Verification (first 5 values):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        float ref = 0.0f;
        for (int k = 0; k < K; ++k) {
            ref += __half2float(h_A[i * K + k]) * __half2float(h_B[k * N + 0]);
        }
        std::cout << "  Idx " << i << ": GPU=" << h_C[i * N] << ", CPU_ref=" << ref << std::endl;
    }

    std::cout << "Tensor Core WMMA Demo completed successfully!" << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}