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

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

// =================================================================
// Split-K GEMM Kernel:
// gridDim.x = (N + TILE_N - 1) / TILE_N
// gridDim.y = (M + TILE_M - 1) / TILE_M
// gridDim.z = SPLIT_K (把 K 轴切分成 SPLIT_K 份并发)
// =================================================================
__global__ void split_k_gemm_atomic_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int split_k)
{
    // 1. 获取 3D Grid 索引
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;
    int k_slice_id = blockIdx.z; // 当前 Block 负责第几段 K 轴

    // 2. 计算当前 Block 负责的 K 轴起止区间 [k_start, k_end)
    int k_chunk = (K + split_k - 1) / split_k;
    int k_start = k_slice_id * k_chunk;
    int k_end   = min(k_start + k_chunk, K);

    // 3. 片上 SRAM 分块
    __shared__ float s_a[TILE_M][TILE_K];
    __shared__ float s_b[TILE_K][TILE_N];

    float partial_acc = 0.0f;

    // 4. 只在自己的 K 轴片段内进行滑动累加
    for (int t = k_start; t < k_end; t += TILE_K) {
        // 协同搬运 A 的分块
        int a_col = t + threadIdx.x;
        if (row < M && a_col < k_end) {
            s_a[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            s_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 协同搬运 B 的分块
        int b_row = t + threadIdx.y;
        if (b_row < k_end && col < N) {
            s_b[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            s_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 内积展开
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            partial_acc += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    // 5. 跨 Block 归约：将部分和原子累加回全局显存的目标位置 C[row, col]
    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], partial_acc);
    }
}


int main() {
    const int M = 16;
    const int N = 16;
    const int K = 4096;          // 模拟极长 K 轴场景
    const int SPLIT_K = 4;       // 沿 K 轴切分成 4 份并发

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);
    std::vector<float> h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    // 关键：原子累加前必须将输出缓冲区清零！
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, 
              (M + TILE_M - 1) / TILE_M, 
              SPLIT_K); // 3D Grid 升维调度

    split_k_gemm_atomic_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, SPLIT_K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // 验证：1.0 * 2.0 * 4096 = 8192.0
    std::cout << "验证结果 C[0]: " << h_C[0] << " (预期: 8192.0)" << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}