#include <iostream>
#include <vector>
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

constexpr int TILE_SIZE = 128; // 每个 Tile 128 个 float

// =================================================================
// 1. Naive Kernel: 传统写法 (寄存器中转 + 串行等待)
// =================================================================
__global__ void naive_pipeline_kernel(
    const float* __restrict__ gmem_in, 
    float* __restrict__ gmem_out, 
    int num_tiles) 
{
    if (threadIdx.x >= 32) return;
    int tid = threadIdx.x;

    // 单份共享内存缓存
    __shared__ alignas(16) float smem_buffer[TILE_SIZE];

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // -------------------------------------------------------------
        // 痛点步骤 1: Global Memory -> 通用寄存器 (占用了 4 个局部 float 寄存器)
        // -------------------------------------------------------------
        float reg_temp[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            reg_temp[i] = gmem_in[tile_idx * TILE_SIZE + tid * 4 + i];
        }

        // -------------------------------------------------------------
        // 痛点步骤 2: 通用寄存器 -> Shared Memory
        // -------------------------------------------------------------
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            smem_buffer[tid * 4 + i] = reg_temp[i];
        }

        // 必须同步，等待所有线程把数据全部从中转寄存器写到 SMEM
        __syncthreads();

        // -------------------------------------------------------------
        // 步骤 3: 从 SMEM 读取并执行计算 (搬运与计算完全串行，无法掩盖延迟)
        // -------------------------------------------------------------
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[i] += smem_buffer[tid * 4 + i];
        }

        // 必须再次同步，防止下一次循环过早覆盖当前还在被读取的 SMEM
        __syncthreads();
    }

    // 写回结果
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        gmem_out[tid * 4 + i] = acc[i];
    }
}

// =================================================================
// 2. cp.async Kernel: 硬件 DMA 异步双缓冲流水线
// =================================================================
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        : : "r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__global__ void cp_async_pipeline_kernel(
    const float* __restrict__ gmem_in, 
    float* __restrict__ gmem_out, 
    int num_tiles) 
{
    if (threadIdx.x >= 32) return;
    int tid = threadIdx.x;

    // 双缓冲 Shared Memory
    __shared__ alignas(16) float smem_buffer[2][TILE_SIZE];

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Prologue: 预取第 0 块
    const void* gmem_ptr_0 = reinterpret_cast<const void*>(&gmem_in[tid * 4]);
    void* smem_ptr_0 = reinterpret_cast<void*>(&smem_buffer[0][tid * 4]);
    cp_async_16(smem_ptr_0, gmem_ptr_0);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    int read_stage = 0;

    // Main Loop: 边计算当前 Tile，边由硬件 DMA 异步搬运下一个 Tile
    for (int tile_idx = 0; tile_idx < num_tiles - 1; ++tile_idx) {
        int write_stage = read_stage ^ 1;

        // 异步搬运下一块数据 (绕过寄存器)
        const void* next_gmem_ptr = reinterpret_cast<const void*>(&gmem_in[(tile_idx + 1) * TILE_SIZE + tid * 4]);
        void* next_smem_ptr = reinterpret_cast<void*>(&smem_buffer[write_stage][tid * 4]);
        cp_async_16(next_smem_ptr, next_gmem_ptr);
        cp_async_commit();

        // 核心重叠：计算当前 read_stage 的数据
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[i] += smem_buffer[read_stage][tid * 4 + i];
        }

        // 等待下一块数据落地
        cp_async_wait<0>();
        __syncthreads();

        read_stage ^= 1;
    }

    // Epilogue: 结算最后一块
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc[i] += smem_buffer[read_stage][tid * 4 + i];
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        gmem_out[tid * 4 + i] = acc[i];
    }
}

// =================================================================
// 3. 性能测试与正确性比对
// =================================================================
int main() {
    const int NUM_TILES = 20000; // 迭代 20,000 个 Tile 放大延迟差距
    const int TOTAL_ELEMENTS = NUM_TILES * TILE_SIZE;

    std::vector<float> h_in(TOTAL_ELEMENTS, 1.0f);
    std::vector<float> h_out_naive(128, 0.0f);
    std::vector<float> h_out_async(128, 0.0f);

    float *d_in, *d_out_naive, *d_out_async;
    CUDA_CHECK(cudaMalloc(&d_in, TOTAL_ELEMENTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_naive, 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_async, 128 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), TOTAL_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm up
    naive_pipeline_kernel<<<1, 32>>>(d_in, d_out_naive, 100);
    cp_async_pipeline_kernel<<<1, 32>>>(d_in, d_out_async, 100);
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------------------------------------
    // 测试 1: Naive Kernel
    // -------------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start));
    naive_pipeline_kernel<<<1, 32>>>(d_in, d_out_naive, NUM_TILES);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_naive = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));

    // -------------------------------------------------------------
    // 测试 2: cp.async Pipeline Kernel
    // -------------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start));
    cp_async_pipeline_kernel<<<1, 32>>>(d_in, d_out_async, NUM_TILES);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_async = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_async, start, stop));

    // 结果验证
    CUDA_CHECK(cudaMemcpy(h_out_naive.data(), d_out_naive, 128 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_async.data(), d_out_async, 128 * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << ">>> 验证结果 (期望值均为 " << NUM_TILES << ".0):" << std::endl;
    std::cout << "Naive 输出 [0]: " << h_out_naive[0] << std::endl;
    std::cout << "Async 输出 [0]: " << h_out_async[0] << std::endl;

    std::cout << "\n>>> 性能对比 (" << NUM_TILES << " 次迭代):" << std::endl;
    std::cout << "Naive Kernel 耗时:    " << ms_naive << " ms" << std::endl;
    std::cout << "cp.async Kernel 耗时: " << ms_async << " ms" << std::endl;
    std::cout << "加速比 (Speedup):     " << (ms_naive / ms_async) << "x" << std::endl;

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out_naive));
    CUDA_CHECK(cudaFree(d_out_async));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}