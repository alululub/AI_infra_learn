#include <chrono>   // C++ 高精度计时
#include <cmath>    // 数学函数 (expf, INFINITY)
#include <cstdlib>  // C 标准库 (malloc, free)
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>


#define warpSize 32  //GPU 架构的 warp 大小为 32

// CUDA 错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)



void softmax_cpu(const float* input, float* output, int batch_size, int num_classes) {
    for (int i = 0; i < batch_size; ++i) {
        float max_val = -INFINITY;
        for (int j = 0; j < num_classes; ++j) {
            max_val = std::max(max_val, input[i * num_classes + j]);
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += std::exp(input[i * num_classes + j] - max_val);
        }

        for (int j = 0; j < num_classes; ++j) {
            output[i * num_classes + j] = std::exp(input[i * num_classes + j] - max_val) / sum_exp;
        }
    }
}

__device__ float ReduceMax(float val)
{
    for (int stride = warpSize/2; stride > 0; stride/=2)
    {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, stride));
    }
    return val;
}

__device__ float ReduceSum(float val)
{
    for (int stride = warpSize/2; stride > 0; stride/=2)
    {
        val += __shfl_down_sync(0xffffffff, val, stride);
    }
    return val;
}

__global__ void softmax_online(const float *input, float *output, int N, int C)
{
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int warp_id   = tid / warpSize;      // 属于第几个 warp
    int warp_lane = tid % warpSize;      // 在 warp 中排第几个
    const float *x = input + idx * C;

    // ---- 第一遍: 在线找最大值 ----
    float max_thread = -INFINITY;
    for (int i = tid; i < C; i += block_size)
    {
        max_thread = fmaxf(max_thread, x[i]);
    }

    float max_warp = ReduceMax(max_thread);
    __shared__ float max_block[32];       // 最多支持 32 个 warp (1024 线程)
    if (warp_lane == 0)
    {
        max_block[warp_id] = max_warp;
    }
    __syncthreads();  // 确保所有 warp 的 max 已写入共享内存

    // 所有线程从共享内存读取并求全局最大值
    float max_val = -INFINITY;
    for (int i = 0; i < blockDim.x / warpSize; ++i)
    {
        max_val = fmaxf(max_val, max_block[i]);
    }

    // ---- 第二遍: 在线计算 exp 和 sum ----
    float sum_thread = 0.0f;
    for (int i = tid; i < C; i += block_size)
    {
        sum_thread += expf(x[i] - max_val);
    }

    float sum_warp = ReduceSum(sum_thread);

    // 跨 warp 归约: 将各 warp 的 sum 存入共享内存
    __shared__ float sum_block[32];
    if (warp_lane == 0)
    {
        sum_block[warp_id] = sum_warp;
    }
    __syncthreads();

    // 所有线程计算全局 sum (只遍历有效 warp 数)
    float sum = 0.0f;
    for (int i = 0; i < blockDim.x / warpSize; ++i)
    {
        sum += sum_block[i];
    }

    // ---- 第三遍: 写入最终结果 ----
    for (int i = tid; i < C; i += block_size)
    {
        output[idx * C + i] = expf(x[i] - max_val) / sum;
    }
}

bool compare_results(const float *cpu, const float *gpu, int N, int C,
                     float epsilon = 1e-3f) {
  for (int i = 0; i < N * C; ++i) {
    if (fabs(cpu[i] - gpu[i]) > epsilon) {  // 误差超过 0.001 就算不一致
      std::cout << "Difference at index " << i << ": CPU=" << cpu[i]
                << ", GPU=" << gpu[i] << ", diff=" << fabs(cpu[i] - gpu[i])
                << std::endl;
      return false;
    }
  }
  return true;
}

int main()
{

int N = 32;     // 32 行 (batch size = 32)
    int C = 4096;   // 每行 4096 个元素 (类别数)

    size_t num_elements = N * C;
    float *inp = (float *)malloc(num_elements * sizeof(float));
    float *out_cpu = (float *)malloc(num_elements * sizeof(float));
    float *out_gpu = (float *)malloc(num_elements * sizeof(float));

    // 初始化: 第 n 行的第 c 个元素 = c
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
        inp[n * C + c] = float(c);
        }
    }

    // ========== CPU 计时 ==========
    auto start_cpu = std::chrono::high_resolution_clock::now();
    softmax_cpu(inp, out_cpu, N, C);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    // ========== GPU 计时 (用 CUDA Events) ==========
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *d_out, *d_inp;
    cudaMalloc((void **)&d_out, N * C * sizeof(float));
    cudaMalloc((void **)&d_inp, N * C * sizeof(float));
    cudaMemcpy(d_inp, inp, N * C * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    int blockSize = 256;      // 每块 256 个线程
    int numBlocks = N;        // N 个块, 每个块处理一行
    softmax_online<<<numBlocks, blockSize>>>(d_inp, d_out, N, C);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);  // 等 GPU 跑完
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);  // 计算时间差

    cudaMemcpy(out_gpu, d_out, N * C * sizeof(float), cudaMemcpyDeviceToHost);

    // ========== 清理 GPU 资源 ==========
    cudaFree(d_out);
    cudaFree(d_inp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // ========== 对比结果 ==========
    bool success = compare_results(out_cpu, out_gpu, N, C);
    std::cout << "Results match: " << (success ? "YES ✓" : "NO ✗") << std::endl;

    // ========== 性能报告 ==========
    std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_time.count() / (gpu_time_ms)) << "x"
                << std::endl;

    // ========== 清理 CPU 资源 ==========
    free(inp);
    free(out_cpu);
    free(out_gpu);
    
    return 0;
}